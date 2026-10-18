//! Native simulation adapters for the standalone GPUI demo applications.

// Rust guideline compliant 2026-02-21
mod adapters;
mod catalog;
#[expect(
    dead_code,
    reason = "shared browser module includes helpers unused by the worker"
)]
#[path = "../../../../apps/sim-demo/src/data.rs"]
mod data;
use std::error::Error;
use std::time::Instant;

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).is_some_and(|s| s == "--catalog") {
        println!(
            "{}",
            serde_json::to_string(&catalog::app(args.get(2).ok_or("Missing simulator")?)?)?
        );
        return Ok(());
    }
    let input = args
        .get(1)
        .ok_or("usage: sonium-demo-workers INPUT.json OUTPUT.json [--validate]")?;
    let output = args.get(2).ok_or("Missing output path")?;
    let project: data::Project = serde_json::from_slice(&std::fs::read(input)?)?;
    catalog::app(&project.simulator)?.validate(&project)?;
    adapters::validate(&project)?;
    if args.iter().any(|s| s == "--validate") {
        return Ok(());
    }
    if std::env::var_os("SONIUM_JOB_PARENT").is_some() {
        std::thread::spawn(|| {
            let _ = std::io::copy(&mut std::io::stdin().lock(), &mut std::io::sink());
            std::process::exit(130);
        });
    }
    rayon::ThreadPoolBuilder::new()
        .num_threads(project.cpu_threads)
        .build_global()?;
    let start = Instant::now();
    println!(
        "{}",
        serde_json::json!({"progress":0.01,"label":"Native solver started"})
    );
    let mut result = adapters::run(&project)?;
    result.elapsed_s = start.elapsed().as_secs_f64();
    result.validate()?;
    std::fs::write(output, serde_json::to_vec(&result)?)?;
    println!(
        "{}",
        serde_json::json!({"progress":1.0,"label":"Validated scientific result"})
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn every_catalog_study_produces_valid_native_outputs() {
        for simulator in ["room", "pcfan", "propeller", "driver", "structural", "flow"] {
            let app = catalog::app(simulator).unwrap();
            for i in 0..app.studies.len() {
                let project = app.project(i);
                app.validate(&project).unwrap();
                adapters::validate(&project).unwrap();
                let result = adapters::run(&project).unwrap();
                result.validate().unwrap();
                assert_eq!(result.project, project);
                assert!(!result.curves.is_empty());
                if simulator == "structural" && project.study == "bracket" {
                    assert!(
                        result
                            .metrics
                            .iter()
                            .find(|m| m.label == "Energy balance error")
                            .unwrap()
                            .value
                            < 1e-8
                    );
                }
            }
        }
    }
    #[test]
    fn rejects_cross_simulator_gpu_and_corrupt_scientific_results() {
        let driver = catalog::app("driver").unwrap();
        let mut project = driver.project(0);
        assert!(catalog::app("room").unwrap().validate(&project).is_err());
        project.backend = "cuda".into();
        assert!(driver.validate(&project).is_err());
        project.backend = "cpu".into();
        project.parameters.insert("mass".into(), "NaN".into());
        assert!(driver.validate(&project).is_err());
        let mut data = adapters::run(&driver.project(0)).unwrap();
        data.surfaces[0].mesh.triangles[0][0] = u32::MAX;
        assert!(data.validate().is_err());
        let mut data = adapters::run(&driver.project(0)).unwrap();
        data.surfaces[0].scalar_frames[0].pop();
        assert!(data.validate().is_err());
        let mut data = adapters::run(&driver.project(0)).unwrap();
        data.curves[0].series[0].values[0] = f64::INFINITY;
        assert!(data.validate().is_err());
    }
}
