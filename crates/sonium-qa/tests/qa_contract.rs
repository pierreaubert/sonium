use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn manifest_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn workspace_justfile() -> PathBuf {
    manifest_dir()
        .parent()
        .and_then(Path::parent)
        .expect("sonium-qa must live two levels below workspace root")
        .join("Justfile")
}

fn registered_bins() -> BTreeMap<PathBuf, String> {
    let manifest_path = manifest_dir().join("Cargo.toml");
    let manifest: toml::Value = fs::read_to_string(&manifest_path)
        .expect("read sonium-qa Cargo.toml")
        .parse()
        .expect("parse sonium-qa Cargo.toml");

    manifest["bin"]
        .as_array()
        .expect("Cargo.toml must contain [[bin]] targets")
        .iter()
        .map(|target| {
            let path = target["path"].as_str().expect("binary path");
            let name = target["name"].as_str().expect("binary name");
            (PathBuf::from(path), name.to_owned())
        })
        .collect()
}

fn qa_source_files() -> BTreeSet<PathBuf> {
    fs::read_dir(manifest_dir().join("bin"))
        .expect("read QA bin directory")
        .map(|entry| entry.expect("read QA source entry").path())
        .filter(|path| path.extension().is_some_and(|extension| extension == "rs"))
        .map(|path| PathBuf::from("bin").join(path.file_name().expect("QA source filename")))
        .collect()
}

#[test]
fn every_qa_source_is_a_registered_binary() {
    let registered: BTreeSet<_> = registered_bins().into_keys().collect();
    let sources = qa_source_files();
    assert_eq!(
        sources, registered,
        "every standalone QA source must be registered; remove obsolete sources instead of silently excluding them"
    );
}

#[test]
fn deprecated_monolithic_fem_qa_is_retired() {
    let legacy_source = Path::new("bin/fem_old_qa.rs");
    let registered = registered_bins();

    assert!(
        !manifest_dir().join(legacy_source).exists(),
        "the deprecated monolithic FEM QA duplicates focused qa-fem-* binaries, takes more than five minutes, and accepts 20-100% relative errors; remove it instead of weakening qa-full"
    );
    assert!(
        !registered.contains_key(legacy_source),
        "qa-old-fem must not remain registered after its obsolete source is retired"
    );
}

#[test]
fn every_registered_binary_has_a_machine_checkable_output_contract() {
    let mut violations = Vec::new();
    let fem_3d_runner_path = manifest_dir().join("src/fem_physics_3d.rs");
    assert!(
        fem_3d_runner_path.exists(),
        "shared FEM 3D QA runner {} is missing: thin qa-fem-* binaries delegate \
         their SUCCESS/FAILED contract to it, so restore the runner instead of \
         weakening this contract test",
        fem_3d_runner_path.display()
    );
    let fem_3d_runner = fs::read_to_string(&fem_3d_runner_path)
        .expect("shared FEM 3D QA runner exists but is unreadable; check file permissions");
    let fem_3d_contract = fem_3d_runner.contains("SUCCESS")
        && fem_3d_runner.contains("FAILED")
        && fem_3d_runner.contains("process::exit(1)");
    // The OpenCourant parity binaries delegate publication to the shared
    // rad gate, which prints SUCCESS/FAILED/SKIPPED and exits accordingly.
    // Accept that delegation only while the gate itself carries the full
    // contract, mirroring the FEM 3D runner exception above.
    let rad_gate_path = manifest_dir().join("src/rad_gate.rs");
    assert!(
        rad_gate_path.exists(),
        "shared rad gate {} is missing: qa-fem-opencourant-* binaries delegate \
         their SUCCESS/FAILED contract to it, so restore the gate instead of \
         weakening this contract test",
        rad_gate_path.display()
    );
    let rad_gate = fs::read_to_string(&rad_gate_path)
        .expect("shared rad gate exists but is unreadable; check file permissions");
    let rad_gate_contract = rad_gate.contains("SUCCESS")
        && rad_gate.contains("FAILED")
        && rad_gate.contains("process::exit(1)");
    for (relative_path, name) in registered_bins() {
        let source = fs::read_to_string(manifest_dir().join(&relative_path))
            .unwrap_or_else(|error| panic!("read {}: {error}", relative_path.display()));
        // The 3D FEM entries are intentionally thin production-form wrappers.
        // Their shared runner owns the structured record and process contract;
        // accepting the wrapper only when that exact runner has the complete
        // contract keeps the check strict without duplicating boilerplate in
        // every physics binary.
        let uses_fem_3d_runner = source.contains("sonium_qa::fem_physics_3d::run_");
        let uses_rad_gate = source.contains("rad::gate::publish_and_exit");
        // A registered binary may also be a thin wrapper over a sibling
        // source pulled in with #[path]; the included file then owns the
        // output contract.
        let path_included_contract = source
            .lines()
            .filter_map(|line| {
                let start = line.find("#[path = \"")? + "#[path = \"".len();
                let rest = &line[start..];
                let end = rest.find('"')?;
                Some(rest[..end].to_string())
            })
            .filter_map(|included| {
                let included_path = manifest_dir().join(&relative_path).parent()?.join(included);
                fs::read_to_string(included_path).ok()
            })
            .any(|included| {
                included.contains("SUCCESS")
                    && included.contains("FAILED")
                    && included.contains("process::exit(1)")
            });
        let delegated_contract = (uses_fem_3d_runner && fem_3d_contract)
            || (uses_rad_gate && rad_gate_contract)
            || path_included_contract;
        let is_compile_only_contract = source.contains("Compile-only");
        let has_success = source.contains("SUCCESS")
            || source.contains("QaResult::passed")
            || is_compile_only_contract
            || delegated_contract;
        let has_failed = source.contains("FAILED")
            || source.contains("QaResult::failed")
            || is_compile_only_contract
            || delegated_contract;
        let has_failure_path = source.contains("process::exit(1)")
            || source.contains("ExitCode::FAILURE")
            || source.contains("return Err(")
            || source.contains("assert!(")
            || source.contains("assert_eq!(")
            || is_compile_only_contract
            || delegated_contract;
        if !has_success {
            violations.push(format!("{name}: missing SUCCESS marker"));
        }
        if !has_failed {
            violations.push(format!("{name}: missing FAILED marker"));
        }
        if !has_failure_path {
            violations.push(format!("{name}: missing non-zero failure path"));
        }
    }

    assert!(
        violations.is_empty(),
        "QA output-contract violations:\n{}",
        violations.join("\n")
    );
}

#[test]
fn full_runner_is_manifest_driven_and_rejects_any_failure_marker() {
    let justfile_path = workspace_justfile();
    let justfile = fs::read_to_string(&justfile_path).expect("read sonium-qa Justfile");

    assert!(
        justfile.contains("qa-full: _qa-run-all"),
        "qa-full must be driven from all [[bin]] targets, including BLR and application QA"
    );
    assert!(
        justfile.contains("if \"FAILED\" in result.stdout:"),
        "the runner must reject every FAILED diagnostic, including indented solver failures"
    );
    assert!(
        justfile.contains("SONIUM_OUTPUT_DIR"),
        "the runner must route generated artifacts to a known writable directory"
    );
    assert!(
        !justfile.contains("line.strip() == \"FAILED\""),
        "line-anchored failure detection can hide per-case non-convergence"
    );
    assert!(
        justfile.contains("unsupported QA result status"),
        "the runner must fail closed on unknown structured-result statuses"
    );
    assert!(
        justfile.contains("if \"SUCCESS\" not in result.stdout and embedded_status != \"passed\":"),
        "the runner must accept a passing structured result without requiring a legacy SUCCESS marker"
    );
    assert!(
        justfile.contains("payload.startswith(\"{\")"),
        "the runner must recognize legacy bare structured QA result lines"
    );
    assert!(
        justfile.contains("case_policy or {\"id\": structured.get(\"case_id\")}"),
        "unmanifested binaries must validate their own structured result identity"
    );
    assert!(
        justfile.contains("threshold_directions")
            && justfile.contains("unsatisfied or ambiguous QA threshold"),
        "the runner must evaluate explicit structured threshold directions"
    );
}

#[test]
fn release_runner_carries_manifest_evidence_into_archived_records() {
    let justfile = fs::read_to_string(workspace_justfile()).expect("read sonium-qa Justfile");

    for field in [
        "case_metadata",
        "reference_checksum",
        "formulation",
        "base_evidence",
        "artifact_checksums",
        "evidence_errors",
    ] {
        assert!(
            justfile.contains(field),
            "release runner must preserve manifest/evidence field {field:?}"
        );
    }
    assert!(
        justfile.contains("case.get(\"id\") == expected_case_id"),
        "explicit case IDs must receive manifest provenance even when the binary name is ambiguous"
    );
}

#[test]
fn manifest_inventory_is_generated_and_checked_by_the_qa_recipe() {
    let root = manifest_dir();
    let readme = fs::read_to_string(root.join("README.md")).expect("read QA README");
    let inventory = fs::read_to_string(root.join("manifest-inventory.md"))
        .expect("read generated QA manifest inventory");
    let justfile = fs::read_to_string(workspace_justfile()).expect("read workspace Justfile");

    assert!(
        readme.contains("[manifest inventory](manifest-inventory.md)"),
        "the QA README must link to the generated manifest inventory instead of hard-coding counts"
    );
    assert!(
        inventory.contains("Generated by scripts/generate_qa_inventory.py"),
        "the inventory must declare its generator"
    );
    assert!(
        justfile.contains("qa-inventory:")
            && justfile.contains("python3 scripts/generate_qa_inventory.py"),
        "the workspace Justfile must expose the stale-inventory check"
    );
    let status = Command::new("python3")
        .arg(
            workspace_justfile()
                .parent()
                .expect("workspace root")
                .join("scripts/generate_qa_inventory.py"),
        )
        .status()
        .expect("run the manifest-inventory freshness check");
    assert!(status.success(), "the generated QA inventory is stale");
}

#[test]
fn native_speaker_workflow_requires_native_aarch64_reference_before_x86() {
    let workflow = fs::read_to_string(
        workspace_justfile()
            .parent()
            .expect("workspace root")
            .join(".gitea/workflows/sonium-sim-speaker-x86.yml"),
    )
    .expect("read native speaker workflow");

    for contract in [
        "native-aarch64-reference:",
        "runs-on: [self-hosted, macos, metal]",
        "test \"$(uname -m)\" = \"arm64\"",
        "test \"$(rustc -vV | sed -n 's/^host: //p')\" = \"aarch64-apple-darwin\"",
        "SONIUM_AARCH64_REFERENCE_ONLY: \"1\"",
        "name: sonium-speaker-native-aarch64-reference",
        "native-x86-64:",
        "needs: native-aarch64-reference",
        "SONIUM_REQUIRE_AARCH64_GOLDEN: \"1\"",
    ] {
        assert!(
            workflow.contains(contract),
            "native speaker workflow is missing required contract {contract:?}"
        );
    }
    assert_eq!(
        workflow.matches("timeout-minutes: 20").count(),
        2,
        "both native speaker workflow jobs must have a bounded timeout"
    );
    assert!(
        workflow.contains(
            "jq -e '(.dispatch | contains([\"simd=neon\"])) and any(.dispatch[]; startswith(\"vendor=\") and (.[7:] | length > 0) and ((.[7:] | ascii_downcase) != \"unknown\"))'"
        ),
        "the native speaker workflow must validate non-placeholder aarch64 dispatch metadata"
    );
}

#[test]
fn accelerator_workflow_consumes_the_current_native_speaker_schema() {
    let workflow = fs::read_to_string(
        workspace_justfile()
            .parent()
            .expect("workspace root")
            .join(".gitea/workflows/compute-accelerators.yml"),
    )
    .expect("read accelerator workflow");

    assert!(
        workflow.contains(
            "test \"$(jq -r .schema_version target/sonium-aarch64-speaker-evidence/speaker_native_x86_golden.json)\" = \"3\""
        ),
        "the accelerator fan-in must validate the current schema-v3 native speaker artifact"
    );
    assert!(
        workflow.contains("sha256sum -c speaker_native_x86_golden.json.sha256"),
        "the accelerator fan-in must verify the accepted native speaker artifact checksum"
    );
    assert!(
        workflow.contains("command -v sha256sum")
            && workflow.contains("shasum -a 256 speaker_native_x86_golden.json")
            && workflow.contains("shasum -a 256 -c speaker_native_x86_golden.json.sha256"),
        "the native macOS speaker producer must fall back to shasum when sha256sum is unavailable"
    );
    assert!(
        workflow.contains(
            "test \"$(jq -r '.projected_driver_model_reference.schema_version' target/sonium-aarch64-speaker-evidence/speaker_native_x86_golden.json)\" = \"2\""
        ),
        "the accelerator fan-in must validate projected DriverModel promotion evidence"
    );
    assert!(
        workflow.contains(
            "jq -e '(.dispatch | contains([\"simd=neon\"])) and any(.dispatch[]; startswith(\"vendor=\") and (.[7:] | length > 0) and ((.[7:] | ascii_downcase) != \"unknown\"))'"
        ),
        "the native aarch64 lane must validate NEON dispatch and a real CPU identity"
    );
    for contract in [
        "test \"$(jq -r .schema_version \"$artifact\")\" = \"3\"",
        "test \"$(jq -r .target_arch \"$artifact\")\" = \"x86_64\"",
        "contains([\"avx2=true\"])",
        "contains([\"fma=true\"])",
    ] {
        assert!(
            workflow.contains(contract),
            "the accelerator x86 speaker lane is missing artifact identity contract {contract:?}"
        );
    }
}

#[test]
fn release_evidence_records_reproducible_artifact_checksums() {
    let justfile_path = workspace_justfile();
    let justfile = fs::read_to_string(&justfile_path).expect("read sonium-qa Justfile");

    assert!(
        justfile.contains("hashlib.sha256"),
        "release evidence must use a cryptographic digest"
    );
    assert!(
        justfile.contains("artifact_checksums"),
        "release evidence must index structured artifacts"
    );
    assert!(
        justfile.contains("checksums.sha256"),
        "release evidence must emit a verifiable checksum inventory"
    );
    assert!(
        justfile.contains("checksum_errors"),
        "checksum failures must be visible in release evidence"
    );
    assert!(
        justfile.contains("nonpassing_required_cases"),
        "release evidence must record required cases that did not pass"
    );
    assert!(
        justfile.contains("threshold_failures"),
        "release evidence must record metrics that violate declared thresholds"
    );
    assert!(
        justfile.contains("SONIUM_OUTPUT_DIR")
            && justfile.contains("artifact_dir = output / \"artifacts\""),
        "release evidence must route result-declared artifacts into its durable bundle"
    );
    assert!(
        justfile.contains("declared artifact is missing")
            && justfile.contains("artifact path escapes bundle")
            && justfile.contains("artifact_errors"),
        "release evidence must fail closed on missing or escaping result-declared artifacts"
    );
}

#[test]
fn release_runner_supports_bounded_parallel_case_execution() {
    let justfile = fs::read_to_string(workspace_justfile()).expect("read workspace Justfile");
    assert!(
        justfile.contains("SONIUM_QA_PARALLELISM")
            && justfile.contains("ThreadPoolExecutor")
            && justfile.contains("as_completed"),
        "release runner must expose bounded parallel case execution"
    );
    assert!(
        justfile.contains("stdout=subprocess.PIPE") && justfile.contains("print(output, end=\"\")"),
        "parallel release runner must preserve deterministic case output"
    );
}

#[test]
fn release_runner_scopes_room_correlation_resource_defaults_to_the_child() {
    let justfile = fs::read_to_string(workspace_justfile()).expect("read workspace Justfile");
    assert!(
        justfile.contains("case[\"bin\"] == \"qa-room-cross-backend-correlation\"")
            && justfile.contains("case_env.setdefault(\"RUST_MIN_STACK\", \"33554432\")")
            && justfile.contains("case_env.setdefault(\"RAYON_NUM_THREADS\", \"2\")")
            && justfile.contains("env=case_env"),
        "RoomSim fan-in resource defaults must be child-local and override-preserving"
    );
}

#[test]
fn release_runner_preserves_declared_features_for_manifest_cases() {
    let justfile = fs::read_to_string(workspace_justfile()).expect("read workspace Justfile");
    let manifest: toml::Value = fs::read_to_string(manifest_dir().join("validation-manifest.toml"))
        .expect("read validation manifest")
        .parse()
        .expect("parse validation manifest");
    let cargo: toml::Value = fs::read_to_string(manifest_dir().join("Cargo.toml"))
        .expect("read sonium-qa Cargo.toml")
        .parse()
        .expect("parse sonium-qa Cargo.toml");

    assert!(
        justfile.contains("features = case.get(\"features\", [])")
            && justfile.contains("_qa-run-features")
            && justfile.contains("\",\".join(features)"),
        "qa-release-critical must execute optional-dependency cases through the feature-aware runner"
    );

    for case in manifest["case"]
        .as_array()
        .expect("manifest must contain cases")
    {
        let Some(features) = case.get("features") else {
            continue;
        };
        for feature in features.as_array().expect("case features array") {
            let feature = feature.as_str().expect("case feature name");
            assert!(
                cargo["features"].get(feature).is_some(),
                "{} declares unknown Cargo feature {feature}",
                case["id"].as_str().unwrap_or("<unknown>")
            );
        }
    }
}

#[test]
fn registered_binary_paths_are_safe_relative_rust_sources() {
    for path in registered_bins().into_keys() {
        assert_eq!(path.extension(), Some(std::ffi::OsStr::new("rs")));
        assert!(path.starts_with(Path::new("bin")));
        assert!(!path.components().any(|part| part.as_os_str() == ".."));
    }
}

#[test]
fn analytical_wave_references_emit_structured_reference_results() {
    for (name, case_id) in [
        ("wave_1d_reference.rs", "wave.1d-reference.v1"),
        (
            "wave_cylinder_2d_reference.rs",
            "wave.cylinder-2d-reference.v1",
        ),
    ] {
        let source = fs::read_to_string(manifest_dir().join("bin").join(name))
            .expect("read analytical wave QA source");
        assert!(
            source.contains("use sonium_qa::QaResult;"),
            "{name} must use the shared structured QA result"
        );
        assert!(
            source.contains(case_id),
            "{name} must emit its manifest-backed reference case ID"
        );
        assert!(
            source.contains("reference_only"),
            "{name} must identify its analytical-only scope"
        );
    }
}

#[test]
fn bem_sota_gate_checks_compressed_action_against_dense_reference() {
    let source = fs::read_to_string(manifest_dir().join("bin/bem_sota_validation.rs"))
        .expect("read BEM SOTA QA source");
    assert!(
        source.contains("build_hmatrix_stabilized"),
        "BEM SOTA validation must use the stabilized production H-matrix builder"
    );
    assert!(
        source.contains("require_operator_accuracy"),
        "BEM SOTA validation must compare compressed actions with an independent operator"
    );
    assert!(
        source.contains("dense_operator"),
        "BEM SOTA validation must retain a dense reference operator"
    );
    for backend in ["dense", "H-matrix", "SLFMM", "MLFMM"] {
        assert!(
            source.contains(&format!("\"{backend}\"")),
            "BEM SOTA validation must run the independent Mie power gate through {backend}"
        );
    }
    assert!(
        source.contains("all_backend_sphere_max_power_relative_error")
            && source.contains("all_backend_sphere_max_radiation_condition_relative_error")
            && source.contains("all_backend_sphere_max_residual"),
        "the structured release result must expose worst power, radiation-condition, and residual errors across backends"
    );
    for metric in [
        "dense_assembly_ms",
        "hmatrix_assembly_compression_ms",
        "slfmm_setup_ms",
        "mlfmm_setup_ms",
        "all_backend_max_krylov_ms",
        "all_backend_max_residual_check_ms",
        "all_backend_max_field_ms",
        "mlfmm_adjoint_ms",
        "hmatrix_memory_bytes",
    ] {
        assert!(
            source.contains(metric),
            "BEM SOTA validation must report {metric} separately"
        );
    }
}

#[test]
fn fem_analytical_gate_emits_manifest_backed_convergence_result() {
    let source = fs::read_to_string(manifest_dir().join("bin/fem_analytical_validation.rs"))
        .expect("read FEM analytical QA source");
    assert!(
        source.contains("fem.helmholtz-mms-convergence.v1"),
        "FEM analytical validation must emit its manifest-backed case ID"
    );
    assert!(
        source.contains("mms_convergence_cases"),
        "FEM analytical validation must report executed MMS coverage"
    );
    assert!(
        source.contains("independent_reference"),
        "FEM analytical validation must identify its independent reference scope"
    );
}

#[test]
fn fem_aeroacoustics_drp_gate_reports_independent_dispersion_reference() {
    let source = fs::read_to_string(manifest_dir().join("bin/fem_aeroacoustics_drp.rs"))
        .expect("read FEM aeroacoustics DRP QA source");
    assert!(
        source.contains("fem.aeroacoustics-drp-reference.v1"),
        "DRP QA must emit its manifest-backed case ID"
    );
    assert!(
        source.contains("NumericalDispersion") || source.contains("dispersion"),
        "DRP QA must report a dispersion reference"
    );
    assert!(
        source.contains("DrpEuler1d::new") && source.contains("advance"),
        "DRP QA must exercise time-domain propagation"
    );
}

#[test]
fn bem_convergence_compares_aligned_complex_surface_fields() {
    let source = fs::read_to_string(manifest_dir().join("bin/bem_convergence_test.rs"))
        .expect("read BEM convergence QA");

    assert!(
        source.contains("weighted_relative_l2_error"),
        "BEM convergence must use triangle-area-weighted complex L2 error"
    );
    assert!(
        source.contains("element.center") && source.contains("element.area"),
        "Mie values and weights must be evaluated at every BEM collocation element"
    );
    assert!(
        !source.contains("mie_avg") && !source.contains("avg_p"),
        "an all-element BEM magnitude average cannot be compared with a few Mie directions"
    );
    assert!(
        source.contains("final_error <= 0.01") && !source.contains("final_error <= 0.05"),
        "a refined low-ka BEM surface field must reach 1% complex error; a 5% endpoint hides formulation-sign regressions"
    );
}

#[test]
fn bem_mesh_refinement_cannot_average_empty_angular_bins() {
    let source = fs::read_to_string(manifest_dir().join("bin/bem_mesh_refinement.rs"))
        .expect("read BEM mesh-refinement QA");

    assert!(
        source.contains("weighted_relative_l2_error"),
        "mesh refinement must compare aligned complex fields over the whole surface"
    );
    for invalid_bin in ["front_p", "side_p", "back_p"] {
        assert!(
            !source.contains(invalid_bin),
            "angular-bin average `{invalid_bin}` can be empty and produce NaN"
        );
    }
}

#[test]
fn mie_validation_has_strict_weighted_accuracy_gates() {
    let source = fs::read_to_string(manifest_dir().join("bin/bem_mie_validation.rs"))
        .expect("read Mie validation QA");

    assert!(
        source.contains("weighted_relative_l2_error"),
        "surface samples require triangle-area weighting"
    );
    assert!(
        source.contains("SWEEP_ERROR_TOLERANCE"),
        "every frequency in the Mie sweep needs an accuracy bound"
    );
    assert!(
        !source.contains("l2_err < 0.45") && !source.contains("l2_err < 0.90"),
        "45-90% error thresholds cannot certify BEM accuracy"
    );
}

#[test]
fn rigid_sphere_field_pattern_is_manifest_backed_with_aligned_release_contract() {
    let source = fs::read_to_string(manifest_dir().join("bin/bem_sphere_field_pattern.rs"))
        .expect("read rigid-sphere field-pattern QA");
    let manifest = fs::read_to_string(manifest_dir().join("validation-manifest.toml"))
        .expect("read validation manifest");
    let value: toml::Value = manifest.parse().expect("parse validation manifest");
    let case = value["case"]
        .as_array()
        .expect("manifest cases")
        .iter()
        .find(|case| case["id"].as_str() == Some("bem.sphere-field-pattern-mie.v1"))
        .expect("manifest rigid-sphere field-pattern case");

    assert_eq!(case["bin"].as_str(), Some("qa-bem-sphere-field-pattern"));
    assert_eq!(
        case["reference"].as_str(),
        Some(
            "sonium-wave::analytical::sphere_scattering_3d independent rigid-sphere Mie field pattern"
        )
    );
    assert_eq!(
        case["reference_checksum"].as_str(),
        Some("sonium-wave::analytical::sphere_scattering_3d:rigid-sphere-neumann:v1")
    );
    assert_eq!(
        case["dimension"].as_str(),
        Some("3D rigid sphere, 7 far-field polar samples at r=max(12R,R+1m), ka=0.5")
    );
    assert_eq!(
        case["tolerance"].as_str(),
        Some("l2_relative_error <= 1.5e-1; max_pointwise_relative_error <= 2.5e-1")
    );
    assert_eq!(case["required_hardware"].as_str(), Some("cpu"));
    assert_eq!(case["tier"].as_str(), Some("physics-gate"));

    let compact: String = source
        .chars()
        .filter(|character| !character.is_whitespace())
        .collect();
    for marker in [
        "constCASE_ID:&str=\"bem.sphere-field-pattern-mie.v1\"",
        "constARTIFACT_PATH:&str",
        "constMAX_L2_RELATIVE_ERROR:f64=1.5e-1",
        "constMAX_POINTWISE_RELATIVE_ERROR:f64=2.5e-1",
        ".metric(\"l2_relative_error\"",
        ".metric(\"max_pointwise_relative_error\"",
        ".maximum(\"l2_relative_error\",MAX_L2_RELATIVE_ERROR)",
        ".maximum(\"max_pointwise_relative_error\",MAX_POINTWISE_RELATIVE_ERROR)",
        ".reference(REFERENCE)",
        ".reference_checksum(REFERENCE_CHECKSUM)",
        ".dimension(DIMENSION)",
        ".formulation(FORMULATION)",
        ".tolerance(TOLERANCE)",
        ".artifact(ARTIFACT_PATH)",
        "binary_output_dir",
        "sphere_scattering_3d",
        "BemProblem::try_rigid_sphere_scattering",
    ] {
        assert!(
            compact.contains(marker),
            "field-pattern QA is missing {marker}"
        );
    }
    assert!(
        !source.contains("penetrable") && !source.contains("transmission"),
        "rigid-sphere field-pattern QA must not claim transmission physics"
    );
}

#[test]
fn mie_ci_uses_measured_affine_linear_bounds_and_no_fail_open_heavy_tier() {
    let source =
        fs::read_to_string(manifest_dir().join("bin/bem_mie_ci.rs")).expect("read Mie CI QA");

    assert!(
        source.contains("threshold_l2: 0.012") && source.contains("threshold_max: 0.018"),
        "the affine-linear kD=5 gate must remain close to its measured 0.81%/1.24% baseline"
    );
    assert!(
        !source.contains("threshold_l2: 0.09")
            && !source.contains("threshold_max: 0.18")
            && !source.contains("threshold_l2: 0.60")
            && !source.contains("threshold_max: 0.80"),
        "9-80% Mie bounds cannot certify a complex acoustic field"
    );
    assert!(
        !source.contains("MIE_HEAVY_FULL") && !source.contains("linear-curved"),
        "remove the uncertifiable heavy-full path and describe the actual affine-linear geometry"
    );
    assert!(
        source.matches("max_relative.is_finite()").count() >= 2,
        "both constant and affine-linear Mie cases must reject non-finite maximum error"
    );
}

#[test]
fn sphere_surface_qas_compare_aligned_complex_fields() {
    let files = [
        "bem_adaptive_beta.rs",
        "bem_angular_validation.rs",
        "bem_audio_frequency_sweep.rs",
        "bem_beta_comparison.rs",
        "bem_basic.rs",
        "bem_bounded_beta.rs",
        "bem_mesh_comparison.rs",
        "bem_mesh_convergence.rs",
        "bem_mesh_topology.rs",
        "bem_optimal_beta.rs",
        "bem_residual_check.rs",
        "bem_simple_sphere.rs",
        "bem_sphere_integration.rs",
    ];
    let mut violations = Vec::new();

    for filename in files {
        let source = fs::read_to_string(manifest_dir().join("bin").join(filename))
            .unwrap_or_else(|error| panic!("read {filename}: {error}"));
        if !source.contains("rigid_sphere_surface_relative_l2_error") {
            violations.push(format!(
                "{filename}: missing triangle-area-weighted complex surface-field error"
            ));
        }
        for forbidden in ["mie_avg", "avg_p -", "bem_avg -"] {
            if source.contains(forbidden) {
                violations.push(format!(
                    "{filename}: `{forbidden}` compares unaligned surface averages"
                ));
            }
        }
    }

    assert!(
        violations.is_empty(),
        "Invalid rigid-sphere surface comparisons:\n{}",
        violations.join("\n")
    );
}

#[test]
fn accuracy_parity_compares_aligned_complex_fields_without_skips() {
    let source = fs::read_to_string(manifest_dir().join("bin/bem_accuracy_parity.rs"))
        .expect("read BEM accuracy-parity QA");

    assert!(
        source.contains("rigid_sphere_surface_relative_l2_error"),
        "surface accuracy must compare every collocation value with the Mie trace at the same angle"
    );
    assert!(
        source.contains("relative_error(&bem_pressure, &analytical.pressure)"),
        "off-surface accuracy must use the complex relative L2 field error"
    );
    assert!(
        !source.contains("analytical_magnitudes") && !source.contains("bem_magnitudes"),
        "magnitude-only comparison hides phase error"
    );
    assert!(
        !source.contains("skipped - small ref value"),
        "accuracy validation cannot silently discard difficult directions"
    );
}

#[test]
fn bem_convergence_study_checks_the_full_complex_field_and_all_solves() {
    let source = fs::read_to_string(manifest_dir().join("bin/bem_convergence_study.rs"))
        .expect("read BEM convergence-study QA");

    assert!(
        source.contains("relative_error(&bem_pressure, &analytical.pressure)"),
        "convergence must be measured on aligned complex angular samples"
    );
    assert!(
        source.contains("errors.windows(2)"),
        "a convergence study must assert improvement under refinement"
    );
    assert!(
        !source.contains("match solver.solve(&problem)"),
        "a failed refinement cannot be printed and ignored"
    );
}

#[test]
fn fastbem_validation_cannot_substitute_accuracy_for_convergence() {
    let source = fs::read_to_string(manifest_dir().join("bin/bem_fastbem_validation.rs"))
        .expect("read FastBEM validation QA");

    assert!(
        source.contains("relative_error(&bem_pressure, &analytical.pressure)"),
        "FastBEM comparison must retain complex phase"
    );
    assert!(
        !source.contains("improving ||"),
        "an accurate endpoint cannot certify a non-convergent refinement sequence"
    );
}

#[test]
fn high_order_quadrature_qa_uses_an_independent_reference() {
    let source = fs::read_to_string(manifest_dir().join("bin/bem_high_order_quad.rs"))
        .expect("read high-order quadrature QA");

    assert!(
        source.contains("gauss_legendre(32)"),
        "high-order triangle rules need an independent tensor-Gauss reference"
    );
    assert!(
        !source.contains("row sum") && !source.contains("K'[1]"),
        "finite-frequency Burton-Miller row sums cannot be checked against the Laplace jump identity"
    );
}

#[test]
fn bem_batch2_checks_radiation_and_burton_miller_against_analytical_fields() {
    let source = fs::read_to_string(manifest_dir().join("bin/bem_validation_batch2.rs"))
        .expect("read BEM validation batch 2");

    assert!(
        source.contains("sphere_scattering_3d"),
        "the Burton-Miller eigenfrequency case must be checked against the rigid-sphere Mie field"
    );
    assert!(
        source.contains("relative_error(&bem_pressures, &analytical.pressure)"),
        "radiation/scattering accuracy must retain complex phase"
    );
    assert!(
        !source.contains("max_bm < 50.0"),
        "a merely bounded Burton-Miller solution does not establish physical correctness"
    );
}

#[test]
fn bem_analytical_reference_qas_exercise_a_production_operator() {
    let mut violations = Vec::new();
    for relative_path in qa_source_files() {
        let Some(filename) = relative_path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if !filename.starts_with("bem_") {
            continue;
        }

        let source = fs::read_to_string(manifest_dir().join(&relative_path))
            .unwrap_or_else(|error| panic!("read {}: {error}", relative_path.display()));
        if !source.contains("sonium_bem::analytical") {
            continue;
        }

        let executes_production_bem = [
            "BemSolver",
            "build_tbem_system",
            "build_hmatrix",
            "build_slfmm_system",
            "build_mlfmm_system",
        ]
        .iter()
        .any(|marker| source.contains(marker));
        if !executes_production_bem {
            violations.push(filename.to_owned());
        }
    }

    assert!(
        violations.is_empty(),
        "BEM QA files importing analytical references must also execute a production BEM operator; analytical-only checks belong to the reference-only wave tier:\n{}",
        violations.join("\n")
    );
}

#[test]
fn tbem_qas_do_not_invert_the_canonical_incident_rhs() {
    let mut violations = Vec::new();
    for relative_path in qa_source_files() {
        let Some(filename) = relative_path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if !filename.starts_with("bem_") {
            continue;
        }

        let source = fs::read_to_string(manifest_dir().join(&relative_path))
            .unwrap_or_else(|error| panic!("read {}: {error}", relative_path.display()));
        let compact: String = source
            .chars()
            .filter(|character| !character.is_whitespace())
            .collect();
        if compact.contains("=p_inc[i]+") {
            violations.push(format!(
                "{filename}: manually assembles a positive incident trace; use IncidentField::compute_rhs_with_beta"
            ));
        }
    }

    assert!(
        violations.is_empty(),
        "TBEM incident-RHS sign violations:\n{}",
        violations.join("\n")
    );
}

#[test]
fn qa_sources_do_not_certify_double_digit_relative_errors() {
    let forbidden = [
        "error < 10.0",
        "error < 15.0",
        "error < 20.0",
        "error < 25.0",
        "error < 50.0",
        "error < 0.10",
        "error < 0.15",
        "error < 0.20",
        "error < 0.25",
        "error < 0.30",
        "error < 0.35",
        "error < 0.40",
        "error < 0.50",
        "rel_err < 0.1",
        "rel_error < 0.10",
        "max_rel_error < 0.20",
        "max_rel_error < 0.30",
        "max_rel_error < 0.35",
        "max_rel_error < 0.5",
    ];
    let mut violations = Vec::new();

    for relative_path in qa_source_files() {
        let source = fs::read_to_string(manifest_dir().join(&relative_path))
            .unwrap_or_else(|error| panic!("read {}: {error}", relative_path.display()));
        for pattern in forbidden {
            if source.contains(pattern) {
                violations.push(format!("{}: `{pattern}`", relative_path.display()));
            }
        }
    }

    assert!(
        violations.is_empty(),
        "Double-digit relative-error gates cannot certify accuracy:\n{}",
        violations.join("\n")
    );
}

#[test]
fn qa_sources_do_not_use_generic_five_percent_escape_hatches() {
    let mut violations = Vec::new();

    for relative_path in qa_source_files() {
        let source = fs::read_to_string(manifest_dir().join(&relative_path))
            .unwrap_or_else(|error| panic!("read {}: {error}", relative_path.display()));
        for (line_index, line) in source.lines().enumerate() {
            // Documentation cannot act as a gate; only code can.
            if line.trim_start().starts_with("//") {
                continue;
            }
            let compact: String = line
                .chars()
                .filter(|character| !character.is_whitespace() && *character != '_')
                .collect();
            let normalized = compact.to_ascii_lowercase();
            let names_an_error = !normalized.contains("outerradius")
                && (["err", "error", "reldiff", "tolerance", "threshold"]
                    .iter()
                    .any(|name| normalized.contains(name))
                    || (normalized.contains("rel") && !normalized.contains("relaxation")));
            // Match the exact 5% literal, not longer decimals that merely
            // start with it (0.0508186), scientific notation (5.0e-3 is
            // 0.5%, 5.0e-5 is 0.005%), or a percentage already spelled out.
            let has_exact_five_percent = compact.match_indices("0.05").any(|(index, _)| {
                let before_ok = index == 0
                    || !compact.as_bytes()[index - 1].is_ascii_digit()
                        && compact.as_bytes()[index - 1] != b'.';
                let after = &compact[index + 4..];
                let after_ok = !after
                    .chars()
                    .next()
                    .is_some_and(|c| c.is_ascii_digit() || c == 'e' || c == 'E');
                before_ok && after_ok
            });
            let has_bare_five_comparison = ["<5.0", "<=5.0"].iter().any(|needle| {
                compact.match_indices(needle).any(|(index, _)| {
                    !compact[index + needle.len()..]
                        .chars()
                        .next()
                        .is_some_and(|c| c.is_ascii_digit() || c == 'e' || c == 'E' || c == '.')
                })
            });
            let is_five_percent_comparison =
                (has_exact_five_percent && !compact.contains("0.05%")) || has_bare_five_comparison;
            if names_an_error && is_five_percent_comparison {
                violations.push(format!(
                    "{}:{}: `{}`",
                    relative_path.display(),
                    line_index + 1,
                    line.trim()
                ));
            }
        }
    }

    assert!(
        violations.is_empty(),
        "Generic 5% gates are too loose to certify convergence; use a measured physics-specific bound or a refinement contract:\n{}",
        violations.join("\n")
    );
}

#[test]
fn every_navier_stokes_solve_fails_closed_on_nonconvergence() {
    let mut violations = Vec::new();
    for relative_path in qa_source_files() {
        let Some(filename) = relative_path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if !filename.starts_with("fem_ns_") {
            continue;
        }

        let source = fs::read_to_string(manifest_dir().join(&relative_path))
            .unwrap_or_else(|error| panic!("read {}: {error}", relative_path.display()));
        let compact: String = source
            .chars()
            .filter(|character| !character.is_whitespace())
            .collect();
        let solve_count = compact.matches("letresult=solve").count();
        let convergence_gate_count = compact.matches("assert!(result.converged").count();
        if convergence_gate_count < solve_count {
            violations.push(format!(
                "{filename}: {solve_count} Navier-Stokes solves but only {convergence_gate_count} fail-closed convergence gates"
            ));
        }
        for forbidden in [
            "WARNING:notconverged",
            "WARNING:didnotconverge",
            "mayormaynotconverge",
        ] {
            if compact.contains(forbidden) {
                violations.push(format!(
                    "{filename}: contains non-failing `{forbidden}` path"
                ));
            }
        }
    }

    assert!(
        violations.is_empty(),
        "Navier-Stokes QA convergence violations:\n{}",
        violations.join("\n")
    );
}

#[test]
fn lid_cavity_reference_points_are_interpolated_not_nearest_sampled() {
    let source = fs::read_to_string(manifest_dir().join("bin/fem_ns_lid_driven_cavity.rs"))
        .expect("read lid-driven cavity QA");
    assert!(
        source.contains("interpolate_centerline"),
        "Ghia reference coordinates must be aligned by interpolation"
    );
    assert!(
        !source.contains("Find nearest centerline node"),
        "nearest-node matching aliases the steep moving-lid boundary layer"
    );
}

#[test]
fn lid_cavity_refinement_reaches_three_percent_of_lid_speed() {
    let source = fs::read_to_string(manifest_dir().join("bin/fem_ns_lid_driven_cavity.rs"))
        .expect("read lid-driven cavity QA");
    let compact: String = source
        .chars()
        .filter(|character| !character.is_whitespace())
        .collect();
    assert!(
        compact.contains("run_cavity(100.0,40)") && compact.contains("fine_error<0.03"),
        "the aligned Ghia centerline comparison must refine to a 3% endpoint"
    );
}

#[test]
fn accuracy_parity_and_poiseuille_use_measured_three_percent_or_better_bounds() {
    let bem = fs::read_to_string(manifest_dir().join("bin/bem_accuracy_parity.rs"))
        .expect("read BEM accuracy-parity QA");
    assert!(
        bem.contains("error < 0.01") && !bem.contains("error < 0.04"),
        "BEM accuracy parity currently resolves every regime below 1%"
    );

    let poiseuille = fs::read_to_string(manifest_dir().join("bin/fem_ns_poiseuille.rs"))
        .expect("read Navier-Stokes Poiseuille QA");
    assert!(
        poiseuille.matches("< 0.03").count() >= 2,
        "Poiseuille L2 and maximum profile errors must both stay below 3%"
    );
}

#[test]
fn mindlin_plate_accuracy_does_not_hide_behind_a_locking_comparison() {
    let source = fs::read_to_string(manifest_dir().join("bin/fem_mindlin_plate_ss_sinusoidal.rs"))
        .expect("read Mindlin plate QA");
    assert!(
        source.contains("(0.1, 40, 0.02)") && source.contains("err_dsg < 0.02"),
        "the thin-plate DSG result must meet 2% analytical accuracy in both validation paths"
    );
}

#[test]
fn dfg_pressure_probes_are_interpolated_not_nearest_sampled() {
    let source = fs::read_to_string(manifest_dir().join("bin/fem_ns_flow_past_cylinder.rs"))
        .expect("read DFG cylinder QA");
    assert!(
        source.contains("interpolate_scalar_at_point"),
        "DFG front/back pressure probes must use P1 interpolation at the benchmark coordinates"
    );
    assert!(
        !source.contains("nearest_pressure"),
        "nearest-node DFG pressure probes alias the curved cylinder boundary"
    );
}

#[test]
fn batch3_never_warns_and_passes_on_known_accuracy_failures() {
    let source = fs::read_to_string(manifest_dir().join("bin/bem_validation_batch3.rs"))
        .expect("read BEM validation batch 3");

    assert!(
        !source.contains("known solver accuracy issue") && !source.contains("WARNING: ka="),
        "every frequency in batch 3 must fail closed on an inaccurate complex field"
    );
    assert!(
        source.contains("relative_error"),
        "batch-3 analytical checks must compare complex fields"
    );
}

#[test]
fn eaa_validation_fails_closed_on_non_finite_surface_pressure() {
    let source = fs::read_to_string(manifest_dir().join("bin/bem_eaa_validation.rs"))
        .expect("read EAA validation QA");
    let compact: String = source
        .chars()
        .filter(|character| !character.is_whitespace())
        .collect();
    assert!(
        !source.contains("Skipping field evaluation assertions"),
        "non-finite EAA surface pressure must fail the QA instead of skipping its field checks"
    );
    assert!(
        !compact.contains("ifhas_nan")
            && compact.contains("assert!(!has_nan")
            && compact.contains("surface_pressure.iter().all("),
        "EAA validation must assert that every solved surface pressure is finite"
    );
}

#[test]
fn eaa_directionality_retains_complex_phase() {
    let source = fs::read_to_string(manifest_dir().join("bin/bem_eaa_validation.rs"))
        .expect("read EAA validation QA");
    assert!(
        source.contains("forward_shadow_complex_difference"),
        "EAA forward/shadow directionality must compare complex scattered pressures"
    );
    assert!(
        !source.contains("(forward_shadow_ratio - 1.0).abs()"),
        "equal forward/shadow magnitudes do not imply equal complex acoustic fields"
    );
}

#[test]
fn eaa_closed_cylinder_uses_complex_refinement_and_symmetry_not_false_duct_radiation() {
    let source = fs::read_to_string(manifest_dir().join("bin/bem_eaa_validation.rs"))
        .expect("read EAA validation QA");

    assert!(
        !source.contains("sound escapes through the duct ends")
            && !source.contains("Expected axial radiation"),
        "a sealed exterior-scattering mesh cannot certify open-duct end radiation"
    );
    assert!(
        source.contains("coarse_scattered")
            && source.contains("medium_scattered")
            && source.contains("fine_scattered")
            && source.contains("relative_error(&medium_scattered, &coarse_scattered)")
            && source.contains("relative_error(&fine_scattered, &medium_scattered)")
            && source.contains("fine_error < coarse_error"),
        "the closed-cylinder case must compare aligned complex scattered fields under refinement"
    );
    assert!(
        source.contains("rotational_symmetry_error"),
        "the axisymmetric cylinder needs a complex rotational-symmetry invariant"
    );
}

#[test]
fn fem_acoustics_report_uses_the_configured_output_directory() {
    let source = fs::read_to_string(manifest_dir().join("bin/fem_acoustics_integration.rs"))
        .expect("read FEM acoustics integration QA");
    assert!(
        source.contains("let output_path = binary_output_dir"),
        "FEM acoustics JSON reports must honor SONIUM_OUTPUT_DIR"
    );
}

#[test]
fn qa_vtu_artifacts_do_not_write_to_the_repository_root() {
    let mut violations = Vec::new();
    for relative_path in qa_source_files() {
        let source = fs::read_to_string(manifest_dir().join(&relative_path))
            .unwrap_or_else(|error| panic!("read {}: {error}", relative_path.display()));
        for (line_index, line) in source.lines().enumerate() {
            if line.contains(".write(\"") && line.contains(".vtu\"") {
                violations.push(format!(
                    "{}:{}: `{}`",
                    relative_path.display(),
                    line_index + 1,
                    line.trim()
                ));
            }
        }
    }
    assert!(
        violations.is_empty(),
        "VTU artifacts must use binary_output_dir/SONIUM_OUTPUT_DIR:\n{}",
        violations.join("\n")
    );
}

#[test]
fn fmm_solver_validation_requires_accuracy_and_convergence() {
    let source = fs::read_to_string(manifest_dir().join("bin/bem_fmm_solver_validation.rs"))
        .expect("read FMM solver validation");
    let compact: String = source
        .chars()
        .filter(|character| !character.is_whitespace())
        .collect();

    assert!(
        !source.contains("rel_error < 0.5"),
        "50% SLFMM/TBEM matvec error cannot certify the approximation"
    );
    assert!(
        !compact.contains("solution.iterations>0||solution.residual<1.0"),
        "an unconverged iterative solve cannot pass for merely making progress"
    );
}

#[test]
fn optimization_qas_do_not_substitute_progress_for_convergence() {
    for filename in ["solvers_optimization.rs", "fem_topology_cantilever.rs"] {
        let source = fs::read_to_string(manifest_dir().join("bin").join(filename))
            .unwrap_or_else(|error| panic!("read {filename}: {error}"));
        let compact: String = source
            .chars()
            .filter(|character| !character.is_whitespace())
            .collect();
        assert!(
            !compact.contains("result.converged||"),
            "{filename}: a non-converged optimization cannot pass on a progress surrogate"
        );
    }
}

#[test]
fn blr_sphere_validation_retains_complex_phase_and_area_weights() {
    let source = fs::read_to_string(manifest_dir().join("bin/blr_bem_sphere_scattering.rs"))
        .expect("read BLR BEM sphere validation");

    assert!(
        source.contains("rigid_sphere_surface_relative_l2_error"),
        "BLR sphere accuracy must compare the area-weighted complex surface field"
    );
    assert!(
        !source.contains("p_surface[i].norm() - p_ana"),
        "magnitude-only BLR/Mie comparison hides phase error"
    );
}

#[test]
fn mie_convergence_records_and_gates_the_refinement_sequence() {
    let source = fs::read_to_string(manifest_dir().join("bin/bem_mie_convergence.rs"))
        .expect("read Mie convergence QA");

    assert!(
        source.contains("errors.windows(2)"),
        "Mie convergence must gate every adjacent refinement"
    );
    assert!(
        !source.contains("< 10%") && !source.contains("< 0.10"),
        "a 10% endpoint cannot certify the refined BEM field"
    );
}

#[test]
fn fmm_smoke_gates_both_approximations_against_tbem() {
    let source =
        fs::read_to_string(manifest_dir().join("bin/bem_fmm_test.rs")).expect("read FMM smoke QA");

    assert!(source.contains("relative_diff_slfmm < 1e-12"));
    assert!(source.contains("relative_diff_mlfmm < 1e-12"));
}

#[test]
fn thermal_qas_use_normalized_accuracy_gates() {
    for filename in ["fem_thermal_1d_transient.rs", "fem_thermal_2d_disk.rs"] {
        let source = fs::read_to_string(manifest_dir().join("bin").join(filename))
            .unwrap_or_else(|error| panic!("read {filename}: {error}"));
        assert!(
            source.contains("normalized_error"),
            "{filename}: absolute kelvin tolerances must be normalized by the analytical temperature scale"
        );
    }
}

#[test]
fn external_flow_benchmarks_require_refinement_and_strict_accuracy() {
    let cavity = fs::read_to_string(manifest_dir().join("bin/fem_ns_lid_driven_cavity.rs"))
        .expect("read cavity QA");
    assert!(cavity.contains("coarse_error") && cavity.contains("fine_error"));
    assert!(!cavity.contains("max_err < 0.15"));

    let cylinder = fs::read_to_string(manifest_dir().join("bin/fem_ns_flow_past_cylinder.rs"))
        .expect("read flow-past-cylinder QA");
    assert!(cylinder.contains("coarse_metrics") && cylinder.contains("fine_metrics"));
    assert!(!cylinder.contains("c_d_error <= 0.10"));
    assert!(!cylinder.contains("delta_p_error <= 0.10"));
}

#[test]
fn potential_flow_gates_velocity_refinement_not_one_coarse_derivative() {
    let source = fs::read_to_string(manifest_dir().join("bin/fem_potential_flow_cylinder.rs"))
        .expect("read potential-flow QA");
    assert!(
        source.contains("velocity_errors.windows(2)"),
        "potential-flow convergence must include the derived velocity field"
    );
    assert!(!source.contains("max_vel_error < 0.08"));
}

#[test]
fn waveholtz_accuracy_never_uses_a_five_percent_escape_hatch() {
    let source = fs::read_to_string(manifest_dir().join("bin/fem_waveholtz_validation.rs"))
        .expect("read WaveHoltz validation");
    assert!(!source.contains("error < 0.05"));
    assert!(source.contains("waveholtz_vs_direct_error"));
}

#[test]
fn acceptance_gates_enforce_their_advertised_tolerances() {
    let galerkin = fs::read_to_string(manifest_dir().join("bin/bem_galerkin_single_layer.rs"))
        .expect("read Galerkin single-layer QA");
    assert!(
        galerkin.contains("non-finite entries"),
        "Galerkin QA must reject non-finite matrix entries"
    );
    assert!(
        galerkin.contains("&physics, 7)"),
        "Galerkin QA must confirm its bound against an independent order-7 reference"
    );
    assert!(
        galerkin.contains(".maximum(\"maximum_matrix_asymmetry\", ASYMMETRY_TOLERANCE)")
            && galerkin.contains(".maximum(\"reference_matrix_asymmetry\", ASYMMETRY_TOLERANCE)"),
        "Galerkin QA must carry structured asymmetry thresholds"
    );

    let parity =
        fs::read_to_string(manifest_dir().join("bin/bem_transmission_kernel_native_parity.rs"))
            .expect("read kernel-native parity QA");
    assert!(
        parity.contains(".maximum(\"forward_action_error\", ACTION_ERROR_TOLERANCE)")
            && parity.contains(".maximum(\"transpose_action_error\", ACTION_ERROR_TOLERANCE)"),
        "kernel-native parity QA must carry structured action-error thresholds"
    );
    assert!(
        parity.contains("forward action error")
            && parity.contains("transpose action error")
            && parity.contains("exceeds"),
        "kernel-native parity QA must fail standalone runs that exceed the bound"
    );

    for filename in [
        "bem_transmission_distributed_rows.rs",
        "bem_transmission_distributed_higher_order_rows.rs",
    ] {
        let source = fs::read_to_string(manifest_dir().join("bin").join(filename))
            .unwrap_or_else(|error| panic!("read {filename}: {error}"));
        assert!(
            source.contains(".maximum(\"maximum_local_action_error\", LOCAL_ACTION_TOLERANCE)")
                && source.contains(".maximum(\"coverage_defects\", 0.0)"),
            "{filename}: row QA must carry structured error and coverage thresholds"
        );
        assert!(
            source.contains("binary_search") && source.contains("coverage_defects != 0"),
            "{filename}: row QA must verify exact unique global row coverage"
        );
    }

    let stokes = fs::read_to_string(manifest_dir().join("bin/fem_stokes_validation.rs"))
        .expect("read Stokes validation QA");
    assert!(
        !stokes.contains("(0.0, 0.0, true)"),
        "Stokes QA must not fabricate zero error for cases it did not run"
    );
    assert!(
        stokes.contains("unknown SONIUM_STOKES_CASE"),
        "Stokes QA must reject unknown case filters instead of omitting every case"
    );
    assert!(
        stokes.contains("executed_cases") && stokes.contains(".minimum(\"executed_cases\", 1.0)"),
        "Stokes QA must report executed counts with a nonzero threshold"
    );

    let numcalc = fs::read_to_string(manifest_dir().join("bin/bem_numcalc_parity.rs"))
        .expect("read NumCalc parity QA");
    let skip_offset = numcalc
        .find("NumCalc not found")
        .expect("NumCalc QA keeps its missing-executable branch");
    let skip_arm: String = numcalc[skip_offset..].lines().take(20).collect();
    assert!(
        skip_arm.contains("QaResult::skipped"),
        "missing NumCalc must emit a structured skipped record"
    );
    assert!(
        skip_arm.contains("process::exit(1)") && !skip_arm.contains("SUCCESS"),
        "missing NumCalc must fail the required parity gate, not print SUCCESS"
    );
}

#[test]
fn partial_implementations_solve_or_declare_their_scope() {
    let beam = fs::read_to_string(manifest_dir().join("bin/fem_beam_validation.rs"))
        .expect("read beam validation");
    assert!(
        beam.contains("BETA1L") && beam.contains("EIGEN_RESIDUAL_TOLERANCE"),
        "beam validation must eigen-solve with the corrected beta^2 formula and a residual gate"
    );
    assert!(
        !beam.contains("1.875_f64.powi(2)"),
        "beam validation must not raise the already-squared beta coefficient again"
    );

    let maxwell = fs::read_to_string(manifest_dir().join("bin/fem_maxwell_validation.rs"))
        .expect("read Maxwell validation");
    assert!(
        maxwell.contains("WAVEGUIDE_KC_TOLERANCE")
            && maxwell.contains("CAVITY_K_TOLERANCE")
            && maxwell.contains("relative_eigen_residual"),
        "Maxwell validation must eigen-solve waveguide and cavity modes with residual gates"
    );

    let magnetostatics =
        fs::read_to_string(manifest_dir().join("bin/fem_magnetostatics_validation.rs"))
            .expect("read magnetostatics validation");
    assert!(
        magnetostatics.contains("DIPOLE_FIELD_TOLERANCE")
            && magnetostatics.contains("TRUE_RESIDUAL_TOLERANCE")
            && magnetostatics.contains("SHIELD_CONTRACTION_TOLERANCE"),
        "magnetostatics validation must solve dipole, wire, shielding, and inductance scenarios"
    );

    let wrappers = fs::read_to_string(manifest_dir().join("bin/fem_eigenphysics_wrappers.rs"))
        .expect("read eigenphysics wrappers");
    assert!(
        wrappers.contains("EM_WAVENUMBER_TOLERANCE")
            && wrappers.contains("EIGEN_RESIDUAL_TOLERANCE"),
        "eigenphysics wrappers must solve the EM leg and gate convergence plus residuals"
    );

    let thermal = fs::read_to_string(manifest_dir().join("bin/fem_thermal_validation.rs"))
        .expect("read thermal validation");
    assert!(
        thermal.contains("quadrature_l2_error")
            && thermal.contains("L2_RATE_MINIMUM")
            && thermal.contains("FINE_L2_MAX_ERROR"),
        "thermal validation must measure the convergence rate with quadrature L2 error"
    );

    let shell = fs::read_to_string(manifest_dir().join("bin/fem_shell_validation.rs"))
        .expect("read shell validation");
    assert!(
        shell.contains("TOL_DEFLECTION") && shell.contains("TOL_ROTATION"),
        "shell validation must solve deflections and gate them against a reference"
    );

    let plasticity = fs::read_to_string(manifest_dir().join("bin/fem_plasticity_validation.rs"))
        .expect("read plasticity validation");
    assert!(
        plasticity.contains("return_mapping_3d")
            && plasticity.contains("TOL_PLASTIC_WORK")
            && plasticity.contains("TOL_UNLOAD_SLOPE"),
        "plasticity validation must run yielding load/unload scenarios through production return mapping"
    );

    let biharmonic = fs::read_to_string(manifest_dir().join("bin/fem_biharmonic_validation.rs"))
        .expect("read biharmonic validation");
    assert!(
        biharmonic.contains("TOL_FINE_U") && biharmonic.contains("TOL_RATE_U"),
        "biharmonic validation must solve a manufactured plate with refinement gates"
    );

    let electroacoustics =
        fs::read_to_string(manifest_dir().join("bin/fem_electroacoustics_validation.rs"))
            .expect("read electroacoustics validation");
    assert!(
        electroacoustics.contains("TOL_RECIPROCITY") && electroacoustics.contains("TOL_POWER"),
        "electroacoustics validation must solve the driven coupled response with reciprocity and power gates"
    );

    let nonlinear =
        fs::read_to_string(manifest_dir().join("bin/fem_nonlinear_solid_validation.rs"))
            .expect("read nonlinear-solid validation");
    assert!(
        nonlinear.contains("fem_newton_solve_with_bcs") && nonlinear.contains("from u=0"),
        "nonlinear-solid validation must solve loaded problems from a nonequilibrium start"
    );
    assert!(
        nonlinear.contains("nonzero iterations"),
        "nonlinear-solid validation must not pass a zero-iteration equilibrium as a solve"
    );

    let acoustics = fs::read_to_string(manifest_dir().join("bin/fem_acoustics_integration.rs"))
        .expect("read acoustics integration");
    assert!(
        acoustics.contains("quasi_1d_error") && acoustics.contains("TOL_QUASI_1D_100HZ"),
        "acoustics integration must compare pressure against an independent reference"
    );
    assert!(
        acoustics.contains("cached sweep solve failed")
            && acoustics.contains("fresh sweep solve failed"),
        "acoustics integration must compare cached sweeps against freshly assembled solves"
    );

    let analytical = fs::read_to_string(manifest_dir().join("bin/fem_analytical_validation.rs"))
        .expect("read analytical validation");
    assert!(
        analytical.contains("assert_nontrivial_solve") && analytical.contains("interior_l2_error"),
        "analytical validation must assert free unknowns and use interior norms"
    );
    assert!(
        analytical.contains("-(kn * x).sin()/denom")
            || analytical.contains("-(kn * x).sin() / denom"),
        "analytical validation must use the FEM-sign Helmholtz exact mode"
    );

    let shoebox = fs::read_to_string(manifest_dir().join("bin/room_shoebox_fem_benchmark.rs"))
        .expect("read shoebox benchmark");
    assert!(
        shoebox.contains("LOBPCG_TOLERANCE") && shoebox.contains("room.shoebox-fem-benchmark.v1"),
        "shoebox benchmark must solve eigenpairs with residual gates"
    );

    let piston =
        fs::read_to_string(manifest_dir().join("bin/speaker_piston_directivity_benchmark.rs"))
            .expect("read piston benchmark");
    assert!(
        piston.contains("RAYLEIGH_OVERKILL_GRID")
            && piston.contains("validate_numerical_rayleigh_radiation"),
        "piston benchmark must converge numerical radiation against an overkill grid"
    );

    let modal = fs::read_to_string(manifest_dir().join("bin/room_modal_validation.rs"))
        .expect("read room modal validation");
    assert!(
        modal.contains("SCOPE: analytical reference only"),
        "room modal validation must declare its reference-only scope"
    );

    for filename in [
        "fem_biharmonic_3d.rs",
        "fem_fsi_mortar_3d.rs",
        "fem_compressible_ns_3d.rs",
    ] {
        let source = fs::read_to_string(manifest_dir().join("bin").join(filename))
            .unwrap_or_else(|error| panic!("read {filename}: {error}"));
        assert!(
            source.contains("deliberately limited") && source.contains("cite this binary as"),
            "{filename}: thin 3D wrapper must declare its limited contract scope"
        );
    }

    for filename in ["fem_axisym.rs", "fem_crash_shells.rs", "fem_foam_laws.rs"] {
        let source = fs::read_to_string(manifest_dir().join("bin").join(filename))
            .unwrap_or_else(|error| panic!("read {filename}: {error}"));
        assert!(
            source.contains("integrity-only"),
            "{filename}: golden audit must declare its integrity-only scope"
        );
    }
    let rupture =
        fs::read_to_string(manifest_dir().join("bin/fem_rupture.rs")).expect("read rupture QA");
    assert!(
        rupture.contains("criterion-replay"),
        "rupture QA must declare its criterion-replay scope"
    );

    let port = fs::read_to_string(manifest_dir().join("bin/speaker_port_nonlinear.rs"))
        .expect("read speaker port QA");
    assert!(
        port.contains("CASE_ID_2D")
            && port.contains("CASE_ID_3D")
            && port.contains("SUBCASE_SUMMARY"),
        "speaker port QA must emit independent 2D and 3D subcase statuses"
    );
    assert!(
        port.contains("ThreeDOutcome::Skipped") && port.contains("covers the 2D subcase only"),
        "speaker port QA must honestly scope whole-binary SUCCESS to 2D when 3D did not run"
    );
    assert!(
        port.contains("ThreeDOutcome::Failed") && port.contains("3D subcase failed"),
        "speaker port QA must fail closed when the executed 3D leg fails"
    );
    assert!(
        port.contains(".minimum(\"executed_3d\", 1.0)"),
        "a 3D pass claim must require the executed 3D flag"
    );
}
