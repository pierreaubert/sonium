//! Re-checks a JLR LBM result bundle against the paper's fixtures.
//!
//! Usage: qa-jlr-lbm <bundle.json> [deviation_budget]
//! Exits non-zero when the bundle's circulation number or annular-gap
//! profile misses the Melchior et al. (2026) gates (experimental band
//! and the authors' 8% deviation claim; see sonium-sim-jlr compare).

// Rust guideline compliant 2026-02-21

use sonium_sim_jlr::case::CaseResult;
use sonium_sim_jlr::compare::compare_point;

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().unwrap_or_else(|| {
        eprintln!("usage: qa-jlr-lbm <bundle.json> [deviation_budget]");
        std::process::exit(2);
    });
    let budget: f64 = args.next().and_then(|s| s.parse().ok()).unwrap_or(0.08);
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        eprintln!("cannot read {path}: {e}");
        std::process::exit(2);
    });
    let bundle: serde_json::Value = serde_json::from_str(&text).unwrap_or_else(|e| {
        eprintln!("invalid bundle JSON: {e}");
        std::process::exit(2);
    });
    let result: CaseResult = serde_json::from_value(bundle["result"].clone()).unwrap_or_else(|e| {
        eprintln!("bundle has no valid result field: {e}");
        std::process::exit(2);
    });
    let cmp = compare_point(&result, budget);
    println!(
        "phi={:.3} kW/m3  profile max dev {:.2}%  mean dev {:.2}%  Ncirc={:.3} vs paper sim {:.2} ({:.2}%)  in exp band: {}  Re dev: {:.2}%/{:.2}%/{:.2}%",
        cmp.specific_power_kw_m3,
        cmp.max_profile_deviation * 100.0,
        cmp.mean_profile_deviation * 100.0,
        cmp.circulation_number,
        cmp.paper_sim_circulation_number,
        cmp.circulation_deviation_vs_paper_sim * 100.0,
        cmp.circulation_in_experimental_band,
        cmp.reynolds_deviation[0] * 100.0,
        cmp.reynolds_deviation[1] * 100.0,
        cmp.reynolds_deviation[2] * 100.0,
    );
    if cmp.passed {
        println!("JLR VALIDATION PASSED (budget {:.0}%)", budget * 100.0);
        println!("SUCCESS");
    } else {
        println!("JLR VALIDATION FAILED (budget {:.0}%)", budget * 100.0);
        std::process::exit(1);
    }
}
