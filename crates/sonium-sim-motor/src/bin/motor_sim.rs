//! `motor-sim` — showcase CLI for the GEO_30MM rotating-machine
//! stack: one operating point, a gamma sweep, an MTPA search, and
//! thermal / rotor-stress checks.
//!
//! Examples:
//!
//! ```text
//! motor-sim point --current 60 --gamma 0 --rpm 15000
//! motor-sim sweep --current 60 --rpm 15000 --gammas=-30,-15,0,15,30
//! motor-sim mtpa --current 60 --rpm 15000
//! motor-sim point --thermal --mech
//! ```

// Rust guideline compliant 2026-02-21

use clap::{Args, Parser, Subcommand};
use sonium_sim_motor::{MotorDesign, MotorReport, MotorSimulator, OperatingPoint, SimOptions};

#[derive(Parser)]
#[command(about = "GEO_30MM showcase motor simulation (sonium-fem rotating machine)")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Solve one operating point.
    Point(PointArgs),
    /// Solve a list of gamma values.
    Sweep(SweepArgs),
    /// Golden-section search for maximum torque over gamma.
    Mtpa(PointArgs),
}

#[derive(Args)]
struct PointArgs {
    /// Terminal phase current, A rms.
    #[arg(long, default_value_t = 60.0)]
    current: f64,
    /// Current angle from the q axis, electrical degrees.
    #[arg(long, default_value_t = 0.0)]
    gamma: f64,
    /// Rotor speed, rpm.
    #[arg(long, default_value_t = 15_000.0)]
    rpm: f64,
    /// Coil temperature, deg C.
    #[arg(long, default_value_t = 120.0)]
    coil_temp: f64,
    /// Also run the steady thermal check.
    #[arg(long)]
    thermal: bool,
    /// Also run the rotor-stress check.
    #[arg(long)]
    mech: bool,
}

#[derive(Args)]
struct SweepArgs {
    #[command(flatten)]
    point: PointArgs,
    /// Comma-separated gamma values, electrical degrees.
    #[arg(long, value_delimiter = ',', default_values_t = vec![-30.0, -15.0, 0.0, 15.0, 30.0])]
    gammas: Vec<f64>,
}

impl PointArgs {
    fn point(&self) -> OperatingPoint {
        OperatingPoint {
            i_phase_rms_a: self.current,
            gamma_deg: self.gamma,
            rpm: self.rpm,
            coil_temp_c: self.coil_temp,
        }
    }
}

fn print_report(report: &MotorReport) {
    let point = report.point;
    println!(
        "I={:.1} A rms  gamma={:.1} deg  {:.0} rpm  (daxis {:.2} deg)",
        point.i_phase_rms_a, point.gamma_deg, point.rpm, report.daxis_deg
    );
    println!(
        "  torque        {:.4} N m   (ripple {:.2} %)",
        report.torque_nm, report.torque_ripple_pct
    );
    println!("  shaft power   {:.1} W", report.shaft_power_w);
    println!("  iron loss     {:.2} W", report.iron_loss_w);
    println!("  copper loss   {:.2} W  (DC)", report.copper_loss_w);
    println!("  V peak        {:.2} V", report.v_peak_v);
    println!(
        "  efficiency    {:.1} %  (Fe+Cu only)",
        report.efficiency_pct
    );
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    let design = MotorDesign::geo30mm()?;
    let mut simulator = MotorSimulator::new(design, SimOptions::default())?;
    match cli.command {
        Command::Point(args) => {
            let report = simulator.simulate(args.point())?;
            print_report(&report);
            if args.thermal {
                // Liquid-jacket film: this operating point dissipates
                // ~112 W from a 30 mm can, far beyond natural
                // convection; the budget line below shows why.
                let thermal = simulator.thermal_check(&report, 1_500.0, 25.0)?;
                println!(
                    "  thermal: coil {:.1} C, magnets {:.1} C, stator {:.1} C, \
                     housing rejects {:.1} W (residual {:.2e} W)",
                    thermal.coil_mean_c,
                    thermal.magnet_mean_c,
                    thermal.stator_mean_c,
                    thermal.housing_heat_w,
                    thermal.heat_residual_w
                );
            }
            if args.mech {
                // Centrifugal only (delta T = 0): the bonded model
                // overstates magnet thermal-mismatch tension versus
                // the reference's pocket-contact model.
                let mech = simulator.mech_check(args.rpm, 0.0)?;
                println!(
                    "  mech: SF min {:.2} (marker {}), limit speed {}, radial growth {:.3} mm",
                    mech.safety_factor_min,
                    mech.limiting_marker,
                    mech.limit_speed_rpm.map_or_else(
                        || "> search range".to_string(),
                        |rpm| format!("{rpm:.0} rpm")
                    ),
                    mech.max_radial_growth_m * 1.0e3
                );
            }
        }
        Command::Sweep(args) => {
            let reports = simulator.sweep_gamma(args.point.point(), &args.gammas)?;
            println!("gamma [deg]   torque [N m]   P_fe [W]   eta [%]");
            for report in &reports {
                println!(
                    "{:10.1}   {:12.4}   {:8.2}   {:7.1}",
                    report.point.gamma_deg,
                    report.torque_nm,
                    report.iron_loss_w,
                    report.efficiency_pct
                );
            }
        }
        Command::Mtpa(args) => {
            let report = simulator.max_torque_gamma(args.point(), -45.0, 45.0, 8)?;
            print_report(&report);
        }
    }
    Ok(())
}
