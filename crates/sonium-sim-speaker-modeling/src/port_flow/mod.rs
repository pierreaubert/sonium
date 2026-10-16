//! Resolved nonlinear port flow: geometry, unsteady Navier–Stokes solve,
//! box/driver coupling, and THD/IMD analysis.
//!
//! `geometry` builds the deterministic 2D slot-port mesh; `geometry3d` builds
//! the current 3D round/flared port mesh. `coupled` advances the resolved
//! Taylor–Hood P2–P1 BDF flow in lockstep with the lumped box/driver ODE,
//! `analysis`/`measure`/`sweep` extract port/diaphragm volume flow, THD/IMD,
//! SPL, and compression sweeps, `baseline` freezes the Step-0 reference case
//! and straight-tube archive, `evaluate` reduces one candidate profile to
//! Tier-2 metrics, and `convergence`/`runtime` gate product
//! promotion on independent spatial and temporal ladders that agree at a
//! common fine state.
//!
//! Validated envelope: the strict QA gates cover the 2D path at a converged
//! timestep. The 3D pipe-flow and exact complex-reference gates are nightly
//! tier (`SPEAKER_PORT_3D=1` in `qa-speaker-port-nonlinear`) and must not be
//! reported passed from 2D evidence. Bounded linear straight-tube/port
//! references (`linear_reference`) do not establish nonlinear 3D port-flow
//! convergence at product scale.

pub mod analysis;
pub mod baseline;
pub mod box_driver;
pub mod campaign;
pub mod convergence;
pub mod coupled;
pub mod duct1d;
pub mod error;
pub mod evaluate;
pub mod finalist;
pub mod geometry;
pub mod geometry3d;
pub mod handoff;
pub mod linear_reference;
pub mod measure;
pub mod optim_gate;
pub mod optimize;
pub mod profile;
pub mod promotion;
pub mod runtime;
pub mod spline_geometry;
pub mod sweep;

pub use baseline::{
    BASELINE_MAX_ENERGY_ALGEBRAIC_CLOSURE, BASELINE_MAX_ENERGY_INTERFACE_RESIDUAL,
    BASELINE_MAX_ENERGY_PHYSICAL_CLOSURE, BASELINE_MAX_LINEAR_THD_FRACTION,
    BASELINE_MAX_LUMPED_LIMIT_RELATIVE_ERROR, BASELINE_MAX_TIER1_FLUX_RELATIVE_DIFFERENCE,
    BASELINE_MAX_TIER1_RELATIVE_RESIDUAL, BaselineGateReport, LumpedLimitBaseline,
    PORT_OPTIM_BASELINE_PROVENANCE, PORT_OPTIM_BASELINE_SCHEMA_VERSION, PortOptimBaseline,
    PortOptimReferenceCase, Tier1BaselinePoint, Tier2BaselineLevel, check_baseline_gates,
    complex_pair, compression_db, duct_bulk_mach_number,
};
pub use campaign::{
    CampaignArchiverError, Criterion1Inputs, EqualAirflowDelta, GateVerdict, LevelDelta,
    STEP7_GRID_LEVELS_N5, STEP7_LHS_COUNT, STEP7_MIN_TIER2_FINALISTS, STEP7_PROFILE_CSV_STATIONS,
    STEP7_REPORT_PROVENANCE, STEP7_REPORT_SCHEMA_VERSION, STEP7_SENSITIVITY_REL_TOL,
    STEP7_TIER1_COARSE_MESH_M, STEP7_TIER2_DRIVES_V_PEAK, SensitivityPoint, Step7Finalist,
    Step7Report, airflow_match_residual_pct, campaign_duct_config, campaign_f_target_hz,
    campaign_profile_options, check_criterion_1, check_criterion_2, coarse_screening_fidelity,
    equal_airflow_voltage, grid_candidates, lhs_candidates, production_fidelity, profile_csv,
    quota_selection, resample_winner_u,
};
pub use convergence::{
    PORT_FLOW_PROMOTION_SCHEMA_VERSION, PortFlowConvergenceLevel, PortFlowPhasor,
    PortFlowPromotionEvidence, PortFlowPromotionGate, PortFlowPromotionReport,
    validate_port_flow_promotion,
};
pub use duct1d::{
    DUCT1D_DEFAULT_SEGMENTS, Duct1dConfig, Duct1dPoint, DuctEnd, FIRST_HIGHER_MODE_ROOT,
    INTERIOR_END_CORRECTION_FACTOR, PortFidelity, compact_port_impedance_pa_s_per_m3,
    cutoff_frequency_hz, duct_inertance_kg_per_m4, effective_mouth_radius_m,
    interior_end_correction_m, select_port_fidelity, solve_duct_1d, tuning_frequency_hz,
};
pub use error::PortFlowError;
pub use evaluate::{
    PortCandidateMetrics, PortEvalDrive, PortEvalFidelity, PortEvalHealth, SecondTone,
    SmallSignalReference, TIER2_EVAL_PROVENANCE, TIER2_EVAL_SCHEMA_VERSION,
    TIER2_LINEAR_RESIDUAL_OK_MAX, TIER2_MIN_SAMPLES_PER_CYCLE, TIER2_THROAT_FRACTION_OF_BORE,
    Tier2EvalContext, Tier2EvalOutcome,
};
pub use finalist::{
    FinalistNoiseLevel, STEP8_MIN_NOISE_SNAPSHOTS, STEP8_NOISE_HARMONICS,
    STEP8_NOISE_OBSERVER_DISTANCE_M, STEP8_NOISE_SNAPSHOTS_PER_CYCLE, STEP8_REPORT_PROVENANCE,
    STEP8_REPORT_SCHEMA_VERSION, STEP8_SPOTCHECK_DISAGREE_REL_TOL, STEP8_SPOTCHECK_DRIVES_V_PEAK,
    STEP8_SPOTCHECK_FLARE_LENGTH_M, STEP8_SPOTCHECK_PROVENANCE, STEP8_SPOTCHECK_TIER,
    SpotCheck3dFidelity, SpotCheck3dLevel, SpotCheck3dRun, SpotVerdict, Step8Report,
    analyze_tail_noise, finalist_noise_spec, noise_observer_1m_on_axis, noise_snapshot_fidelity,
    noise_snapshot_stride, relative_ratio_disagreement, run_3d_spot_check, spline_3d_geometry,
    spot_drive_frequency_hz, spotcheck_template, tail_snapshot_window, tail_snapshots,
};
pub use geometry::{
    MARKER_BOX_CAP, MARKER_EXTERIOR_CAP, MARKER_WALL, SlotPortGeometry, build_slot_port_mesh,
};
pub use geometry3d::{
    RoundPortGeometry, SplineRoundPortGeometry, build_round_port_mesh,
    build_spline_round_port_mesh, refine_round_port_mesh_projected,
};
pub use handoff::{
    WINNER_EVIDENCE_PATHS, WINNER_F_TARGET_HZ, WINNER_N_CONTROL_POINTS, WINNER_PROFILE_PROVENANCE,
    WINNER_PROFILE_SAMPLE_STATIONS, WINNER_PROFILE_SCHEMA_VERSION, WINNER_TIER1_FLOW_RATIO,
    WINNER_TIER1_FLUX_RELATIVE_DIFFERENCE, WINNER_TIER1_IMPEDANCE_PA_S_PER_M3,
    WINNER_TIER1_MESH_ELEMENTS, WINNER_TIER1_MESH_NODES, WINNER_TIER1_RELATIVE_RESIDUAL,
    WINNER_TIER1_SCALAR_DOFS, WINNER_U, WinnerProfile, reproduce_winner_tier1, winner_profile,
    winner_profile_for_case, winner_template, winner_tier1_frequency_hz, winner_tier1_geometry,
    winner_tier1_mesh_node_counts, winner_u,
};
pub use linear_reference::{
    ComplexPortTransfer, LinearReferenceConfig, ReferenceRadiusProfile,
    RoundPortLinearReferenceConfig, SplinePortLinearReferenceConfig,
    solve_round_port_linear_reference, solve_spline_port_linear_reference,
};
pub use optim_gate::{
    STEP10_CASE_ID, STEP10_MIN_TIER0_FEASIBLE, STEP10_REPORT_PROVENANCE,
    STEP10_REPORT_SCHEMA_VERSION, Step10GateConfig, Step10GateReport, Step10Tier0Stage,
    Step10Tier1Stage, Step10Tier2Fidelity, Step10Tier2Stage, run_step10_gate,
};
pub use optimize::{
    AlBobyqaConfig, BudgetUsage, CBO_TIER2_BUDGET_THRESHOLD, CONSTRAINT_NAMES, CboSearchConfig,
    DEFAULT_AIRFLOW_MIN_RATIO, DEFAULT_CACHE_QUANTIZATION_STEP, DEFAULT_COMPRESSION_MAX_DB,
    DEFAULT_TUNING_TOLERANCE_HZ, Duct1dTier0Evaluator, EngineChoice, EngineUsed, EvalTier,
    EvaluatedCandidate, FeasibilitySpec, FunnelConfig, FunnelOutcome, FunnelReport,
    HistoryOutcomeKind, HistoryRecord, OptimHistory, PARETO_VIOLATION_PENALTY,
    PORT_OPTIM_HISTORY_PROVENANCE, PORT_OPTIM_HISTORY_SCHEMA_VERSION, ParetoCandidate,
    ParetoResult, ParetoSearchConfig, PortOptimError, PortOptimFunnel, REJECTED_OBJECTIVE,
    RankedCandidate, ScalarSearchConfig, ScalarWeights, ScalarizedResult, SplineTier1Evaluator,
    StageReport, Tier0Evaluator, Tier0Obs, Tier1Evaluator, Tier1Obs, Tier2Evaluator,
    Tier2HarnessEvaluator, TierBudgets, campaign_template, free_variable_bounds, pack_u,
    tier2_slot_template,
};
pub use profile::{
    ClampedCubicSpline, DEFAULT_SHAPE_CURVATURE_BUDGET_M3, DEFAULT_SHAPE_SLOPE_MAX, KnotSpacing,
    PROFILE_VALIDATION_GRID_POINTS, PchipSpline, ProfileOptions, ProfileSplineKind, ShapeLimits,
    SplinePortProfile,
};
pub use promotion::{
    FinestPairChange, MeshIndependenceCheck, PromotionExtendedChanges, PromotionGateMirror,
    PromotionLadderConfig, PromotionLevelDiagnostics, PromotionMeshStats, PromotionReductionInputs,
    PromotionReportMirror, STEP9_MAX_CAP_CONTINUITY_RELATIVE_ERROR,
    STEP9_MAX_FINEST_PAIR_RELATIVE_CHANGE, STEP9_MAX_FLOW_RATIO_RELATIVE_ERROR,
    STEP9_MAX_IMPEDANCE_RATIO_RELATIVE_ERROR, STEP9_MAX_MACH, STEP9_MAX_THD_FRACTION,
    STEP9_MAX_TRANSFER_IDENTITY_RELATIVE_ERROR, STEP9_PROMOTION_PROVENANCE,
    STEP9_REPORT_PROVENANCE, STEP9_REPORT_SCHEMA_VERSION, STEP9_SPATIAL_MESH_SIZES_M,
    STEP9_TEMPORAL_SAMPLES_PER_CYCLE, STEP9_THD_HARMONICS, STEP9_THD_RESOLUTION_FLOOR,
    STEP9_TIER1_FLARE_LENGTH_M, STEP9_TIER1_MESH_SIZE_M, Step9LadderOutcome, Step9LinearGateInputs,
    Step9LinearGateReport, Step9Report, Step9Verdict, Tier1TransferPoint, build_promotion_case,
    check_extended_changes, check_finest_pair, check_linear_transfer_gate, check_thd_finest_pair,
    evaluate_step9_verdict, min_profile_area_m2, reduce_promotion_trace, run_step9_ladder,
    solve_step9_tier1, step9_promotion_gate, step9_structural_gate,
};
pub use runtime::{
    PortFlowRuntimeCase, PortFlowRuntimePlan, PortFlowRuntimePromotion,
    PortFlowRuntimePromotionArtifact, PortFlowRuntimeRefinement, replay_port_flow_promotion,
    run_port_flow_promotion, run_port_flow_promotion_with_observer,
};
pub use spline_geometry::{
    DEFAULT_MORPH_SMOOTHING_PASSES, SplinePortGeometry, min_triangle_area_m2,
};
