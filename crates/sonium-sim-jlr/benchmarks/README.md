# Benchmarks

Run from the `sonium/` workspace:

    cargo run --release -p sonium-sim-jlr --features cuda --bin jlr-bench -- --cuda --steps 500
    cargo run --release -p sonium-sim-jlr --bin jlr-bench -- --steps 20 --dx-mm 3.0

Recorded 2026-10-16 on the workspace RTX PRO 6000 Blackwell (driver
610.57, CUDA 13.3): CUDA f32 full level-I grid 594 MLUPS with the
central-moment collision (1038 MLUPS for the BGK kernel); CPU f64
rayon at dx = 3 mm 113 MLUPS. Timing includes a device synchronise
after the timed loop; an unsynchronised number is not a measurement.
