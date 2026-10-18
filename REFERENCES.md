# References

Academic and standards references cited (or directly relied on) by the sonium workspace.
Each entry lists where it is used. Citations live as `//` or `//!` comments at the head of
the relevant module; this file is the index. Inclusion does not imply endorsement — it
documents the literature the implementation was designed against.

Items marked **(canonical)** are the standard reference for a named method that appears
in the code but isn't cited by file/year in the source; included so a reader can find the
underlying algorithm.

---

## Standards

- **ISO 3382-1:2009** — *Acoustics — Measurement of room acoustic parameters.*
  Room-acoustic descriptors (T60, EDT, C50/C80, …).
  `crates/sonium-fem/src/physics/acoustics/room/metrics.rs`
- **ISO 9613-1:1993** — *Acoustics — Attenuation of sound during propagation outdoors.*
  Atmospheric absorption (oxygen/nitrogen relaxation model).
  `crates/sonium-fem/src/physics/acoustics/room/atmosphere.rs`,
  `../sonium-sim-room/src/fem_runner.rs`,
  `crates/sonium-wasm/src/lib.rs`
- **IEC 61672-1:2013** — *Sound level meters — Specifications.*
  A-weighting filter shape.
  `crates/sonium-fem/src/physics/acoustics/sound_metrics.rs`,
  `../sonium-sim-pcfan/src/{model,evaluation}.rs`
- **AES69** — *SOFA file format for spatial-audio data (HRTF/DRIR).*
  `crates/sonium-xem/src/io/sofa.rs`,
  `crates/sonium-bem/src/numcalc/mod.rs`,
  `crates/sonium-bem/tests/mesh2hrtf_pipeline.rs`
- **CEA-2034** — *Standard Method of Measurement for In-Home Loudspeakers* (spinorama).
  `../sonium-sim-speaker/src/cea2034_output.rs`,
  `../sonium-sim-speaker/src/optimization.rs`,
  `crates/sonium-bem/src/speaker/`
- **ITU-R BS.775 / ITU-R BS.2051** — Multichannel sound system loudspeaker placement.
  `../sonium-sim-room/src/scenarios.rs`

## Books

- **Bowman, J. J., Senior, T. B. A., & Uslenghi, P. L. E.** (1987).
  *Electromagnetic and Acoustic Scattering by Simple Shapes.* Hemisphere.
  `crates/sonium-wave/src/analytical/solutions_3d.rs`
- **Morse, P. M., & Ingard, K. U.** (1968).
  *Theoretical Acoustics.* McGraw-Hill.
  `crates/sonium-wave/src/analytical/solutions_3d.rs`
- **Bruneau, M.** (2006).
  *Fundamentals of Acoustics.* ISTE.
  `crates/sonium-fem/src/physics/acoustics/viscothermal.rs`
- **Senturia, S. D.** (2001).
  *Microsystem Design.* Springer.
  `crates/sonium-fem/src/physics/electrostatic.rs`
- **Beranek, L. L., & Mellow, T. J.** (2012).
  *Acoustics: Sound Fields, Transducers and Vibration.* Academic Press, Ch. 6.
  `crates/sonium-fem/src/physics/electrostatic.rs`
- **Sauter, S., & Schwab, C.** (2011).
  *Boundary Element Methods.* Springer.
  `crates/sonium-bem/src/core/integration/duffy.rs`
- **Simo, J. C., & Hughes, T. J. R.** (1998).
  *Computational Inelasticity.* Springer.
  `crates/sonium-fem/src/physics/plasticity/return_mapping.rs`
- **Kampinga, W. R.** (2010).
  *Viscothermal acoustics using finite elements.* PhD thesis, University of Twente.
  `crates/sonium-fem/src/physics/acoustics/viscothermal.rs`
- **Beltman, W. M.** (1999).
  *Viscothermal wave propagation including acousto-elastic interaction.* PhD thesis,
  University of Twente.
  `crates/sonium-fem/src/physics/acoustics/viscothermal.rs`
- **Hoemmen, M.** (2010).
  *Communication-Avoiding Krylov Subspace Methods.* PhD thesis, UC Berkeley.
  `crates/sonium-solvers/src/iterative/ca_cg.rs`

---

## Iterative & Krylov solvers (sonium-solvers)

- **Saad, Y., & Schultz, M. H.** (1986).
  "GMRES: A generalized minimal residual algorithm for solving nonsymmetric linear
  systems." *SIAM J. Sci. Stat. Comput.* 7(3), 856–869.
  `crates/sonium-solvers/src/iterative/gmres.rs`,
  `crates/sonium-compute/src/gmres.rs` (Saad §9.3 right-preconditioned form)
- **Saad, Y.** (1993).
  "A flexible inner–outer preconditioned GMRES algorithm." *SIAM J. Sci. Comput.* 14(2).
  `crates/sonium-solvers/src/iterative/fgmres.rs`
- **Saad, Y.** *Iterative Methods for Sparse Linear Systems.* (canonical — ILUT
  drop-tolerance algorithm used by `IlutPreconditioner`.)
  `crates/sonium-solvers/src/preconditioners/ilut.rs`
- **Morgan, R. B.** (2002).
  "GMRES with Deflated Restarting." *SIAM J. Sci. Comput.* 24(1), 20–37.
  Used both as the GMRES-DR algorithm and for harmonic-Ritz vector extraction in
  GCRO-DR.
  `crates/sonium-solvers/src/iterative/gmres_dr.rs`,
  `crates/sonium-solvers/src/iterative/gcro_dr.rs`
- **Parks, M. L., de Sturler, E., Mackey, G., Johnson, D. D., & Maiti, S.** (2006).
  "Recycling Krylov Subspaces for Sequences of Linear Systems."
  *SIAM J. Sci. Comput.* 28(5), 1651–1674. (GCRO-DR)
  `crates/sonium-solvers/src/iterative/gmres_recycled.rs`,
  `crates/sonium-solvers/src/iterative/gcro_dr.rs`
- **Panagiotopoulos, D., Deckers, E., & Desmet, W.** (2020 / 2022).
  Practical GCRO-DR for acoustic FEM frequency sweeps. *JCAM / CMAME.*
  `crates/sonium-solvers/src/iterative/gcro_dr.rs`
- **Sonneveld, P., & van Gijzen, M. B.** (2008).
  "IDR(s): A family of simple and fast algorithms for solving large nonsymmetric
  systems of linear equations." *SIAM J. Sci. Comput.* 31(2).
  `crates/sonium-solvers/src/iterative/idrs.rs`
- **Ghysels, P., Ashby, T., Meerbergen, K., & Vanroose, W.** (2013).
  "Hiding global communication latency in the GMRES algorithm on massively parallel
  machines." *SIAM J. Sci. Comput.* 35.
  `crates/sonium-solvers/src/iterative/gmres_pipelined.rs`
- **Cools, S., & Vanroose, W.** (2017).
  "The communication-hiding pipelined BiCGStab method for the parallel solution of
  large unsymmetric linear systems." *Parallel Comput.* 65.
  `crates/sonium-solvers/src/iterative/bicgstab_pipelined.rs`
- **Carson, E., & Higham, N. J.** (2018).
  "Accelerating the solution of linear systems by iterative refinement in three
  precisions." *SIAM J. Sci. Comput.* 40(2), Algorithm 5.3. (FGMRES-IR)
  `crates/sonium-compute/src/gmres_ir.rs`,
  `crates/sonium-bem/src/core/solver/fmm_interface.rs`,
  `crates/sonium-fem/src/physics/stokes/navier_stokes.rs`,
  `crates/sonium-solvers/src/bin/gmres_ir_scaling.rs`

## Helmholtz preconditioners (FEM)

- **Erlangga, Y. A., Oosterlee, C. W., & Vuik, C.** (2006).
  "A class of preconditioners for the Helmholtz equation." *Comput. Methods Appl. Mech.
  Engrg.* 195. (Shifted-Laplacian / CSLP)
  `crates/sonium-fem/src/solver/mod.rs`,
  `crates/sonium-fem/src/assembly/assembler.rs`,
  `../sonium-sim-room/src/fem_runner.rs` (used as the default precond shift)
- **Erlangga, Y. A., & Nabben, R.** (2008).
  Deflation-based preconditioning for the Helmholtz equation. Combined with:
  **Gaul, A., Gutknecht, M. H., Liesen, J., & Nabben, R.** (2013).
  "A framework for deflated and augmented Krylov subspace methods."
  *SIAM J. Matrix Anal. Appl.* 34(2).
  `crates/sonium-solvers/src/iterative/gmres_deflated.rs`,
  `crates/sonium-fem/src/solver/deflation.rs`
- **Galkowski, J., Gander, M. J., et al.** (2024).
  "Optimised Schwarz methods with PML transmission conditions for the Helmholtz
  equation." arXiv:2408.16580.
  `crates/sonium-fem/src/physics/acoustics/schwarz_pml/{mod,config,solver,…}.rs`,
  `crates/sonium-fem/src/solver/mod.rs`
- **Cui, Y., & Jiang, Z.** (2024).
  "Neural Multigrid for Heterogeneous Helmholtz Equations." arXiv:2404.02493.
  `crates/sonium-fem/src/physics/acoustics/neural_multigrid/{mod,adr,chebyshev,
  config,eikonal,hierarchy,solver}.rs`
- **Appelo, D., Garcia, F., & Runborg, O.** (2020).
  "WaveHoltz: Iterative Solution of the Helmholtz Equation via the Wave Equation."
  *SIAM J. Sci. Comput.*
  **Appelo, D., Banks, J. W., Henshaw, W. D., & Schillinger, D.** (2025).
  "A Multi-Frequency Helmholtz Solver Based on the WaveHoltz Algorithm."
  `crates/sonium-fem/src/physics/acoustics/waveholtz/`,
  `crates/sonium-fem/benches/waveholtz_*`
- **Zhao, H.** (2005).
  "A fast sweeping method for Eikonal equations." *Math. Comp.* 74(250).
  `crates/sonium-fem/src/physics/acoustics/neural_multigrid/eikonal.rs`

## AMG / multigrid

- **Henson, V. E., & Yang, U. M.** (2002).
  "BoomerAMG: A parallel algebraic multigrid solver and preconditioner."
  *Appl. Numer. Math.* 41(1). (HYPRE coarsening algorithm.)
  `crates/sonium-solvers/src/preconditioners/amg/coarsening.rs`

## Eigenvalue solvers

- **Knyazev, A. V.** (2001).
  "Toward the Optimal Preconditioned Eigensolver: Locally Optimal Block
  Preconditioned Conjugate Gradient Method." *SIAM J. Sci. Comput.*
  `crates/sonium-solvers/src/eigen/lobpcg.rs`

## Direct solvers / low-rank compression

- **Amestoy, P., Buttari, A., L'Excellent, J.-Y., & Mary, T.** (2019).
  "Performance and Scalability of the Block Low-Rank Multifrontal Factorization on
  Multicore Architectures." *ACM TOMS* 45(1). (MUMPS-BLR.)
  `crates/sonium-solvers/src/direct/multifrontal_blr/{front,blr,hss,panel,
  truncation,…}.rs`

## BEM / boundary integrals (sonium-bem)

- **Burton, A. J., & Miller, G. F.** (1971) **(canonical)**.
  "The application of integral equation methods to the numerical solution of some
  exterior boundary-value problems." *Proc. R. Soc. London A* 323. (Burton-Miller
  hypersingular formulation used throughout TBEM/FMM assembly.)
  `crates/sonium-bem/src/core/assembly/{mod,mlfmm}.rs`, BEM examples and QA bins.
- **Duffy, M. G.** (1982).
  "Quadrature over a pyramid or cube of integrands with a singularity at a vertex."
  *SIAM J. Numer. Anal.* 19(6), 1260–1262.
  `crates/sonium-bem/src/core/integration/duffy.rs`
- **Lebedev, V. I.** (1976).
  Sphere-quadrature symmetry-class weights.
  `crates/sonium-bem/src/core/integration/lebedev.rs`
- **Cheng, H., Crutchfield, W. Y., Gimbutas, Z., Greengard, L., et al.** (2006).
  "A wideband fast multipole method for the Helmholtz equation in three dimensions."
  *J. Comput. Phys.* 216. (Wideband M2L kernel.)
  `crates/sonium-bem/src/core/assembly/mlfmm.rs`
- **FastBEM Verification Cases.** https://fastbem.com/VerificationCases.html
  `../sonium-accuracy/crates/sonium-qa/bin/bem_fastbem_validation.rs`

## FFT (sonium-fft)

- **Cooley, J. W., & Tukey, J. W.** (1965).
  "An algorithm for the machine calculation of complex Fourier series."
  *Math. Comp.* 19(90), 297–301.
  `crates/sonium-fft/`
- **Stockham, T. G.** (1966).
  "High-speed convolution and correlation." *Proc. AFIPS Spring Joint Comput. Conf.*
  (Auto-sort mixed-radix driver.)
  `crates/sonium-fft/src/stockham.rs`
- **Bluestein, L. I.** (1970).
  "A linear filtering approach to the computation of the discrete Fourier transform."
  *IEEE Trans. Audio Electroacoust.* (chirp-Z transform — handles non-smooth lengths.)
  `crates/sonium-fft/src/bluestein.rs`
- **Wang, H., et al.** (2023).
  "OpenFFT: An Adaptive Tuning Framework for 3D FFT on ARM Multicore CPUs."
  *ICS '23.* DOI:10.1145/3577193.3593735.
  `crates/sonium-fft/README.md`

## Hardware kernels

- **Deng, Y., et al.** (2025).
  "Demystifying ARM SME to Optimize General Matrix Multiplications."
  arXiv:2512.21473.
  `crates/sonium-compute/src/sme2/`

## Porous / viscothermal / impedance models

- **Delany, M. E., & Bazley, E. N.** (1970).
  "Acoustical properties of fibrous absorbent materials." *Applied Acoustics* 3(2).
  `crates/sonium-fem/src/physics/acoustics/impedance.rs`,
  `crates/sonium-xem/src/config.rs`
- **Miki, Y.** (1990).
  "Acoustical properties of porous materials — Modifications of Delany-Bazley models."
  *J. Acoust. Soc. Jpn.* 11(1).
  `crates/sonium-fem/src/physics/acoustics/impedance.rs`,
  `crates/sonium-xem/src/config.rs`
- **Stinson, M. R.** (1991).
  "The propagation of plane sound waves in narrow and wide circular tubes, and
  generalization to uniform tubes of arbitrary cross-sectional shape." *JASA* 89(2).
  `crates/sonium-fem/src/physics/acoustics/viscothermal.rs`

## Edge diffraction

- **Svensson, U. P., Fred, R. I., & Vanderkooy, J.** (1999).
  "An analytic secondary source model of edge diffraction impulse responses." *JASA.*
  `crates/sonium-wasm/src/lib.rs`

## Aeroacoustics & flow-acoustic coupling

- **Gutin, L.** (1948).
  "On the sound field of a rotating propeller."
  `crates/sonium-fem/src/physics/aeroacoustics/propeller_source.rs`
- **Lighthill, M. J.** (1952).
  "On sound generated aerodynamically. I. General theory." *Proc. R. Soc. A* 211.
  Quadrupole radiation, exact and far-field forms.
  `crates/sonium-fem/src/physics/acoustics/aeroacoustics/lighthill.rs`,
  `crates/sonium-fem/src/physics/acoustics/aeroacoustics/lighthill_2d_axi.rs`,
  `crates/sonium-fem/src/physics/acoustics/aeroacoustics/distributed.rs`,
  `crates/sonium-fem/src/physics/acoustics/aeroacoustics/snapshot.rs`
- **Curle, N.** (1955) **(canonical)**.
  "The influence of solid boundaries upon aerodynamic sound." *Proc. R. Soc. A* 231.
  `../sonium-sim-pcfan/src/evaluation.rs` (Curle's equation form for TE noise)
- **Ingard, U.** (1959).
  "Influence of fluid motion past a plane boundary on sound reflection, absorption,
  and transmission." *JASA* 31.
  `crates/sonium-fem/src/physics/aeroacoustics/boundary.rs`
- **Tyler, J. M., & Sofrin, T. G.** (1962).
  "Axial flow compressor noise studies." *SAE Trans.* 70.
  Rotor-stator spinning-mode orders and annular-duct cut-on.
  `crates/sonium-wave/src/analytical/annular_duct_modes.rs`,
  `crates/sonium-fem/src/physics/acoustics/aeroacoustics/duct_modes.rs`,
  `crates/sonium-fem/src/physics/acoustics/aeroacoustics/propeller_source.rs`,
  `crates/sonium-sim-pcfan/src/annular_fwh.rs`
- **Sharland, I. J.** (1964).
  "Sources of noise in axial flow fans." *J. Sound Vib.* 1(3).
  `../sonium-sim-pcfan/src/evaluation.rs`
- **Lowson, M. V.** (1965).
  "The sound field for singularities in motion."
  `crates/sonium-fem/src/physics/aeroacoustics/propeller_source.rs`
- **Ffowcs Williams, J. E., & Hawkings, D. L.** (1969).
  "Sound generation by turbulence and surfaces in arbitrary motion."
  *Phil. Trans. R. Soc. A* 264. Stationary permeable surfaces here;
  moving-surface kinematics are out of scope.
  `crates/sonium-fem/src/physics/acoustics/aeroacoustics/fwh.rs`,
  `crates/sonium-fem/src/physics/acoustics/aeroacoustics/fwh_2d_axi.rs`,
  `crates/sonium-fem/src/physics/acoustics/aeroacoustics/snapshot.rs`,
  `crates/sonium-sim-speaker-modeling/src/port_noise.rs`,
  `crates/sonium-sim-pcfan/src/annular_fwh.rs`,
  `crates/sonium-sim-propellers/src/fwh_reference.rs`
- **Goldstein, M. E.** (1976).
  *Aeroacoustics.* McGraw-Hill. Uniform-flow convected wave equation and
  Garrick-triangle Green's function.
  `crates/sonium-wave/src/special/helmholtz.rs`,
  `crates/sonium-fem/src/physics/acoustics/aeroacoustics/fwh_2d_axi.rs`
- **Hanson, D. B.** (1980).
  "Helicoidal surface theory for harmonic noise of propellers." *JASA* 68(2).
  `crates/sonium-fem/src/physics/aeroacoustics/propeller_source.rs`
- **Myers, M. K.** (1980).
  "On the acoustic boundary condition in the presence of flow." *J. Sound Vib.* 71(3).
  `crates/sonium-fem/src/physics/aeroacoustics/boundary.rs`
- **Brooks, T. F., Pope, D. S., & Marcolini, M. A.** (1989).
  *Airfoil Self-Noise and Prediction.* NASA RP-1218. (BPM TBL-TE noise model.)
  `../sonium-sim-pcfan/src/{model,evaluation}.rs`
- **Tam, C. K. W., & Webb, J. C.** (1993).
  "Dispersion-relation-preserving finite difference schemes for computational
  acoustics." *J. Comput. Phys.* 107. Seven-point optimized stencil with
  selective artificial damping.
  `crates/sonium-fem/src/physics/acoustics/aeroacoustics/drp_euler.rs`,
  `crates/sonium-fem/src/physics/acoustics/aeroacoustics/drp_structured.rs`,
  `crates/sonium-fem/src/physics/acoustics/aeroacoustics/drp_axi.rs`
- **Gottlieb, S., & Shu, C.-W.** (1998).
  "Total variation diminishing Runge-Kutta schemes." *Math. Comp.* 67.
  SSPRK(3,3) time stepping for the DG-LEE solver.
  `crates/sonium-fem/src/physics/acoustics/aeroacoustics/dg_lee.rs`
- **Cockburn, B., & Shu, C.-W.** (2001).
  "Runge-Kutta discontinuous Galerkin methods for convection-dominated
  problems." *J. Sci. Comput.* 16. Strong-form upwind DG with exact
  Riemann flux.
  `crates/sonium-fem/src/physics/acoustics/aeroacoustics/dg_lee.rs`
- **Fukano, T., & Jang, C.-M.** (2004).
  Tip-leakage vortex broadband-noise empirical model.
  `../sonium-sim-pcfan/src/evaluation.rs`
- **Hesthaven, J. S., & Warburton, T.** (2008).
  *Nodal Discontinuous Galerkin Methods.* Springer. Nodal basis,
  lift operators, and the strong/weak-form equivalence fixing the
  surface-correction sign.
  `crates/sonium-fem/src/physics/acoustics/aeroacoustics/dg_lee.rs`
- **Brambley, E. J.** (2009).
  "Fundamental problems with the model of uniform flow over acoustic linings."
  *J. Sound Vib.* 322.
  `crates/sonium-fem/src/physics/aeroacoustics/boundary.rs`

## Loudspeaker / lumped-circuit acoustic models

- **Roozen, N. B., et al.** (2008).
  "Reduction of bass-reflex port nonlinearities." *JASA.*
  `crates/sonium-fem/src/physics/lumped_circuit/acoustic_elements.rs`

## Stokes / Navier–Stokes / turbulence

- **Tezduyar, T. E., & Osawa, Y.** (2000).
  "Finite element stabilization parameters computed from element matrices and
  vectors." *Comput. Methods Appl. Mech. Engrg.* 190(3–4).
  `crates/sonium-fem/src/physics/stokes/stabilisation.rs`
- **Codina, R.** (2001).
  "A stabilized finite element method for generalized stationary incompressible
  flows." *Comput. Methods Appl. Mech. Engrg.* 190.
  `crates/sonium-fem/src/physics/stokes/stabilisation.rs`
- **Menter, F. R.** (1994) **(canonical)**.
  "Two-equation eddy-viscosity turbulence models for engineering applications."
  *AIAA J.* 32(8). (k-ω SST.)
  `crates/sonium-fem/src/physics/stokes/turbulence.rs`

## Plates, shells, beams, plasticity

- **Batoz, J.-L., Bathe, K.-J., & Ho, L.-W.** (1980).
  "A study of three-node triangular plate bending elements." *Int. J. Numer. Methods
  Engrg.* 15(12). (DKT — discrete Kirchhoff triangle.)
  `crates/sonium-fem/src/physics/shell/dkt.rs`
- **Engel, G., Garikipati, K., Hughes, T. J. R., Larson, M. G., Mazzei, L., & Taylor, R. L.**
  (2002).
  "Continuous/discontinuous finite element approximations of fourth-order elliptic
  problems in structural and continuum mechanics with applications to thin beams
  and plates, and strain gradient elasticity." *Comput. Methods Appl. Mech. Engrg.*
  191. (Biharmonic SIP.)
  `crates/sonium-fem/src/physics/biharmonic.rs`

## Optimization

- **Tanabe, R., & Fukunaga, A. S.** (2014).
  "Improving the search performance of SHADE using linear population size reduction."
  *IEEE CEC.* (L-SHADE.)
  `../sonium-sim-pcfan/src/optimizer.rs`,
  `../sonium-sim-pcfan/src/bin/pcfan_opt.rs`
- **Gao, F., & Han, L.** (2012).
  "Implementing the Nelder–Mead simplex algorithm with adaptive parameters."
  *Comput. Optim. Appl.* 51(1).
  `crates/sonium-solvers/src/optimization/unconstrained/nelder_mead_impl.rs`

## Time integration

- **Newmark, N. M.** (1959) **(canonical)**.
  "A method of computation for structural dynamics." *J. Eng. Mech. Div., ASCE*
  85(EM3). (Newmark-β implicit time integration.)
  `crates/sonium-fem/src/physics/acoustics/time_domain.rs`,
  `crates/sonium-fem/src/physics/acoustics/waveholtz/time_stepper.rs`,
  `crates/sonium-fem/src/physics/electroacoustics/time_domain.rs`,
  `crates/sonium-fem/src/physics/lumped_circuit/`

## Benchmark / validation references

- **De Vahl Davis, G.** (1983).
  "Natural convection of air in a square cavity: A bench mark numerical solution."
  *Int. J. Numer. Methods Fluids* 3(3). (Nu ≈ 2.243 at Ra=10⁴.)
  `crates/sonium-fem/examples/ns_natural_convection.rs`
- **Armaly, B. F., Durst, F., Pereira, J. C. F., & Schönung, B.** (1983).
  "Experimental and theoretical investigation of backward-facing step flow."
  *J. Fluid Mech.* 127. (x_r/h ≈ 6 at Re=100.)
  `crates/sonium-fem/examples/ns_backward_step.rs`
- **Ghia, U., Ghia, K. N., & Shin, C. T.** (1982).
  "High-Re solutions for incompressible flow using the Navier–Stokes equations and
  a multigrid method." *J. Comput. Phys.* 48, 387–411.
  `../sonium-accuracy/crates/sonium-qa/bin/fem_ns_lid_driven_cavity.rs`
- **Schäfer, M., & Turek, S.** (1996).
  DFG flow-past-cylinder benchmark.
  `../sonium-accuracy/crates/sonium-qa/bin/fem_ns_flow_past_cylinder.rs`,
  `crates/sonium-mesher/src/mesh/generators.rs`
- **Pironneau, O.** (1973).
  "On optimum profiles in Stokes flow." *J. Fluid Mech.* 59(1). (Ogival drag-optimal
  profile.)
  `crates/sonium-fem/examples/stokes_shape_opt.rs`
- **Pironneau, O.** (1974).
  "On optimum design in fluid mechanics." *J. Fluid Mech.* 64(1).
  (Fixed-area drag minimisation applied here to steady Navier–Stokes flow.)
  `crates/sonium-fem/src/physics/fluid/stokes/drag_shape_opt.rs`,
  `crates/sonium-fem/examples/ns_shape_opt.rs`

## Comparison codes (calibration references — not redistributed)

- **PETSc** — iteration-count and accuracy parity benchmark.
  `crates/sonium-solvers/benches/{petsc_compare_bench,ffi}/`,
  `crates/sonium-solvers/src/bin/benchmark/`
- **HYPRE** — Chebyshev / AMG default parameters (`λ_max / λ_min = 30`, 5 smoother
  steps, MIS-2 coarsening).
  `crates/sonium-solvers/src/preconditioners/{chebyshev,amg}/`,
  `crates/sonium-fem/src/physics/stokes/navier_stokes.rs`
- **MUMPS-BLR** — Block Low-Rank multifrontal ordering pattern reference.
  `crates/sonium-solvers/src/direct/multifrontal_blr/front.rs`

---

## How to update this file

When adding a new module that builds on a paper, book, or standard:

1. Cite it in the module's top-of-file `//!` doc comment with author, year, and venue.
2. Add an entry here under the appropriate section. Keep one line per where-used path
   and group multiple sites for the same reference together.
3. If the reference is for a *named* method (e.g. "Burton-Miller", "Newmark-β") but the
   source files only mention the name, mark the entry **(canonical)** so future readers
   know it was added retroactively rather than from a direct in-source citation.
