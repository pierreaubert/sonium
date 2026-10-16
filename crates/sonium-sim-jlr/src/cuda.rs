//! CUDA D3Q19 solver mirroring the CPU core in f32.
//!
//! Contains the NVRTC-compiled stream/collide kernels and their host
//! driver. The kernels implement the same semantics as [`crate::lbm`]
//! (BGK collision, Smagorinsky LES via the non-equilibrium second
//! moment, halfway bounce-back, equilibrium plug inlet, zero-gradient
//! outlet) in single precision; CPU↔GPU parity is gated on a shortened
//! case before any production timing is reported. Averaging of the axial
//! velocity happens on-device so production runs never download the
//! distribution field.

// Rust guideline compliant 2026-02-21

use std::sync::Arc;

use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg as _,
};
use cudarc::nvrtc::{CompileOptions, compile_ptx_with_opts};

/// Launch configuration with 256-thread blocks (the collide kernel's
/// register use rules out 1024-thread blocks).
fn launch_cfg(n: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (n.div_ceil(256), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    }
}

use crate::lbm::Cell;

/// CUDA kernel source, compiled at solver construction via NVRTC.
const KERNEL_SRC: &str = r#"
__constant__ int CX[19] = {1,-1,0,0,0,0,1,-1,1,-1,1,-1,1,-1,0,0,0,0,0};
__constant__ int CY[19] = {0,0,1,-1,0,0,1,1,-1,-1,0,0,0,0,1,-1,1,-1,0};
__constant__ int CZ[19] = {0,0,0,0,1,-1,0,0,0,0,1,1,-1,-1,1,1,-1,-1,0};
__constant__ float WT[19] = {
  0.0555555558f,0.0555555558f,0.0555555558f,0.0555555558f,0.0555555558f,0.0555555558f,
  0.0277777778f,0.0277777778f,0.0277777778f,0.0277777778f,0.0277777778f,0.0277777778f,
  0.0277777778f,0.0277777778f,0.0277777778f,0.0277777778f,0.0277777778f,0.0277777778f,
  0.3333333333f};
__constant__ int OPP[19] = {1,0,3,2,5,4,9,8,7,6,13,12,11,10,17,16,15,14,18};
__device__ float feq_i(int i, float rho, float ux, float uy, float uz) {
  float cu = CX[i]*ux + CY[i]*uy + CZ[i]*uz;
  float u2 = ux*ux + uy*uy + uz*uz;
  return WT[i]*rho*(1.0f + 3.0f*cu + 4.5f*cu*cu - 1.5f*u2);
}
extern "C" __global__ void lbm_init(float* f, const unsigned char* cells, int n) {
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  if (idx >= n) return;
  for (int i = 0; i < 19; i++) f[idx*19 + i] = feq_i(i, 1.0f, 0.0f, 0.0f, 0.0f);
}
__constant__ int MEXP[19][3] = {
  {0,0,0},{1,0,0},{0,1,0},{0,0,1},{2,0,0},{0,2,0},{0,0,2},{1,1,0},{1,0,1},{0,1,1},
  {2,1,0},{2,0,1},{1,2,0},{0,2,1},{1,0,2},{0,1,2},{2,2,0},{0,2,2},{2,0,2}};
__device__ int moment_index_d(int a, int b, int c) {
  for (int k = 0; k < 19; k++)
    if (MEXP[k][0]==a && MEXP[k][1]==b && MEXP[k][2]==c) return k;
  return -1;
}
__device__ float binom_d(int n, int k) {
  if (k > n) return 0.0f;
  float r = 1.0f;
  for (int i = 0; i < k; i++) r *= (float)(n-i)/(float)(i+1);
  return r;
}
__device__ float pow_u(float base, int e) {
  return e==0 ? 1.0f : (e==1 ? base : base*base);
}
extern "C" __global__ void lbm_collide(
    const float* __restrict__ f, float* __restrict__ fcol,
    const unsigned char* __restrict__ cells, int n,
    float tau0, float cs_smag, float fx, float fy, float fz, int regularized,
    int central, float bulk_tau, float ghost_rate, const float* __restrict__ minv) {
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  if (idx >= n) return;
  const float* src = f + idx*19;
  float* dst = fcol + idx*19;
  if (cells[idx] == 1) { for (int i = 0; i < 19; i++) dst[i] = src[i]; return; }
  float rho = 0.0f; float mx = 0.0f, my = 0.0f, mz = 0.0f;
  for (int i = 0; i < 19; i++) { rho += src[i]; mx += CX[i]*src[i]; my += CY[i]*src[i]; mz += CZ[i]*src[i]; }
  float inv = 1.0f/rho;
  float ux = (mx + 0.5f*fx)*inv, uy = (my + 0.5f*fy)*inv, uz = (mz + 0.5f*fz)*inv;
  if (central) {
    float kap[19] = {0.0f};
    for (int i = 0; i < 19; i++) {
      float dx = CX[i]-ux, dy = CY[i]-uy, dz = CZ[i]-uz;
      for (int k = 0; k < 19; k++)
        kap[k] += src[i]*pow_u(dx,MEXP[k][0])*pow_u(dy,MEXP[k][1])*pow_u(dz,MEXP[k][2]);
    }
    float tau = tau0;
    if (cs_smag > 0.0f) {
      float pxx = kap[4]-rho*0.3333333333f, pyy = kap[5]-rho*0.3333333333f, pzz = kap[6]-rho*0.3333333333f;
      float pp = pxx*pxx+pyy*pyy+pzz*pzz + 2.0f*(kap[7]*kap[7]+kap[8]*kap[8]+kap[9]*kap[9]);
      float q = sqrtf(2.0f*pp);
      float rhs = cs_smag*cs_smag*q/(2.0f*rho*0.1111111111f);
      tau = 0.5f*(tau0 + sqrtf(tau0*tau0 + 4.0f*rhs));
    }
    float omega_nu = 1.0f/tau;
    float omega_bulk = 1.0f/bulk_tau;
    float mean = (kap[4]+kap[5]+kap[6])/3.0f;
    float mean_new = mean - omega_bulk*(mean - rho*0.3333333333f);
    float knew[19];
    for (int k = 0; k < 19; k++) {
      int order = MEXP[k][0]+MEXP[k][1]+MEXP[k][2];
      float keq = (k==0) ? rho : (k>=4 && k<=6) ? rho*0.3333333333f
                : (k>=16) ? rho*0.1111111111f : 0.0f;
      if (order >= 3) knew[k] = kap[k] - ghost_rate*(kap[k]-keq);
      else if (order == 2 && k >= 7) knew[k] = kap[k] - omega_nu*kap[k];
      else knew[k] = kap[k];
    }
    for (int kk = 4; kk <= 6; kk++) {
      float dev = kap[kk]-mean;
      knew[kk] = mean_new + dev - omega_nu*dev;
    }
    // Fourth-order cumulant targets from the relaxed second moments.
    knew[16] = (knew[4]*knew[5] + 2.0f*knew[7]*knew[7])/rho;
    knew[18] = (knew[4]*knew[6] + 2.0f*knew[8]*knew[8])/rho;
    knew[17] = (knew[5]*knew[6] + 2.0f*knew[9]*knew[9])/rho;
    float raw[19] = {0.0f};
    for (int k = 0; k < 19; k++) {
      int a=MEXP[k][0], b=MEXP[k][1], c=MEXP[k][2];
      float sum = 0.0f;
      for (int al=0; al<=a; al++) for (int be=0; be<=b; be++) for (int ga=0; ga<=c; ga++) {
        int sk = moment_index_d(al,be,ga);
        if (sk < 0) continue;
        sum += binom_d(a,al)*binom_d(b,be)*binom_d(c,ga)
             * pow_u(ux,a-al)*pow_u(uy,b-be)*pow_u(uz,c-ga)*knew[sk];
      }
      raw[k] = sum;
    }
    raw[1] += fx; raw[2] += fy; raw[3] += fz;
    for (int i = 0; i < 19; i++) {
      float fi = 0.0f;
      for (int k = 0; k < 19; k++) fi += minv[i*19+k]*raw[k];
      dst[i] = fi;
    }
    return;
  }
  float feq[19];
  for (int i = 0; i < 19; i++) feq[i] = feq_i(i, rho, ux, uy, uz);
  float pij[6] = {0.0f,0.0f,0.0f,0.0f,0.0f,0.0f};
  if (cs_smag > 0.0f || regularized) {
    for (int i = 0; i < 19; i++) {
      float d = src[i] - feq[i];
      pij[0] += CX[i]*CX[i]*d; pij[1] += CY[i]*CY[i]*d; pij[2] += CZ[i]*CZ[i]*d;
      pij[3] += CX[i]*CY[i]*d; pij[4] += CX[i]*CZ[i]*d; pij[5] += CY[i]*CZ[i]*d;
    }
  }
  float tau = tau0;
  if (cs_smag > 0.0f) {
    float pp = pij[0]*pij[0]+pij[1]*pij[1]+pij[2]*pij[2]
             + 2.0f*(pij[3]*pij[3]+pij[4]*pij[4]+pij[5]*pij[5]);
    float q = sqrtf(2.0f*pp);
    float rhs = cs_smag*cs_smag*q/(2.0f*rho*0.1111111111f);
    tau = 0.5f*(tau0 + sqrtf(tau0*tau0 + 4.0f*rhs));
  }
  float omega = 1.0f/tau;
  float pref = 1.0f - 0.5f*omega;
  for (int i = 0; i < 19; i++) {
    float cu = CX[i]*ux + CY[i]*uy + CZ[i]*uz;
    float guo = ((CX[i]-ux)*fx + (CY[i]-uy)*fy + (CZ[i]-uz)*fz)*3.0f
              + cu*(CX[i]*fx + CY[i]*fy + CZ[i]*fz)*9.0f;
    float s = src[i];
    if (regularized) {
      float q = (CX[i]*CX[i]-0.3333333333f)*pij[0] + (CY[i]*CY[i]-0.3333333333f)*pij[1]
              + (CZ[i]*CZ[i]-0.3333333333f)*pij[2]
              + 2.0f*(CX[i]*CY[i]*pij[3] + CX[i]*CZ[i]*pij[4] + CY[i]*CZ[i]*pij[5]);
      s = feq[i] + WT[i]*4.5f*q;
    }
    dst[i] = s - omega*(s - feq[i]) + pref*WT[i]*guo;
  }
}
extern "C" __global__ void lbm_stream(
    const float* __restrict__ fcol, float* __restrict__ fnext,
    const unsigned char* __restrict__ cells, int nx, int ny, int nz,
    float ramp, const float* __restrict__ inlet_vel, float ouvx, float ouvy, float ouvz, int onx, int ony, int onz,
    int perx, int pery, int perz) {
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  int n = nx*ny*nz;
  if (idx >= n) return;
  unsigned char cell = cells[idx];
  float* dst = fnext + idx*19;
  if (cell == 1) { for (int i = 0; i < 19; i++) dst[i] = fcol[idx*19+i]; return; }
  if (cell == 2) {
    float ux = inlet_vel[idx*3]*ramp, uy = inlet_vel[idx*3+1]*ramp, uz = inlet_vel[idx*3+2]*ramp;
    for (int i = 0; i < 19; i++) dst[i] = feq_i(i, 1.0f, ux, uy, uz);
    return;
  }
  int x = idx % nx; int y = (idx/nx) % ny; int z = idx/(nx*ny);
  if (cell == 3) {
    int ix = min(max(x+onx, 0), nx-1), iy = min(max(y+ony, 0), ny-1), iz = min(max(z+onz, 0), nz-1);
    int s = ix + nx*(iy + ny*iz);
    for (int i = 0; i < 19; i++) dst[i] = fcol[s*19+i];
    return;
  }
  float rho = 0.0f;
  for (int i = 0; i < 19; i++) rho += fcol[idx*19+i];
  for (int i = 0; i < 19; i++) {
    int sx = x - CX[i], sy = y - CY[i], sz = z - CZ[i];
    if (perx) sx = ((sx % nx) + nx) % nx;
    if (pery) sy = ((sy % ny) + ny) % ny;
    if (perz) sz = ((sz % nz) + nz) % nz;
    if (sx < 0 || sx >= nx || sy < 0 || sy >= ny || sz < 0 || sz >= nz) {
      dst[i] = fcol[idx*19 + OPP[i]];
      continue;
    }
    int s = sx + nx*(sy + ny*sz);
    if (cells[s] == 2) {
      // Non-equilibrium extrapolation inlet (Guo et al. 2002).
      float ux = inlet_vel[s*3]*ramp, uy = inlet_vel[s*3+1]*ramp, uz = inlet_vel[s*3+2]*ramp;
      float rho_d = 0.0f, mx = 0.0f, my = 0.0f, mz = 0.0f;
      for (int j = 0; j < 19; j++) {
        float fj = fcol[idx*19+j];
        rho_d += fj; mx += CX[j]*fj; my += CY[j]*fj; mz += CZ[j]*fj;
      }
      float feq_b = feq_i(i, rho_d, ux, uy, uz);
      float feq_d = feq_i(i, rho_d, mx/rho_d, my/rho_d, mz/rho_d);
      dst[i] = feq_b + (fcol[idx*19+i] - feq_d);
    }
    else if (cells[s] == 3) {
      // Fixed-flow outlet: NEEM with local density, imposed velocity.
      float rho_d = 0.0f, mx = 0.0f, my = 0.0f, mz = 0.0f;
      for (int j = 0; j < 19; j++) {
        float fj = fcol[idx*19+j];
        rho_d += fj; mx += CX[j]*fj; my += CY[j]*fj; mz += CZ[j]*fj;
      }
      float feq_b = feq_i(i, rho_d, ouvx, ouvy, ouvz);
      float feq_d = feq_i(i, rho_d, mx/rho_d, my/rho_d, mz/rho_d);
      dst[i] = feq_b + (fcol[idx*19+i] - feq_d);
    }
    else if (cells[s] == 1) dst[i] = fcol[idx*19 + OPP[i]];
    else dst[i] = fcol[s*19 + i];
  }
}
extern "C" __global__ void lbm_accum_uz(
    const float* __restrict__ f, const unsigned char* __restrict__ cells,
    float* __restrict__ avg_uz, int n) {
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  if (idx >= n || cells[idx] == 1) return;
  float rho = 0.0f, mz = 0.0f;
  for (int i = 0; i < 19; i++) { rho += f[idx*19+i]; mz += CZ[i]*f[idx*19+i]; }
  avg_uz[idx] += mz/rho;
}
"#;

/// Runs one operating point on the CUDA solver (uniform grid).
///
/// Mirrors [`crate::case::run_case_cpu`]; the refined level-II patch is a
/// CPU-path feature, so CUDA production runs use the uniform level-I grid
/// and say so in their provenance.
///
/// # Errors
/// Returns [`CudaError`] on device, compile, launch, or transfer failure.
pub fn run_case_cuda(
    cfg: &crate::case::CaseConfig,
    op: crate::case::OperatingPoint,
    geom: &crate::geometry::JlrGeometry,
) -> CudaResult<crate::case::CaseResult> {
    use crate::lbm::tau_for_viscosity;
    let dx_m = cfg.dx_mm * 1.0e-3;
    let nu_lb = cfg.nu_m2_s * cfg.dt_s / (dx_m * dx_m);
    let tau = tau_for_viscosity(nu_lb);
    let mut geom = geom.clone();
    geom.rotation_deg = cfg.rotation_deg;
    let (cells, nx, ny, nz) = geom.voxelise(cfg.dx_mm);
    let a_feed = std::f64::consts::PI * (geom.nozzle_inlet_diameter_mm * 1.0e-3 / 2.0).powi(2);
    let v_feed_lb = (op.external_flow_m3_s / a_feed) * cfg.dt_s / dx_m;
    let inlet_field: Option<Vec<[f64; 3]>> =
        if cfg.inlet_profile == crate::case::InletProfile::Parabolic {
            let r_feed_mm = geom.nozzle_inlet_diameter_mm / 2.0;
            let mut field = vec![[0.0; 3]; nx * ny * nz];
            for z in 0..nz {
                for y in 0..ny {
                    for x in 0..nx {
                        let idx = x + nx * (y + ny * z);
                        if cells[idx] != crate::lbm::Cell::Inlet {
                            continue;
                        }
                        let xm = (x as f64 + 0.5) * cfg.dx_mm - nx as f64 * cfg.dx_mm / 2.0;
                        let ym = (y as f64 + 0.5) * cfg.dx_mm - ny as f64 * cfg.dx_mm / 2.0;
                        let r = (xm * xm + ym * ym).sqrt();
                        let shape = 2.0 * (1.0 - (r / r_feed_mm).powi(2)).max(0.0);
                        field[idx] = [0.0, 0.0, -v_feed_lb * shape];
                    }
                }
            }
            Some(field)
        } else {
            None
        };
    let mut lat = CudaLattice::with_inlet_field(
        nx,
        ny,
        nz,
        tau,
        cfg.smagorinsky,
        cfg.regularized,
        [0.0; 3],
        [0.0, 0.0, -v_feed_lb],
        [0, 0, 1],
        [false; 3],
        &cells,
        inlet_field.as_deref(),
    )?;
    lat.set_central_moment(cfg.central_moment);
    {
        let a_patch =
            std::f64::consts::PI * (geom.baffle_diameter_mm * 1.0e-3 / 2.0).powi(2);
        let v_out_lb = (op.external_flow_m3_s / a_patch) * cfg.dt_s / dx_m;
        lat.set_outlet_velocity([0.0, 0.0, -v_out_lb]);
    }
    let total_steps = (cfg.total_time_s / cfg.dt_s).round() as u64;
    let avg_start_step = (cfg.averaging_start_s / cfg.dt_s).round() as u64;
    let ramp_steps = cfg.ramp_time_s / cfg.dt_s;
    let t0 = std::time::Instant::now();
    for step in 0..total_steps {
        let ramp = ((step as f64 + 1.0) / ramp_steps).min(1.0);
        lat.step(ramp, step >= avg_start_step)?;
        if cfg.log_every_steps > 0 && step % cfg.log_every_steps == 0 {
            eprintln!("jlr-cuda step {step}/{total_steps} t={:.3}s", step as f64 * cfg.dt_s);
        }
    }
    let wall = t0.elapsed().as_secs_f64();
    let vel_scale = dx_m / cfg.dt_s;
    let avg_uz: Vec<f64> = lat
        .download_avg_uz()?
        .iter()
        .map(|v| v * vel_scale)
        .collect();
    Ok(crate::case::extract_result(cfg, op, &geom, &avg_uz, nx, ny, nz, total_steps, wall))
}

/// Failure of CUDA solver construction or execution.
#[derive(Debug)]
pub struct CudaError(pub String);

impl std::fmt::Display for CudaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CUDA solver error: {}", self.0)
    }
}
impl std::error::Error for CudaError {}

type CudaResult<T> = Result<T, CudaError>;

/// GPU lattice mirroring [`crate::lbm::Lattice`] in f32.
#[derive(Debug)]
pub struct CudaLattice {
    stream: Arc<CudaStream>,
    collide_fn: CudaFunction,
    stream_fn: CudaFunction,
    accum_fn: CudaFunction,
    f: CudaSlice<f32>,
    f_col: CudaSlice<f32>,
    f_next: CudaSlice<f32>,
    avg_uz: CudaSlice<f32>,
    minv: CudaSlice<f32>,
    central: i32,
    bulk_tau: f32,
    ghost_rate: f32,
    outlet_velocity: [f32; 3],
    inlet_vel: CudaSlice<f32>,
    cells: CudaSlice<u8>,
    nx: usize,
    ny: usize,
    nz: usize,
    tau: f32,
    smagorinsky: f32,
    regularized: i32,
    force: [f32; 3],
    outlet_normal: [i32; 3],
    periodic: [i32; 3],
    steps: u64,
    avg_count: u64,
}

impl CudaLattice {
    /// Builds the GPU lattice on device 0 and initialises rest state.
    ///
    /// # Errors
    /// Returns [`CudaError`] if no CUDA device, NVRTC compilation, or a
    /// device allocation fails.
    pub fn new(
        nx: usize,
        ny: usize,
        nz: usize,
        tau: f64,
        smagorinsky: Option<f64>,
        force: [f64; 3],
        inlet_velocity: [f64; 3],
        outlet_normal: [i32; 3],
        cells: &[Cell],
    ) -> CudaResult<Self> {
        Self::with_periodic(nx, ny, nz, tau, smagorinsky, true, force, inlet_velocity, outlet_normal, [false; 3], cells)
    }

    /// Builds the GPU lattice with per-axis periodic wrapping.
    ///
    /// # Errors
    /// Returns [`CudaError`] if no CUDA device, NVRTC compilation, or a
    /// device allocation fails.
    #[allow(clippy::too_many_arguments)]
    pub fn with_periodic(
        nx: usize,
        ny: usize,
        nz: usize,
        tau: f64,
        smagorinsky: Option<f64>,
        regularized: bool,
        force: [f64; 3],
        inlet_velocity: [f64; 3],
        outlet_normal: [i32; 3],
        periodic: [bool; 3],
        cells: &[Cell],
    ) -> CudaResult<Self> {
        Self::with_inlet_field(nx, ny, nz, tau, smagorinsky, regularized, force, inlet_velocity, outlet_normal, periodic, cells, None)
    }

    /// Builds the GPU lattice with a per-cell inlet velocity field.
    ///
    /// # Errors
    /// Returns [`CudaError`] if no CUDA device, NVRTC compilation, or a
    /// device allocation fails.
    #[allow(clippy::too_many_arguments)]
    pub fn with_inlet_field(
        nx: usize,
        ny: usize,
        nz: usize,
        tau: f64,
        smagorinsky: Option<f64>,
        regularized: bool,
        force: [f64; 3],
        inlet_velocity: [f64; 3],
        outlet_normal: [i32; 3],
        periodic: [bool; 3],
        cells: &[Cell],
        inlet_vel_cells: Option<&[[f64; 3]]>,
    ) -> CudaResult<Self> {
        let ctx = CudaContext::new(0).map_err(|e| CudaError(format!("{e:?}")))?;
        let (major, minor) = ctx
            .compute_capability()
            .map_err(|e| CudaError(format!("{e:?}")))?;
        let arch = match (major, minor) {
            (12, _) => "compute_120",
            (10, _) => "compute_100",
            (9, _) => "compute_90",
            (8, 9) => "compute_89",
            (8, _) => "compute_80",
            _ => "compute_80",
        };
        let opts = CompileOptions { arch: Some(arch), ..Default::default() };
        let ptx = compile_ptx_with_opts(KERNEL_SRC, opts)
            .map_err(|e| CudaError(format!("NVRTC: {e:?}")))?;
        let module = ctx.load_module(ptx).map_err(|e| CudaError(format!("{e:?}")))?;
        let stream = ctx.default_stream();
        let n = nx * ny * nz;
        let cells_u8: Vec<u8> = cells.iter().map(|c| *c as u8).collect();
        let mut cells_dev = stream
            .alloc_zeros::<u8>(n)
            .map_err(|e| CudaError(format!("{e:?}")))?;
        stream
            .memcpy_htod(&cells_u8, &mut cells_dev)
            .map_err(|e| CudaError(format!("{e:?}")))?;
        let mut inlet_host = vec![0.0f32; n * 3];
        for (idx, v) in inlet_host.chunks_mut(3).enumerate() {
            let src = inlet_vel_cells
                .map(|f| f[idx])
                .unwrap_or(inlet_velocity);
            v[0] = src[0] as f32;
            v[1] = src[1] as f32;
            v[2] = src[2] as f32;
        }
        let mut inlet_dev = stream
            .alloc_zeros::<f32>(n * 3)
            .map_err(|e| CudaError(format!("{e:?}")))?;
        stream
            .memcpy_htod(&inlet_host, &mut inlet_dev)
            .map_err(|e| CudaError(format!("{e:?}")))?;
        let inv = crate::lbm::moment_inverse();
        let mut minv_host = vec![0.0f32; 19 * 19];
        for i in 0..19 {
            for k in 0..19 {
                minv_host[i * 19 + k] = inv[i][k] as f32;
            }
        }
        let mut minv_dev = stream
            .alloc_zeros::<f32>(19 * 19)
            .map_err(|e| CudaError(format!("{e:?}")))?;
        stream
            .memcpy_htod(&minv_host, &mut minv_dev)
            .map_err(|e| CudaError(format!("{e:?}")))?;
        let mut lat = Self {
            collide_fn: module.load_function("lbm_collide").map_err(|e| CudaError(format!("{e:?}")))?,
            stream_fn: module.load_function("lbm_stream").map_err(|e| CudaError(format!("{e:?}")))?,
            accum_fn: module
                .load_function("lbm_accum_uz")
                .map_err(|e| CudaError(format!("{e:?}")))?,
            stream,
            f: ctx
                .default_stream()
                .alloc_zeros::<f32>(n * 19)
                .map_err(|e| CudaError(format!("{e:?}")))?,
            f_col: ctx
                .default_stream()
                .alloc_zeros::<f32>(n * 19)
                .map_err(|e| CudaError(format!("{e:?}")))?,
            f_next: ctx
                .default_stream()
                .alloc_zeros::<f32>(n * 19)
                .map_err(|e| CudaError(format!("{e:?}")))?,
            avg_uz: ctx
                .default_stream()
                .alloc_zeros::<f32>(n)
                .map_err(|e| CudaError(format!("{e:?}")))?,
            minv: minv_dev,
            central: 0,
            bulk_tau: 1.0,
            ghost_rate: 1.0,
            outlet_velocity: [0.0; 3],
            inlet_vel: inlet_dev,
            cells: cells_dev,
            nx,
            ny,
            nz,
            tau: tau as f32,
            smagorinsky: smagorinsky.unwrap_or(0.0) as f32,
            regularized: regularized as i32,
            force: [force[0] as f32, force[1] as f32, force[2] as f32],
            outlet_normal,
            periodic: [periodic[0] as i32, periodic[1] as i32, periodic[2] as i32],
            steps: 0,
            avg_count: 0,
        };
        let init_fn = module.load_function("lbm_init").map_err(|e| CudaError(format!("{e:?}")))?;
        let n_i32 = n as i32;
        unsafe {
            lat.stream
                .launch_builder(&init_fn)
                .arg(&mut lat.f)
                .arg(&lat.cells)
                .arg(&n_i32)
                .launch(launch_cfg(n as u32))
                .map_err(|e| CudaError(format!("{e:?}")))?;
        }
        lat.stream.synchronize().map_err(|e| CudaError(format!("{e:?}")))?;
        Ok(lat)
    }

    /// Number of completed steps.
    #[must_use]
    pub fn steps(&self) -> u64 {
        self.steps
    }

    /// Selects the central-moment (cascaded) collision on device.
    pub fn set_central_moment(&mut self, on: bool) {
        self.central = i32::from(on);
    }

    /// Sets the imposed fixed-flow outlet velocity (lattice units).
    pub fn set_outlet_velocity(&mut self, u: [f64; 3]) {
        self.outlet_velocity = [u[0] as f32, u[1] as f32, u[2] as f32];
    }

    /// Blocks until all queued kernels complete (timing boundary).
    ///
    /// # Errors
    /// Returns [`CudaError`] on a synchronisation failure.
    pub fn synchronize(&self) -> CudaResult<()> {
        self.stream.synchronize().map_err(|e| CudaError(format!("{e:?}")))
    }

    /// Advances one step; accumulates mean axial velocity if requested.
    ///
    /// # Errors
    /// Returns [`CudaError`] on a kernel launch failure.
    pub fn step(&mut self, ramp: f64, accumulate: bool) -> CudaResult<()> {
        let n = (self.nx * self.ny * self.nz) as i32;
        let (nx, ny, nz) = (self.nx as i32, self.ny as i32, self.nz as i32);
        let (tau, cs) = (self.tau, self.smagorinsky);
        let reg = self.regularized;
        let [fx, fy, fz] = self.force;
        unsafe {
            self.stream
                .launch_builder(&self.collide_fn)
                .arg(&self.f)
                .arg(&mut self.f_col)
                .arg(&self.cells)
                .arg(&n)
                .arg(&tau)
                .arg(&cs)
                .arg(&fx)
                .arg(&fy)
                .arg(&fz)
                .arg(&reg)
                .arg(&self.central)
                .arg(&self.bulk_tau)
                .arg(&self.ghost_rate)
                .arg(&self.minv)
                .launch(launch_cfg(n as u32))
                .map_err(|e| CudaError(format!("{e:?}")))?;
        }
        let ramp_f = ramp as f32;
        let [onx, ony, onz] = self.outlet_normal;
        unsafe {
            self.stream
                .launch_builder(&self.stream_fn)
                .arg(&self.f_col)
                .arg(&mut self.f_next)
                .arg(&self.cells)
                .arg(&nx)
                .arg(&ny)
                .arg(&nz)
                .arg(&ramp_f)
                .arg(&self.inlet_vel)
                .arg(&self.outlet_velocity[0])
                .arg(&self.outlet_velocity[1])
                .arg(&self.outlet_velocity[2])
                .arg(&onx)
                .arg(&ony)
                .arg(&onz)
                .arg(&self.periodic[0])
                .arg(&self.periodic[1])
                .arg(&self.periodic[2])
                .launch(launch_cfg(n as u32))
                .map_err(|e| CudaError(format!("{e:?}")))?;
        }
        std::mem::swap(&mut self.f, &mut self.f_next);
        if accumulate {
            unsafe {
                self.stream
                    .launch_builder(&self.accum_fn)
                    .arg(&self.f)
                    .arg(&self.cells)
                    .arg(&mut self.avg_uz)
                    .arg(&n)
                    .launch(launch_cfg(n as u32))
                    .map_err(|e| CudaError(format!("{e:?}")))?;
            }
            self.avg_count += 1;
        }
        self.steps += 1;
        Ok(())
    }

    /// Downloads the time-averaged axial velocity in lattice units.
    ///
    /// # Errors
    /// Returns [`CudaError`] on a transfer failure.
    pub fn download_avg_uz(&mut self) -> CudaResult<Vec<f64>> {
        self.stream.synchronize().map_err(|e| CudaError(format!("{e:?}")))?;
        let host: Vec<f32> = self
            .stream
            .clone_dtoh(&self.avg_uz)
            .map_err(|e| CudaError(format!("{e:?}")))?;
        let denom = (self.avg_count.max(1)) as f64;
        Ok(host.iter().map(|v| *v as f64 / denom).collect())
    }

    /// Downloads the instantaneous density field.
    ///
    /// # Errors
    /// Returns [`CudaError`] on a transfer failure.
    pub fn download_density(&mut self) -> CudaResult<Vec<f64>> {
        self.stream.synchronize().map_err(|e| CudaError(format!("{e:?}")))?;
        let n = self.nx * self.ny * self.nz;
        let host: Vec<f32> = self
            .stream
            .clone_dtoh(&self.f)
            .map_err(|e| CudaError(format!("{e:?}")))?;
        Ok((0..n)
            .map(|idx| (0..19).map(|i| host[idx * 19 + i] as f64).sum())
            .collect())
    }

    /// Downloads the instantaneous macroscopic velocity in lattice units.
    ///
    /// Intended for small parity cases; downloads the full distribution
    /// field.
    ///
    /// # Errors
    /// Returns [`CudaError`] on a transfer failure.
    pub fn download_velocity(&mut self) -> CudaResult<Vec<[f64; 3]>> {
        self.stream.synchronize().map_err(|e| CudaError(format!("{e:?}")))?;
        let n = self.nx * self.ny * self.nz;
        let host: Vec<f32> = self
            .stream
            .clone_dtoh(&self.f)
            .map_err(|e| CudaError(format!("{e:?}")))?;
        // D3Q19 z-directions, kept in sync with crate::lbm::CZ.
        const CZL: [i32; 19] = [0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, 1, -1, -1, 1, 1, -1, -1, 0];
        const CXL: [i32; 19] = [1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0, 0];
        const CYL: [i32; 19] = [0, 0, 1, -1, 0, 0, 1, 1, -1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 0];
        Ok((0..n)
            .map(|idx| {
                let mut rho = 0.0f64;
                let mut m = [0.0f64; 3];
                for i in 0..19 {
                    let v = host[idx * 19 + i] as f64;
                    rho += v;
                    m[0] += CXL[i] as f64 * v;
                    m[1] += CYL[i] as f64 * v;
                    m[2] += CZL[i] as f64 * v;
                }
                [m[0] / rho, m[1] / rho, m[2] / rho]
            })
            .collect())
    }
}
