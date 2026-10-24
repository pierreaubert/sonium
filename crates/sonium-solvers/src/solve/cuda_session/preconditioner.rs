//! Concrete retained preconditioners support checked numeric replacement.

// Rust guideline compliant 2026-02-21
use super::*;
use crate::preconditioners::{CudaIluPreconditioner, CudaSchwarzPreconditioner};
use sonium_compute::ChebyshevBackend;

pub(super) enum SessionPreconditioner<T>
where
    T: ComplexField + DeviceScalar<Real = <T as ComplexField>::Real>,
{
    Identity(IdentityDevicePreconditioner),
    Prepared(Box<dyn DevicePreconditioner<CudaBackend, T>>),
    Jacobi(Box<PointJacobiBackend<CudaBackend, T>>),
    Amg(Box<BackendAmgPreconditioner<CudaBackend, T>>),
    Chebyshev(Box<ChebyshevBackend<'static, CudaBackend, T>>),
    Ilu(Box<CudaIluPreconditioner<T>>),
    Schwarz(Box<CudaSchwarzPreconditioner<T>>),
}

impl<T> SessionPreconditioner<T>
where
    T: ComplexField + DeviceScalar<Real = <T as ComplexField>::Real>,
{
    pub(super) fn is_prepared(&self) -> bool {
        matches!(self, Self::Prepared(_))
    }
}

fn amg_config(solver: LinearSolverType) -> AmgConfig {
    if solver == LinearSolverType::CgAmg {
        AmgConfig::for_backend_pcg()
    } else {
        AmgConfig::default()
    }
}

impl<T> SessionPreconditioner<T>
where
    T: ComplexField + DeviceScalar<Real = <T as ComplexField>::Real>,
    <T as ComplexField>::Real: FromPrimitive + Sync + Send,
{
    pub(super) fn try_new(
        backend: &CudaBackend,
        canonical: &CsrMatrix<T>,
        solver: LinearSolverType,
    ) -> Result<Self, LinearSolveError> {
        Ok(match solver {
            LinearSolverType::GmresSchwarz | LinearSolverType::FgmresSchwarz => {
                Self::Schwarz(Box::new(
                    CudaSchwarzPreconditioner::try_from_csr_auto(backend, canonical, 1)
                        .map_err(backend_error)?,
                ))
            }
            LinearSolverType::GmresIlu => Self::Ilu(Box::new(
                CudaIluPreconditioner::try_from_csr(backend, canonical).map_err(backend_error)?,
            )),
            LinearSolverType::GmresChebyshev => {
                let config = ChebyshevConfig::default();
                let degree = config.degree;
                let ratio =
                    <<T as DeviceScalar>::Real as FromPrimitive>::from_f64(config.eigenvalue_ratio)
                        .ok_or_else(|| backend_error("Chebyshev interval is not representable"))?;
                // Use the same checked real-SPD admission and spectral estimate
                // as the CPU facade, before any device preconditioner upload.
                let host = try_chebyshev_preconditioner(canonical, config)?;
                let diagonal = (0..canonical.num_rows)
                    .map(|row| canonical.get(row, row))
                    .collect::<Vec<_>>();
                let matrix = PersistentDeviceCsr::try_from_host_csr_ref(
                    backend,
                    canonical.as_host_csr_ref(),
                )
                .map_err(backend_error)?;
                Self::Chebyshev(Box::new(
                    ChebyshevBackend::try_from_owned_csr(
                        backend,
                        matrix,
                        &diagonal,
                        degree,
                        ratio,
                        host.lambda_max(),
                    )
                    .map_err(backend_error)?,
                ))
            }
            LinearSolverType::GmresJacobi => {
                let diagonal = (0..canonical.num_rows)
                    .map(|row| canonical.get(row, row))
                    .collect::<Vec<_>>();
                Self::Jacobi(Box::new(
                    PointJacobiBackend::try_from_diagonal(backend, &diagonal)
                        .map_err(backend_error)?,
                ))
            }
            LinearSolverType::GmresAmg | LinearSolverType::CgAmg => Self::Amg(Box::new(
                BackendAmgPreconditioner::try_from_csr(backend, canonical, amg_config(solver))
                    .map_err(backend_error)?,
            )),
            _ => Self::Identity(IdentityDevicePreconditioner),
        })
    }

    pub(super) fn action(&self) -> &dyn DevicePreconditioner<CudaBackend, T> {
        match self {
            Self::Identity(action) => action,
            Self::Prepared(action) => action.as_ref(),
            Self::Jacobi(action) => action.as_ref(),
            Self::Amg(action) => action.as_ref(),
            Self::Chebyshev(action) => action.as_ref(),
            Self::Ilu(action) => action.as_ref(),
            Self::Schwarz(action) => action.as_ref(),
        }
    }

    // False means existing device storage was refreshed (or identity retained).
    // A changed AMG hierarchy requires replacement even with fixed finest CSR.
    pub(super) fn refresh(
        &mut self,
        backend: &CudaBackend,
        canonical: &CsrMatrix<T>,
        values: &CudaBuffer<T>,
        solver: LinearSolverType,
        chebyshev_bound: Option<<T as DeviceScalar>::Real>,
    ) -> Result<bool, LinearSolveError> {
        match self {
            Self::Schwarz(action) => {
                let host = try_schwarz_preconditioner(canonical, 1)?;
                if action
                    .try_refresh_from_host(backend, &host)
                    .map_err(backend_error)?
                {
                    Ok(false)
                } else {
                    **action = CudaSchwarzPreconditioner::try_from_host(backend, &host)
                        .map_err(backend_error)?;
                    Ok(true)
                }
            }
            Self::Ilu(action) => {
                if action
                    .try_refresh_from_csr(backend, canonical)
                    .map_err(backend_error)?
                {
                    Ok(false)
                } else {
                    **action = CudaIluPreconditioner::try_from_csr(backend, canonical)
                        .map_err(backend_error)?;
                    Ok(true)
                }
            }
            Self::Chebyshev(action) => {
                let bound = chebyshev_bound.ok_or_else(|| {
                    backend_error("Chebyshev refresh requires a validated spectral bound")
                })?;
                let diagonal = (0..canonical.num_rows)
                    .map(|row| canonical.get(row, row))
                    .collect::<Vec<_>>();
                // Reuse the physical matrix upload as the numeric source; no
                // temporary allocation or second host matrix upload is needed.
                action
                    .try_update_numeric(backend, values, &diagonal, bound)
                    .map_err(backend_error)?;
                Ok(false)
            }
            Self::Identity(_) => Ok(false),
            Self::Prepared(_) => Err(LinearSolveError::InvalidConfig(
                "prepared preconditioning requires explicit replacement or lagging".into(),
            )),
            Self::Jacobi(action) => {
                let diagonal = (0..canonical.num_rows)
                    .map(|row| canonical.get(row, row))
                    .collect::<Vec<_>>();
                action
                    .try_update_diagonal(backend, &diagonal)
                    .map_err(backend_error)?;
                Ok(false)
            }
            Self::Amg(action) => {
                let host = try_amg_preconditioner(canonical, amg_config(solver))?;
                if action
                    .try_refresh_from_host_amg(backend, &host)
                    .map_err(backend_error)?
                {
                    Ok(false)
                } else {
                    **action = BackendAmgPreconditioner::try_from_host_amg(backend, &host)
                        .map_err(backend_error)?;
                    Ok(true)
                }
            }
        }
    }
}
