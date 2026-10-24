//! Failure and recovery contracts for full-cycle vector reservations.

// Rust guideline compliant 2026-02-21
use super::*;
use crate::{CpuBackend, MemoryPoolConfig};

#[test]
fn reservation_checks_capacity_and_refunds_partial_quota_on_clear() {
    let pool = MemoryPool::new(CpuBackend, MemoryPoolConfig::with_cap(64)).unwrap();
    let mut workspace = GmresWorkspace::<_, f64>::new(8);
    assert!(workspace.try_reserve_cycle(&pool, 0, true).is_err());
    assert!(workspace.try_reserve_cycle(&pool, usize::MAX, true).is_err());
    assert_eq!(workspace.cached_buffer_count().unwrap(), 0);
    assert!(workspace.try_reserve_cycle(&pool, 4, true).is_err());
    assert_eq!(workspace.cached_buffer_count().unwrap(), 1);
    workspace.clear().unwrap();
    assert_eq!(workspace.cached_buffer_count().unwrap(), 0);
    assert!(pool.try_alloc::<f64>(8).is_ok());
}
