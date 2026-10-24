//! Electromagnetics physics modules (stage-0 grouping).
//!
//! Each sub-module implements the generic form traits for one physics
//! domain; see `super` for the layer contract.

// Rust guideline compliant 2026-02-21

pub mod eddy_current;
pub mod electrostatic;
pub mod magnetostatics;
pub mod maxwell;
pub mod rotating_machine;
pub mod semiconductor;
