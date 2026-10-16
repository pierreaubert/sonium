//! Typed errors for `sonium-materials` (Microsoft Rust: `thiserror`, no panics on input).

/// Every fallible operation in this crate.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum MaterialError {
    /// Schema version in a serialised pack is not supported.
    #[error("unsupported material schema version {found}, expected {expected}")]
    SchemaVersion {
        /// Version found in the file.
        found: u32,
        /// Version this build supports.
        expected: u32,
    },
    /// Checksum verification failed — file was tampered or corrupted.
    #[error("material pack checksum mismatch: stored {stored}, computed {computed}")]
    ChecksumMismatch {
        /// Checksum stored in the file.
        stored: String,
        /// Checksum recomputed over the payload.
        computed: String,
    },
    /// A required field failed its physical-domain validation.
    #[error("invalid material {material_id}: field `{field}`: {reason}")]
    InvalidField {
        /// Material identifier.
        material_id: String,
        /// Offending field name.
        field: &'static str,
        /// Human-readable reason.
        reason: String,
    },
    /// Duplicate material id inside a pack.
    #[error("duplicate material id `{0}` in pack")]
    DuplicateId(String),
    /// Material id not present in a pack.
    #[error("material id `{0}` not found in pack")]
    NotFound(String),
    /// An adapter cannot be satisfied from the record — fail-closed named gap.
    ///
    /// `gap` is a stable machine-readable token (for example
    /// `missing_elastic_isotropic`) so callers can match on it.
    #[error("adapter `{adapter}` cannot convert material `{material_id}`: gap `{gap}`: {detail}")]
    AdapterGap {
        /// Adapter name, for example `PanelMaterial`.
        adapter: &'static str,
        /// Material that could not be converted.
        material_id: String,
        /// Stable gap token.
        gap: &'static str,
        /// Human-readable detail.
        detail: String,
    },
    /// Interpolation request outside the tabulated domain (no extrapolation).
    #[error("interpolation out of domain for material `{material_id}`: {detail}")]
    OutOfDomain {
        /// Material identifier.
        material_id: String,
        /// Detail including requested vs available range.
        detail: String,
    },
    /// B-H curve monotonicity / domain violation (mirrors `sonium-fem` `BhCurve`).
    #[error("invalid B-H curve for material `{material_id}`: {detail}")]
    InvalidBhCurve {
        /// Material identifier.
        material_id: String,
        /// Detail.
        detail: String,
    },
    /// A computed screening draft was asked to auto-promote — always refused.
    #[error("screening draft from {source_name} ({external_id}) must never auto-promote: {detail}")]
    ScreeningPromotionRefused {
        /// Source database name.
        source_name: String,
        /// External identifier.
        external_id: String,
        /// Detail.
        detail: String,
    },
    /// Serialisation failure.
    #[error("material serialisation failed: {0}")]
    Serialisation(String),
    /// Deserialisation failure.
    #[error("material deserialisation failed: {0}")]
    Deserialisation(String),
}
