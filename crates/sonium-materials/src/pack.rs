//! Versioned, checksummed material packs (JSON + TOML).
//!
//! A pack is the exchange unit. The checksum is SHA-256 over the canonical
//! JSON of the payload (schema version + name + materials sorted by id),
//! so any byte-level tampering or silent edit is detected on load. Loading
//! always validates every record and verifies the checksum — fail closed.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{Material, MaterialError};

/// Current material-pack schema version.
pub const SCHEMA_VERSION: u32 = 1;

/// A versioned collection of [`Material`] records with a content checksum.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MaterialPack {
    /// Schema version (must equal [`SCHEMA_VERSION`]).
    pub schema_version: u32,
    /// Pack name, for example `sonium-seed-pack`.
    pub name: String,
    /// Materials (any order on disk; canonicalised by id for the checksum).
    pub materials: Vec<Material>,
    /// Lowercase hex SHA-256 of the canonical payload (see module docs).
    pub checksum: String,
}

#[derive(Serialize)]
struct CanonicalPayload<'a> {
    schema_version: u32,
    name: &'a str,
    materials: Vec<&'a Material>,
}

impl MaterialPack {
    /// Build a pack and compute its checksum. Validates every record.
    pub fn new(name: &str, mut materials: Vec<Material>) -> Result<Self, MaterialError> {
        materials.sort_by(|a, b| a.id.cmp(&b.id));
        for w in materials.windows(2) {
            if w[0].id == w[1].id {
                return Err(MaterialError::DuplicateId(w[0].id.clone()));
            }
        }
        for m in &materials {
            m.validate()?;
        }
        let mut pack = Self {
            schema_version: SCHEMA_VERSION,
            name: name.to_owned(),
            materials,
            checksum: String::new(),
        };
        pack.checksum = pack.compute_checksum()?;
        Ok(pack)
    }

    /// Recompute the checksum over the canonical payload.
    pub fn compute_checksum(&self) -> Result<String, MaterialError> {
        let mut sorted: Vec<&Material> = self.materials.iter().collect();
        sorted.sort_by(|a, b| a.id.cmp(&b.id));
        let payload = CanonicalPayload {
            schema_version: self.schema_version,
            name: &self.name,
            materials: sorted,
        };
        let bytes = serde_json::to_vec(&payload)
            .map_err(|e| MaterialError::Serialisation(e.to_string()))?;
        let digest = Sha256::digest(&bytes);
        Ok(digest.iter().map(|b| format!("{b:02x}")).collect())
    }

    /// Verify schema version, checksum, duplicates, and every record.
    pub fn verify(&self) -> Result<(), MaterialError> {
        if self.schema_version != SCHEMA_VERSION {
            return Err(MaterialError::SchemaVersion {
                found: self.schema_version,
                expected: SCHEMA_VERSION,
            });
        }
        let computed = self.compute_checksum()?;
        if computed != self.checksum {
            return Err(MaterialError::ChecksumMismatch {
                stored: self.checksum.clone(),
                computed,
            });
        }
        let mut ids: Vec<&str> = self.materials.iter().map(|m| m.id.as_str()).collect();
        ids.sort_unstable();
        for w in ids.windows(2) {
            if w[0] == w[1] {
                return Err(MaterialError::DuplicateId(w[0].to_owned()));
            }
        }
        for m in &self.materials {
            m.validate()?;
        }
        Ok(())
    }

    /// Look up a material by id.
    pub fn get(&self, id: &str) -> Result<&Material, MaterialError> {
        self.materials
            .iter()
            .find(|m| m.id == id)
            .ok_or_else(|| MaterialError::NotFound(id.to_owned()))
    }

    /// Serialise to pretty JSON (checksum included).
    pub fn to_json(&self) -> Result<String, MaterialError> {
        serde_json::to_string_pretty(self).map_err(|e| MaterialError::Serialisation(e.to_string()))
    }

    /// Parse JSON and verify (checksum + validation). Fail closed.
    pub fn from_json(text: &str) -> Result<Self, MaterialError> {
        let pack: Self =
            serde_json::from_str(text).map_err(|e| MaterialError::Deserialisation(e.to_string()))?;
        pack.verify()?;
        Ok(pack)
    }

    /// Serialise to TOML (checksum included).
    pub fn to_toml(&self) -> Result<String, MaterialError> {
        toml::to_string(self).map_err(|e| MaterialError::Serialisation(e.to_string()))
    }

    /// Parse TOML and verify (checksum + validation). Fail closed.
    pub fn from_toml(text: &str) -> Result<Self, MaterialError> {
        let pack: Self =
            toml::from_str(text).map_err(|e| MaterialError::Deserialisation(e.to_string()))?;
        pack.verify()?;
        Ok(pack)
    }
}
