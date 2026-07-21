//! Typed PSTT records. No paper pool lists, date windows, or file paths.

use serde::{Deserialize, Serialize};

/// Lowercase 0x-prefixed 20-byte Ethereum address as text.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
#[serde(transparent)]
pub struct Address(pub String);

impl Address {
    pub fn normalize(raw: &str) -> crate::pstt::error::Result<Self> {
        let s = raw.trim().to_ascii_lowercase();
        if !(s.starts_with("0x") && s.len() == 42 && s[2..].chars().all(|c| c.is_ascii_hexdigit()))
        {
            return Err(crate::pstt::error::PsttError::parse(format!(
                "invalid address: {raw}"
            )));
        }
        Ok(Self(s))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for Address {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReferenceKind {
    LastTrade,
    Vwap1s,
}

impl ReferenceKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::LastTrade => "last_trade",
            Self::Vwap1s => "vwap1s",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockHeader {
    pub block_number: u64,
    pub block_hash: String,
    pub parent_hash: String,
    pub timestamp_unix: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwapEvent {
    pub pool: Address,
    pub block: u64,
    pub ts: i64,
    pub tx_index: Option<u32>,
    pub log_index: Option<u32>,
    pub token0: Address,
    pub token1: Address,
    pub amount0: i128,
    pub amount1: i128,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct AggTrade {
    pub timestamp_secs: f64,
    pub price: f64,
    pub quantity: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrientedFill {
    pub pool: Address,
    pub timestamp_unix: f64,
    pub week: String,
    pub q: f64,
    pub p_exec: f64,
    pub direction: f64,
    pub invert: bool,
    pub cex_symbol: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
pub struct WeeklyPrimitives {
    pub l: f64,
    pub a: f64,
    pub b: f64,
    pub s: f64,
    pub observed_mass: u64,
    pub service_q2: f64,
    pub fill_count: u64,
    pub matched_count: u64,
}

impl WeeklyPrimitives {
    pub fn accumulate_mark(&mut self, ell: f64, q: f64) {
        self.l += ell;
        self.a += ell.max(0.0);
        self.b += (-ell).max(0.0);
        self.s += q;
        self.observed_mass += 1;
        self.service_q2 += q * q;
        self.matched_count += 1;
    }

    pub fn identity_residual(&self) -> f64 {
        (self.l - (self.a - self.b)).abs()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeeklyRow {
    pub pool: String,
    pub pair: String,
    pub fee: u32,
    pub reference: String,
    pub week: String,
    #[serde(rename = "L")]
    pub l: f64,
    #[serde(rename = "A")]
    pub a: f64,
    #[serde(rename = "B")]
    pub b: f64,
    #[serde(rename = "S")]
    pub s: f64,
    pub observed_mass: u64,
    pub service_q2: f64,
    pub fill_count: u64,
    pub matched_count: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoolRecord {
    pub pool: Address,
    pub pair: String,
    pub fee: u32,
    pub token0: Address,
    pub token1: Address,
    pub factory: Address,
    pub canonical_factory: bool,
    pub canonical_tokens: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContrastRecord {
    pub contrast_id: String,
    pub pair: String,
    pub base: String,
    pub quote: String,
    pub cex_symbol: String,
    pub invert: bool,
    pub lower_fee: u32,
    pub higher_fee: u32,
    pub pool_lower: Address,
    pub pool_higher: Address,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoolStalenessStats {
    pub blocks: u64,
    pub joined: u64,
    pub coverage: f64,
    pub q50_seconds: Option<f64>,
    pub q90_seconds: Option<f64>,
    pub q99_seconds: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileDigest {
    pub path: String,
    pub bytes: u64,
    pub sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    pub files: Vec<FileDigest>,
    pub file_count: usize,
    pub total_bytes: u64,
}
