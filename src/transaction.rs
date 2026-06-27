use crate::actor::ActorId;
use crate::amount::{BasisPoints, TokenAmount};
use crate::swap::SwapDirection;
use serde::{Deserialize, Serialize};

mod as_u64 {
    use serde::{Deserialize, Deserializer, Serializer};
    pub fn serialize<S: Serializer>(v: &u128, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_u64(*v as u64)
    }
    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<u128, D::Error> {
        Ok(u64::deserialize(d)? as u128)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum Transaction {
    CreatePool {
        #[serde(with = "as_u64")]
        reserve_x: TokenAmount,
        #[serde(with = "as_u64")]
        reserve_y: TokenAmount,
        fee_bps: BasisPoints,
    },
    AddLiquidity {
        actor: ActorId,
        #[serde(with = "as_u64")]
        amount_x: TokenAmount,
        #[serde(with = "as_u64")]
        amount_y: TokenAmount,
    },
    RemoveLiquidity {
        actor: ActorId,
        #[serde(with = "as_u64")]
        lp_shares: TokenAmount,
    },
    SwapExactIn {
        actor: ActorId,
        direction: SwapDirection,
        #[serde(with = "as_u64")]
        amount_in: TokenAmount,
        #[serde(with = "as_u64")]
        min_amount_out: TokenAmount,
    },
    ExternalPriceMove {
        new_price: f64,
    },
    ArbitrageUntilNoProfit {
        max_steps: u32,
    },
    ReportLpPerformance {
        actor: ActorId,
    },
}
