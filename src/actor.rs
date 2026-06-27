use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ActorId {
    Lp1,
    Trader1,
    Arbitrageur1,
    Pool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActorRole {
    LiquidityProvider,
    Trader,
    Arbitrageur,
}
