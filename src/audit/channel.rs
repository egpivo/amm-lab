//! Channel variables and how the public record grades them (Paper C, Section 2.1).

/// A channel variable in the AMM fee-revenue system.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Channel {
    TakeRate,          // rho: LP share of collected fees (LP-side)
    TraderFee,         // c: trader-facing fee (trader-side)
    LiquiditySupply,   // L
    JitResponse,       // J
    RoutingResponse,   // R
    OrderFlow,         // Q
    TraderComposition, // C
    LpWelfare,         // W
    Lvr,               // LVR
}

/// Which market path a channel sits on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Path {
    LpSide,
    TraderSide,
}

impl Channel {
    /// The path a channel belongs to; welfare is the shared terminal node.
    pub fn path(self) -> Path {
        use Channel::*;
        match self {
            TakeRate | LiquiditySupply | JitResponse | LpWelfare => Path::LpSide,
            TraderFee | RoutingResponse | OrderFlow | TraderComposition | Lvr => Path::TraderSide,
        }
    }
}

/// Observability of a variable in the public on-chain record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Observability {
    Measured,      // read directly from logs
    Reconstructed, // computable from logs (e.g. active liquidity)
    Latent,        // no image in the record (trader type, decision set)
}

/// Level at which the intervention behind a variable varies in the record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Support {
    Pool,
    Route,
    Trader,
    Venue,
    None,
}

/// The two maps the audit reads: observability O(v) and support S(v).
pub fn observability(v: Channel) -> Observability {
    use Channel::*;
    match v {
        TakeRate | TraderFee | LiquiditySupply | JitResponse => Observability::Measured,
        LpWelfare => Observability::Reconstructed,
        RoutingResponse | OrderFlow | TraderComposition | Lvr => Observability::Latent,
    }
}

/// Support under the protocol-fee switch: the take-rate varies at the pool level;
/// the trader-facing fee does not vary within fixed-fee history.
pub fn support_under_fee_switch(v: Channel) -> Support {
    match v {
        Channel::TakeRate => Support::Pool,
        Channel::TraderFee => Support::None,
        _ => Support::None,
    }
}
