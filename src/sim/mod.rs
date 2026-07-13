//! Equilibrium-consistent execution-routing simulator.
//!
//! Closed-loop market lab: an execution trader's actions move pool
//! inventory, dynamic fees, noise routing, and arbitrage response. See
//! `.local/rl_equilibrium/plan.md` for scope, claims, and boundaries.

pub mod amm;
pub mod arbitrage;
pub mod env;
pub mod execution_agent;
pub mod fee;
pub mod noise;
pub mod oracle;
pub mod planner;
pub mod q_learner;
