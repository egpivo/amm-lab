use crate::actor::ActorId;
use crate::amount::{TokenAmount, TokenId};
use crate::error::AmmError;
use std::collections::HashMap;

pub struct Ledger {
    balances: HashMap<(ActorId, TokenId), TokenAmount>,
}

impl Default for Ledger {
    fn default() -> Self {
        Self::new()
    }
}

impl Ledger {
    pub fn new() -> Self {
        Ledger {
            balances: HashMap::new(),
        }
    }

    pub fn mint(&mut self, actor: ActorId, token: TokenId, amount: TokenAmount) {
        *self.balances.entry((actor, token)).or_insert(0) += amount;
    }

    pub fn debit(
        &mut self,
        actor: ActorId,
        token: TokenId,
        amount: TokenAmount,
    ) -> Result<(), AmmError> {
        let bal = self.balances.entry((actor, token)).or_insert(0);
        if *bal < amount {
            return Err(AmmError::InsufficientLiquidity);
        }
        *bal -= amount;
        Ok(())
    }

    pub fn credit(&mut self, actor: ActorId, token: TokenId, amount: TokenAmount) {
        *self.balances.entry((actor, token)).or_insert(0) += amount;
    }

    pub fn balance(&self, actor: ActorId, token: TokenId) -> TokenAmount {
        *self.balances.get(&(actor, token)).unwrap_or(&0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mint_and_balance() {
        let mut ledger = Ledger::new();
        ledger.mint(ActorId::Lp1, TokenId::X, 1_000_000);
        assert_eq!(ledger.balance(ActorId::Lp1, TokenId::X), 1_000_000);
    }

    #[test]
    fn test_debit_success() {
        let mut ledger = Ledger::new();
        ledger.mint(ActorId::Trader1, TokenId::Y, 500_000);
        ledger.debit(ActorId::Trader1, TokenId::Y, 200_000).unwrap();
        assert_eq!(ledger.balance(ActorId::Trader1, TokenId::Y), 300_000);
    }

    #[test]
    fn test_debit_insufficient() {
        let mut ledger = Ledger::new();
        ledger.mint(ActorId::Trader1, TokenId::X, 100);
        assert!(ledger.debit(ActorId::Trader1, TokenId::X, 200).is_err());
        assert_eq!(ledger.balance(ActorId::Trader1, TokenId::X), 100);
    }

    #[test]
    fn test_balance_zero_for_unknown() {
        let ledger = Ledger::new();
        assert_eq!(ledger.balance(ActorId::Arbitrageur1, TokenId::X), 0);
    }
}
