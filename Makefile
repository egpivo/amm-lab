# amm-lab — repo-wide targets.
# RL-equilibrium paper:  make -C scripts/rl_equilibrium help

.PHONY: help build test scenarios rl-help

help:
	@echo "amm-lab — three tracks:"
	@echo "  1. AMM scenarios (practice)     make scenarios"
	@echo "  2. Paper — causality            see README §2, scripts/causality/"
	@echo "  3. Paper — RL equilibrium       make rl-help"
	@echo ""
	@echo "  make build     cargo build --release --bins"
	@echo "  make test      cargo test"
	@echo "  make scenarios controlled-pool scenario runner"
	@echo "  make rl-help   RL-equilibrium paper pipeline"

build:
	cargo build --release --bins

test:
	cargo test

scenarios:
	scripts/run_all_scenarios.sh

rl-help:
	$(MAKE) -C scripts/rl_equilibrium help
