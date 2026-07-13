# RL-equilibrium paper pipeline (execution routing DQN).
# Run from repo root:  make -C scripts/rl_equilibrium <target>
# Artifacts: data/rl_equilibrium/

REPO_ROOT := $(abspath ../..)
OUT := $(REPO_ROOT)/data/rl_equilibrium
CARGO := cd $(REPO_ROOT) && cargo

.PHONY: help build test train-dqn m3r-final m4 ladder figures manifest verify

help:
	@echo "RL-equilibrium paper pipeline (scripts/rl_equilibrium)"
	@echo ""
	@echo "  make build        cargo build --release --bins"
	@echo "  make test           cargo test"
	@echo "  make train-dqn      six DQN checkpoints (~15 min each)"
	@echo "  make m3r-final      M3R eval matrices + final paper block"
	@echo "  make m4             LP/JIT sensitivity append"
	@echo "  make ladder         final 1000-seed ladder (forced terminal)"
	@echo "  make figures        regenerate PNGs from out/"
	@echo "  make manifest       write m3r_run_manifest.json"
	@echo "  make verify         check frozen artifact numbers"
	@echo ""
	@echo "Outputs: $(OUT)/"

build:
	$(CARGO) build --release --bins

test:
	$(CARGO) test

train-dqn: build
	python3 dqn_train.py --episodes 12000 --tag dynamic_duopoly
	python3 dqn_train.py --episodes 12000 --tag completion_aware --train-penalty 0.08
	python3 dqn_train.py --episodes 12000 --tag order_random --agent-order random
	python3 dqn_train.py --episodes 12000 --tag order_after --agent-order after
	python3 dqn_train.py --episodes 12000 --tag constant_duopoly --mode constant_duopoly
	python3 dqn_train.py --episodes 12000 --tag dynamic_monopoly --mode dynamic_monopoly

m3r-final: build
	$(REPO_ROOT)/target/release/rl_equilibrium_completion --n-seeds 500
	$(REPO_ROOT)/target/release/rl_equilibrium_reference --n-seeds 300
	$(REPO_ROOT)/target/release/rl_equilibrium_planner --n-val 200 --n-seeds 300
	python3 dqn_m3r_eval.py --n-seeds 300 --n-completion 500
	$(REPO_ROOT)/target/release/rl_equilibrium_reference --seed-base 90000 --n-seeds 1000 \
	  --out $(OUT)/m3r_reference_final.csv
	$(REPO_ROOT)/target/release/rl_equilibrium_planner --n-val 200 --n-seeds 1000 --seed-base 90000 \
	  --out-name m3r_stochastic_planner_final.csv
	python3 dqn_final_block.py

m4: build
	$(REPO_ROOT)/target/release/rl_equilibrium_sensitivity --n-seeds 500
	python3 dqn_m4_eval.py

ladder: build
	$(REPO_ROOT)/target/release/rl_equilibrium_ladder --seed-base 90000 --n-seeds 1000 \
	  --completion-rule forced_terminal \
	  --q-coarse $(OUT)/q_table_dynamic_duopoly.json \
	  --q-fine $(OUT)/q_table_dynamic_duopoly_fine.json \
	  --out-name final_ladder.csv

figures:
	python3 plot_final_ladder.py
	python3 m3_figures.py
	python3 m3r_figures.py

manifest:
	python3 make_manifest.py

verify:
	python3 verify_paper_artifacts.py
