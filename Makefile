# RL-equilibrium paper pipeline. Order matters: Rust runners create the
# CSVs, DQN eval scripts append to them (and refuse to append twice).
RL := scripts/rl_equilibrium
OUT := experiments/rl_execution/out

.PHONY: build test train-dqn m3r-final m4 ladder figures manifest verify

build:
	cargo build --release --bins

test:
	cargo test

# five DQN variants used by M3R/M4 (~15 min each; run in parallel manually
# if desired)
train-dqn: build
	cd $(RL) && python3 dqn_train.py --episodes 12000 --tag dynamic_duopoly
	cd $(RL) && python3 dqn_train.py --episodes 12000 --tag completion_aware --train-penalty 0.08
	cd $(RL) && python3 dqn_train.py --episodes 12000 --tag order_random --agent-order random
	cd $(RL) && python3 dqn_train.py --episodes 12000 --tag order_after --agent-order after
	cd $(RL) && python3 dqn_train.py --episodes 12000 --tag constant_duopoly --mode constant_duopoly
	cd $(RL) && python3 dqn_train.py --episodes 12000 --tag dynamic_monopoly --mode dynamic_monopoly

# M3R evaluation matrices + final untouched block
m3r-final: build
	./target/release/rl_equilibrium_completion --n-seeds 500
	./target/release/rl_equilibrium_reference --n-seeds 300
	./target/release/rl_equilibrium_planner --n-val 200 --n-seeds 300
	cd $(RL) && python3 dqn_m3r_eval.py --n-seeds 300 --n-completion 500
	./target/release/rl_equilibrium_reference --seed-base 90000 --n-seeds 1000 \
	  --out $(OUT)/m3r_reference_final.csv
	./target/release/rl_equilibrium_planner --n-val 200 --n-seeds 1000 --seed-base 90000 \
	  --out-name m3r_stochastic_planner_final.csv
	cd $(RL) && python3 dqn_final_block.py

m4: build
	./target/release/rl_equilibrium_sensitivity --n-seeds 500
	cd $(RL) && python3 dqn_m4_eval.py

# final paper ladder (untouched seeds, forced terminal)
ladder: build
	./target/release/rl_equilibrium_ladder --seed-base 90000 --n-seeds 1000 \
	  --completion-rule forced_terminal \
	  --q-coarse $(OUT)/q_table_dynamic_duopoly.json \
	  --q-fine $(OUT)/q_table_dynamic_duopoly_fine.json \
	  --out-name final_ladder.csv

figures:
	cd $(RL) && python3 plot_final_ladder.py && python3 m3_figures.py && python3 m3r_figures.py

manifest:
	cd $(RL) && python3 make_manifest.py

verify:
	cd $(RL) && python3 verify_paper_artifacts.py
