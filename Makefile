CONFIG ?= configs/pong.yaml
PY     ?= .venv/bin/python

.PHONY: setup smoke train-ez train-dqn train demo live clean

setup:
	bash setup.sh

# 2-minute end-to-end check of every code path (tiny nets, tiny runs)
smoke:
	$(PY) -m src.efficientzero.train --config $(CONFIG) --smoke
	$(PY) -m src.dqn.train           --config $(CONFIG) --smoke
	$(PY) -m demo.build_demo         --config $(CONFIG) --episodes 1 --showcase-sims 4

train-ez:
	$(PY) -m src.efficientzero.train --config $(CONFIG)

train-dqn:
	$(PY) -m src.dqn.train --config $(CONFIG)

# run both at once (needs ~2 GPUs or a lot of patience on one)
train:
	bash scripts/train_all.sh $(CONFIG)

demo:
	$(PY) -m demo.build_demo --config $(CONFIG)

live:
	#$(PY) -m demo.live_demo --ckpt runs/ez_pong/checkpoints/ez_final.pt \
	#                        --ckpt2 runs/dqn_pong/checkpoints/dqn_final.pt
	$(PY) -m demo.live_demo --ckpt runs/ez_pong/checkpoints/ez_step00100000.pt \
	                        --ckpt2 runs/dqn_pong/checkpoints/dqn_step01000000.pt

clean:
	rm -rf demo_out/_cards
