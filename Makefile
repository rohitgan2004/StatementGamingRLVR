# LEANDRIFT: statement gaming in RLVR autoformalize-then-prove pipelines.
#
# Local (no GPU / no Lean) targets validate the whole pipeline with the
# simulation policy and the mock Lean backend.  Cloud targets run the real
# Qwen2.5 GRPO training; see docs/CLOUD.md.

PY ?= python
CONFIG ?= configs/base.yaml
CORPUS ?= data/corpus/corpus.json
ARM ?= E
STEPS ?=

.PHONY: help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

.PHONY: setup
setup: ## Install the env/reward-loop package (no GPU)
	$(PY) -m pip install -e .

.PHONY: setup-trl
setup-trl: ## Install the single-GPU TRL/QLoRA training extras
	$(PY) -m pip install -e ".[trl]"

.PHONY: setup-tinker
setup-tinker: ## Install the Thinking Machines Tinker training extras
	$(PY) -m pip install -e ".[tinker]"

.PHONY: corpus
corpus: ## Build the difficulty-calibrated corpus (simulation calibrator)
	leandrift gen-corpus --config $(CONFIG)

.PHONY: validate
validate: ## Validate DRIFTCLASS against the exact MICROPROOF oracle (Appendix A)
	leandrift validate-driftclass

.PHONY: test
test: ## Run the unit test suite
	pytest

.PHONY: reproduce
reproduce: ## Full LOCAL simulation: all arms + baselines + Figure 2 + Tables 4/5
	leandrift validate-driftclass
	leandrift reproduce $(if $(STEPS),--steps $(STEPS),)
	@echo "Figures in ./figures, tables in ./outputs"

.PHONY: figures
figures: ## Regenerate figures/tables from logged runs
	leandrift figures --config $(CONFIG)

# ---- real (cloud GPU) runs -----------------------------------------------------
.PHONY: calibrate
calibrate: ## Real base-model corpus calibration (needs a GPU + Lean)
	$(PY) -m leandrift.corpus.calibrate --config $(CONFIG) --out $(CORPUS)

.PHONY: sft
sft: ## Train the SFT-on-genuine-proofs baseline (TRL/QLoRA)
	$(PY) -m leandrift.rl.sft --config $(CONFIG) --corpus $(CORPUS)

.PHONY: train-tinker
train-tinker: ## Real GRPO on Tinker: make train-tinker ARM=E
	$(PY) -m leandrift.rl.train_tinker --config configs/arm_$(ARM).yaml --corpus $(CORPUS) $(if $(STEPS),--steps $(STEPS),)

.PHONY: train-trl
train-trl: ## Real GRPO on a single GPU (TRL): make train-trl ARM=E
	$(PY) -m leandrift.rl.train_trl --config configs/arm_$(ARM).yaml --corpus $(CORPUS) $(if $(STEPS),--steps $(STEPS),)

.PHONY: lean-build
lean-build: ## Fetch + build the pinned Lean/Mathlib/REPL environment
	cd lean && lake update && lake exe cache get && lake build

.PHONY: clean
clean: ## Remove runs/outputs/figures artifacts
	rm -rf runs outputs figures data/corpus/corpus.json
