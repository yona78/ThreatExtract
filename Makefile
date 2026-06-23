UV_CACHE_DIR ?= .uv-cache
UV_RUN = UV_CACHE_DIR=$(UV_CACHE_DIR) uv run --with-requirements requirements-research.txt
PYTHON ?= $(UV_RUN) python
PYTEST ?= $(UV_RUN) pytest
REPORTS ?= reports/fork1
DNRTI_DIR ?= data/dnrti
CACHE_DIR ?= model_cache/fork1
DEVICE ?= mps
SEED ?= 20260621

RUN = $(PYTHON) -m fork1.run_experiment --dnrti-dir $(DNRTI_DIR) --out-dir $(REPORTS) --offline --cache-dir $(CACHE_DIR)
SYNTH = $(PYTHON) -m fork1.synthesis --reports-dir $(REPORTS) --out-dir $(REPORTS) --dnrti-dir $(DNRTI_DIR) --cache-dir $(CACHE_DIR)

.PHONY: setup-research test legacy e1 e2 e3 e4 e5 e6 e7 e8 e9 e10 e11 figures reproduce

setup-research:
	uv pip install -r requirements-research.txt

test:
	$(PYTEST) -q

legacy:
	$(PYTHON) benchmark.py --dnrti-dir $(DNRTI_DIR) --out-dir $(REPORTS) --models securebert,cyner --split test --subset-sizes 10,100,all --repeats 1 --seed $(SEED) --device $(DEVICE) --allow-device-fallback --offline --cache-dir $(CACHE_DIR)

e1:
	$(RUN) --sweep methodology --device $(DEVICE)

e2:
	$(RUN) --sweep preprocessing --device $(DEVICE)

e3:
	$(RUN) --sweep protocol --device $(DEVICE)

e4:
	$(RUN) --sweep subset --device $(DEVICE)

e5:
	@echo "E5 statistical separability is regenerated as CI/McNemar/flip columns by e1-e4 and e9."

e6:
	$(RUN) --sweep intrinsic --device $(DEVICE)

e7:
	$(RUN) --sweep operational --device both

e8:
	$(SYNTH)

e9:
	$(RUN) --sweep robustness --device $(DEVICE)

e10:
	$(RUN) --sweep calibration --device $(DEVICE)

e11:
	$(SYNTH)

figures: e2 e4 e7 e8 e10

reproduce: test legacy e1 e2 e3 e4 e5 e6 e7 e8 e9 e10 e11
