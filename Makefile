PYTHON ?= python3
CMAKE ?= cmake
.PHONY: demo train export build verify stress test benchmark sanitized check real-data-check clean

# Full synthetic contract: supervised training -> export -> compile -> parity.
demo:
	$(PYTHON) -m snn_pipeline.train --demo --epochs 30
	$(PYTHON) -m snn_pipeline.export
	$(MAKE) build
	$(MAKE) verify
	$(MAKE) stress

export:
	$(PYTHON) -m snn_pipeline.export

build:
	$(CMAKE) -S deployment -B build -DCMAKE_BUILD_TYPE=Release
	$(CMAKE) --build build -j 2

verify:
	$(PYTHON) -m snn_pipeline.verify

stress:
	$(PYTHON) -m snn_pipeline.stress

test: build
	$(PYTHON) -m pytest tests -v
	ctest --test-dir build --output-on-failure

benchmark: build
	./build/snn_benchmark

sanitized:
	$(CMAKE) -S deployment -B build/sanitized -DSNN_SANITIZERS=ON -DCMAKE_BUILD_TYPE=Debug
	$(CMAKE) --build build/sanitized -j 2
	ctest --test-dir build/sanitized --output-on-failure

check: demo test

# Validates actual labeled EEG data. Does not silently invent labels or subjects.
real-data-check:
	@test -n "$(DATA)" || (echo "Provide DATA=path/to/windows.npz" >&2; exit 2)
	$(PYTHON) -m snn_pipeline.validate_dataset "$(DATA)"

clean:
	rm -rf build artifacts
