PYTHON ?= python3
.PHONY: test check parser
check: test parser
test:
	PYTHONPATH=. $(PYTHON) -m pytest -q
parser:
	cmake -S cpp -B cpp/build
	cmake --build cpp/build -j
