# Falls back to the TerraLingua venv when there is no root .venv
PYTHON ?= $(firstword $(wildcard .venv/bin/python3) code/terralingua/.venv/bin/python)
NAME ?= hou_sim
TICKS ?= 50

# Proxy plus our memory runner on the newer engine: make run NAME=world7 TICKS=300
run: code/__main__.py
	$(PYTHON) $< $(NAME) $(TICKS)

# Same, on the older engine with the settings of the graded anchor_v1 worlds
run-anchor: code/__main__.py
	$(PYTHON) $< $(NAME) $(TICKS) anchor

# Proxy plus the fork's paper_core demo
run-fork: code/__main__.py
	$(PYTHON) $< fork


clean:
	-rm -rf logs/*
