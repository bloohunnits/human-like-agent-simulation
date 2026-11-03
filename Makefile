PYTHON=.venv/bin/python3

run: code/__main__.py
	$(PYTHON) $<


clean:
	-rm -rf logs/*
