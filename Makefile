CODE_DIR=code/
PROXY_DIR=$(SRC_DIR)integration_layer/
TL_DIR ?= $(SRC_DIR)terralingua-relational-memory/

MODEL_PORT=8080
TL_PORT=9000

PROXY_LOG_FILE=logs/reverse-proxy.log
PROXY_PID_FILE=logs/reverse-proxy.pid


proxy-up:
	PYTHONPATH=$(CODE_DIR) .venv/bin/python $(CODE_DIR)/__main__.py \
		-p $(MODEL_PORT) \
		> $(PROXY_LOG_FILE) 2>&1 & echo $$! > $(PROXY_PID_FILE) &

proxy-down:
	@if [ -f $(PROXY_PID_FILE) ]; then \
		kill "$$(cat $(PROXY_PID_FILE))" 2>/dev/null || true; \
		rm -f $(PROXY_PID_FILE); \
	fi
	-kill -9 $(lsof -ti :$(TL_PORT)) # TODO This lives in a subprocess, would prefer to do something else

tl-paper-core: $(TL_DIR)Makefile-terralingua
	make -C $(TL_DIR) -f Makefile-terralingua $@

clean:
	-rm $(PROXY_LOG_FILE) $(PROXY_PID_FILE)
