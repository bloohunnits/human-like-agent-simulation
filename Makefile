SRC_DIR=code/
PROXY_DIR=$(SRC_DIR)integration_layer/
TL_DIR ?= $(SRC_DIR)terralingua-relational-memory/


proxy-up:
	python $(PROXY_DIR)server.py

tl-paper-core: $(TL_DIR)Makefile-terralingua
	make -C $(TL_DIR) -f Makefile-terralingua $@
