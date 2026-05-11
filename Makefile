# hash-miner Makefile.
#
# Usage:
#   make build           debug build
#   make release         optimized build (default target)
#   make install         symlink hash-miner into $(PREFIX)/bin
#   make uninstall       remove the symlink
#   make setup           release + install (one-shot first-time setup)
#   make wallet          generate a new address + private key
#   make selftest        verify CUDA kernel matches host keccak
#   make bench           run a 30 s hash-rate benchmark
#   make mine            start the miner (needs $(PRIVKEY_ENV) exported)
#   make devices         list visible CUDA devices
#   make clean           cargo clean
#   make help            show this help

PREFIX     ?= /usr/local
BIN_DIR    ?= $(PREFIX)/bin
CARGO      ?= cargo
NAME       := hash-miner
RELEASE_BIN := $(CURDIR)/target/release/$(NAME)
CONFIG     ?= ./config.toml
PRIVKEY_ENV ?= MINER1_PRIVKEY
BENCH_SECS ?= 30
BENCH_ITERS ?= 64

.PHONY: all build release install uninstall setup wallet selftest bench mine devices clean help

all: release

build:
	$(CARGO) build

release: $(RELEASE_BIN)

$(RELEASE_BIN):
	$(CARGO) build --release

install: release
	@mkdir -p $(BIN_DIR)
	@ln -sfn $(RELEASE_BIN) $(BIN_DIR)/$(NAME)
	@echo "linked $(BIN_DIR)/$(NAME) -> $(RELEASE_BIN)"
	@echo "try:   $(NAME) --help"

uninstall:
	@rm -f $(BIN_DIR)/$(NAME)
	@echo "removed $(BIN_DIR)/$(NAME)"

setup: install
	@echo ""
	@echo "next steps:"
	@echo "  1. make wallet              # generate an address"
	@echo "  2. cp config.example.toml config.toml && \$$EDITOR config.toml"
	@echo "  3. export $(PRIVKEY_ENV)=0x...     # the private key from step 1"
	@echo "  4. make mine"

wallet: release
	@$(RELEASE_BIN) wallet new

selftest: release
	@$(RELEASE_BIN) selftest --device 0

bench: release
	@$(RELEASE_BIN) bench --seconds $(BENCH_SECS) --iters $(BENCH_ITERS)

devices: release
	@$(RELEASE_BIN) devices

mine: release
	@if [ -z "$$$(PRIVKEY_ENV)" ]; then \
		echo "error: $(PRIVKEY_ENV) is not set"; \
		echo "  export $(PRIVKEY_ENV)=0x<your hex private key>"; \
		exit 1; \
	fi
	@if [ ! -f $(CONFIG) ]; then \
		echo "error: $(CONFIG) not found"; \
		echo "  cp config.example.toml $(CONFIG) && \$$EDITOR $(CONFIG)"; \
		exit 1; \
	fi
	$(RELEASE_BIN) mine --config $(CONFIG)

clean:
	$(CARGO) clean

help:
	@awk '/^# *Usage:/,/^$$/{print}' $(firstword $(MAKEFILE_LIST)) | sed 's/^# *//'
