# Cross-platform distribution builder for cnnaio (the vision server).
#
# The whole project is pure Go: wazero runs the ncnn wasm, and every model + the
# wasm itself is embedded via go:embed. So the binary is fully self-contained and
# cross-compiles with CGO disabled — no toolchain needed per target.
#
# Usage:
#   make all        # build windows + macos + linux bundles into ./dist
#   make windows    # just Windows (amd64 + arm64)
#   make macos      # just macOS  (amd64 + arm64)
#   make linux      # just Linux  (amd64 + arm64)
#   make clean      # remove ./dist
#
# Each target produces, per OS/arch, a ./dist/<name>-<os>-<arch>/ folder (server
# binary + README + sample images) and a matching .tar.gz.
#
# Recipes use POSIX shell utilities (mkdir/cp/tar); on Windows run from Git Bash.

SHELL    := bash
BINARY   := cnnaio
PKG      := ./
DIST     := dist
README   := $(firstword $(wildcard README.md cmd/README.md))
EXTRAS   := $(README) docs #tests    # shipped next to the binary
GOFLAGS  := -trimpath -ldflags "-s -w"
export CGO_ENABLED := 0
.DEFAULT_GOAL := all

.PHONY: all windows macos linux clean help

help:
	@echo "targets: all | windows | macos | linux | clean"

all: windows macos linux

windows:
	@$(MAKE) --no-print-directory bundle GOOS=windows GOARCH=amd64 EXT=.exe
	@$(MAKE) --no-print-directory bundle GOOS=windows GOARCH=arm64 EXT=.exe

macos:
	@$(MAKE) --no-print-directory bundle GOOS=darwin GOARCH=amd64
	@$(MAKE) --no-print-directory bundle GOOS=darwin GOARCH=arm64

linux:
	@$(MAKE) --no-print-directory bundle GOOS=linux GOARCH=amd64
	@$(MAKE) --no-print-directory bundle GOOS=linux GOARCH=arm64

# Internal: build one OS/arch bundle. Parameters come in as make variables.
DIR := $(BINARY)-$(GOOS)-$(GOARCH)
bundle:
	@echo ">> $(GOOS)/$(GOARCH)"
	@rm -rf "$(DIST)/$(DIR)"
	@mkdir -p "$(DIST)/$(DIR)"
	GOOS=$(GOOS) GOARCH=$(GOARCH) go build $(GOFLAGS) -o "$(DIST)/$(DIR)/$(BINARY)$(EXT)" $(PKG)
	@cp -r $(EXTRAS) "$(DIST)/$(DIR)/"
	@tar -czf "$(DIST)/$(DIR).tar.gz" -C "$(DIST)" "$(DIR)"
	@echo "   -> $(DIST)/$(DIR).tar.gz"

clean:
	@rm -rf "$(DIST)"
	@echo "removed $(DIST)/"
