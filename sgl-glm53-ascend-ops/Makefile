BUILD_DIR := build
LIB_NAME  := sgl_glm53_ascend_ops_lib
LIB_SO    := $(BUILD_DIR)/$(LIB_NAME).so
PKG_DIR   := src/sgl_glm53_ascend_ops

.PHONY: all build configure install clean

# Default: fast incremental rebuild (requires prior configure)
all: $(BUILD_DIR)/build.ninja
	@ninja -C $(BUILD_DIR)
	@ln -sf $(abspath $(LIB_SO)) $(PKG_DIR)/$(LIB_NAME).so

# Full configure + build
build: configure
	@ninja -C $(BUILD_DIR)
	@ln -sf $(abspath $(LIB_SO)) $(PKG_DIR)/$(LIB_NAME).so

configure:
	cmake -B $(BUILD_DIR) -G Ninja

# One-time setup: editable Python install + build
install: build
	pip install -e . --no-build-isolation

clean:
	rm -rf $(BUILD_DIR)
	rm -f $(PKG_DIR)/$(LIB_NAME).so

# Guard: if build.ninja doesn't exist yet, run configure first
$(BUILD_DIR)/build.ninja:
	@$(MAKE) configure
