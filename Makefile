# Cross-platform Makefile for the Sequence engine.
#
# Targets:
#   make            - build all C++ binaries (test, eval_dump, bench, compete, tune, ablation)
#   make python     - build the pybind11 extension into python/ (needs pybind11)
#   make test       - build and run the basic test harness
#   make codegen    - regenerate src/board_data.{hpp,cpp} from python/board_layout.py
#   make clean      - rm -rf build, rm python/_seqengine*.so / .pyd
#
# Cross-platform shell: shell commands (mkdir, rm) are routed through
# small Python helpers in tools/ to avoid sh-vs-cmd.exe differences.
# Override PY/CXX from the environment if your defaults differ.
#
# Windows: tested with MinGW make + g++. MSVC `nmake` won't work — get
# GNU Make from MSYS2 or chocolatey. Required tools: g++, python, and
# pybind11 (`pip install pybind11`) for the `python` target.

# --- toolchain detection ---------------------------------------------------

ifeq ($(OS),Windows_NT)
    PY ?= python
else
    PY ?= python3
endif

CXX      ?= g++
CXXFLAGS ?= -std=c++20 -O3 -Wall -Wextra -Wpedantic -Wno-missing-field-initializers -pthread
BUILD    := build

# Extra link flags for the pybind11 module. On Windows (MinGW/MSYS2) we
# need to statically link libgcc, libstdc++, and libwinpthread into the
# .pyd so it has no external DLL dependencies — otherwise Python's
# `import` fails with "DLL load failed" when those DLLs aren't on PATH.
# Linux/macOS shared libs don't need this.
#
# Two subtleties that bit us before:
#
#   1. -static-libstdc++ pulls libstdc++.a, which itself contains
#      references to pthread symbols. If the linker resolves those
#      against libwinpthread.dll.a (the import library) before our
#      explicit static archive is processed, the .pyd ends up depending
#      on libwinpthread-1.dll regardless.
#   2. Plain -Wl,-Bstatic -lwinpthread -Wl,-Bdynamic doesn't always
#      win the search-order tug-of-war on MSYS2/UCRT64 because the .a
#      and .dll.a sit in the same directory.
#
# Fix: --whole-archive forces the linker to pull every object from
# libwinpthread.a into the output unconditionally. That seeds the
# symbol table with the static pthread symbols BEFORE libstdc++ is
# processed, so libstdc++'s pthread references resolve against the
# already-present static symbols and the DLL form is never consulted.
ifeq ($(OS),Windows_NT)
    PY_LDFLAGS = -static-libgcc -static-libstdc++ \
                 -Wl,-Bstatic -Wl,--whole-archive -lwinpthread -Wl,--no-whole-archive \
                 -Wl,-Bdynamic
else
    PY_LDFLAGS =
endif

# Helpers for cross-platform shell operations. Make recipes are short,
# Python startup cost (~30ms) is negligible for our build sizes.
MKDIR_P = $(PY) tools/mkpath.py
RM_RF   = $(PY) tools/rmpath.py


# --- engine sources --------------------------------------------------------

ENGINE_SRCS := src/state.cpp src/board_data.cpp src/search.cpp
ENGINE_OBJS := $(ENGINE_SRCS:%.cpp=$(BUILD)/%.o)
ENGINE_HDRS := src/state.hpp src/types.hpp src/board_data.hpp \
               src/eval.hpp src/rng.hpp src/search.hpp

# Position-independent objects for the shared library backing the Python
# binding. (PIC is a no-op on Windows but harmless.)
ENGINE_PIC_OBJS := $(ENGINE_SRCS:%.cpp=$(BUILD)/pic/%.o)

# Objects built with -DSEQ_PROFILE, isolated under build/profile/ so the
# instrumented build doesn't clobber the regular .o cache. The profile
# macros in src/profile.hpp expand to RAII timers only when this define
# is set; otherwise they are `(void)0`.
ENGINE_PROFILE_OBJS := $(ENGINE_SRCS:%.cpp=$(BUILD)/profile/%.o)


# --- output binaries -------------------------------------------------------

TEST_BIN      := $(BUILD)/test_basic
EVAL_DUMP_BIN := $(BUILD)/eval_dump
BENCH_BIN     := $(BUILD)/bench
COMPETE_BIN   := $(BUILD)/compete
TUNE_BIN      := $(BUILD)/tune
ABLATION_BIN  := $(BUILD)/ablation
PROFILE_BIN   := $(BUILD)/profile

# Python extension. EXT_SUFFIX is e.g. .cpython-312-x86_64-linux-gnu.so,
# .cp312-win_amd64.pyd, .cpython-312-darwin.so — whatever the host Python
# expects to find on disk.
PY_EXT_SUFFIX := $(shell $(PY) tools/build_config.py ext_suffix)
PY_INCLUDES   := $(shell $(PY) tools/build_config.py py_includes)
PY_LINK       := $(shell $(PY) tools/build_config.py py_link)
PY_MODULE     := python/_seqengine$(PY_EXT_SUFFIX)


# --- targets ---------------------------------------------------------------

.PHONY: all test codegen validate bench compete tune ablation profile python pydeps clean

all: $(TEST_BIN) $(EVAL_DUMP_BIN) $(BENCH_BIN) $(COMPETE_BIN) $(TUNE_BIN) $(ABLATION_BIN)

# Generic compile rule. $(@D) is GNU Make for "directory of the target".
$(BUILD)/%.o: %.cpp $(ENGINE_HDRS)
	@$(MKDIR_P) $(@D)
	$(CXX) $(CXXFLAGS) -Isrc -c $< -o $@

$(BUILD)/pic/%.o: %.cpp $(ENGINE_HDRS)
	@$(MKDIR_P) $(@D)
	$(CXX) $(CXXFLAGS) -fPIC -Isrc -c $< -o $@

# Engine objects compiled with the profile instrumentation enabled. Kept
# in build/profile/ so they don't share a cache with the regular build —
# `make` and `make profile` can coexist without rebuilding everything
# on each switch.
$(BUILD)/profile/%.o: %.cpp $(ENGINE_HDRS) src/profile.hpp
	@$(MKDIR_P) $(@D)
	$(CXX) $(CXXFLAGS) -DSEQ_PROFILE -Isrc -c $< -o $@

$(TEST_BIN): tests/test_basic.cpp $(ENGINE_OBJS) $(ENGINE_HDRS)
	@$(MKDIR_P) $(BUILD)
	$(CXX) $(CXXFLAGS) -Isrc tests/test_basic.cpp $(ENGINE_OBJS) -o $@

$(EVAL_DUMP_BIN): tools/eval_dump.cpp $(ENGINE_OBJS) $(ENGINE_HDRS)
	@$(MKDIR_P) $(BUILD)
	$(CXX) $(CXXFLAGS) -Isrc tools/eval_dump.cpp $(ENGINE_OBJS) -o $@

$(BENCH_BIN): tools/bench.cpp $(ENGINE_OBJS) $(ENGINE_HDRS)
	@$(MKDIR_P) $(BUILD)
	$(CXX) $(CXXFLAGS) -Isrc tools/bench.cpp $(ENGINE_OBJS) -o $@

$(COMPETE_BIN): tools/compete.cpp $(ENGINE_OBJS) $(ENGINE_HDRS)
	@$(MKDIR_P) $(BUILD)
	$(CXX) $(CXXFLAGS) -Isrc tools/compete.cpp $(ENGINE_OBJS) -o $@

$(TUNE_BIN): tools/tune.cpp $(ENGINE_OBJS) $(ENGINE_HDRS)
	@$(MKDIR_P) $(BUILD)
	$(CXX) $(CXXFLAGS) -Isrc tools/tune.cpp $(ENGINE_OBJS) -o $@

$(ABLATION_BIN): tools/ablation.cpp $(ENGINE_OBJS) $(ENGINE_HDRS)
	@$(MKDIR_P) $(BUILD)
	$(CXX) $(CXXFLAGS) -Isrc tools/ablation.cpp $(ENGINE_OBJS) -o $@

# Profile driver — built against the instrumented engine objects, so the
# RAII timers in src/profile.hpp actually record. The driver itself is
# also compiled with -DSEQ_PROFILE so its `#ifndef SEQ_PROFILE` warning
# path stays inert in the intended build.
$(PROFILE_BIN): tools/profile.cpp $(ENGINE_PROFILE_OBJS) $(ENGINE_HDRS) src/profile.hpp
	@$(MKDIR_P) $(BUILD)
	$(CXX) $(CXXFLAGS) -DSEQ_PROFILE -Isrc tools/profile.cpp $(ENGINE_PROFILE_OBJS) -o $@

# pybind11 module. The .so/.pyd lands in python/ so `import _seqengine`
# from anything in that directory just works.
#
# Note we strip -pthread from CXXFLAGS for the *link* line on Windows.
# -pthread implicitly adds -lpthread, which on MSYS2 resolves to a
# wrapper that depends on libwinpthread.dll.a (the import library,
# pulls the DLL at runtime). The static libwinpthread.a in PY_LDFLAGS
# satisfies the same symbols statically — but only if -lpthread isn't
# also dragging in the DLL form. Object files were already compiled
# with -pthread so any thread-local-storage etc. is in place; we just
# don't want the import-library form at link time.
ifeq ($(OS),Windows_NT)
    PY_LINK_FLAGS := $(filter-out -pthread,$(CXXFLAGS))
else
    PY_LINK_FLAGS := $(CXXFLAGS)
endif

$(PY_MODULE): bindings/seqengine_py.cpp $(ENGINE_PIC_OBJS) $(ENGINE_HDRS)
	@$(MKDIR_P) python
	$(CXX) $(PY_LINK_FLAGS) -fPIC -shared -Isrc $(PY_INCLUDES) \
	    bindings/seqengine_py.cpp $(ENGINE_PIC_OBJS) $(PY_LINK) $(PY_LDFLAGS) -o $@
	@$(PY) tools/finalize_pyd.py $@

test: $(TEST_BIN)
	./$(TEST_BIN)

# Cross-check src/eval.hpp against game_engine.py on random positions.
validate: $(EVAL_DUMP_BIN)
	$(PY) tools/validate_eval.py

bench: $(BENCH_BIN)
	./$(BENCH_BIN)

compete: $(COMPETE_BIN)
	./$(COMPETE_BIN)

tune: $(TUNE_BIN)
	./$(TUNE_BIN)

ablation: $(ABLATION_BIN)
	./$(ABLATION_BIN)

# Play 5 full MCTS-vs-MCTS games with the profile instrumentation
# active, then print a sorted breakdown of where time was spent and how
# many times each scope fired per game. Overrideable:
#   make profile PROFILE_ARGS="10 2000"   # 10 games, 2000 iters/move
profile: $(PROFILE_BIN)
	./$(PROFILE_BIN) $(PROFILE_ARGS)

python: $(PY_MODULE)

# Diagnostic: list the .pyd/.so's external DLL/shared-lib dependencies.
# Useful for debugging "DLL load failed" errors on Windows — anything
# in this list other than KERNEL32, USER32, msvcrt, ucrtbase, api-ms-*,
# and pythonNNN.dll is a problem (means the static-link flags missed
# something). Uses objdump from MinGW or binutils.
pydeps: $(PY_MODULE)
	@echo "Dependencies of $(PY_MODULE):"
	@objdump -p $(PY_MODULE) | $(PY) -c "import sys; [print('  ' + l.split()[-1]) for l in sys.stdin if 'DLL Name' in l or 'NEEDED' in l]"

codegen:
	$(PY) tools/gen_board_data.py

clean:
	$(RM_RF) $(BUILD) python/_seqengine.so python/_seqengine.pyd \
	         python/_seqengine.cpython-*.so python/_seqengine.cp*-win_amd64.pyd
