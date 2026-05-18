"""
Post-build step for the pybind11 .pyd on Windows.

We aggressively try to statically link the MinGW runtime (libgcc,
libstdc++, libwinpthread) into the .pyd, but some MSYS2/UCRT64 toolchain
combinations still produce a .pyd that depends on libwinpthread-1.dll
(typically because the static libwinpthread.a is itself a thin shim or
the toolchain's libstdc++ pulls pthread symbols in a way that resolves
against the DLL despite our --whole-archive flag).

If the .pyd still has DLL dependencies that aren't part of the standard
Windows runtime, we copy them into the .pyd's directory. Windows
searches the same directory when resolving DLLs for a loaded module, so
this makes `import _seqengine` work regardless of PATH.

On Linux/macOS this is a no-op.

Usage:  python tools/finalize_pyd.py <path/to/module>
"""
import os
import re
import shutil
import subprocess
import sys


# DLLs we expect to be available on any modern Windows install. Anything
# not in this list is a candidate for copying.
KNOWN_SYSTEM_DLLS = {
    'kernel32.dll', 'user32.dll', 'msvcrt.dll', 'ucrtbase.dll',
    'advapi32.dll', 'ws2_32.dll', 'bcrypt.dll', 'shell32.dll',
    'ole32.dll', 'oleaut32.dll',
}

def is_system_dll(name):
    n = name.lower()
    if n in KNOWN_SYSTEM_DLLS:
        return True
    if n.startswith('api-ms-win-'):    # CRT facade DLLs, always present
        return True
    if re.match(r'python\d+\.dll$', n): # the running interpreter
        return True
    return False


def list_dependencies(pyd_path):
    """Return the list of DLL names the .pyd depends on, via objdump."""
    try:
        out = subprocess.check_output(
            ['objdump', '-p', pyd_path],
            stderr=subprocess.DEVNULL).decode('utf-8', errors='replace')
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    deps = []
    for line in out.splitlines():
        line = line.strip()
        if line.startswith('DLL Name:'):
            deps.append(line.split(':', 1)[1].strip())
    return deps


def find_dll_in_gcc_path(name):
    """Use gcc to locate a DLL on the toolchain's bin path."""
    cxx = os.environ.get('CXX', 'g++')
    # gcc -print-search-dirs gives library/program search paths in a
    # parseable format. The "programs:" line is what we want (DLLs sit
    # next to the toolchain binaries on MinGW/MSYS2).
    try:
        out = subprocess.check_output(
            [cxx, '-print-search-dirs'],
            stderr=subprocess.DEVNULL).decode('utf-8', errors='replace')
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    for line in out.splitlines():
        if line.startswith('programs:'):
            paths = line.split('=', 1)[1].split(os.pathsep)
        elif line.startswith('libraries:'):
            paths = line.split('=', 1)[1].split(os.pathsep)
        else:
            continue
        for d in paths:
            # MSYS2 paths can be relative with .. in them; normalize.
            d = os.path.normpath(d)
            cand = os.path.join(d, name)
            if os.path.exists(cand):
                return cand
            # Some setups have the DLLs in ../bin relative to a library dir.
            cand2 = os.path.normpath(os.path.join(d, '..', 'bin', name))
            if os.path.exists(cand2):
                return cand2
    return None


def main():
    if len(sys.argv) != 2:
        sys.stderr.write(__doc__)
        sys.exit(2)
    pyd = sys.argv[1]
    if not os.path.exists(pyd):
        sys.exit(0)

    # No-op on non-Windows: the .so will use the host's dynamic linker
    # with normal /etc/ld.so or rpath rules. Nothing to copy.
    if sys.platform != 'win32' and not pyd.endswith('.pyd'):
        return

    deps = list_dependencies(pyd)
    if deps is None:
        # objdump unavailable — silently skip. The build itself succeeded;
        # if there's a runtime DLL problem, mcts.py's diagnostic will
        # surface it.
        return

    out_dir = os.path.dirname(os.path.abspath(pyd))
    copied = []
    missing = []
    for dep in deps:
        if is_system_dll(dep):
            continue
        # Already next to the .pyd? Nothing to do.
        if os.path.exists(os.path.join(out_dir, dep)):
            continue
        src = find_dll_in_gcc_path(dep)
        if src is None:
            missing.append(dep)
            continue
        try:
            shutil.copy2(src, out_dir)
            copied.append(dep)
        except OSError as e:
            missing.append(f"{dep} ({e})")

    if copied:
        print(f"  finalize_pyd: copied {len(copied)} runtime DLL(s) to "
              f"{out_dir}: {', '.join(copied)}")
    if missing:
        print(f"  finalize_pyd: WARNING — could not locate: "
              f"{', '.join(missing)}. `import` will likely fail; install "
              f"these DLLs or add their directory to PATH.")


if __name__ == '__main__':
    main()
