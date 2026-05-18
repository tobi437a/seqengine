"""
Emit one piece of build configuration to stdout. Used by the Makefile so
the build works on Linux/macOS/Windows without depending on python3-config
(which doesn't exist on Windows).

Usage:
    python tools/build_config.py ext_suffix
        -> e.g. ".cpython-312-x86_64-linux-gnu.so" on Linux
                ".cp312-win_amd64.pyd"             on Windows
                ".cpython-312-darwin.so"           on macOS

    python tools/build_config.py py_includes
        -> "-I<python_include> -I<pybind11_include>" suitable for g++ -I flags

    python tools/build_config.py py_link
        -> link line for the Python library. Empty on Linux/macOS (the
           extension's symbols resolve at load time); on Windows we need
           to link libpython explicitly with MinGW.
"""
import os
import sys
import sysconfig


def die(msg):
    sys.stderr.write(msg.rstrip() + '\n')
    sys.exit(1)


def main():
    if len(sys.argv) != 2:
        die(__doc__)
    what = sys.argv[1]

    if what == 'ext_suffix':
        # EXT_SUFFIX is the canonical extension Python's import machinery
        # looks for. SO is the legacy single-extension on some platforms.
        s = sysconfig.get_config_var('EXT_SUFFIX') \
            or sysconfig.get_config_var('SO') \
            or '.so'
        sys.stdout.write(s)

    elif what == 'py_includes':
        parts = ['-I' + sysconfig.get_paths()['include']]
        try:
            import pybind11
        except ImportError:
            die("pybind11 is not installed. Run:  pip install pybind11")
        parts.append('-I' + pybind11.get_include())
        sys.stdout.write(' '.join(parts))

    elif what == 'py_link':
        # On Linux and macOS, an extension's references to the CPython API
        # are resolved against the host interpreter at load time — no
        # explicit linkage needed.
        if sys.platform != 'win32':
            sys.stdout.write('')
            return
        # On Windows (esp. MinGW), the import library must be linked.
        # Search candidates: $prefix/libs (cpython install layout) and
        # the value of LIBDIR from sysconfig if it points anywhere real.
        libdirs = []
        for cand in (os.path.join(sys.prefix, 'libs'),
                     sysconfig.get_config_var('LIBDIR') or ''):
            if cand and os.path.isdir(cand) and cand not in libdirs:
                libdirs.append(cand)
        ver = f"{sys.version_info.major}{sys.version_info.minor}"
        flags = []
        for d in libdirs:
            flags.append('-L' + d)
        flags.append(f'-lpython{ver}')
        sys.stdout.write(' '.join(flags))

    else:
        die(f"unknown config key: {what!r}")


if __name__ == '__main__':
    main()
