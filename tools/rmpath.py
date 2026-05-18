"""rm -rf (with glob), cross-platform. Usage: python rmpath.py PATH [PATH...]
Each PATH may contain glob characters (*, ?). Missing files are ignored.
"""
import glob
import os
import shutil
import sys

for pat in sys.argv[1:]:
    if any(c in pat for c in '*?['):
        paths = glob.glob(pat)
    else:
        paths = [pat]
    for p in paths:
        if os.path.isdir(p):
            shutil.rmtree(p)
        elif os.path.exists(p):
            os.remove(p)
