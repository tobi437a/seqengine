"""mkdir -p, cross-platform. Usage: python mkpath.py DIR [DIR...]"""
import os
import sys

for p in sys.argv[1:]:
    if p:
        os.makedirs(p, exist_ok=True)
