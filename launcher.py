#!/usr/bin/env python
# coding: utf-8

import sys

if len(sys.argv) < 2:
    print("Give at least one value of r. Exiting.")
    exit()

try:
    rv = [float(rr) for rr in  sys.argv[1:]]
except:
    print('Some error occured parsing args. Exiting.'); exit()

print(rv)






import subprocess

procs = []
for r in rv:
    proc = subprocess.Popen([sys.executable, '/scratch/scarpolini/lorenz/db_lorenz.py', str(r)])#, stdout=PIPE)
    procs.append(proc)

for proc in procs:
    proc.wait()

print('All done. Exiting.')
exit()
