# Copyright (c) 2013 G. Peter Lepage. 
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version (see <http://www.gnu.org/licenses/>).
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

PYTHON = python
PLOT = True

run:
	$(PYTHON) simple.py $(PLOT) > simple.tmp
	- diff -w simple.out simple.tmp
	$(PYTHON) slow.py > slow.tmp     # about 6s
	- diff -w slow.out slow.tmp
	$(PYTHON) faster.py > faster.tmp
	- diff -w faster.out faster.tmp
	$(PYTHON) fastest.py > fastest.tmp
	- diff -w fastest.out fastest.tmp
	$(PYTHON) path-integral.py $(PLOT) > path-integral.tmp  # about 16s
	- diff -w path-integral.out path-integral.tmp
	- $(PYTHON) parallel-pools.py > parallel-pools.tmp
	- diff -w parallel-pools.out parallel-pools.tmp
	- mpirun -np 4 $(PYTHON) parallel-mpi.py > parallel-mpi.tmp
	- diff -w parallel-mpi.out parallel-mpi.tmp
time:
	time $(MAKE) PYTHON=$(PYTHON) PLOT=False run

update-all:
	$(PYTHON) simple.py > simple.out
	$(PYTHON) slow.py > slow.out
	$(PYTHON) faster.py > faster.out
	$(PYTHON) fastest.py > fastest.out
	$(PYTHON) parallel-pools.py > parallel-pools.out 
	- mpirun -np 4 parallel-pools.py > parallel-pools.out 

clean:
	rm -f *.pyc *.tmp *.p *.prof

