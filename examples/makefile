# Copyright (c) 2013-2018 G. Peter Lepage.
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
FORTRAN = gfortran
PLOT = True
PYVERSION:=$(shell python -c 'import sys; print(sys.version_info[0])')

ifeq (2, $(PYVERSION))
	SO = so
else
	SO = cpython-36m-darwin.so
endif

run:
	$(PYTHON) simple.py $(PLOT) > simple.tmp
	- diff -w simple.out simple.tmp
	$(PYTHON) slow.py > slow.tmp
	- diff -w slow.out slow.tmp
	$(PYTHON) faster.py > faster.tmp
	- diff -w slow.out faster.tmp
	$(PYTHON) fastest.py > fastest.tmp
	- diff -w fastest.out fastest.tmp
	$(PYTHON) bayes.py $(PLOT) > bayes.tmp
	- diff -w bayes.out bayes.tmp
	$(PYTHON) distribution.py > distribution.tmp
	- diff -w distribution.out distribution.tmp
	$(PYTHON) path-integral.py $(PLOT) > path-integral.tmp 
	- diff -w path-integral.out path-integral.tmp
	$(PYTHON) parallel-pools.py > parallel-pools.tmp
	- diff -w parallel-pools.out parallel-pools.tmp
	$(PYTHON) ridge.py 8 > ridge.tmp
	- diff -w ridge.out ridge.tmp
	# mpi test
	# - mpirun -np 2 -outfile-pattern=mpi-ridge.tmp.%r $(PYTHON)  ridge.py
	- mpirun -np 8 $(PYTHON)  ridge.py 'mpi' > mpi-ridge.tmp
	- diff -w ridge.out mpi-ridge.tmp

_time:
	$(PYTHON) simple.py $(PLOT) > simple.tmp
	$(PYTHON) slow.py > slow.tmp
	$(PYTHON) faster.py > faster.tmp
	$(PYTHON) fastest.py > fastest.tmp
	$(PYTHON) bayes.py $(PLOT) > bayes.tmp
	$(PYTHON) distribution.py > distribution.tmp
	$(PYTHON) path-integral.py $(PLOT) > path-integral.tmp 
	$(PYTHON) parallel-pools.py > parallel-pools.tmp
	$(PYTHON) ridge.py 8 > ridge.tmp
	# skip mpi test because highly variable

time:
	time make _time PLOT=False

test-linking:
	# ----------------------------------------
	$(PYTHON) cfcn_cffi-build.py
	$(PYTHON) cfcn-cffi.py > cfcn-cffi.tmp
	- diff -w cfcn.out cfcn-cffi.tmp
	-rm cfcn_cython*.so
	$(PYTHON) cfcn-cython.py > cfcn-cython.tmp
	- diff -w cfcn.out cfcn-cython.tmp
	# ----------------------------------------
	$(FORTRAN) -c ffcn.f -o ffcn.o
	$(PYTHON) ffcn_cffi-build.py
	$(PYTHON) ffcn-cffi.py > ffcn-cffi.tmp
	- diff -w cfcn.out ffcn-cffi.tmp
	# ----------------------------------------
	-f2py -m ffcn_f2py --quiet -c ffcn.f
	$(PYTHON) ffcn-f2py.py > ffcn-f2py.tmp
	- diff -w cfcn.out ffcn-f2py.tmp

update-all:
	$(PYTHON) simple.py False > simple.out
	$(PYTHON) slow.py > slow.out
	$(PYTHON) faster.py > faster.out
	$(PYTHON) fastest.py > fastest.out
	$(PYTHON) bayes.py False > bayes.out
	$(PYTHON) distribution.py > distribution.out
	$(PYTHON) path-integral.py False > path-integral.out
	$(PYTHON) parallel-pools.py > parallel-pools.out
	$(PYTHON) ridge.py > ridge.out

clean:
	rm -f *.pyc *.tmp *.tmp.0 *.tmp.1 *.tmp.mpi *.p *.prof *.1.0 *.1.1 *.1.2 *.pyxbldc *.so

