# Copyright (c) 2013-2016 G. Peter Lepage.
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
	$(PYTHON) ridge.py > ridge.tmp
	- diff -w ridge.out ridge.tmp
	# mpi tests
	- mpirun -np 4 $(PYTHON) ridge.py > mpi-ridge.tmp
	- diff -w mpi-ridge.out mpi-ridge.tmp
	- mpirun -np 2 -output-filename mpi-slow $(PYTHON) slow.py
	- diff -w slow.out mpi-slow.1.0
	- diff -w mpi-slow.1.0 mpi-slow.1.1
	- mpirun -np 2 -output-filename mpi-faster $(PYTHON) faster.py
	- diff -w faster.out mpi-faster.1.0
	- diff -w mpi-faster.1.0 mpi-faster.1.1
	- mpirun -np 2 -output-filename mpi-fastest $(PYTHON) fastest.py
	- diff -w fastest.out mpi-fastest.1.0
	- diff -w mpi-fastest.1.0 mpi-fastest.1.1

time:
	time $(MAKE) PYTHON=$(PYTHON) PLOT=False run

test-linking: cfcn_f2py.so cfcn_ctypes.so ffcn_f2py.so
	$(PYTHON) cfcn-f2py.py > cfcn-f2py.tmp
	- diff -w cfcn.out cfcn-f2py.tmp
	$(PYTHON) cfcn-ctypes.py > cfcn-ctypes.tmp
	- diff -w cfcn.out cfcn-ctypes.tmp
	$(PYTHON) cfcn-cython.py > cfcn-cython.tmp
	- diff -w cfcn.out cfcn-cython.tmp
	$(PYTHON) ffcn-f2py.py > ffcn-f2py.tmp
	- diff -w cfcn.out ffcn-f2py.tmp

cfcn_f2py.so : cfcn.c cfcn_f2py.pyf
	f2py -m cfcn_f2py -c cfcn_f2py.pyf cfcn.c

cfcn_ctypes.so : cfcn.c
	# the following works on OSX and probably also most UNIX systems,
	# but probably not on windows machines
	cc -fPIC -shared -o cfcn_ctypes.so cfcn.c

ffcn_f2py.so : ffcn.f
	f2py -m ffcn_f2py -c ffcn.f
	-rm ffcn_f2py.pyxbldc

update-all:
	$(PYTHON) simple.py > simple.out
	$(PYTHON) slow.py > slow.out
	$(PYTHON) faster.py > faster.out
	$(PYTHON) fastest.py > fastest.out
	$(PYTHON) parallel-pools.py > parallel-pools.out
	$(PYTHON) ridge.py > ridge.out
	- mpirun -np 4 $(PYTHON) ridge.py > mpi-ridge.out

clean:
	rm -f *.pyc *.tmp *.p *.prof *.1.0 *.1.1 *.1.2 *.pyxbldc *.so

