# Created by G. Peter Lepage (Cornell University) in 12/2013.
# Copyright (c) 2013-18 G. Peter Lepage.
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
PIP = $(PYTHON) -m pip
PYTHONVERSION = python`$(PYTHON) -c 'import platform; print(platform.python_version())'`
VERSION = `$(PYTHON) -c 'import vegas; print (vegas.__version__)'`

DOCFILES :=  $(shell ls doc/source/conf.py doc/source/*.{rst,out,png})
SRCFILES := $(shell ls setup.py src/vegas/*.{py,pyx})
CYTHONFILES := src/vegas/_vegas.c

install-user : 
	$(PIP) install . --user --no-cache-dir

install install-sys : 
	$(PIP) install . --no-cache-dir

uninstall :			# mostly works (may leave some empty directories)
	- $(PIP) uninstall vegas

update :
	make uninstall install

rebuild:
	rm -rf $(CYTHONFILES)
	make uninstall install 

src/vegas/_vegas.c : src/vegas/_vegas.pyx src/vegas/_vegas.pxd
	cd src/vegas; cython _vegas.pyx

.PHONY : doc 

doc-html doc :
	make doc/html/index.html

doc/html/index.html : $(SRCFILES) $(DOCFILES) setup.cfg
	sphinx-build -b html doc/source doc/html

clear-doc:
	rm -rf doc/html; 

doc-zip doc.zip:
	cd doc/html; zip -r doc *; mv doc.zip ../..

doc-all: doc-html # doc-pdf

sdist:	$(CYTHONFILES)	# source distribution
	$(PYTHON) setup.py sdist
	# $(PYTHON) -m build --sdist

.PHONY: tests

tests:
	@echo 'N.B. Some tests involve random numbers and so fail occasionally'
	@echo '     (less than 1 in 100 times) due to multi-sigma fluctuations.'
	@echo '     Run again if any test fails.'
	@echo ''
	$(PYTHON) -m unittest discover

run run-examples:
	$(MAKE) -C examples PYTHON=$(PYTHON) PLOT=True run

test-linking:
	$(MAKE) -C examples PYTHON=$(PYTHON) test-linking

time:
	time $(MAKE) -C examples PYTHON=$(PYTHON) PLOT=False run

upload-twine: 
	twine upload dist/vegas-$(VERSION).tar.gz

upload-git: 
	echo  "version $(VERSION)"
	make doc-html
	git diff --exit-code
	git diff --cached --exit-code
	git push origin master

tag-git:
	echo  "version $(VERSION)"
	git tag -a v$(VERSION) -m "version $(VERSION)"
	git push origin v$(VERSION)

test-download:
	-$(PIP) uninstall vegas
	$(PIP) install vegas --no-cache-dir

test-readme:
	python setup.py --long-description | rst2html.py > README.html

clean:
	rm -f -r build
	rm -rf __pycache__
	rm -f *.so *.tmp *.pyc *.prof .coverage doc.zip
	rm -f -r dist
	# $(MAKE) -C doc/source clean
	# $(MAKE) -C tests clean
	# $(MAKE) -C examples clean

