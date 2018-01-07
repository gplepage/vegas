""" to build modules in module_list:   python setup.py build_ext --inplace

Created by G. Peter Lepage (Cornell University) in 12/2013.
Copyright (c) 2013-18 G. Peter Lepage.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
any later version (see <http://www.gnu.org/licenses/>).

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
"""

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import sys
import numpy

VEGAS_VERSION = '3.3.2'

# create vegas/version.py so vegas knows its version number
with open("src/vegas/_version.py","w") as version_file:
    version_file.write(
        "# File created by vegas setup.py\nversion = '%s'\n"
        % VEGAS_VERSION
        )

ext_args = dict(
    libraries=[],
    include_dirs=[numpy.get_include()],
    library_dirs=[],
    runtime_library_dirs=[],
    extra_link_args=[],
    )
data_files = [('vegas_include', ['vegas.pxd'])]
ext_modules = [
    Extension('vegas._vegas', ['src/vegas/_vegas.pyx'], **ext_args),
    ]

# fix for ReadTheDocs support
# import os
# if os.environ.get('READTHEDOCS') == 'True':
#     requires = ["cython (>=0.17)","numpy (>=1.7)"]
#     install_requires = ['cython>=0.17', 'numpy>=1.7']
# else:
requires = ["cython (>=0.17)","numpy (>=1.7)", "gvar (>=8.0)"]
install_requires = ['cython>=0.17', 'numpy>=1.7', 'gvar>=8.0']

setup(
	name='vegas',
	version=VEGAS_VERSION,
	description='Tools for adaptive multidimensional Monte Carlo integration.',
	author='G. Peter Lepage',
	author_email='g.p.lepage@cornell.edu',
	packages=['vegas'],
    package_dir=dict(vegas='src/vegas'),
    package_data=dict(vegas=['../vegas.pxd','_vegas.pxd']),
    ext_modules=cythonize(ext_modules),
    install_requires=install_requires, # for pip
    requires=requires, # for disutils
    url="https://github.com/gplepage/vegas.git",
    license='GPLv3+',
    platforms='Any',
    long_description="""
    This package provides tools evaluating multidimensional
    integrals numerically using an enhanced version of
	the adaptive Monte Carlo vegas algorithm (G. P. Lepage,
	J. Comput. Phys. 27(1978) 192).
    """,
    classifiers = [                     #
        'Development Status :: 5 - Production/Stable',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: POSIX',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Cython',
        'Topic :: Scientific/Engineering'
        ]

	)
