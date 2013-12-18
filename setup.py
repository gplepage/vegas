""" to build modules in module_list:   python setup.py build_ext --inplace """

from distutils.core import setup
from Cython.Build import cythonize
import sys
import numpy

VEGAS_VERSION = '1.0'

# create vegas/version.py so vegas knows its version number 
with open("src/vegas/_version.py","w") as version_file:
    version_file.write(
        "# File created by vegas setup.py\nversion = '%s'\n" 
        % VEGAS_VERSION
        )


include_dirs = [numpy.get_include()]
data_files = [('vegas_include', ['vegas.pxd'])]

ext_modules = cythonize('src/vegas/_vegas.pyx') + cythonize('examples/kinoshita/kinoshita.pyx')

packages = ['vegas']
package_dir = dict(vegas='src/vegas')
package_data = dict(vegas=['../vegas.pxd','_vegas.pxd'])

setup(
	name='vegas',
	version=VEGAS_VERSION,
	author='G. Peter Lepage',
	author_email='g.p.lepage@cornell.edu',
	include_dirs=include_dirs,
	packages=packages,
    package_dir=package_dir,
    package_data=package_data,
    ext_modules=ext_modules,
	)
