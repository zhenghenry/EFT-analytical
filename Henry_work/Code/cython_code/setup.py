from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy
import sys

import Cython.Compiler.Options
Cython.Compiler.Options.annotate = True

# ext = Extension("test_gmpy2", ["test_gmpy2.pyx"], include_dirs=sys.path, libraries=['gmp', 'mpfr', 'mpc'])

# setup(
# 	name = "cython_gmpy_test",
# 	ext_modules = cythonize([ext],
# 							include_path = sys.path,
# 							annotate = True),
# )

#extensions = Extension("babispython_v10_cython",["babispython_v10_cython.pyx"])
setup(
	name = "babiscython_v4_ubuntu",
	ext_modules = cythonize(
		Extension(
			"babiscython_v4_ubuntu",
			sources = ["babiscython_v4_ubuntu.pyx"],
			include_dirs=[numpy.get_include()],
			libraries = ['gmp', 'mpfr', 'mpc']
		),
		annotate = True),
	install_requires = ["numpy"],
	zip_safe = False
)

# python setup.py build_ext --inplace