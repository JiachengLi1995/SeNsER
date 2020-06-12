from distutils.core import setup, Extension
from Cython.Build import cythonize

extensions = [
    Extension(
        'pytrie',
        ['pytrie.pyx'],
        language = 'c++',
        extra_compile_args = ['-std=c++11', '-Wno-sign-compare'],
        extra_link_args=['-std=c++11', '-Wno-sign-compare']
    )
]

setup(
    ext_modules = cythonize(extensions),
    version='0.0.3',
)

