# https://stackoverflow.com/questions/4505747/how-should-i-structure-a-python-package-that-contains-cython-code
from setuptools import Extension, find_packages, setup

use_cython = True
try:
    from Cython.Distutils import build_ext
except ImportError:
    use_cython = False

cmdclass = {}
ext_modules = []

if use_cython:
    ext_modules.append(Extension("kdelearn.cutils", ["kdelearn/cutils.pyx"]))
    cmdclass.update({"build_ext": build_ext})
else:
    ext_modules.append(Extension("kdelearn.cutils", ["kdelearn/cutils.c"]))

with open("VERSION", "r") as f:
    version = f.read().strip()

with open("README.md", "r") as f:
    long_description = f.read()

with open("requirements.txt") as f:
    install_requires = f.read().split("\n")

setup(
    name="kdelearn",
    version=version,
    author="Krystian Franus",
    author_email="krystian.franus@gmail.com",
    description="Short description of ml methods based on kernel density estimation",
    long_description=long_description,
    license="MIT",
    packages=find_packages(),
    install_requires=install_requires,
    cmdclass=cmdclass,
    ext_modules=ext_modules,
    classifiers=[
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires=">=3.6",
)
