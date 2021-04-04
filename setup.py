#!/usr/bin/env python3

import os
from setuptools import setup, find_packages

# get key package details from py_pkg/__version__.py
about = {}  # type: ignore
here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, "apl", "__version__.py")) as f:
    exec(f.read(), about)

# load the README file and use it as the long_description for PyPI
with open("README.md", "r") as f:
    readme = f.read()

requirements = ["numpy~=1.20,<2.0.0", "scikit-learn~=0.24.1,<1.0.0", "scipy"]
dev_requirements = ["pytest", "pylint", "black", "rope"]
extra_requirements = ["streamlit", "matplotlib", "plotly"]

# package configuration - for reference see:
# https://setuptools.readthedocs.io/en/latest/setuptools.html#id9
setup(
    name=about["__title__"],
    version=about["__version__"],
    description=about["__description__"],
    url=about["__url__"],
    author=about["__author__"],
    author_email=about["__author_email__"],
    long_description=readme,
    long_description_content_type="text/markdown",
    include_package_data=True,
    # python_requires=">=3.6.*",
    packages=find_packages(),
    install_requires=requirements,
    extras_require={"dev": dev_requirements, "extras": extra_requirements},
    zip_safe=False,
    # classifiers=[
    #     "Development Status :: 4 - Beta",
    #     "Intended Audience :: Developers",
    #     "Programming Language :: Python :: 3.7",
    # ],
    # keywords="package development template",
)