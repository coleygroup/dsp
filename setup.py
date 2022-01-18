"""
boip
Bayesian Optimization with Input Pruning
"""
from setuptools import setup, find_packages

short_description = __doc__.split("\n")

try:
    with open("README.md", "r") as handle:
        long_description = handle.read()
except:
    long_description = "\n".join(short_description[2:])


setup(
    name="boip",
    author="david graff",
    author_email="deg711@g.harvard.edu",
    description=short_description[0],
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    packages=find_packages(),
    include_package_data=True,
    url="https://github.com/coleygroup/boip",
    platforms=["Linux", "Mac OS-X", "Unix"],
    python_requires=">=3.8",
    version="alpha",
    entry_points = {
        'console_scripts': ['boip=boip.cli.main:main'],
    },
    requires=["numpy", "tqdm"]
)
