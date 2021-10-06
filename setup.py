from setuptools import setup

setup(
    name='smlm_publication',
    version='0.9.1',
    packages=['smlm', 'smlm.styles'],
    url='https://github.com/npeschke/smlm',
    license='BSD-3-Clause',
    author='Nicolas Peschke',
    author_email='peschke@stud.uni-heidelberg.de',
    description='Performs Voronoi density analysis on SMLM data',
    install_requires=[
        "matplotlib~=3.3.4",
        "seaborn~=0.11.0",
        "numpy~=1.20.2",
        "cmasher~=1.6.0",
        "pandas~=1.1.2",
        "scipy~=1.6.2",
        "hdbscan~=0.8.26",
        "scikit-learn~=0.23.2",
        "setuptools~=52.0.0",
    ]
)
