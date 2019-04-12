"""
qc-python
Instructional Python Implementation of Quantum Chemistry Methods
"""
from setuptools import setup, find_packages

DOCLINES = __doc__.split("\n")

setup(
    name='qc_python',
    author='Yudong Qiu',
    description=DOCLINES[0],
    long_description="\n".join(DOCLINES[2:]),
    version='0.0.1',
    packages=find_packages(),
    entry_points={'console_scripts': [
        'qc_python-hf = qc_python.hf:main',
    ]},
    package_data={'qc_python': ['basis/*.basis']},
    url='https://github.com/yudongqiu/qc-python',
    install_requires=[
        'scipy',
        'numpy',
        'numba'
    ]
)
