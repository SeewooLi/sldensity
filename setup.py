from setuptools import find_packages, setup
setup(
    name='sldensity',
    packages=find_packages(include=['sldensity']),
    version='0.4.3',
    url='http://gitlab.o/fredric/sldensity.git',
    description='Skewed logistic distribution for FADU NAND-TF',
    author='Seewoo Fredric Li',
    license='MIT',
    install_requires=[],
    setup_requires=['pytest-runner'],
    tests_require=['pytest==4.4.1'],
    test_suite='tests',
)