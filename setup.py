from setuptools import setup

import versioneer


setup(
    name='tensorpotential',
    #version=versioneer.get_version(),
    version='0.2.0+19.gfa5b8b0',
    cmdclass=versioneer.get_cmdclass(),
    packages=['tensorpotential', 'tensorpotential.utils', 'tensorpotential.potentials', 'tensorpotential.functions'],
    package_dir={'': 'src'},
    url='',
    license='',
    author='Anton Bochkarev',
    author_email='',
    description='',
    python_requires='<3.11',
    install_requires=
    [
    'scipy',
    'tensorflow',
    'numpy<=1.26.4',
    'pandas<=2.0',
    'ase'
    ]
)

