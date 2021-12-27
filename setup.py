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
    python_requires='<3.9',
    install_requires=
    [
    'scipy>=1.4.1',
    'tensorflow<=2.7',
    'numpy==1.19.4',
    'pandas',
    'ase'
    ]
)

