import platform

from setuptools import setup

import versioneer

install_requires = [
    "scipy",
    "numpy",
    "pandas",
    "ase",
]
if platform.system() == "Darwin" and platform.machine() == "arm64":
    # M1/M2 Mac
    install_requires.append("tensorflow-macos<=2.9.1")
else:
    install_requires.append("tensorflow<=2.9.1")


setup(
    name="tensorpotential",
    # version=versioneer.get_version(),
    version="0.2.0+19.gfa5b8b0",
    cmdclass=versioneer.get_cmdclass(),
    packages=[
        "tensorpotential",
        "tensorpotential.utils",
        "tensorpotential.potentials",
        "tensorpotential.functions",
    ],
    package_dir={"": "src"},
    url="",
    license="",
    author="Anton Bochkarev",
    author_email="",
    description="",
    python_requires="<3.10",
    install_requires=install_requires,
)
