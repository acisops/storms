#!/usr/bin/env python
from setuptools import setup

entry_points = {
    "console_scripts": [
        "calc_p3_fluence = storms.apps.calc_p3_fluence:main",
        "make_storm_plots = storms.apps.make_storm_plots:main",
    ],
}

setup(
    name="storms",
    packages=["storms", "storms.txings_proxy", "storms.apps"],
    use_scm_version=True,
    setup_requires=["setuptools_scm", "setuptools_scm_git_archive"],
    description="Solar storm analysis for ACIS Ops",
    author="John ZuHone",
    author_email="john.zuhone@cfa.harvard.edu",
    url="http://github.com/acisops/storms",
    install_requires=["numpy>=1.12.1", "requests", "astropy"],
    classifiers=[
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
    ],
    entry_points=entry_points,
)
