"""Compatibility shim for legacy editable installs.

Project metadata lives in pyproject.toml. This file exists so environments
whose build backend lacks PEP 660's build_editable hook can still fall back to
setuptools' legacy editable install path.
"""

from setuptools import setup


setup()
