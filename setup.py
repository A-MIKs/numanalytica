#!/usr/bin/env python

"""Minimal setup.py compatibility shim.

The actual package configuration is in pyproject.toml.
This file exists only for backward compatibility with older build tools.
"""

from setuptools import setup

# All configuration is in pyproject.toml
if __name__ == "__main__":
    setup()
