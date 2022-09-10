#!/usr/bin/env python
"""Setup script for the py-pip-install-test package."""

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

requirements = ['pytest==7.1.1']

test_requirements = ['pandas', 'matplotlib', 'sklearn',
                     'streamlit', 'sql', 'pytest>=3', ]

setup(
    author="Michael Getachew",
    email="mikygetyos@gmail.com",
    python_requires='>=3.6',
    description="Rosseman Pharmaceutical Sales prediction",
    install_requires=requirements,
    long_description=readme,
    include_package_data=True,
    keywords='sales_prediction, deep_learning, unit_testing, pytest',
    name='Rossmann',
    packages=find_packages(include=['src', 'src.*']),
    test_suite='Tests',
    tests_require=test_requirements,
    version='0.1.0',
    zip_safe=False,
)