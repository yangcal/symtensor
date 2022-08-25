from setuptools import setup, find_packages
import os

PACKAGE_NAME = 'symtensor'

VERSION_PATH = os.path.join(os.path.dirname(__file__), PACKAGE_NAME, 'VERSION')
with open(VERSION_PATH) as version_file:
    VERSION = version_file.read().strip()

setup(
    name=PACKAGE_NAME,
    version=VERSION,
    packages=find_packages(exclude=[]),
    package_data={
        PACKAGE_NAME: ['VERSION'],
    },
    install_requires=[
        'numpy',
        'tensorbackends',
    ],
)
