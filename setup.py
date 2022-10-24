from setuptools import find_packages
from setuptools import setup

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name="safeworder",
    version="1.0.2",
    description="Replace dirty strings with clean ones",
    license='MIT',
    author="Tommel",
    package_data={
      'safeworder': ['*'],
    },
    url="https://github.com/Tommel71/safeworder",
    zip_safe=False,
    include_package_data=True,
    packages=find_packages(),
    install_requires=required
)