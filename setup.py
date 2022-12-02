from setuptools import find_packages
from setuptools import setup

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name="eventstudystatistics",
    version="1.0.0",
    description="statistics for event studies",
    license='MIT',
    author="Tommel",
    package_data={
      'eventstudystatistics': ['*'],
    },
    url="https://github.com/Tommel71/event-study-statistics",
    zip_safe=False,
    include_package_data=True,
    packages=find_packages(),
    install_requires=required
)