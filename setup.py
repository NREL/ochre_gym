import io
import os
import re
from setuptools import setup, find_packages

# Read the version from the __init__.py file without importing it
def read(*names, **kwargs):
    with io.open(
            os.path.join(os.path.dirname(__file__), *names),
            encoding=kwargs.get("encoding", "utf8")
    ) as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")

with open('requirements.txt') as f:
    reqs = f.read().splitlines()


setup(name='ochre_gym',
      version=find_version('ochre_gym', '__init__.py'),
      description='A Gymnasium environment based on the purely Python-based OCHRE residential energy building simulator.',
      author='Patrick Emami, Xiangyu Zhang, Peter Graf',
      author_email='pemami@nrel.gov, Xiangyu.Zhang@nrel.gov, Peter.Graf@nrel.gov',
      url='https://github.nrel.gov/NREL/ochre_gym',
      python_requires='>=3.9',
      install_requires=reqs,
      long_description=open('README.md').read(),
      long_description_content_type='text/markdown',
      packages=find_packages(include=['ochre_gym',
                                      'ochre_gym.spaces'],
                             exclude=['test']),
      package_data={'ochre_gym': ['buildings/*/*', 'buildings/defaults.toml', 'energy_price/*/*']},
      license='BSD 3-Clause',
      keywords=['reinforcement learning', 'hvac', 'building', 'building energy simulation'],
      classifiers=[
          "Development Status :: 3 - Alpha",
          "Intended Audience :: Science/Research",
          "License :: OSI Approved :: BSD License",
          "Natural Language :: English",
          "Programming Language :: Python :: 3",
          "Topic :: Scientific/Engineering :: Artificial Intelligence",
      ]
)
