from setuptools import setup, find_packages
import re

package_name = "rputils"

VERSIONFILE=package_name+"/_version.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    verstr = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in ", VERSIONFILE)

setup(name=package_name,
      version= verstr,
      description='Library of rock physics utilities',
      author='Chance Amos',
      author_email='camos@mines.edu',
      packages = find_packages(),
      install_requires=[
          'numpy',
          'scipy'])
