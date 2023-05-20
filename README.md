# RPutils
## Python library for extraterrestrial rock physics

This library contains several rock physics models, utilities, and functions to help describe the elastic properties of rocks and granular media. Much of this code is based on equations presented in The Rock Physics Handbook (Mavko et al.), see docstrings for additional references.

This initial purpose of this library is to support rock physics modeling of lunar regolith simulant. More information can be found in the [LunarIce repository](https://github.com/astroChance/LunarIce)

Installation instructions:
Clone this repo to a local directory. From the command line, activate your Python environment and cd to the location where you saved this repo. Enter `python setup.py bdist_wheel`, then cd to the newly created "dist" directory. Enter `pip install {copy the name of the wheel file}`. The library should now be accessible. The current structure can be conveniently imported as `import rputils.rputils as rp`
