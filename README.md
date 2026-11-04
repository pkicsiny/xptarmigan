# XPtarmigan

XPtarmigan is an interface between [Xsuite](https://github.com/xsuite) and [Ptarmigan](https://github.com/tgblackburn/ptarmigan) which simulates nonlinear inverse Compton scattering between e+/e- and laser gamma photons. The package build follows [XCain](https://gitlab.cern.ch/mihofer/xcain)".

## Getting started

To run, first install the required python packages by running 

`python -m pip install .`.

Note that as of time of writing, only Python 3.11 is supported.

For using, install a `LaserInteraction` element in the beamline following

```python
import xptarmigan as xpt

xpt_laser = xpt.LaserInteraction(
        'laser',
        PTARMIGAN_EXECUTABLE,
        EXAMPLE_LASER_PARAMETER,
        seed=1)
```
