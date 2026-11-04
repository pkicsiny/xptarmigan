import pytest
from xptarmigan import laser_interaction as xpt
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from schema import SchemaError
import xpart as xp
import xobjects as xo
from xptarmigan import laser_interaction as xpt

INPUT_DIR = Path(__file__).parent / "input"
PTARMIGAN_EXECUTABLE = Path(__file__).parent.parent.parent / "ptarmigan_executable" / "ptarmigan_compiled.mac"
XSUITE_COORDINATES = ['x', 'px', 'y', 'py', 'zeta', 'delta']

EXAMPLE_LASER_PARAMETER = {
        'pulse_energy_j': 10e-3,         
           'w0_waist_um': 1.8,            
         'wavelength_um': .8,            
'fwhm_pulse_duration_fs': 48,   
          'polarization': "linear",             
             'angle_deg': 180+28.1,
      'ebeam_offset_x_m': 0,
      'ebeam_offset_y_m': 0,
      'ebeam_offset_z_m': 0,
         'dt_multiplier': 1,
              'rng_seed': 42,
}

def make_beam(q0, n_part=1000, seed=42):
    """
    helper function to create an xpart beam
    """

    rng = np.random.default_rng(seed)
    kw = {k: 1e-6 * rng.standard_normal(n_part)
          for k in ("x", "px", "y", "py", "zeta", "delta")}
    return xp.Particles(
        _context=xo.ContextCpu(), q0=q0, p0c=10e9, mass0=xp.ELECTRON_MASS_EV,
        weight=1.55e11 / n_part, **kw)


@pytest.mark.parametrize("q0", [-1, 1])
def test_run_ptarmigan(q0, tmp_path):
    """
    test that ptarmigan runs and
    the laser changes coordinates and
    the beam loses momentum
    """

    particles_in = make_beam(q0)
    particles_in_df = particles_in.to_pandas()

    particles = particles_in.copy()
    active_range = particles.get_active_particle_id_range()

    xpt.xsuite_to_ptarmigan_h5(particles, active_range, tmp_path)

    xpt.prepare_ptarmigan_input(EXAMPLE_LASER_PARAMETER, particles, 1, tmp_path)
    assert (tmp_path / xpt.PTARMIGAN_INPUT_FILENAME).exists()

    xpt.run_ptarmigan(PTARMIGAN_EXECUTABLE, tmp_path)
    assert (tmp_path / xpt.PTARMIGAN_OUTPUT_FILENAME).exists()
    assert (tmp_path / xpt.PTARMIGAN_LOG_FILENAME).exists()

    xpt.ptarmigan_h5_to_xsuite(particles, active_range, tmp_path)
    particles_out_df = particles.to_pandas()

    assert particles_out_df.delta.mean() < particles_in_df.delta.mean(), \
        "Beam final mean momentum higher than at start!"

    with h5py.File(tmp_path / xpt.PTARMIGAN_OUTPUT_FILENAME, "r") as f:
        assert f["final-state/photon/weight"].shape[0] > 0, "No photons were emitted!"


def test_run_ptarmigan_invalid_charge(tmp_path):
    """
    test that an invalid ebeam charge such as 0 fails
    """

    particles = make_beam(q0=0)
    active_range = particles.get_active_particle_id_range()
    with pytest.raises(ValueError, match="q0"):
        xpt.xsuite_to_ptarmigan_h5(particles, active_range, tmp_path)


def test_run_ptarmigan_empty(tmp_path):
    """
    test that ptarmigan dump and readback
    gives back same ebeam coords
    when no laser is present
    """

    particles_in = make_beam(-1)
    particles_in_df = particles_in.to_pandas()

    particles = particles_in.copy()
    active_range = particles.get_active_particle_id_range()

    xpt.xsuite_to_ptarmigan_h5(particles, active_range, tmp_path)

    laser_dict = EXAMPLE_LASER_PARAMETER.copy()
    laser_dict["pulse_energy_j"] = 0
    xpt.prepare_ptarmigan_input(laser_dict, particles, 1, tmp_path)
    assert (tmp_path/xpt.PTARMIGAN_INPUT_FILENAME).exists()

    xpt.run_ptarmigan(PTARMIGAN_EXECUTABLE, tmp_path)
    assert (tmp_path/xpt.PTARMIGAN_OUTPUT_FILENAME).exists()
    assert (tmp_path/xpt.PTARMIGAN_LOG_FILENAME).exists()

    xpt.ptarmigan_h5_to_xsuite(particles, active_range, tmp_path)
    particles_out_df = particles.to_pandas()

    np.testing.assert_allclose(
            particles_out_df.loc[:, XSUITE_COORDINATES].to_numpy(),
            particles_in_df.loc[:, XSUITE_COORDINATES].to_numpy(),
            rtol=0, atol=1e-12,
    )

    with h5py.File(tmp_path / xpt.PTARMIGAN_OUTPUT_FILENAME) as f:
        w_gamma = f["final-state/photon/weight"]
        assert len(w_gamma)==0, "Photons were emitted!"


def test_check_too_few_macroparts(tmp_path):
    """
    test that xptarmigan correctly requires
    minimum 1000 ebeam macroparticles
    """   
 
    particles = make_beam(q0=-1, n_part=999)

    xpt_laser = xpt.LaserInteraction(
        'laser',
        PTARMIGAN_EXECUTABLE,
        EXAMPLE_LASER_PARAMETER,
        1,
        )

    with pytest.raises(ValueError, match="too low"):
        xpt_laser.track(particles)


def test_laser_dict():
    """
    test laser dict keys
    """

    xpt.LASER_PARAMETER_SCHEMA.validate(EXAMPLE_LASER_PARAMETER)
    failing_dict = EXAMPLE_LASER_PARAMETER.copy()

    failing_dict.pop('angle_deg') # removing optional key should pass
    xpt.LASER_PARAMETER_SCHEMA.validate(failing_dict)

    failing_dict.pop('pulse_energy_j') # removing mandatory key should fail
    with pytest.raises(SchemaError):
        xpt.LASER_PARAMETER_SCHEMA.validate(failing_dict)

