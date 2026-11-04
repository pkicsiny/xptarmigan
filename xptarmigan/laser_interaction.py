import numpy as np
import tempfile, subprocess
from pathlib import Path
import pandas as pd
import h5py
import os
from schema import Schema, Use, And, Optional

to_float = lambda x: float(x)
to_int = lambda x: int(x)

LASER_PARAMETER_SCHEMA = Schema(
    {
        'pulse_energy_j':                         Use(to_float,  error='Invalid pulse energy. Unit: [J]'),
        'w0_waist_um':                            Use(to_float,  error='Invalid laser w0 (1/e2) spot size. Unit: [um]'),
        'wavelength_um':                          Use(to_float,  error='Invalid laser wave length. Unit: [um]'),
        'fwhm_pulse_duration_fs':                 Use(to_float,  error='Invalid FWHM pulse duration. Unit: [fs]'),
        'polarization':                           And(lambda x: x in ("linear", "circular"),  error='Invalid polarization.'),
        Optional(       'angle_deg', default=0.): Use(to_float,  error='Invalid angle. Unit: [deg]'),
        Optional('ebeam_offset_x_m', default=0.): Use(to_float,  error='Invalid ebeam IP shift x. Unit: [m]'),
        Optional('ebeam_offset_y_m', default=0.): Use(to_float,  error='Invalid ebeam IP shift y. Unit: [m]'),
        Optional('ebeam_offset_z_m', default=0.): Use(to_float,  error='Invalid ebeam IP shift z. Unit: [m]'),
        Optional(   'dt_multiplier', default=1.): Use(to_float,  error='Invalid dt multiplier. Unit: [1]'),
        #Optional(        'rng_seed', default=1): Use(to_int,  error='Invalid rng seed. Unit: [1]'),
        
    }, ignore_extra_keys=True
)

PTARMIGAN_INPUT_FILENAME  = 'ptarmigan_input.yml'
PTARMIGAN_BEAM_FILENAME   = 'ptarmigan_input_ebeam.h5'
PTARMIGAN_LOG_FILENAME    = 'ptarmigan_log.txt'
PTARMIGAN_OUTPUT_FILENAME = 'ptarmigan_output_particles.h5'

def xsuite_to_ptarmigan_h5(
    particles,
    particle_range,
    temp_dir,
    s_ref=0,            # [m] distance upstream of laser focus (z = +s_ref for a -z beam)
    beam_axis="-z",
    rotate_about_y=True,  # keep right-handed frame when going from +s to -z
):
    p = particles
    m = np.arange(*particle_range)

    if p.q0==1:
        species="positron"
    elif p.q0==-1:
        species="electron"
    else:
        raise ValueError("particles.q0 must be either 1 or -1!")

    # drop lost particles
    alive = np.asarray(p.state)[m] > 0
    m = m[alive]

    beta0   = np.asarray(p.beta0)
    p0c   = np.asarray(p.p0c)[m]    / 1e6   # [MeV]
    E_tot = np.asarray(p.energy)[m] / 1e6   # [MeV]
    x     = np.asarray(p.x)[m]
    y     = np.asarray(p.y)[m]
    px    = np.asarray(p.px)[m]
    py    = np.asarray(p.py)[m]
    zeta  = np.asarray(p.zeta)[m]
    delta = np.asarray(p.delta)[m]
    w     = np.asarray(p.weight)[m]
    
    ###########
    # convert #
    ###########
    
    # momentum from [rad] to [MeV/c]
    p_tot   = p0c * (1 + delta)
    px_phys = px * p0c
    py_phys = py * p0c
    pz_phys = np.sqrt(p_tot**2 - px_phys**2 - py_phys**2)

    # move from s based (xsuite) to t based (ptarmigan) system
    x_phys = x + px_phys/pz_phys * zeta
    y_phys = y + py_phys/pz_phys * zeta
    
    if beam_axis == "-z": # ebeam goes from right to left
        z_phys  = s_ref - zeta
        pz_phys = -pz_phys
        if rotate_about_y:
            x_phys, px_phys = -x_phys, -px_phys
    else:  # ebeam goes from left to right
        z_phys = -s_ref + zeta

    ct = 0*(s_ref-z_phys)/beta0
    
    # in ptarmigan t is the independent variable 
    momentum = np.column_stack([E_tot, px_phys, py_phys, pz_phys])
    position = np.column_stack([ct, x_phys, y_phys, z_phys])
    weight   = w.reshape(-1, 1).astype("f8")

    ##############
    # dump to h5 #
    ##############
    
    utf8 = h5py.string_dtype(encoding="utf-8")
    with h5py.File(os.path.join(temp_dir, PTARMIGAN_BEAM_FILENAME), "w") as f:
        f.create_dataset(           "beam_axis", data=beam_axis, dtype=utf8)
        f.create_dataset("config/unit/momentum", data=  "MeV/c", dtype=utf8)
        f.create_dataset("config/unit/position", data=      "m", dtype=utf8)

        g = f.create_group(f"final-state/{species}")
        g.create_dataset(  "weight", data=  weight, dtype="f8")
        g.create_dataset("momentum", data=momentum, dtype="f8")
        g.create_dataset("position", data=position, dtype="f8")

    return momentum, position, weight

def prepare_ptarmigan_input(ptarmigan_parameters_dict: dict, particles, seed: int, working_directory: Path) -> None:
    
    ####################
    # get beam species #
    ####################
   
    if particles.q0==1:
        species="positron"
    elif particles.q0==-1:
        species="electron"
    else:
        raise ValueError("particles.q0 must be either 1 or -1!")

    ####################
    # get laser params #
    ####################
    
    angle_deg              = ptarmigan_parameters_dict[             'angle_deg']  # [deg]
    pulse_energy_j         = ptarmigan_parameters_dict[        'pulse_energy_j']  # [J]
    w0_waist_um            = ptarmigan_parameters_dict[           'w0_waist_um']  # [um]
    wavelength_um          = ptarmigan_parameters_dict[         'wavelength_um']  # [um]
    fwhm_pulse_duration_fs = ptarmigan_parameters_dict['fwhm_pulse_duration_fs']  # [fs]
    polarization           = ptarmigan_parameters_dict[          'polarization']  # [string]
    ebeam_offset_x_m       = ptarmigan_parameters_dict[      'ebeam_offset_x_m']  # [m]
    ebeam_offset_y_m       = ptarmigan_parameters_dict[      'ebeam_offset_y_m']  # [m]
    ebeam_offset_z_m       = ptarmigan_parameters_dict[      'ebeam_offset_z_m']  # [m]
    dt_multiplier          = ptarmigan_parameters_dict[         'dt_multiplier']  # [1]
    #rng_seed               = ptarmigan_parameters_dict[              'rng_seed']  # [1]

    a0 = 6.6e-6 * wavelength_um/w0_waist_um * np.sqrt(pulse_energy_j/(fwhm_pulse_duration_fs*1e-15))
    
    ###############################
    # create ptarmigan input file #
    ###############################
    
    with open(Path(working_directory)/PTARMIGAN_INPUT_FILENAME, 'w', encoding='utf-8') as ptarmigan_input:
        ptarmigan_input.write(f"control:\n")
        ptarmigan_input.write(f"  rng_seed: {seed}\n")
        ptarmigan_input.write(f"  dt_multiplier: {dt_multiplier}\n")
        ptarmigan_input.write(f"  lcfa: false\n")
        ptarmigan_input.write(f"  radiation_reaction: true\n")
        ptarmigan_input.write(f"  pair_creation: false\n")
        ptarmigan_input.write(f"\n")
        ptarmigan_input.write(f"laser:\n")
        ptarmigan_input.write(f"  a0: {a0}\n")
        ptarmigan_input.write(f"  wavelength: {wavelength_um} * micro\n")
        ptarmigan_input.write(f"  fwhm_duration: {fwhm_pulse_duration_fs} * femto\n")
        ptarmigan_input.write(f"  waist: {w0_waist_um} * micro\n")
        ptarmigan_input.write(f"  polarization: {polarization}\n")
        ptarmigan_input.write(f"\n")
        ptarmigan_input.write(f"beam:\n")
        ptarmigan_input.write(f"  from_hdf5:\n")
        ptarmigan_input.write(f"    file: {PTARMIGAN_BEAM_FILENAME}\n")
        ptarmigan_input.write(f"    distance_bt_ips: 0.0\n")
        ptarmigan_input.write(f"  species: {species}\n")
        ptarmigan_input.write(f"  collision_angle: {angle_deg} * degree\n")
        ptarmigan_input.write(f"  offset: [{ebeam_offset_x_m}, {ebeam_offset_y_m}, {ebeam_offset_z_m}]\n")
        ptarmigan_input.write(f"\n")
        ptarmigan_input.write(f"output:\n")
        ptarmigan_input.write(f"  ident: ptarmigan_output\n")
        ptarmigan_input.write(f"  coordinate_system: beam\n") # transforms back ebeam coords to +z
        ptarmigan_input.write(f"  discard_background: false\n") # true discards particles without collision
        ptarmigan_input.write(f"  dump_all_particles: hdf5\n")
        ptarmigan_input.write(f"  file_format: fits\n")
        ptarmigan_input.write(f"  photon:\n")
        ptarmigan_input.write(f"    - energy\n")
        ptarmigan_input.write(f"    - energy:(auto; auto; angle in 0, 100 * micro)\n")
        ptarmigan_input.write(f"    - angle_x:angle_y:(auto; auto; angle in 0, 20 * micro)\n")

            
def run_ptarmigan(ptarmigan_path: Path, working_directory: Path) -> None:
    
    # ./target/release/ptarmigan examples/hdf5/stage1_import.yml
    input_file = working_directory / PTARMIGAN_INPUT_FILENAME

    with open(working_directory / PTARMIGAN_LOG_FILENAME, 'w', encoding='utf-8') as log:
        result = subprocess.run(
            [str(ptarmigan_path), str(input_file)],
            cwd=working_directory,
            stdout=log,
            stderr=subprocess.PIPE,
            text=True,
        )
    if result.returncode != 0:
        raise RuntimeError(f"ptarmigan at {working_directory} exited with {result.returncode}\n--- stderr ---\n{result.stderr}")
        
def ptarmigan_h5_to_xsuite(
    particles, 
    particle_range, 
    temp_dir: Path,
    s_ref=0.0,
):
    """
    This assumes that the 'coordinate_system: beam'
    option is used in the ptarmigan input.
    """
    
    m = range(particle_range[0], particle_range[1])
    p0c_ev   = particles.p0c[0]
    beta0 = particles.beta0[0]

    if particles.q0==1:
        species="positron"
    elif particles.q0==-1:
        species="electron"
    else:
        raise ValueError("particles.q0 must be either 1 or -1!")

    with h5py.File(str(temp_dir/PTARMIGAN_OUTPUT_FILENAME), 'r') as f:
        g = f[f"final-state/{species}"]
        pos = g["position"][:]
        mom = g["momentum"][:]
        w   = g[  "weight"][:]

        ##############
        # 4-position #
        ##############

        ct = pos[:, 0] # time at end of simulation
        x_phys = pos[:, 1]
        y_phys = pos[:, 2]
        z_phys = pos[:, 3]

        ##############
        # 4-momentum #
        ##############

        E_tot_ev= mom[:, 0] * 1e6  # [eV/c]
        px_phys = mom[:, 1] * 1e6  # [eV/c]
        py_phys = mom[:, 2] * 1e6  # [eV/c]
        pz_phys = mom[:, 3] * 1e6  # [eV/c]

        ################
        # t to s based #
        ################

        dz = s_ref - z_phys 
        dx_dz = px_phys / pz_phys
        dy_dz = py_phys / pz_phys

        x_out = x_phys + dx_dz * dz
        y_out = y_phys + dy_dz * dz

        # time at which each particle reaches z = s_ref
        beta_z   = pz_phys / E_tot_ev
        ct_cross = ct + (s_ref - z_phys) / beta_z
        zeta_out = -beta0 * ct_cross

        # from [ev/c] to [rad]
        p_tot     = np.sqrt(px_phys**2 + py_phys**2 + pz_phys**2)
        delta_out =   p_tot / p0c_ev - 1
        px_out    = px_phys / p0c_ev
        py_out    = py_phys / p0c_ev
        
        # place back into xpart buffers
        particles.x    [m] = x_out
        particles.px   [m] = px_out
        particles.y    [m] = y_out
        particles.py   [m] = py_out
        particles.zeta [m] = zeta_out
        particles.delta[m] = delta_out

def get_active_particle_id_range(particles):
    """
    Get the range of particle indices of active particles.
    """
    ctx2np = particles._buffer.context.nparray_from_context_array
    idx_active = np.argwhere(ctx2np(particles.state) > 0)
    return idx_active[0][0], idx_active[-1][0] + 1

    
class LaserInteraction():
    def __init__(self, name, ptarmigan_path: Path, laser_parameter: dict, seed: int):
        self.ptarmigan_path = Path(ptarmigan_path).absolute()
        self.laser_parameter = LASER_PARAMETER_SCHEMA.validate(laser_parameter)
        self.seed = seed
        self.is_on = True

    def track(self, particles):
        if not self.is_on:
            return
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)

            particles.reorganize()
            particle_range = get_active_particle_id_range(particles)

            n_particles = particle_range[1] - particle_range[0]
            if n_particles <= 1000:
                raise ValueError(
                    f"Number of particles too low: got {n_particles}, need more than 1000."
                )

            xsuite_to_ptarmigan_h5(particles, particle_range, temp_dir)
            prepare_ptarmigan_input(self.laser_parameter, particles, self.seed, temp_dir)
            run_ptarmigan(self.ptarmigan_path, temp_dir)
            ptarmigan_h5_to_xsuite(particles, particle_range, temp_dir)
        self.seed = self.seed + 1
