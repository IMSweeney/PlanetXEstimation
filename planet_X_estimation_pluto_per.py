import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from uncertainties import ufloat

import horizons_API as hapi
import genetic_solver as gen
import logginglite


# Constants
M_sun = 1.98847e30  # [kg] From wikipedia, +/- 0.00007e30
G = 6.6743015e-11  # [m^3/kg-s^2] From wikipedia

au_per_m = 6.6845871226706E-12  # [AU/m]
s_per_d = 86400  # [s/d]
G_au_d = G * (au_per_m ** 3) * (s_per_d ** 2)  # [au^3/kg-s^2]


log = logginglite.create_logger(__name__, level='INFO')


def raw_ephem_to_vec(obj):
    # Process data into vectors/ generate accelerations
    dat = obj['data']
    dat['r_i'] = dat.apply(
        lambda x: np.array([x['X'], x['Y'], x['Z']]), axis=1)
    dat['v_i'] = dat.apply(
        lambda x: np.array([x['VX'], x['VY'], x['VZ']]), axis=1)

    # Don't need to / time since units are au/d and points are 1d steps
    dat['a_i'] = dat['v_i'].diff() / dat['JDTDB'].diff()


def score_params(params, objects):
    # params are the masses of each object (ateroids[0:n] + planet_x)
    M_px = params[-1]

    ##
    # Calculate the expected acceleration for each asteroid based on Px and Sun
    # Score the error in actual vs expected accel for each asteroid
    dat_p_x = calculate_planet_x_position(objects, params, M_px)

    # calculate_acceleration(objects, masses, dat_p_x, M_px)
    calculate_acceleration_all_ast(objects, masses, dat_p_x, M_px)

    score = 0
    for i, obj in enumerate(objects):
        dat = obj['data']

        dat['sq_err'] = (dat['a_i'] - dat['a_calc']) ** 2
        score += dat['sq_err'].apply(np.linalg.norm).sum()

    return score ** (1 / 2)


# UNFINISHED
def calculate_planet_x_pos_w_orbits(objects, obrit, true_anamaly):
    # orbit = dict
    #   period - 3375   [yrs]
    #   a      - 225    [AU]
    #   e      - 0.62   []
    #   p      - 86     [AU]
    #   l      - 19.5   [deg]
    #   om     - 95.5   [deg]
    #   w      - 153.5  [deg]
    #
    # {\displaystyle r=a\,{1-e^{2} \over 1+e\cos \nu }\,\!}
    dat_p_x = objects[0]['data']['Calendar Date (TDB)'].to_frame()

    dat_p_x['R'] = dat_p_x.apply(transform_to_cartesian, axis=1)

    return dat_p_x


# UNFINISHED
def transform_to_cartesian(orbit, true_anamaly):
    pass


def plot_pos(dat_p_x, r_vec):
    # Plot planetX location wrt time
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    x = dat_p_x[r_vec].apply(lambda i: i[0])
    y = dat_p_x[r_vec].apply(lambda i: i[1])
    z = dat_p_x[r_vec].apply(lambda i: i[2])
    ax.plot(x, y, z)
    plt.show()


def calc_acc_perturb_obs(obj):
    data = obj['data']
    log.info('{}, Mass: {}'.format(obj['Name'], obj['Mass']))
    data['a_sun'] = (
        (G_au_d * M_sun * data['r_i']) /
        (data['r_i'].apply(np.linalg.norm)**3)).shift()
    data['a_perturb'] = data['a_i'] - data['a_sun']

    log.info('average perturbation from the keplerian orbit is:\n{}'.format(
        data['a_perturb'].mean()))
    # TODO: direction of perturbation


def calc_perturb_planets(target, objects):
    data = target['data']
    data['a_per_planets'] = data.apply(
        lambda x: np.array([0, 0, 0]), axis=1)
    pd.set_option('mode.chained_assignment', None)
    data['a_per_planets'].iloc[0] = np.nan
    for obj in objects:
        if obj == target:
            continue
        mass = obj['Mass']
        data['dr'] = data['r_i'] - obj['data']['r_i']
        data['a_per_planets'] += (
            (G_au_d * mass * data['dr']) /
            (data['dr'].apply(np.linalg.norm)**3)).shift()

    log.info('average perturbation due to planets is:\n{}'.format(
        data['a_per_planets'].mean()))
    log.info('normalized error in perturbation is:\n{} AU/d^2'.format(
        ((data['a_perturb'] - data['a_per_planets']) / data['a_perturb'])
        .mean()))


def calc_perturb_ast(target, objects):
    pass


def calc_perturb_px(target, px):
    pass


if __name__ == '__main__':
    ##
    # Load and process ephemeris data for planets
    planet_data_dir = r'data\horizons\planets'
    planets = pd.read_csv(r'data\planets.csv', sep=', ', engine='python')
    planets['Mass'] = planets['Mass'].astype(float)
    planets = planets.to_dict(orient='records')
    masses = [obj['Mass'] for obj in planets]

    for obj in planets:
        if pd.isna(obj['horizons']):
            log.error('Object {} missing horizons id'.format(obj['Name']))
            exit()
        obj['data'] = hapi.get_obj_data(
            planet_data_dir, obj['horizons'], stepsize='50d', overwrite=False,
            start_time='1910-01-01-00-00-00', stop_time='2050-01-01-00-00-00')
        raw_ephem_to_vec(obj)

    pluto = [obj for obj in planets if obj['Name'] == 'Pluto'][0]
    log.info(pluto['data']['Calendar Date (TDB)'].head())
    calc_acc_perturb_obs(pluto)
    calc_perturb_planets(pluto, planets)

    plot_pos(pluto['data'].iloc[1:], 'r_i')
    plot_pos(pluto['data'].iloc[1:], 'a_sun')
    dat = pluto['data']
    dat['a_sun_norm'] = dat['a_sun'].apply(np.linalg.norm)
    ax = plt.axes()
    ax.plot(dat['a_sun_norm'])
    plt.show()
    exit()

    # Initial score
    score = score_params(masses, objects)
    print('Initial score: {} AU/d^2'.format(score))
    print('Vec: {}'.format(masses))

    # Genetic solver
    # solver = gen.GeneticSolver(masses, score_params, args=[objects],
    #                            pop_size=100, num_iterations=100)

    # best, score = solver.solve()
    # print('Best score: {} AU/d^2'.format(score))
    # print('Vec: {}'.format(best))

    # Plot
    dat_px = calculate_planet_x_position(objects, masses, M_px)
    plot_px_pos(dat_px)
