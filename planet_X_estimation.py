import logging

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import horizons_API as hapi

import genetic_solver as gen

G = 6.6743015e-11  # [m^3] From wikipedia
M_sun = 1.989e30  # [kg] From wikipedia


def create_logger(name, level=logging.INFO, fh=None):
    # create logger with 'spam_application'
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    if fh:
        # create file handler which logs even debug messages
        fh = logging.FileHandler('out.log')
        fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)


def load_object_list_jpl():
    df = pd.read_csv(r'data\jpl_tno_db.csv')
    df['horizons'] = "DES=+" + df['spkid'].astype(str)

    # could filter objects based on parameters
    # df = df.loc[(
    #     (df['a'] > 150) &
    #     ((0.75 < df['e']) & (df['e'] < 0.98)) &
    #     ((3 < df['i']) & (df['i'] < 25)) &
    #     ((200 < df['w']) & (df['w'] < 350))
    # )]

    # For now choose the 12 obj from paper and merge in phys data
    # Merge in mass/diameter data
    masses = pd.read_csv(r'data\TNO_parameters.txt')
    df = df.merge(masses, on='spkid', how='inner')

    print('{} objects found'.format(len(df.index)))
    print(df[['full_name', 'Mass(kg)', 'horizons']].head())

    return df.to_dict(orient='records')


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


def calculate_planet_x_position(objects, masses, M_px):
    ##
    # For each point in time, Calculate Planet_X location from EQ31:
    # EQ31: m_p^2 * R_p = -M * sum(m_i * r_si)
    # Where:
    #   - m_p  is the mass of planet X
    #   - R_p  is the Range vector of planet X
    #   - M    is the mass of the Sun
    #   - m_i  is the mass of the satelite, i
    #   - r_si is the range vector of satelite i w.r.t the Sun
    dat_p_x = objects[0]['data']['Calendar Date (TDB)'].to_frame()
    dat_p_x['sum_mi_ri'] = dat_p_x.apply(
        lambda x: np.array([0, 0, 0]), axis=1)

    for i, obj in enumerate(objects):
        dat = obj['data']
        dat_p_x['sum_mi_ri'] += masses[i] * dat['r_i']
    dat_p_x['R'] = dat_p_x['sum_mi_ri'] * -1 * M_sun / M_px
    return dat_p_x


def calculate_acceleration(objects, masses, dat_p_x, M_px):
    for i, obj in enumerate(objects):
        dat = obj['data']
        dat['r_px'] = dat['r_i'] - dat_p_x['R']
        dat['F_sun'] = (G * M_sun * masses[i] / (dat['r_i']**2)).shift()
        dat['F_px'] = (G * M_px * masses[i] / (dat['r_px']**2)).shift()
        dat['a_calc'] = dat['F_sun'] + dat['F_px'] / masses[i]
        # NOTE: mass of each asteroid seems to be irrelivant...


def calculate_acceleration_all_ast(objects, masses, dat_p_x, M_px):
    for i, obj in enumerate(objects):
        dat = obj['data']

        # Accel comp from sun and p_x
        dat['r_px'] = dat['r_i'] - dat_p_x['R']
        dat['a_sun'] = (G * M_sun / (dat['r_i']**2)).shift()
        dat['a_px'] = (G * M_px / (dat['r_px']**2)).shift()
        dat['a_calc'] = dat['a_sun'] + dat['a_px']
        for j, obj2 in enumerate(objects):
            dat2 = obj2['data']
            if i == j:
                dat['a_{}'.format(j)] = 0
            else:
                dat['a_{}'.format(j)] = (
                    G * masses[j] / ((dat['r_i'] - dat2['r_i'])**2)).shift()
            dat['a_calc'] = dat['a_calc'] + dat['a_{}'.format(j)]

        dat['sun_contrib'] = dat.apply(
            lambda x: np.divide(x['a_sun'], x['a_calc']), axis=1)
        avg_sun_contrib = dat['sun_contrib'].mean()
        print('sun contributes {} of accel'.format(avg_sun_contrib))

        dat['px_contrib'] = dat.apply(
            lambda x: np.divide(x['a_px'], x['a_calc']), axis=1)
        avg_px_contrib = dat['px_contrib'].mean()
        print('px contributes {} of accel'.format(avg_px_contrib))


def plot_px_pos(dat_p_x):
    # Plot planetX location wrt time
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    x = dat_p_x['R'].apply(lambda i: i[0])
    y = dat_p_x['R'].apply(lambda i: i[1])
    z = dat_p_x['R'].apply(lambda i: i[2])
    ax.plot(x, y, z)
    plt.show()


if __name__ == '__main__':
    ##
    # Meta_data (masses)
    objects = load_object_list_jpl()
    M_px = 5.388e25  # [kg] Estimate from paper

    ##
    # Load and process ephemeris data
    data_dir = r'data\horizons\etc'
    for obj in objects:
        if pd.isna(obj['horizons']):
            exit('Object {} missing horizons id'.format(obj['full_name']))
        obj['data'] = hapi.get_obj_data(data_dir, obj['horizons'])
        raw_ephem_to_vec(obj)

    masses = [obj['Mass(kg)'] for obj in objects] + [M_px]

    # Initial score
    score = score_params(masses, objects)
    print('Initial score: {}'.format(score))
    print('Vec: {}'.format(masses))

    # # Genetic solver
    # solver = gen.GeneticSolver(masses, score_params, args=[objects],
    #                            pop_size=100, num_iterations=1000)

    # best, score = solver.solve()
    # print('Best score: {}'.format(score))
    # print('Vec: {}'.format(best))

    # Plot
    dat_px = calculate_planet_x_position(objects, masses, M_px)
    plot_px_pos(dat_px)
