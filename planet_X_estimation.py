import os
import pandas as pd
import numpy as np
import re

import matplotlib.pyplot as plt
import horizons_API as hapi

import genetic_solver as gen

G = 6.6743015e-11  # [m^3]


def load_object_list_jpl():
    df = pd.read_csv(r'data\jpl_tno_db.csv')
    df['horizons'] = "DES=+" + df['spkid'].astype(str)

    # df = df.loc[(
    #     (df['a'] > 150) &
    #     ((0.75 < df['e']) & (df['e'] < 0.98)) &
    #     ((3 < df['i']) & (df['i'] < 25)) &
    #     ((200 < df['w']) & (df['w'] < 350))
    # )]

    # TODO: Merge in mass/diameter data
    masses = pd.read_csv(r'data\TNO_parameters.txt')
    df = df.merge(masses, on='spkid', how='inner')

    print('{} objects found'.format(len(df.index)))
    print(df[['full_name', 'Mass(kg)', 'horizons']].head())

    return df.to_dict(orient='records')


def load_object_list():
    # r'data\jpl_tno_db.csv'
    # df['horizons'] = "DES=+"+df['spkid'].astype(str)

    object_df = pd.read_csv(r'data\TNO_parameters.txt')
    outer_asteroids_df = pd.read_csv(r'data\any_outer_asteroids.csv')

    object_df['pdes'] = object_df['Object']
    outer_asteroids_df['pdes'] = outer_asteroids_df['pdes'].map(
        lambda x: x.replace(' ', '_'))

    object_df = object_df.merge(outer_asteroids_df, how='left', on='pdes')
    object_df = object_df[['Object', 'horizons', 'Mass(kg)']]
    print(object_df.head())

    object_list = object_df.to_dict(orient='records')
    return object_list


def score_params(params, objects):
    # params are the masses of each object (ateroids[0:n] + planet_x)
    M_sun = 1.989e30  # kg
    M_px = params[-1]

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
        dat_p_x['sum_mi_ri'] += params[i] * dat['r_i']
    dat_p_x['R'] = dat_p_x['sum_mi_ri'] * -1 * M_sun / M_px

    ##
    # Calculate the expected acceleration for each asteroid based on Px and Sun
    # Score the error in actual vs expected accel for each asteroid
    score = 0
    for i, obj in enumerate(objects):
        dat = obj['data']
        dat['r_px'] = dat['r_i'] - dat_p_x['R']
        dat['F_sun'] = (G * M_sun * params[i] / (dat['r_i']**2)).shift()
        dat['F_px'] = (G * M_px * params[i] / (dat['r_px']**2)).shift()
        dat['a_calc'] = dat['F_sun'] + dat['F_px'] / params[i]
        # NOTE: mass of each asteroid seems to be irrelivant...

        dat['sq_err'] = (dat['a_i'] - dat['a_calc']) ** 2
        score += dat['sq_err'].apply(np.linalg.norm).sum()

    return score ** (1 / 2)


def test():
    ##
    # Meta_data (masses)
    objects = load_object_list_jpl()
    M_sun = 1.989e30  # kg
    M_px = 5.388e25  # kg

    ##
    # Location data (X, Y, Z from Sun)
    data_dir = r'data\horizons\etc'
    for obj in objects:
        if pd.notna(obj['horizons']):
            data = hapi.get_obj_data(data_dir, obj['horizons'])
            obj['data'] = data

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

    for obj in objects:
        obj_name = obj['full_name']
        mass = obj[r'Mass(kg)']
        dat = obj['data']
        dat['r_i'] = dat.apply(
            lambda x: np.array([x['X'], x['Y'], x['Z']]), axis=1)
        dat['v_i'] = dat.apply(
            lambda x: np.array([x['VX'], x['VY'], x['VZ']]), axis=1)

        # Don't need to / time since units are au/d and points are 1d steps
        dat['a_i'] = dat['v_i'].diff()

        dat_p_x['sum_mi_ri'] += mass * dat['r_i']

    dat_p_x['R'] = dat_p_x['sum_mi_ri'] * -1 * M_sun / M_px
    print(dat_p_x.head())

    # Plot planetX location wrt time
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    x = dat_p_x['R'].apply(lambda i: i[0])
    y = dat_p_x['R'].apply(lambda i: i[1])
    z = dat_p_x['R'].apply(lambda i: i[2])
    ax.plot(x, y, z)
    plt.show()


if __name__ == '__main__':
    # test()

    ##
    # Meta_data (masses)
    objects = load_object_list_jpl()
    M_sun = 1.989e30  # kg
    M_px = 5.388e25  # kg

    ##
    # Location data (X, Y, Z from Sun)
    data_dir = r'data\horizons\etc'
    for obj in objects:
        if pd.notna(obj['horizons']):
            data = hapi.get_obj_data(data_dir, obj['horizons'])
            obj['data'] = data

    # Process data into vectors/ generate accelerations
    for obj in objects:
        obj_name = obj['full_name']
        mass = obj[r'Mass(kg)']
        dat = obj['data']
        dat['r_i'] = dat.apply(
            lambda x: np.array([x['X'], x['Y'], x['Z']]), axis=1)
        dat['v_i'] = dat.apply(
            lambda x: np.array([x['VX'], x['VY'], x['VZ']]), axis=1)

        # Don't need to / time since units are au/d and points are 1d steps
        dat['a_i'] = dat['v_i'].diff()

    masses = [obj['Mass(kg)'] for obj in objects] + [M_px]

    score = score_params(masses, objects)
    print('Initial score: {}'.format(score))
    print('Vec: {}'.format(masses))

    solver = gen.GeneticSolver(masses, score_params, args=[objects],
                               pop_size=100, num_iterations=1000)

    best, score = solver.solve()
    print('Best score: {}'.format(score))
    print('Vec: {}'.format(best))
