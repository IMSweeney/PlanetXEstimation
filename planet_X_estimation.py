import os
import pandas as pd
import numpy as np
import re

import matplotlib.pyplot as plt


import horizons_API as hapi


def load_object_list_jpl():
    df = pd.read_csv(r'data\jpl_tno_db.csv')
    df['horizons'] = "DES=+" + df['spkid'].astype(str)

    df = df.loc[(
        (df['a'] > 150) &
        ((0.75 < df['e']) & (df['e'] < 0.98)) &
        ((3 < df['i']) & (df['i'] < 25)) &
        ((200 < df['w']) & (df['w'] < 350))
    )]
    print('{} objects found'.format(len(df.index)))
    # print(df['full_name'])

    # TODO: Merge in mass/diameter data

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


if __name__ == '__main__':
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
        if obj_name in ['Planet_X_(6)', 'Planet_X_(12)']:
            continue
        dat = obj['data']
        dat['r_i'] = dat.apply(
            lambda x: np.array([x['X'], x['Y'], x['Z']]), axis=1)
        dat['v_i'] = dat.apply(
            lambda x: np.array([x['VX'], x['VY'], x['VZ']]), axis=1)

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
pass
