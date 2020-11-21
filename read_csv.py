import os
import pandas as pd
import numpy as np
import re


def read_ephemeris(location, name):
    '''
    Example row
    2459172.500000000 = A.D. 2020-Nov-19 00:00:00.0000 TDB
     X = 4.821981172317374E+01 Y = 6.530051732740368E+01 Z =-2.192485635872903E+01
     VX=-1.432129363419439E-03 VY= 1.862287144980514E-03 VZ= 6.304524709497195E-04
     LT= 4.856248394831282E-01 RG= 8.408333444979316E+01 RR= 4.605993101835660E-04
    '''
    name = 'HORIZON_' + name + '.txt'
    dat_file = os.path.join(location, name)

    data = []

    with open(dat_file, 'r') as f:
        line = f.readline()

        # Seek for '$$SOE'
        while not re.match(r'[$][$]SOE', line):
            line = f.readline()

        line = f.readline()
        # Read data in the data section
        while not re.match(r'[$][$]EOE', line):
            t = re.match(r'.+(?= = )', line).group()
            line = f.readline()
            X = re.search(r'(?<=X =).+?[+-]\d\d', line).group()
            Y = re.search(r'(?<=Y =).+?[+-]\d\d', line).group()
            Z = re.search(r'(?<=Z =).+?[+-]\d\d', line).group()
            line = f.readline()
            VX = re.search(r'(?<=VX=).+?[+-]\d\d', line).group()
            VY = re.search(r'(?<=VY=).+?[+-]\d\d', line).group()
            VZ = re.search(r'(?<=VZ=).+?[+-]\d\d', line).group()
            line = f.readline()
            LT = re.search(r'(?<=LT=).+?[+-]\d\d', line).group()
            RG = re.search(r'(?<=RG=).+?[+-]\d\d', line).group()
            RR = re.search(r'(?<=RR=).+?[+-]\d\d', line).group()
            data.append({
                'time': t,
                'X': float(X), 'Y': float(Y), 'Z': float(Z),
                'VX': VX, 'VY': VY, 'VZ': VZ,
                # 'LT': LT, 'RG': RG, 'RR': RR
            })
            line = f.readline()
            # input()

    return data


if __name__ == '__main__':
    ##
    # Meta_data (masses)
    df = pd.read_csv('TNO_parameters.txt')
    df = df.rename(columns={' Mass(kg)': 'Mass(kg)', ' h(m*2/sec)': 'h(m*2/sec)'})
    masses = df.to_dict(orient='records')
    M_sun = 1.989e30  # kg
    M_px = 5.388e25  # kg

    ##
    # Location data (X, Y, Z from Sun)
    eph_data = {}
    data_location = 'data'
    for i, row in df.iterrows():
        obj_name = row['Object']
        if obj_name in ['Planet_X_(6)', 'Planet_X_(12)']:
            continue
        dat = read_ephemeris(data_location, obj_name)
        dat = pd.DataFrame(dat)
        eph_data[obj_name] = dat

    ##
    # For each point in time, Calculate Planet_X location from EQ31:
    # EQ31: m_p^2 * R_p = -M * sum(m_i * r_si)
    # Where:
    #   - m_p  is the mass of planet X
    #   - R_p  is the Range vector of planet X
    #   - M    is the mass of the Sun
    #   - m_i  is the mass of the satelite, i
    #   - r_si is the range vector of satelite i w.r.t the Sun

    dat_p_x = list(eph_data.values())[0]['time'].to_frame()
    dat_p_x['sum_mi_ri'] = dat_p_x.apply(lambda x: np.array([0, 0, 0]), axis=1)

    for i, row in df.iterrows():
        obj_name = row['Object']
        mass = row[r'Mass(kg)']
        if obj_name in ['Planet_X_(6)', 'Planet_X_(12)']:
            continue
        dat = eph_data[obj_name]
        dat['r_i'] = dat.apply(lambda x: np.array([x['X'], x['Y'], x['Z']]), axis=1)
        dat['v_i'] = dat.apply(lambda x: np.array([x['VX'], x['VY'], x['VZ']]), axis=1)

        dat_p_x['sum_mi_ri'] += mass * dat['r_i']

    dat_p_x['R'] = dat_p_x['sum_mi_ri'] * -1 * M_sun / M_px
    print(dat_p_x.head())
pass
