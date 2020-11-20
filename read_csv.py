import os
import pandas as pd
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
            # VX = re.search(r'(?<=VX=).+?[+-]\d\d', line).group()
            # VY = re.search(r'(?<=VY=).+?[+-]\d\d', line).group()
            # VZ = re.search(r'(?<=VZ=).+?[+-]\d\d', line).group()
            line = f.readline()
            # LT = re.search(r'(?<=LT=).+?[+-]\d\d', line).group()
            # RG = re.search(r'(?<=RG=).+?[+-]\d\d', line).group()
            # RR = re.search(r'(?<=RR=).+?[+-]\d\d', line).group()
            data.append({
                'time': t,
                'X': float(X), 'Y': float(Y), 'Z': float(Z),
                # 'VX': VX, 'VY': VY, 'VZ': VZ,
                # 'LT': LT, 'RG': RG, 'RR': RR
            })
            line = f.readline()
            # input()

    return data


if __name__ == '__main__':
    df = pd.read_csv('TNO_parameters.txt')

    eph_data = {}
    data_location = 'data'
    for i, row in df.iterrows():
        obj_name = row['Object']
        if obj_name in ['Planet_X_(6)', 'Planet_X_(12)']:
            continue
        dat = read_ephemeris(data_location, obj_name)
        dat = pd.DataFrame(dat)
        eph_data[obj_name] = dat
        print(dat.head())
        exit()


pass
