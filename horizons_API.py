# From:
#   https://github.com/eleanorlutz/asteroids_atlas_of_space/blob/main/3_fetch_data.ipynb

import numpy as np
import pandas as pd
import time
import urllib.request
import os.path

import logging


def get_link(objid, start_time, stop_time, stepsize, center):
    link = "https://ssd.jpl.nasa.gov/horizons_batch.cgi?batch=1"
    link += "&COMMAND='" + str(objid) + "'"
    link += "&MAKE_EPHEM='YES'"
    link += "&TABLE_TYPE='VECTORS'"
    link += "&START_TIME='" + start_time + "'"
    link += "&STOP_TIME='" + stop_time + "'"
    link += "&STEP_SIZE='" + stepsize + "'"
    link += "&OUT_UNITS='AU-D'"
    link += "&REF_PLANE='ECLIPTIC'"
    link += "&REF_SYSTEM='J2000'"
    link += "&VEC_LABELS='YES'"
    link += "&VEC_DELTA_T='NO'"
    link += "&OBJ_DATA='YES'"
    link += "&VEC_TABLE='3'"  # Type 3 for state vector (x, v, r, ...)
    link += "&VECT_CORR='NONE'"
    link += "&CSV_FORMAT='YES'"
    link += "&CENTER='" + center + "'"
    return(link)


def get_data(link, savename, objid):
    values = {'name': 'Ian Sweeney',
              'location': 'Portland, OR, USA',
              'language': 'Python'}
    headers = {'User-Agent': "PlanetX Search-Sim bot"}
    data = urllib.parse.urlencode(values).encode('UTF-8')

    if os.path.isfile(savename):
        return

    try:
        output = urllib.request.urlopen(link)
        output = [x.decode('UTF-8') for x in output]
        output = [x.strip() for x in output]

        if '$$SOE' in output:
            header = output[output.index('$$SOE') - 2].split(',')
            header = [x.lstrip() for x in header]

            content = output[output.index('$$SOE') + 1: output.index('$$EOE') - 1]
            content = [x.split(',') for x in content]

            df = pd.DataFrame(content, columns=header)
            df.to_csv(savename, index=False)
        else:
            print(objid, 'request successful but output not expected format')
            print(link)
            raise TypeError('Bad http formatt')
    except:
        print(objid, 'request unsuccessful')
        print(link)
        raise


def query_horizons(readname, savename_head, stepsize='1d', center='@sun'):
    if not os.path.isfile(savename_head[:-1] + ".zip"):
        print("---\nNow analyzing", readname)
        df = pd.read_csv(readname, low_memory=False)

        # Check for duplicated names before running script
        dupl = df.duplicated('horizons')
        if sum(dupl) > 1:
            print(sum(dupl), 'names in the series are duplicated, of', len(df), 'total')
            print(df[dupl == True])

        # Check to see if any names are NaN values
        print(df['horizons'].astype(str).replace(' ', '').isnull().sum(),
              'null values in horizons query list')

        for index, row in df.iterrows():
            if (index % 500 == 0) and (index != 0):
                print(index, 'items analyzed!')
            start_time = str(row['begin_time']).replace(' ', '')
            stop_time = str(row['end_time']).replace(' ', '')
            objid = str(row['horizons']).replace(' ', '')
            savename = savename_head + objid + '.csv'

            if not os.path.isfile(savename):
                link = get_link(objid, start_time, stop_time, stepsize, center)
                get_data(link, savename, objid)

                # sleep to be polite to HORIZONS servers
                sleeptime = min(20, max(1, np.random.normal(loc=10, scale=5)))
                time.sleep(sleeptime)

                if index == 0:
                    # Save parameters of 1st item to txt file for later checking
                    txtname = savename_head + 'PARAMETERS.txt'
                    with open(txtname, "w") as f:
                        f.write(link)
    else:
        raise ValueError("Please unzip the data files that already exist")
    print('ALL ITEMS ANALYZED!')


def planets_example():
    '''
    Get orbital coordinates from HORIZONS server
    '''
    center = '@sun'
    stepsize = '1d'

    # readname = './data/moons.csv'
    # savename_head = './data/moons/'
    # query_horizons(readname, savename_head, stepsize=stepsize, center=center)

    readname = './data/planets.csv'
    savename_head = './data/horizons/planets/'
    query_horizons(readname, savename_head, stepsize=stepsize, center=center)

    # readname = './data/large_asteroids.csv'
    # savename_head = './data/large_asteroids/'
    # query_horizons(readname, savename_head, stepsize=stepsize, center=center)

    # readname = './data/large_comets.csv'
    # savename_head = './data/large_comets/'
    # query_horizons(readname, savename_head, stepsize=stepsize, center=center)

    # readname = './data/small_asteroids.csv'
    # savename_head = './data/small_asteroids/'
    # query_horizons(readname, savename_head, stepsize=stepsize, center=center)

    # readname = './data/any_outer_asteroids.csv'
    # savename_head = './data/any_outer_asteroids/'
    # query_horizons(readname, savename_head, stepsize=stepsize, center=center)

    # readname = './data/any_inner_asteroids.csv'
    # savename_head = './data/any_inner_asteroids/'
    # query_horizons(readname, savename_head, stepsize=stepsize, center=center)


def download_obj_data(savedir, objid, start_time, stop_time,
                      stepsize='1d', center='@sun', overwrite=False):
    savename = os.path.join(savedir, objid + '.csv')
    link = get_link(objid, start_time, stop_time, stepsize, center)
    if os.path.isfile(savename) and not overwrite:
        print('data file {} already exists'.format(savename))
    else:
        get_data(link, savename, objid)


def get_obj_data(savedir, objid, overwrite=False):
    start_time = '2020-11-01-00-00-00'
    stop_time = '2020-12-01-00-00-00'
    savename = os.path.join(savedir, str(objid) + '.csv')
    if not os.path.isfile(savename) or overwrite:
        download_obj_data(savedir, objid, start_time, stop_time,
                          overwrite=overwrite)

    return pd.read_csv(savename)


if __name__ == '__main__':
    # planets_example()
    savedir = r'./data/horizons/etc/'
    objid = '2012_VP113'
    df = get_obj_data(savedir, objid, overwrite=True)
    print(df.head())
