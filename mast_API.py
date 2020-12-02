# From:
#   https://mast.stsci.edu/api/v0/MastApiTutorial.html

import sys
import os
import time
import re
import json

try:  # Python 3.x
    from urllib.parse import quote as urlencode
    from urllib.request import urlretrieve
except ImportError:  # Python 2.x
    from urllib import pathname2url as urlencode
    from urllib import urlretrieve

try:  # Python 3.x
    import http.client as httplib
except ImportError:  # Python 2.x
    import httplib

import astropy
from astropy.table import Table
import numpy as np

import matplotlib.pyplot as plt

import pprint
pp = pprint.PrettyPrinter(indent=4)


def mastQuery(request):
    """Perform a MAST query.

        Parameters
        ----------
        request (dictionary): The MAST request json object

        Returns head,content where head is the response HTTP headers
        and content is the returned data
    """

    server = 'mast.stsci.edu'

    # Grab Python Version
    version = ".".join(map(str, sys.version_info[:3]))

    # Create Http Header Variables
    headers = {"Content-type": "application/x-www-form-urlencoded",
               "Accept": "text/plain",
               "User-agent": "python-requests/" + version}

    # Encoding the request as a json string
    requestString = json.dumps(request)
    requestString = urlencode(requestString)

    # opening the https connection
    conn = httplib.HTTPSConnection(server)

    # Making the query
    conn.request("POST", "/api/v0/invoke", "request=" + requestString, headers)

    # Getting the response
    resp = conn.getresponse()
    head = resp.getheaders()
    content = resp.read().decode('utf-8')

    # Close the https connection
    conn.close()

    return head, content


def view_cone_query(ra, dec):
    cachebreaker = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    mastRequest = {
        'service': 'Mast.Caom.Cone',
        'params': {
            'ra': ra,
            'dec': dec,
            'radius': 0.2},
        'format': 'json',
        'pagesize': 2000,
        'page': 1,
        'removenullcolumns': True,
        'cachebreaker': cachebreaker}

    headers, mastDataString = mastQuery(mastRequest)

    mastData = json.loads(mastDataString)
    return mastData


def fits_view(filename):
    with astropy.io.fits.open(filename) as hdul:
        print(hdul.info())
        print(hdul[0].header['DATE'])
    pass

    image_data = astropy.io.fits.getdata(filename)
    plt.imshow(image_data, cmap='gray')
    plt.colorbar()
    plt.show()


def test():
    # step 1 - Name Resolver
    objectOfInterest = 'M101'

    resolverRequest = {
        'service': 'Mast.Name.Lookup',
        'params': {'input': objectOfInterest,
                   'format': 'json'}}

    headers, resolvedObjectString = mastQuery(resolverRequest)

    resolvedObject = json.loads(resolvedObjectString)

    # pp.pprint(resolvedObject)

    objRa = resolvedObject['resolvedCoordinate'][0]['ra']
    objDec = resolvedObject['resolvedCoordinate'][0]['decl']

    # step 2 - MAST Query (cone)
    #

    mastData = view_cone_query(objRa, objDec)

    # print(mastData.keys())
    print("Query status:", mastData['status'])

    # pp.pprint(mastData['fields'][:5])
    obs = 1400
    pp.pprint(mastData['data'][obs])

    # step 3 - Getting data products
    #
    obsid = mastData['data'][obs]['obsid']
    productRequest = {
        'service': 'Mast.Caom.Products',
        'params': {'obsid': obsid},
        'format': 'json',
        'pagesize': 100,
        'page': 1}

    headers, obsProductsString = mastQuery(productRequest)

    obsProducts = json.loads(obsProductsString)

    print("Number of data products:", len(obsProducts["data"]))
    print("Product information column names:")
    pp.pprint(obsProducts['fields'])


if __name__ == '__main__':
    # test()
    filename = r'images\SPITZER_S2_14799616_0009_8_E7468022_coa2d.fits'
    fits_view(filename)
