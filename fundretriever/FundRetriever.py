#!/usr/bin/env python

import requests
import pytz
import pandas as pd

from bs4 import BeautifulSoup
from datetime import datetime
import pandas_datareader.data as web


SITE = "http://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
#sd_0 = datetime(1970, 1, 1)
#ed_0 = datetime(1980, 1, 1)

#sd_1 = datetime(1980, 1, 1)
#ed_1 = datetime(1990, 1, 1)

#sd_2 = datetime(1990, 1, 1)
#ed_2 = datetime(2000, 1, 1)

#sd_3 = datetime(2000, 1, 1)
#ed_3 = datetime(2010, 1, 1)

sd = datetime(1900, 1, 1)
ed = datetime.now()


def scrape_list(site):
    hdr = {'User-Agent': 'Mozilla/5.0'}
    req = requests.get(site, headers=hdr)
    soup = BeautifulSoup(req.content, "html.parser")

    table = soup.find('table', {'class': 'wikitable sortable'})
    sector_tickers = dict()
    for row in table.findAll('tr'):
        col = row.findAll('td')
        if len(col) > 0:
            sector = str(col[3].string.strip()).lower().replace(' ', '_')
            ticker = str(col[0].string.strip())
            if sector not in sector_tickers:
                sector_tickers[sector] = list()
            sector_tickers[sector].append(ticker)
    return sector_tickers


def download_ohlc(sector_tickers):
    sector_ohlc = {}
    for sector, tickers in sector_tickers.items():
        print('Downloading data from Yahoo for %s sector' % sector)
        data = web.DataReader(tickers, 'yahoo', sd, ed)
        #data1 = web.DataReader(tickers, 'google', sd_1, ed_1)
        # data2 = web.DataReader(tickers, 'google', sd_2, ed_2)
        # data3 = web.DataReader(tickers, 'google', sd_3, ed_3)
        # data4 = web.DataReader(tickers, 'google', sd_4, ed_4)
        # data = data0.add(data1, axis=1)
        # data = data.add(data2, axis=1)
        # data = data.add(data3, axis=1)
        # data = data.add(data4, axis=1)

        data.fillna(method='ffill', inplace=True)
        data.fillna(method='bfill', inplace=True)
        data.dropna(axis=2, inplace=True)
        for item in ['Open', 'High', 'Low']:
            data[item] = data[item] * data['Adj Close'] / data['Close']
        data.rename(items={'Open': 'open', 'High': 'high', 'Low': 'low',
                           'Adj Close': 'close', 'Volume': 'volume'},
                    inplace=True)
        data.drop(['Close'], inplace=True)
        sector_ohlc[sector] = data
    print('Finished downloading data')
    return sector_ohlc


def store_HDF5(sector_ohlc, path):
    with pd.get_store(path) as store:
        for sector, ohlc in sector_ohlc.items():
            store[sector] = ohlc


def get_snp500():
    sector_tickers = scrape_list(SITE)
    sector_ohlc = download_ohlc(sector_tickers)
    store_HDF5(sector_ohlc, 'snp500.h5')


if __name__ == '__main__':
    get_snp500()
