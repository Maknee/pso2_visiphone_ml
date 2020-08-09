from bs4 import BeautifulSoup
import requests
import urllib3
import datetime
import pandas as pd
import csv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from sklearn.preprocessing import StandardScaler
import sklearn


class Item:
    def __init__(self, name, ship, lowest_price, average_price, date):
        self.name = name
        self.ship = ship
        self.lowest_price = lowest_price
        self.average_price = average_price
        self.date = date

    def df(self):
        df = pd.DataFrame([self.__dict__])
        return df

    def to_csv(self):
        s = f'{self.name},{self.ship},{self.lowest_price},{self.average_price},{self.date}'
        return s


def ConvertItemsToTimeSeries(items_2s, input_size, output_size):
    scalar = sklearn.preprocessing.MinMaxScaler()

    input_series = []
    output_series = []
    for name, group in items_2s.groupby('name'):
        values = []
        for _, d in group.groupby(['ship', 'date'], sort=True):
            price = d['lowest_price'].to_numpy()
            if price.shape[0] != 1:
                break
            values.append(price)

        if len(values) == 0:
            continue

        values = np.array(values)
        values = scalar.fit_transform(values)

        for i in range(len(values) - input_size):  # - output_size):
            inputs = values[i:i+input_size]
            input_series.append(inputs)

    input_series = np.array(input_series)
    output_series = np.array(output_series)
    input_series = input_series.reshape(-1, 1, input_size)
    output_series = output_series.reshape(-1, 1, output_size)

    return input_series, output_series, scalar


def y_fmt(y, pos):
    decades = [1e9, 1e6, 1e3, 1e0, 1e-3, 1e-6, 1e-9]
    suffix = ["G", "M", "k", "", "m", "u", "n"]
    if y == 0:
        return str(0)
    for i, d in enumerate(decades):
        if np.abs(y) >= d:
            val = y/float(d)
            signf = len(str(val).split(".")[1])
            if signf == 0:
                return '{val:d} {suffix}'.format(val=int(val), suffix=suffix[i])
            else:
                if signf == 1:
                    if str(val).split(".")[1] == "0":
                        return '{val:d} {suffix}'.format(val=int(round(val)), suffix=suffix[i])
                tx = "{"+"val:.{signf}f".format(signf=signf) + "} {suffix}"
                return tx.format(val=val, suffix=suffix[i])
    return y


def plot_example(dataset, model, scalar, title, num_instances=3):
    data = dataset.take(num_instances)
    for x in data:
        x = x[:num_instances]
        y_pred = model.predict(x)
        x = tf.reshape(x[0], (-1,))

        input_size = x.shape[0]
        output_size = y_pred.shape[1]

        input_indices = range(input_size)
        output_indices = range(input_size, input_size + output_size)

        x = scalar.inverse_transform(x.numpy().reshape(-1, 1)).reshape(-1,)
        y_pred = scalar.inverse_transform(y_pred.reshape(-1, 1)).reshape(-1,)
        plt.plot(input_indices, x, label='Inputs', marker='.', zorder=-10)
        plt.scatter(output_indices, y_pred, marker='X', edgecolors='k', label='Predictions', c='#ff7f0e', s=64)
        plt.gca().yaxis.set_major_formatter(FuncFormatter(y_fmt))
        plt.gca().set_title(title)
        plt.gca().set_xlabel('Time')
        plt.gca().set_ylabel('Price')


def GetItemPrices(url, s=None):
    req = None
    while True:
        try:
            if s is None:
                req = requests.get(url)
            else:
                req = s.get(url)
            if req.status_code != requests.codes.ok:
                return None
            break
        except Exception as e:
            print(f'Got exception {e} with {url}')
            continue
    soup = BeautifulSoup(req.content, features="lxml")

    title = soup.select('title')[0].text
    name_with_ship = title[title.find('| ') + 2:]
    name = name_with_ship[:name_with_ship.find(' (Ship')]
    ship = int(name_with_ship[name_with_ship.find(' (Ship') + 7:name_with_ship.rfind(')')])

    bodies = soup.select('tbody')
    if len(bodies) == 0:
        return []

    body = bodies[0]
    items = []
    for tr in body.select('tr'):
        tds = tr.select('td')

        lowest = int(tds[0].text.replace(',', ''))
        average = int(tds[1].text.replace(',', ''))
        date = datetime.datetime.strptime(tds[2].text, '%H:%S %b %d')

        item = Item(name, ship, lowest, average, date)
        items.append(item)

    return items
