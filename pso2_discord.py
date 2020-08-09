import os
import discord
import random
import requests
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import tensorflow as tf

from pso2_ml import *

TOKEN = 'SET_YOUR_DISCORD_TOKEN_HERE'

client = discord.Client()


def get_local_embed_image(path):
    file = discord.File(path, filename=path)
    embed = discord.Embed()
    embed.set_image(url='attachment://' + path)
    return file, embed


gru_10_10_model = tf.keras.models.load_model('models/gru_model_10_10')
gru_10_1_model = tf.keras.models.load_model('models/gru_model_10_1')


@client.event
async def on_message(message):
    if message.author == client.user:
        return

    request = message.content
    channel = message.channel
    author = str(message.author)

    if request.startswith('!pso2_predict_price'):
        try:
            values = request.split(' ')
            if len(values) < 2:
                await channel.send('Format: !pso2_predict_price https://pso2market.com/item/{ship}/{item_id}`')
                return
            url = ' '.join(values[1:])

            print(f'{author} is requesting `{url}`')
            items = GetItemPrices(url)
            items_df = pd.concat([i.df() for i in items])
            name = items[0].name

            async def ProduceResults(items_df, input_size, output_size, model, name):
                input_series, output_series, scalar = ConvertItemsToTimeSeries(items_df, input_size, output_size)
                if input_series.shape[0] < input_size:
                    await channel.send(f'Not enough data points for `{name}`')
                    return
                dataset = tf.data.Dataset.from_tensor_slices((input_series[-1:]))
                plot_example(dataset, model, scalar, name, 1)
                filename = f'image_{input_size, output_size}.png'
                plt.savefig(filename)
                (f, e) = get_local_embed_image(filename)
                await channel.send(file=f)
                plt.clf()

            await ProduceResults(items_df, 10, 1, gru_10_1_model, name)
            await ProduceResults(items_df, 10, 10, gru_10_10_model, name)

        except Exception as e:
            await channel.send('Got an error!: ' + str(e))

client.run(TOKEN)
