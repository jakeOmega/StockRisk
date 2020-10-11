# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 14:09:46 2020

@author: jakef
"""

import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt

seq_length = 90
batch_size = 8
bins = 1000
feature_number = 2

def split_input_target(chunk):
    N = len(chunk) - 1
    input_chunk = chunk[:-1]
    target_chunk = chunk[1:]
    gaussian = tfp.distributions.Normal(0, bins/1000)
    input_chunk, target_chunk = tf.reshape(input_chunk, (N, feature_number, 1)), tf.reshape(target_chunk, (N, feature_number, 1))
    input_chunk, target_chunk = tf.math.subtract(input_chunk, tf.range(bins, dtype='int64')), tf.math.subtract(target_chunk, tf.range(bins, dtype='int64'))
    input_chunk, target_chunk = gaussian.prob(tf.cast(input_chunk, tf.float32)), gaussian.prob(tf.cast(target_chunk, tf.float32))
    return input_chunk, target_chunk

def generate_sequence(model, start_sequence, num_generate, parallel_generate, temperature = 1.0):
    input_eval = tf.one_hot(start_sequence[:seq_length], bins, axis=1)
    input_eval = tf.stack([input_eval]*parallel_generate, axis=0)
    input_eval = tf.transpose(input_eval, [0, 1, 3, 2])
    values_generated = []
    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        predictions = predictions / temperature
        predicted_id = [tf.random.categorical(predictions[i][j], num_samples=1)[-1, 0].numpy() for i in range(parallel_generate) for j in range(feature_number)]
        input_eval = tf.concat([input_eval[:,1:,:,:], tf.expand_dims(tf.expand_dims(tf.one_hot(predicted_id, bins, axis=1), 0), 0)], axis = 1)
        values_generated.append([predicted_id])
    return (np.array(values_generated).reshape(num_generate, feature_number, parallel_generate)).transpose()

def gen_emp_func(data):
    sorted_data = np.sort(data)
    xp = np.linspace(0, 1, len(sorted_data))
    return lambda x: np.interp(x, xp, sorted_data)

def gen_inv_emp_func(data):
    sorted_data = np.sort(data)
    xp = np.linspace(0, 1, len(sorted_data))
    return lambda x: np.interp(x, sorted_data, xp)

def growth_to_values(initial_value, growth_data):
    return np.append(np.array([initial_value]), initial_value * np.cumprod(growth_data + 1))

historical_data = pd.read_csv( "E:\\Libraries\\Downloads\\^GSPC.csv", parse_dates=["Date"])
historical_data = historical_data.set_index("Date")
historical_data["Growth"] = (historical_data["Adj Close"] - historical_data["Adj Close"].shift(1))/np.max([historical_data["Adj Close"], historical_data["Adj Close"].shift(1)], axis=0)
historical_data["Vol_Growth"] = (historical_data.Volume - historical_data.Volume.shift(1))/np.max([historical_data.Volume, historical_data.Volume.shift(1)], axis=0)
historical_data = historical_data[(historical_data.Growth.notnull()) & (historical_data.Vol_Growth.notnull())]
pct_to_val_growth = gen_emp_func(historical_data.Growth)
val_to_pct_growth = gen_inv_emp_func(historical_data.Growth)
pct_to_val_volume = gen_emp_func(historical_data.Vol_Growth)
val_to_pct_volume = gen_inv_emp_func(historical_data.Vol_Growth)
historical_data["Growth_Percentile"] = val_to_pct_growth(historical_data["Growth"])
historical_data["Volume_Percentile"] = val_to_pct_volume(historical_data["Vol_Growth"])
historical_data["Growth_Bin"] = np.digitize(historical_data.Growth_Percentile, np.linspace(0, 1, bins)) #Make this geomspace 0.001, 0.01, 0.1, ..., 0.9, 0.99, 0.999 ?
historical_data["Volume_Bin"] = np.digitize(historical_data.Volume_Percentile, np.linspace(0, 1, bins))

raw_dataset = tf.data.Dataset.from_tensor_slices(np.array(historical_data[["Growth_Bin", "Volume_Bin"]]))
sequences = raw_dataset.batch(seq_length+1, drop_remainder=True)
dataset = sequences.map(split_input_target)
dataset = dataset.batch(batch_size, drop_remainder=True)

model = keras.Sequential()
model.add(keras.layers.Reshape((seq_length, bins * feature_number)))
model.add(keras.layers.LSTM(2000, return_sequences=True))
model.add(keras.layers.Dense(bins * feature_number))
model.add(keras.layers.Reshape((seq_length, feature_number, bins)))
model.build(input_shape=(batch_size, seq_length, feature_number, bins))

model.compile(optimizer=keras.optimizers.RMSprop(), loss=keras.losses.CategoricalCrossentropy(from_logits=True))
history = model.fit(dataset, epochs=500).history["loss"]
plt.plot(history)
plt.yscale("log")
plt.show()


parallel_runs = 1
serial_runs = 10000
seq_growth = []
seq_volume = []
input_data = np.array(historical_data[["Growth_Bin", "Volume_Bin"]])
for run_num in range(serial_runs):
    print(run_num)
    current_seq = generate_sequence(model, input_data, 365, parallel_runs)/bins
    current_seq_growth = [pct_to_val_growth(current_seq[i][0]) for i in range(parallel_runs)]
    current_seq_growth = [growth_to_values(historical_data["Adj Close"][-1], current_seq_growth[i]) for i in range(parallel_runs)]
    seq_growth += current_seq_growth
    current_seq_volume = [pct_to_val_volume(current_seq[i][1]) for i in range(parallel_runs)]
    current_seq_volume = [growth_to_values(historical_data["Volume"][-1], current_seq_volume[i]) for i in range(parallel_runs)]
    seq_volume += current_seq_volume
[plt.plot(seq_growth[i]) for i in range(parallel_runs * serial_runs)]
plt.show()
plt.hist([x[-1] for x in seq_growth], bins=30)
plt.show()
seq_by_date = np.transpose(np.array(seq_growth))
min_sim, p5, mean, p95, max_sim = list(map(np.min, seq_by_date)), list(map(lambda x: np.percentile(x, 5), seq_by_date)), list(map(np.mean, seq_by_date)), list(map(lambda x: np.percentile(x, 95), seq_by_date)), list(map(np.max, seq_by_date))
plt.plot(min_sim)
plt.plot(p5)
plt.plot(mean)
plt.plot(p95)
plt.plot(max_sim)
plt.show()

[plt.plot(seq_volume[i]) for i in range(parallel_runs * serial_runs)]
plt.yscale("log")
plt.show()
plt.hist([x[-1] for x in seq_volume], bins=30)
plt.show()
seq_by_date = np.transpose(np.array(seq_volume))
min_sim, p5, mean, p95, max_sim = list(map(np.min, seq_by_date)), list(map(lambda x: np.percentile(x, 5), seq_by_date)), list(map(np.mean, seq_by_date)), list(map(lambda x: np.percentile(x, 95), seq_by_date)), list(map(np.max, seq_by_date))
plt.plot(min_sim)
plt.plot(p5)
plt.plot(mean)
plt.plot(p95)
plt.plot(max_sim)
plt.yscale("log")
plt.show()