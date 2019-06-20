from __future__ import absolute_import, division, print_function

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

dataset = pd.read_pickle('dfTotal.pickle')
#reset the index
dataset = dataset.reset_index()
#keep only a few columns
colList = ['isBachelor', 'numInSeason', 'year', 'viewers(millions)']
dataset = dataset[colList]


train_dataset = dataset.sample(frac=0.6,random_state=0)
test_cv_dataset = dataset.drop(train_dataset.index)
test_dataset = test_cv_dataset.sample(frac=0.5, random_state=0)
cv_dataset = test_cv_dataset.drop(test_dataset.index)
sns.pairplot(train_dataset[colList], diag_kind="kde")

train_stats = train_dataset.describe()
train_stats.pop("viewers(millions)")
train_stats = train_stats.transpose()
train_stats

train_labels = train_dataset.pop('viewers(millions)')
test_labels = test_dataset.pop('viewers(millions)')

def norm(x):
  return (x - train_stats['mean']) / train_stats['std']
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

def build_model():
  model = keras.Sequential([
    layers.Dense(64, activation=tf.nn.relu, input_shape=[len(train_dataset.keys())]),
    layers.Dense(64, activation=tf.nn.relu),
    layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mean_squared_error',
                optimizer=optimizer,
                metrics=['mean_absolute_error', 'mean_squared_error'])
  return model

model = build_model()

# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

EPOCHS = 1000
# The patience parameter is the amount of epochs to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
history = model.fit(
  normed_train_data, train_labels,
  epochs=EPOCHS, validation_split = 0.2, verbose=0,
  callbacks=[early_stop, PrintDot()])

def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch
  
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error [$viewers(millions)^2$]')
  plt.plot(hist['epoch'], hist['mean_squared_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_squared_error'],
           label = 'Val Error')
  plt.legend()
  plt.show()

plot_history(history)

loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=1)

print("Testing set Mean Abs Error: {:5.2f} viewers(millions)".format(mae))

test_predictions = model.predict(normed_test_data).flatten()
plt.figure()
ax = plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [viewers(millions)]')
plt.ylabel('Predictions [viewers(millions)]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
ax = plt.plot([-100, 100], [-100, 100])
plt.figure()
error = test_predictions - test_labels
ax1 = plt.hist(error, bins = 25)
plt.xlabel("Prediction Error [viewers(millions)]")
plt.ylabel("Count")