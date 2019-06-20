from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.feature_selection import SelectFromModel as SFM
import pandas as pd
import pyBach


tf.enable_eager_execution()

df = pd.read_pickle('dfTotal.pickle')
df = df.reset_index(drop=True)

dfTrainAll, dfTestAll, dfCVAll = pyBach.splitDfs(df, rsInt=0)
#Pop off the viwers(millions) label
trainLabels = dfTrainAll.pop('viewers(millions)')


model = keras.Sequential([
        layers.Dense(64, activation=tf.nn.relu,
                     input_shape=[len(dfTrainAll.keys())]),
                     layers.Dense(64, activation=tf.nn.relu),
                     layers.Dense(1)
                     ])
optimizer = tf.keras.optimizers.RMSprop(0.001)

model.compile(loss='mean_squared_error',
                optimizer=optimizer,
                metrics=['mean_absolute_error', 'mean_squared_error'])

wrappedModel = KerasRegressor(model)


#create StandardScalar instance
scaler = StandardScaler()
normNd = scaler.fit_transform(dfTrainAll)

