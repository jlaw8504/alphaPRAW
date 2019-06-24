import pyBach
import pandas as pd
import seaborn as sns

featureLists = []
testLosses = []
cvLosses = []
models = [];
scalers = [];
labelList = ['Show Only', 'Episode Data', 'Reddit Counts', 'Reddit Sentiments', 'All']
featureLists.append(['isBachelor', 'viewers(millions)'])
featureLists.append(['isBachelor', 'numOverall', 'numInSeason', 
                     'season', 'year', 'viewers(millions)'])
featureLists.append(['isBachelor', 'numOverall', 'numInSeason', 
                     'season', 'year', 'subNum', 'commentNum', 'viewers(millions)'])
featureLists.append(['isBachelor', 'numOverall', 'numInSeason', 
                     'season', 'year', 'subNum', 'commentNum', 
                     'meanTitleSentiment', 'stdTitleSentiment', 
                     'meanCommentSentiment', 'stdCommentSentiment','viewers(millions)'])

df = pd.read_pickle('dfTotal.pickle')
df = df.reset_index(drop=True)

featureLists.append(df.columns)

dfTrainAll, dfTestAll, dfCVAll = pyBach.splitDfs(df, rsInt=0)

for features in featureLists:
    dfTrain = dfTrainAll[features]
    dfCV = dfCVAll[features]
    trainLabels = dfTrain['viewers(millions)']
    scaler, model = pyBach.trainNN(dfTrain)
    #collect models
    models.append(model)
    #collect scalers, probably not necessary...
    scalers.append(scaler)
    #pop off dfCV labels
    cvLabels = dfCV.pop('viewers(millions)')
    #normalized dfCV using scaler
    trainNorm = scaler.transform(dfTrain)
    cvNorm = scaler.transform(dfCV)
    testLoss, _, _ = model.evaluate(trainNorm, trainLabels, verbose=0)
    cvLoss, _, _ = model.evaluate(cvNorm, cvLabels, verbose=0)
    testLosses.append(testLoss)
    cvLosses.append(cvLoss)


zippedList = list(zip(testLosses + cvLosses,
                      ['Train']*len(testLosses) + ['Validation']*len(cvLosses),
                      labelList + labelList))

# Create a dataframe from zipped list
dfObj = pd.DataFrame(zippedList, columns = ['loss' , 'type', 'label'],) 
ax = sns.barplot(x='label', y='loss', hue='type', data=dfObj)


import matplotlib.pyplot as plt
# so it looks like Episode Data alone has be cross-validation performance
bestModel = models[2]
bestScalar = scalers[2]
bestFeatures = featureLists[2]

#lets see the history of the bestModel
history = bestModel.history
# "borrowed" from https://www.tensorflow.org/tutorials/keras/basic_regression
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

## Use test data to predict performance on data model has not seen
dfTest = dfTestAll[bestFeatures]
testLabels = dfTest.pop('viewers(millions)')
testNorm = bestScalar.transform(dfTest)
testPredictions = bestModel.predict(testNorm).flatten()

plt.scatter(testLabels, testPredictions)
plt.xlabel('True Values [viewers(millions)]')
plt.ylabel('Predictions [viewers(milllions)]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])
plt.show()

error = testPredictions - testLabels
percError = error/testLabels*100
plt.hist(percError, bins = 25)
plt.xlabel("Prediction Percent Error [viewers(millions)]")
_ = plt.ylabel("Count")
plt.show()