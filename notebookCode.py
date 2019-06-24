#%%
#Using https://www.tensorflow.org/tutorials/keras/basic_regression as template NN-regression analysis
import pyBach #my module
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
#%%
dfList = []
bachSeasons = range(19,24)
etteSeasons = range(10,15)

for bachSeason in bachSeasons:
    dfList.append(pyBach.scrapeWikiEpiTable(bachSeason, keepTitle=True))
for etteSeason in etteSeasons:
    dfList.append(pyBach.scrapeWikiEpiTable(etteSeason, show='bachelorette', keepTitle=True))

df = pd.concat(dfList)
df.reset_index(drop=True, inplace=True)
# add one hot encoding to label each episode
df = pyBach.appendEpiTypes(df)
#%%
sns.pairplot(df[['isBachelor', 'numOverall', 'numInSeason', 'year', 'viewers(millions)']], diag_kind="kde")
plt.show()


#dfAll = pyBach.appendRedditStats(df)
#remove title column
#dfTotal.drop('title', axis=1, inplace=True)

dfTotal = pd.read_pickle('dfAll.pickle')
dfTotal.reset_index(drop=True, inplace=True)
#%%
#sns.pairplot(dfTotal[['numOverall', 'numInSeason', 'tellAll', 'finale',
                     #'season', 'year', 'subNum', 'commentNum', 'viewers(millions)']], diag_kind="kde")
#plt.show()
                     #%%
#sns.pairplot(dfTotal[['year', 'subNum', 'commentNum', 
                     #'meanTitleSentiment', 'stdTitleSentiment', 
                     #'meanCommentSentiment', 'stdCommentSentiment','viewers(millions)']], diag_kind="kde")
#plt.show()
#%%                     
#establish feature subsets
featureLists = []
featureLists.append(['isBachelor', 
                     'viewers(millions)'])
featureLists.append(['isBachelor', 'numOverall', 'numInSeason', 
                     'year', 'viewers(millions)'])
featureLists.append(['isBachelor', 'numOverall', 'numInSeason', 
                     'year', 'normal', 'tellAll', 'finale', 'afterFinale', 'viewers(millions)'])
featureLists.append(['isBachelor', 'numOverall', 'numInSeason',
                     'normal', 'tellAll', 'finale', 'afterFinale',
                     'year', 'subNum', 'commentNum', 'viewers(millions)'])
featureLists.append(['isBachelor', 'numOverall', 'numInSeason',
                     'normal', 'tellAll', 'finale', 'afterFinale',
                     'year', 'subNum', 'commentNum', 
                     'meanTitleSentiment', 'stdTitleSentiment', 
                     'meanCommentSentiment', 'stdCommentSentiment','viewers(millions)'])
featureLists.append(dfTotal.columns)
print(dfTotal.columns)
#%%
dfTrainAll, dfTestAll, dfCVAll = pyBach.splitDfs(dfTotal, rsInt=0)
#%%
#Instantiate lists to keep trained models and scalers for normalizing data
testLosses = []
cvLosses = []
models = [];
scalers = [];
#Train the models on subsets of features
for features in featureLists:
    dfTrain = dfTrainAll[features]
    dfCV = dfCVAll[features]
    trainLabels = dfTrain['viewers(millions)']
    #There is an early stop parameter in pyBach.trainNN (100 epochs w/o improvement), prevents overfitting
    scaler, model = pyBach.trainNN(dfTrain, epochNum=10000)
    #collect models
    models.append(model)
    #collect scalers, probably not necessary...
    scalers.append(scaler)
    #pop off dfCV labels
    cvLabels = dfCV.pop('viewers(millions)')
    #normalized dfCV using scaler
    trainNorm = scaler.transform(dfTrain)
    cvNorm = scaler.transform(dfCV)
    testLoss, _, _ = model.evaluate(trainNorm, trainLabels, verbose=0) #loss is Mean Squared Error
    cvLoss, _, _ = model.evaluate(cvNorm, cvLabels, verbose=0)
    testLosses.append(testLoss)
    cvLosses.append(cvLoss)
#%%
labelList = ['Show Only\n(underfitting)', 'Episode Data', 'Episode Data and Type', 'Reddit Counts', 'Reddit Sentiments', 'All']
zippedList = list(zip(testLosses + cvLosses,
                      ['Train']*len(testLosses) + ['Validation']*len(cvLosses),
                      labelList + labelList))
#%%
# Create a dataframe from zipped list
dfObj = pd.DataFrame(zippedList, columns = ['loss' , 'type', 'label'],) 
fig, ax = plt.subplots()
fig.set_figwidth(20)
ax = sns.barplot(x='label', y='loss', hue='type', data=dfObj, ax=ax)
ax.set_ylabel('Loss (Mean Squared Error)')
ax.set_xlabel('')
bestIdx = np.array(cvLosses).argmin()
ax.set_title('Best performing feature set on Validation set is: ' + labelList[bestIdx] + '; loss: '
             + str(format(cvLosses[bestIdx], '.2f')))
#%%
bestModel = models[bestIdx]
bestScaler = scalers[bestIdx]
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
#%%
## Use test data to predict performance on data model has not seen
bestFeatures = featureLists[bestIdx]
dfTest = dfTestAll[bestFeatures]
testLabels = dfTest.pop('viewers(millions)')
testNorm = bestScaler.transform(dfTest)
testPredictions = bestModel.predict(testNorm).flatten()
#%%
# "borrowed" from https://www.tensorflow.org/tutorials/keras/basic_regression
plt.scatter(testLabels, testPredictions)
plt.xlabel('True Values [viewers(millions)]')
plt.ylabel('Predictions [viewers(milllions)]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])
# Give R^2 value in title
SST = sum((testLabels-testLabels.mean())**2)
SSE = sum((testLabels - testPredictions)**2)
Rsq = (SST-SSE)/SST
plt.title('R-Square is: ' + str(format(Rsq, '.2f')))
plt.show()
#%%
error = testLabels - testPredictions
mae = abs(error).mean()
n, myBins, patches = plt.hist(error, label='Error')
plt.xlabel("Prediction Error")
_ = plt.ylabel("Count")
plt.title('Sample Size: ' + str(len(testLabels)) + '; Mean Error: ' + str(format(error.mean(), '.2f')))
testError = testLabels - testLabels.mean()
maeTest = abs(testError).mean()
print('Mean Absolute Error if mean of test set is used as prediction : ' + str(format(maeTest, '.2f')) + ' million viewers')
print('Mean Absolute Error of model: ' + str(format(mae, '.2f')) + ' million viewers')
if error.mean() > 0.01:
    print('Model is underestimating viewer numbers')
elif error.mean() < -0.01:
    print('Model is overestimating viewer numbers')
else:
    print('Model is not over/underestimating viewer numbers')
plt.show()
#%%