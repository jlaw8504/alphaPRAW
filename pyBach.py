# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 19:14:31 2019

@author: joshl

Functions to scrape Wikipedia and Reddit for Bachelor data
"""

def scrapeWikiEpiTable(seasonNum, show='bachelor', keepTitle=False,
                       keepDescription=False):
    """Scrape Wikipedia for episode table of given season

    Uses beautiful soup to scrape the episode table from a given season page of
    The Bachelor or The Bachelorette.

    Args:
        seasonNum : An integer variable specifying which season of The
        Bachelor or The Bachelorette to scrape.
    
        show : A string specifying which show to scrape. Set to either
        'bachelor' or 'bachelorette'

        keepTitle : A boolean varible specifying if title is included in
        dataframe

        keepDescription : A boolen variable specifying if description is
        included in dataframe

    Returns:
        A Pandas dataframe of the episode table with the following headers:

            ['numOverall', 'numInSeason',
               'year', 'month', 'day', 'title'
              'viewers(millions)', 'description', 'posix time', 'isBachelor',
              'season']

        Additionally, title and description can be added to dataframe using
        keepTitle=True and keepDescription=True
        
    Raises:
        ValueError : If no or multiple episode tables are found for a given
        season
        
        
    """
    import requests
    from bs4 import BeautifulSoup
    import numpy as np
    import pandas as pd
    import re
    import datetime as dt
    import time

    if show == 'bachelor':
        showStr = 'Bachelor'
    elif show == 'bachelorette':
        showStr = 'Bachelorette'
    else:
        raise ValueError('Show variable must be bachelor or bachelorette.')
    website_url = requests.get(
        'https://en.wikipedia.org/wiki/The_' + 
        showStr + '_(season_' + str(seasonNum) +')'
        ).text
    soup = BeautifulSoup(website_url, features='lxml')
    table_classes = {'class': 'wikiepisodetable'}
    wikitables = soup.findAll("table", table_classes)
    if len(wikitables) == 1:
        table = wikitables[0]
    elif len(wikitables) == 0:
        raise ValueError('I did not find a wikiepisode table!')
    else:
        raise ValueError('I found multiple wikiepisode tables!')
    
    #Parse the column headers and row headers
    colHeads = [h.getText().lower() for h in table.findAll('th', {'scope': 'col'})]
    rowHeads = [h.getText().lower() for h in table.findAll('th', {'scope': 'row'})]
    # pull out on the headers with scope='col', those are the 
    cellList = [c.getText().lower() for c in table.findAll('td')]
    
    #check that row header number * col header number equals cell number
    if len(colHeads) * len(rowHeads) != len(cellList):
        raise ValueError('Missing values in episode table!')

    #change cell list into numpy array and reshape into pandas dataframe
    cellArray = np.array(cellList).reshape(len(rowHeads), len(colHeads))
    #Have to append rowHead to cellArray since wikiepidsodetables include the
    #row header as part of the row itself
    rowArray = np.array(rowHeads).reshape(len(rowHeads),1)
    finalArray = np.hstack((rowArray, cellArray))
    #add the description entry to column headers
    colHeads.append('description')
    df = pd.DataFrame(finalArray, columns=colHeads)
    
    #Clean the viewership number and convert the original air date to Posix time
    posixTimes = []
    #extract year of original airdate
    years = []
    #extract month
    months = []
    #extract days
    days = []
    for idx in df.index:
        # remove the reference number in brackets
        df['u.s. viewers(millions)'][idx] = re.sub(
                r" ?\[[^)]+\]", "", df['u.s. viewers(millions)'][idx])
        # find the date in yyyy-mm-dd format
        m = re.search(r'\d\d\d\d-\d\d-\d\d', df['original air date'][idx])
        pTime = int(
                time.mktime(
                dt.datetime.strptime(
                m.group(), '%Y-%m-%d').timetuple()))
        posixTimes.append(pTime)
        years.append(dt.datetime.fromtimestamp(pTime).year)
        months.append(dt.datetime.fromtimestamp(pTime).month)
        days.append(dt.datetime.fromtimestamp(pTime).day)
    # join time date to df
    df = df.join(pd.DataFrame({'posix time': posixTimes}))
    df = df.join(pd.DataFrame({'year': years}))
    df = df.join(pd.DataFrame({'month': months}))
    df = df.join(pd.DataFrame({'day': days}))
    # join show label to df
    df = df.join(pd.DataFrame({'isBachelor' : [float(show=='bachelor')]*df.shape[0]}))
    # join season to df
    df = df.join(pd.DataFrame({'season' : [seasonNum]*df.shape[0]}))
    # set dtype of viewers to float
    df['viewers(millions)'] = df['u.s. viewers(millions)'].astype('float')
    # set dtype of no. inseason to int
    df['numInSeason'] = df['no. inseason'].astype('float')
    df['numOverall'] = df['no.overall'].astype('float')
    # specify columns to include
    columnList = ['numOverall', 'numInSeason',
               'year', 'month', 'day',
              'posix time', 'isBachelor',
              'season', 'viewers(millions)']
    if keepTitle:
        columnList.append('title')
    if keepDescription:
        columnList.append('description')
    return df[columnList]
# keywords to filter in title are 'tell all', 'final rose' and 'finale'
    
def filterFinalRoseAndTellAll(df):
    """
    Filter out final rose and tell all episodes
    
    Args :
        df : A dataframe generated by scrapeWikiEpiTable.
    
    Returns :
        A dataframe without final rose and tell episode rows
    """
    df = df[~df.title.str.contains('final rose')]
    df = df[~df.title.str.contains('tell all')]
    return df

def appendEpiTypes(df):
    """
    Use one-hot encoding to label episodes as 'tellAll' or 'finale'
    
    Args :
        df : A dataframe generated by scrapeWikiEpiTable
    
    Returns :
        A dataframe with final rose and tell all episode columns appended
    """
    tellAll = df.title.str.contains('tell all')
    finale = df.title.str.contains('finale')
    afterFinale = df.title.str.contains('after the final rose')
    premiere = df.numInSeason == 1
    normal = ~(premiere | finale | tellAll | afterFinale)
    df['premiere'] = premiere.astype('float')
    df['tellAll'] = tellAll.astype('float')
    df['finale'] = finale.astype('float')
    df['afterFinale'] = afterFinale.astype('float')
    df['normal'] = normal.astype('float')
    return df

def appendStartTime(df, deltaDay = 6):
    """
    Parse the posix time of each episode's original airdate append a 'start
    time' column to the dataframe
    
    Args :
        df : A dataframe generated by scrapewikiEpiTable.
        
        deltaDay : An int variable specifying how many days to subtract from 
        original airdate time.
    
    Returns :
        A dataframe with the 'start time' column appended
    """
    import datetime as dt
    
    #parse the posix time column
    times = df.loc[:, 'posix time']
    # loop through times and append to timeList
    timeList = []
    for time in times:
        timeUTC = dt.datetime.utcfromtimestamp(time)
        startTime = timeUTC - dt.timedelta(days=deltaDay)
        timeList.append(int(startTime.timestamp()))
    #append timeList to dataframe as column
    df['startEpoch'] = timeList
    return df

def appendEndTime(df, deltaDay = 0):
    """
    Parse the posix time of each episode's original airdate append a 'start
    time' column to the dataframe
    
    Args :
        df : A dataframe generated by scrapewikiEpiTable.
        
        deltaDay : An int variable specifying how many days to subtract from 
        original airdate time.
    
    Returns :
        A dataframe with the 'start time' column appended
    """
    import datetime as dt
    
    #parse the posix time column
    times = df.loc[:, 'posix time']
    # loop through times and append to timeList
    timeList = []
    for time in times:
        timeUTC = dt.datetime.utcfromtimestamp(time)
        startTime = timeUTC - dt.timedelta(days=deltaDay)
        timeList.append(int(startTime.timestamp()))
    #append timeList to dataframe as column
    df['endEpoch'] = timeList
    return df

def getRedditSubsList(startEpoch, endEpoch, prawUserAgent = 'alpha'):
    """
    Returns a list of Reddit submissions from the bachelor subreddit
    
    Args :
        startEpoch : An int variable of posix time. Inidcates start time of
        reddit submissions.
        
        endEpoch : An int variable of posix time. Indicates end time of reddit
        submissions.
        
        prawUserAgent : The name of the reddit app. This should be specified in
        a praw.ini text file see: 
        https://praw.readthedocs.io/en/latest/getting_started/configuration/prawini.html
    
    Returns :
        subList : A list of reddit submission objects from PRAW
    """
    import praw
    from psaw import PushshiftAPI


    r = praw.Reddit(prawUserAgent)
    api = PushshiftAPI(r)
    subList = list(api.search_submissions(
        before = endEpoch,
        after = startEpoch,
        subreddit = 'thebachelor'))
    return subList

def appendRedditStats(df, startDayDelta = 6, endDayDelta = 0, prawUserAgent = 'alpha', appendSubs = False):
    """
    Appends statistics of submissions titles and comments to provided dataframe
    
    Args :
        df : A dataframe generated by scrapewikiEpiTable.
        
        startEpoch : An int variable of posix time. Inidcates start time of
        reddit submissions.
        
        endEpoch : An int variable of posix time. Indicates end time of reddit
        submissions.
        
        prawUserAgent : The name of the reddit app. This should be specified in
        a praw.ini text file see: 
        https://praw.readthedocs.io/en/latest/getting_started/configuration/prawini.html
        
        appendSubs : Boolean variable specifying if the subsList parsed by
        PRAW should be appended to dataframe
    
    Returns :
        df : Original dataframe with columns of reddit statistics appended
    """
    from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
    import numpy as np
    
    #add start and end epochs to df
    df = appendEndTime(df, deltaDay=endDayDelta)
    df = appendStartTime(df, deltaDay=startDayDelta)
    sia = SIA()

    # instantiate lists
    subCnts = []
    titleSentMeans = []
    titleSentStds = []
    commentCnts = []
    commentSentMeans = []
    commentSentStds = []
    if appendSubs:
        subListList = []
    
    #set up a progress reporter
    rowNum= df.shape[0]
    cnt = 1
    for _, row in df.iterrows():
        print('On row: ' + str(cnt) + ' of ' + str(rowNum))
        subList = getRedditSubsList(
                        # use int as, for some unknown reason, these are now
                        # float values
                        int(row['startEpoch']), int(row['endEpoch']),
                        prawUserAgent = prawUserAgent)
        subCnts.append(len(subList))
        if appendSubs:
            subListList.append(subList)
        if len(subList) > 0:
            titleSentList = []
            bodySentList = []
            subCnt = 1
            for sub in subList:
                print('\tParsing ' + str(subCnt) + ' of ' + str(len(subList))
                + ' submissions')
                title = sub.title
                pol_score = sia.polarity_scores(title)
                titleSentList.append(pol_score['compound'])
                sub.comments.replace_more(limit=None)
                subCnt += 1
                for comment in sub.comments.list():
                    body = comment.body
                    pol_score = sia.polarity_scores(body)
                    bodySentList.append(pol_score['compound'])
            # process comment body list
            commentCnts.append(len(bodySentList))
            if len(bodySentList) > 0:
                bodyArray = np.array(bodySentList)
                commentSentMeans.append(bodyArray.mean())
                commentSentStds.append(bodyArray.std())
            else:
                commentSentMeans.append(-2)
                commentSentStds.append(0)
            # process sub title lists
            titleSentArray = np.array(titleSentList)
            titleSentMeans.append(titleSentArray.mean())
            titleSentStds.append(titleSentArray.std())
        else:
            titleSentMeans.append(-2)
            titleSentStds.append(0)
            commentCnts.append(0)
            commentSentMeans.append(-2)
            commentSentStds.append(0)
            
        cnt += 1
        # convert the sentiment lists to numpy and grab means and stds
#        bodySentArray = np.array(bodySentList)
#        titleSentArray = np.array(titleSentList)
    df['subNum'] = subCnts
    df['commentNum'] = commentCnts
    df['meanTitleSentiment'] = titleSentMeans
    df['stdTitleSentiment'] = titleSentStds
    df['meanCommentSentiment'] = commentSentMeans
    df['stdCommentSentiment'] = commentSentStds
    if appendSubs:
        df['subList'] = subListList
    return df

def splitDfs(df, rsInt=0):
    """
    Splits dataframe into train, test, and cross validation sets
    
    Args :
        df : A dataframe generated by scrapeWikipediaTable
        
        rsInt : An integer specifying the random state of sampling
        a dataframe
    
    Returns :
        dfTrain : dataframe for training, 0.6 of original dataframe
        
        dfTest : dataframe for testing, 0.2 of original dataframe
        
        dfCV : dataframe for cross validation, 0.2 of orignal dataframe
    """
    dfTrain = df.sample(frac=0.6, random_state=rsInt)
    dfNotTrain = df.drop(dfTrain.index)
    dfCV = dfNotTrain.sample(frac=0.5, random_state=rsInt)
    dfTest = dfNotTrain.drop(dfCV.index)
    
    return dfTrain, dfTest, dfCV

def trainNN(dfTrain, epochNum=1000):
    """
    Normalized the dfTrain dataframe with sklearn's StandardScaler class. Then
    trains a nerual network on the features listed in dfTrain dataframe.
    
    Args :
        dfTrain : A dataframe generated by scrapeWikiEpiTable for training
        
        epochNum : An integer specifying number of epochs
        
    Returns :
        scaler : The StandardScaler instance fit with training instance
        model : The neural network model. Loss is mean squared error. Metrics
        in the model are: mean_absolute_error and mean_squared_error
        
    """
    from sklearn.preprocessing import StandardScaler
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    
    #Pop off the viwers(millions) label
    trainLabels = dfTrain.pop('viewers(millions)')
    #create StandardScalar instance
    scaler = StandardScaler()
    normNd = scaler.fit_transform(dfTrain)
    # build model
    model = keras.Sequential([
    layers.Dense(32, activation=tf.nn.sigmoid,
                 input_shape=[len(dfTrain.keys())]),
    layers.Dense(32, activation=tf.nn.sigmoid),
    layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='mean_squared_error',
                optimizer=optimizer,
                metrics=['mean_absolute_error', 'mean_squared_error'])
    
    # Display training progress by printing a single dot for each completed epoch
    # shamelessly stolen from: https://www.tensorflow.org/tutorials/keras/basic_regression
    class PrintDot(keras.callbacks.Callback):
      def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print('')
        print('.', end='')
    
    # The patience parameter is the amount of epochs to check for improvement
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=100)
    model.fit(
    normNd, trainLabels,
    epochs=epochNum, validation_split = 0.2, verbose=0,
    callbacks=[early_stop, PrintDot()])
    
    return scaler, model
