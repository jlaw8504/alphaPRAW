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
    import sys
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
        print('Cell number does not match header number')
        sys.exit(1)
    
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
    
def filterData(df):
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

def appendRedditStats(df, startDayDelta = 6, endDayDelta = 0, prawUserAgent = 'alpha'):
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
            bodyArray = np.array(bodySentList)
            commentSentMeans.append(bodyArray.mean())
            commentSentStds.append(bodyArray.std())
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
    return df
