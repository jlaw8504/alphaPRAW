# So I added one hot encoding and really don't feel like bricking up
# my laptop for 3 hours, so I'll just concat the reddit data to a newly
# made dataframe scraped from wikiepisode table data

import pyBach
import pandas as pd

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

#load in the old reddit dataframe
dfReddit = pd.read_pickle('dfTotal.pickle')
dfReddit = dfReddit[[
        'subNum',
        'commentNum',
        'meanTitleSentiment',
        'stdTitleSentiment',
        'meanCommentSentiment',
        'stdCommentSentiment'
        ]]
#reset the index for both reddit and df
df.reset_index(inplace=True, drop=True)
dfReddit.reset_index(inplace=True, drop=True)
#use pd.concat to join them
dfAll = pd.concat([df, dfReddit], axis=1)
# drop the title column
dfAll.drop('title', axis=1, inplace=True)
dfAll.to_pickle('dfAll.pickle')
dfAll.to_csv('dfAll.csv')