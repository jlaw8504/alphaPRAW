import pyBach
import pandas as pd

dfList = []
bachSeasons = range(19,24)
etteSeasons = range(10,15)

for bachSeason in bachSeasons:
    dfList.append(pyBach.scrapeWikiEpiTable(bachSeason))
for etteSeason in etteSeasons:
    dfList.append(pyBach.scrapeWikiEpiTable(etteSeason, show='bachelorette'))

df = pd.concat(dfList)
df.reset_index()

dfTotal = pyBach.appendRedditStats(df)