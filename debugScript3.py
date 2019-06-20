
import pyBach
df = pyBach.scrapeWikiEpiTable(10, show='bachelorette')
df = pyBach.appendRedditStats(df, appendSubs=True)