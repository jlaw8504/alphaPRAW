#! python3
import praw
from psaw import PushshiftAPI
import datetime as dt


r = praw.Reddit('alpha')
api = PushshiftAPI(r)

#define a list of bachelor winners
winner = 'Jef'
start_epoch = int(dt.datetime(2017, 5,1).timestamp())
end_epoch = int(dt.datetime(2017, 7, 1).timestamp())
subList = list(api.search_submissions(
        before = end_epoch,
        after = start_epoch,
        subreddit = 'thebachelor',
        limit=10000))
