# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 11:14:46 2019

@author: lawrimor
"""

import requests
from bs4 import BeautifulSoup
import sys
import numpy as np
import pandas as pd
import re
import datetime as dt
import time

website_url = requests.get(
        'https://en.wikipedia.org/wiki/The_Bachelor_(season_18)'
        ).text
soup = BeautifulSoup(website_url, features='lxml')
table_classes = {'class': 'wikiepisodetable'}
wikitables = soup.findAll("table", table_classes)
if len(wikitables) == 1:
    table = wikitables[0]
else:
    print('I found multiple tables!')
    sys.exit(1)

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
for idx in df.index:
    df['u.s. viewers(millions)'][idx] = re.sub(
            r" ?\[[^)]+\]", "", df['u.s. viewers(millions)'][idx])
    m = re.search(r'\d\d\d\d-\d\d-\d\d', df['original air date'][idx])
    posixTimes.append(
            int(
            time.mktime(
            dt.datetime.strptime(
            m.group(), '%Y-%m-%d').timetuple())))
# join posixTimes to df
df = df.join(pd.DataFrame({'posix time': posixTimes}))
# set dtype of viewers to float
df['u.s. viewers(millions)'] = df['u.s. viewers(millions)'].astype('float')