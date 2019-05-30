#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import json



# Using the top 10 most complete stock data we had gathered in milestone 3
# choose ten companies with relative complete data.
# 7181       274
# 1066       275
# 06516F     275
# 4707       276
# 06516G     276
# 9873       276
# 3662WC     277
# 06515B     278
# 3662       279
# 06516J     280

# Read the data as dataframe
df = pd.read_csv("E:/Google Drive/Data Science/7005 Data Mining/Milestone 3/stock_df.csv")

# First stock data will be used to demonstrate the process for interpretation
df1 = df[df['stockCode']=='7181']
df1['stockDate2'] = df1['stockDate']+' '+df1['stockTime']
df1.stockDate2 = pd.to_datetime(df1.stockDate2)
df1.stockDate = pd.to_datetime(df1.stockDate)
df1 = df1[['stockCode','stockName','stockDate','stockDate2','Open','High','Low','Last','Chg %']]
df1.head(5)


# In[5]:


# Plot the time-series graph of stock price of #7181

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('pylab', 'inline')
pylab.rcParams['figure.figsize'] = (15, 9) 
# df1['Last'].plot(grid=True)
name = df1['stockName'].values[0]
plt.plot('stockDate2', 'Last', data=df1)
plt.title(f'{name}')
plt.xlabel('Date')
plt.ylabel('Price')
plt.grid(True)


# In[124]:


# Draw the trend line to observe the pattern/trend of the stock.

import numpy as np
from matplotlib.pyplot import plot, grid, show
from pandas_datareader import data, wb
import pandas as pd
from datetime import datetime, timedelta

def trendGen(xDat, window=1.0/3.0, needPlot=True):

    x = np.array(xDat)
    xLen = len(x)
    window = window * xLen
    window = int(window)

    # find index of min and max
    absMax = np.where(x == max(x))[0][0]
    absMin = np.where(x == min(x))[0][0]

    if absMax + window > xLen:
        xmax = max(x[0:(absMax - window)])
    else:
        xmax = max(x[(absMax + window):])


    if absMin - window < 0:
        xmin = min(x[(absMin + window):])
    else:
        xmin = min(x[0:(absMin - window)])

    xmax = np.where(x == xmax)[0][0]  # index of the 2nd max
    xmin = np.where(x == xmin)[0][0]  # index of the 2nd min

    # Create the trend lines
    # rise over run
    slopeMax = (x[absMax] - x[xmax]) / (absMax - xmax)
    slopeMin = (x[absMin] - x[xmin]) / (absMin - xmin)
    amax     = x[absMax] - (slopeMax * absMax)
    amin     = x[absMin] - (slopeMin * absMin)
    bmax     = x[absMax] + (slopeMax * (xLen - absMax))
    bmin     = x[absMin] + (slopeMax * (xLen - absMin))
    maxline  = np.linspace(amax, bmax, xLen)
    minline  = np.linspace(amin, bmin, xLen)

    trends = np.transpose(np.array((x, maxline, minline)))

    trends = pd.DataFrame(trends, index=np.arange(0, len(x)),
                          columns=['Data', 'Resistance', 'Support'])

    if needPlot:
        plot(trends)
        grid()
        show()

    return trends, slopeMax, slopeMin


def findTops(x, window=1.0/3, charts=True):

    x = np.array(x)
    xLen = len(x)

    if window < 1:
        window = int(window * xLen)

    sigs = np.zeros(xLen, dtype=float)

    i = window

    while i != xLen:
        if x[i] > max(x[i-window:i]): sigs[i] = 1
        elif x[i] < min(x[i-window:i]): sigs[i] = -1
        i += 1

    xmin = np.where(sigs == -1.0)[0]
    xmax = np.where(sigs == 1.0)[0]

    ymin = x[xmin]
    ymax = x[xmax]

    if charts is True:
        plot(x)
        plot(xmin, ymin, 'ro')
        plot(xmax, ymax, 'go')
        show()

    return sigs

def trendSegments(x, segs=2, charts=True):
    y = np.array(x)

    # Implement trendlines
    segs = int(segs)
    maxima = np.ones(segs)
    minima = np.ones(segs)
    segsize = int(len(y)/segs)
    for i in range(1, segs+1):
        ind2 = i*segsize
        ind1 = ind2 - segsize
        maxima[i-1] = max(y[ind1:ind2])
        minima[i-1] = min(y[ind1:ind2])

    # Find the indexes of the maximums
    x_maxima = np.ones(segs)
    x_minima = np.ones(segs)
    for i in range(0, segs):
        x_maxima[i] = np.where(y == maxima[i])[0][0]
        x_minima[i] = np.where(y == minima[i])[0][0]

    if charts:
        plot(y)
        grid()

    for i in range(0, segs-1):
        maxslope = (maxima[i+1] - maxima[i]) / (x_maxima[i+1] - x_maxima[i])
        a_max = maxima[i] - (maxslope * x_maxima[i])
        b_max = maxima[i] + (maxslope * (len(y) - x_maxima[i]))
        upperline = np.linspace(a_max, b_max, len(y))

        minslope = (minima[i+1] - minima[i]) / (x_minima[i+1] - x_minima[i])
        a_min = minima[i] - (minslope * x_minima[i])
        b_min = minima[i] + (minslope * (len(y) - x_minima[i]))
        lowerline = np.linspace(a_min, b_min, len(y))

        if charts:
            plot(upperline, 'g')
            plot(lowerline, 'm')

    if charts:
        show()

    # OUTPUT
    return x_maxima, maxima, x_minima, minima


last = df1.groupby('stockDate').last()
dat = pd.Series(last.Last.values, index=last.stockDate2)

trendSegments(dat)
trendGen(dat)
findTops(dat)


# In[1]:


# The trend lines are hardly of any use since we have too little data to work on. Stock data requires at least a year worth of
# data for the trend lines to be worth looking at. 
# News data will be used here to see if it has any affection on stock price.


import json
import pandas as pd


# Load the json file
with open('E:/Google Drive/Data Science/7005 Data Mining/Milestone 3/news.json') as json_file:
    data = json.loads(json_file.read())

# Put the news content into csv file to avoid reading from json file over and over again since its semi-structured format
# make it harder for us to retrieve data
header_list = ['name', 'code', 'date', 'time', 'content']
news_df = pd.DataFrame(columns=header_list)
news_cnt = []

for i in data:
    name = i['stock_name']
    code = i['stock_code']
    date = i['publish_date']
    time = i['publish_time']
    content = i['Content']
    news_cnt.append([name, code, date, time, content])
#     if code =='7181':
#         print([name, code, date, time, content])

news_df_part = pd.DataFrame(news_cnt, columns=header_list)
news_df = news_df.append(news_df_part)

news_df.to_csv('C:/Users/Yik/Desktop/news.csv', encoding='utf-8')


# In[2]:


import pandas as pd
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords


#Extract the text from content and do analysis
news_csv = pd.read_csv("C:/Users/Yik/Desktop/news.csv", index_col=0)
news1 = news_csv[news_csv['code']=='7181']

#Get a proper date format to be used later
news1['stockdate'] = pd.to_datetime(news1.date)

#Strip the text and remove all the unnecessary symbols
news1['content'] = news1.content.str.replace("[^\w\s]"," ").str.lower().str.rstrip("\n\r")

#Tokenization
news1['content'] = news1.apply(lambda row: nltk.word_tokenize(row['content']), axis=1)

#Remove stop words
stop = stopwords.words('english')
news1['content'] = news1['content'].apply(lambda x: [item for item in x if item not in stop])

news1 = news1[['name','code','stockdate','content']]
print(news1)


# In[3]:


#Start sentiment analysis by using SentimentIntensityAnalyzer of Vader
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA

sia = SIA()
results = []

#Convert list to strings and apply polarity scores to know if the news is good or bad
news1['content2'] = news1['content'].str.join(" ")
news1['Negative'] = news1['content2'].apply(lambda x: sia.polarity_scores(x)['neg'])
news1['Neutral'] = news1['content2'].apply(lambda x: sia.polarity_scores(x)['neu'])
news1['Positive'] = news1['content2'].apply(lambda x: sia.polarity_scores(x)['pos'])
news1['Compound'] = news1['content2'].apply(lambda x: sia.polarity_scores(x)['compound'])
# print(news1['content2'])

news1.head()

# Our dataframe consists of four columns from the sentiment 
# scoring: Neu, Neg, Pos and compound. 
# The first three represent the sentiment score percentage of each category in our headline,
# and the compound single number that scores the sentiment.
# `compound` ranges from -1 (Extremely Negative) to 1 (Extremely Positive).
# 


# In[6]:


# We will consider posts with a compound value greater than 0.2 as positive and less than -0.2 as negative.
# Greater will be blue, neutral is black, negative is red

news1['label'] = 'k'
news1.loc[news1['Compound'] > 0.2, 'label'] = 'b'
news1.loc[news1['Compound'] < 0.2, 'label'] = 'r'
news1.head()


# In[31]:


# Try to plot the graph and the stock news together, blue line indicates good news, black line indicates neutral,
# red line indicated bad news

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('pylab', 'inline')
pylab.rcParams['figure.figsize'] = (15, 9) 
# df1['Last'].plot(grid=True)
name = df1['stockName'].values[0]
plt.plot('stockDate2', 'Last', data=df1)
plt.title(f'{name}')
plt.xlabel('Date')
plt.ylabel('Price')
plt.grid(True)

# Prepare the location of the line of the news on x-axis and the color of each of the line
xposition = news1.stockdate
xcolor = news1.label
y=0

# Put each of the line on the stock price graph
for xp in xposition:
    plt.axvline(x = xp , color = xcolor[xcolor.index[y]], linestyle='--')
    print(xcolor[xcolor.index[y]])
    y += 1


# In[ ]:


# As we can see, every time good news happen, the stock price goes up. This is a very simple interpretation of the stock price
# with stock news. Repeat the process with all other available stock with the same code.

