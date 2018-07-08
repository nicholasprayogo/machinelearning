import bs4 as bs
import pickle
import requests
import os
import datetime as dt
import pandas_datareader.data as web
from matplotlib import style
import matplotlib.dates as mdates
import calendar
import pandas as pd
import numpy as np
def get_tickers():
    page=requests.get('http://thestockmarketwatch.com/markets/pre-market/today.aspx')
    soup=bs.BeautifulSoup(page.text, 'html.parser')
    table= soup.find('table', id='tblMovers')
    stocks=table.find_all('td',{'class':'tdSymbol'})

    tickers=[item.find(class_='symbol').get_text() for item in stocks]
    print(tickers)

    links=[]
    for item in stocks:
        #find all with a tag, then get href property of a
        link=item.find('a')
        links.append('thestockmarketwatch.com'+link.get('href'))
    #return(tickers)
    premarket_vol=[]
    premarket_last=[]
    premarket_high=[]
    premarket_low=[]
    last_news=[]
    last_news_date=[]
    for ticker in tickers:
        link=requests.get('https://www.nasdaq.com/symbol/'+ ticker +'/premarket-chart')
        link=bs.BeautifulSoup(link.text, 'html.parser')
        premarket_vol.append(link.find(id='quotes_content_left_lblVolume').get_text())
        premarket_last.append(link.find(id='quotes_content_left_lblLastsale').get_text())
        premarket_high.append(link.find(id='quotes_content_left_lblHighprice').get_text())
        premarket_low.append(link.find(id='quotes_content_left_lblLowprice').get_text())

        link2=requests.get('https://www.nasdaq.com/symbol/'+ticker)
        link2=bs.BeautifulSoup(link2.text,'html.parser')
        news=link2.find(id='CompanyNewsCommentary')
        last_news.append(news.find('a').get('href'))
        last_news_date.append(news.find('small').get_text())
    df2=pd.DataFrame({'PreMarket Volume':premarket_vol, 'PreMarket Low':premarket_low, 'PreMarket High':premarket_high, 'PreMarket Last':premarket_last, 'Symbol':tickers})
    #can use dictionary but become alphabetical order
    pd.options.display.max_colwidth = 200
    pd.options.display.max_columns=200
    pd.options.display.expand_frame_repr=True

    # pd.set_option('expand_frame_repr', True)
    # pd.set_option('max_columns',200)
    #change max width so can display full text

    df=pd.DataFrame(np.column_stack([tickers, premarket_vol, premarket_last, premarket_low, premarket_high, last_news, last_news_date]), columns=['Symbol', 'PreMarket Vol', 'PreMarket Last', 'Premarket Low', 'PreMarket High', 'Latest News', 'News Date and Source'])
    print(df)

get_tickers()

def analyze():
    style.use('ggplot')
    today=dt.date.today()
    day=calendar.day_name[today.weekday()]

    #get weekday before
    if day=="Sunday":
        yesterday=today - dt.timedelta(days=2)
    elif day=="Monday":
        yesterday=today - dt.timedelta(days=3)
    else:
        yesterday=today - dt.timedelta(days=1)
    #use timedelta for difference
    start=yesterday
    end=dt.datetime.now()
    df=web.DataReader(tickers[0], 'morningstar', start, end)
    print(df)


