# For going through pages
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
# for crawling
from bs4 import BeautifulSoup
import requests
import pandas as pd
import csv
import os
from sqlalchemy import create_engine
import schedule
import time


def job():
    # Getting list of company urls=======================
    # Add driver to PATH before starting
    driver = webdriver.Chrome()
    driver.get("https://www.thestar.com.my/business/marketwatch/stock-list/")
    delay = 5  # seconds

    # Initialize CSV file in writing mode
    csv_file = open('swc.csv', 'a', newline='')
    csv_writer = csv.writer(csv_file)

    # Loop alphabet list and open URL
    url_prefix = "https://www.thestar.com.my/business/marketwatch/stock-list/?alphabet="
    alphabetic = driver.find_elements_by_xpath('//div[@class="btn-group btn-group-sm"]//a[@class="btn btn-default"]')
    alphalist = []
    stock_url_list = []
    for alpha in alphabetic:
        alphalist.append(url_prefix + alpha.text)

    ncount = 0
    for url in alphalist:
        driver.get(url)
        try:
            myElem = WebDriverWait(driver, delay).until(EC.presence_of_element_located(
                (By.XPATH, '//div[@id="stocksect"]//div[@id="active"]//tbody//tr[@class="linedlist"]//td//a[@href]')))
            ncount += 1
            print(f"Page {ncount} is ready!")
        except TimeoutException:
            print("Loading page {ncount} took too much time!")
        #
        # #Retrieve URL for all stocks under the page with same alphabet
        stocks = driver.find_elements_by_xpath(
            '//div[@id="stocksect"]//div[@id="active"]//tbody//tr[@class="linedlist"]//td//a[@href]')
        for stock in stocks:
            stock_url_list.append(stock.get_attribute("href"))
            csv_writer.writerow([stock.get_attribute("href")])
            # print(stock.get_attribute("href"))

    csv_file.close()
    print(f'Total stock urls found:{len(stock_url_list)}.')
    driver.close()


    # Web data scraping ===================================================================================================
    print('Begin data scraping for all pages...')
    #stock_url_list = ['https://www.thestar.com.my/business/marketwatch/stocks/?qcounter=SAPNRG']
    header_list = ['stockCode', 'stockName', 'stockDate', 'stockTime', 'Open', 'High', 'Low', 'Last', 'Chg', 'Chg %',
               'Vol(\'00)', 'Buy/Vol(\'00)', 'Sell/Vol(\'00)']
    stock_df = pd.DataFrame(columns=header_list)

    count = 0
    for url in stock_url_list:
        data = requests.get(url)
        soup = BeautifulSoup(data.text, 'html.parser')

        value_list = []

        stock_code = soup.find('ul', class_='stock-code')
        stock_code_strip = stock_code.find_all('li', class_='f14')[1].text.strip().split(':')[1].strip()
        stock_name = soup.find_all('h1', class_='stock-profile f16')[0].text.strip()
        stock_timestamp = soup.find_all('p', {'class', 'timestamp'})[0].text.strip()
        stock_time = stock_timestamp.split('|')[1].strip()
        stock_date = stock_timestamp.split('|')[0].strip().split(':')[1].strip()

        value_list.append([stock_code_strip, stock_name, stock_date, stock_time])

        stock_price_table = soup.find('table', class_='market-trans bot-15')

        td_list = []

        for tr in stock_price_table.find_all('tr'):
            for td in tr.find_all('td'):
                td_list.append(td.text.strip())

        td_list = td_list[9:18]
        for each in td_list:
            value_list[0].append(each)

        stock_df_part = pd.DataFrame(
            value_list,
            columns=header_list
        )
        stock_df = stock_df.append(stock_df_part)
        count += 1

        print(f'Pages crawled: {count}')
        # end of stock_url_list

    print('Web crawling completed...')
    print(f'Total pages crawled: {count}')
    # print(stock_df)

    # For exporting CSV File
    # Check if file exist
    # IMPORTANT NOTE: CHANGE THE FILE PATH AND CSV PATH ACCORDINGLY
    file_path = 'C:/Users/seong/OneDrive - Configura Sverige AB/web_crawler/stock_df.csv'
    file_exist = os.path.isfile(file_path)

    if file_exist is True:
        print('csv_file already exist, appending Dataframe ...')
        stock_df.to_csv(file_path, index=False, mode='a', header=False)
    else:
        stock_df.to_csv(file_path, index=False)

    # For writing dataframe into sql table
    # Format of engine input
    # 'mysql+pymysql://<username>:<password>@<server>:<port>/<database>'
    engine = create_engine('mysql+pymysql://root@localhost:3306/stock')
    # df.to_sql(name=<table>,....)
    stock_df.to_sql(name='stockdata', con=engine, if_exists='append', index=False)

job()
schedule.every(5).minutes.do(job)

while True:
    schedule.run_pending()
    time.sleep(1)
