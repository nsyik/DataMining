# For going through pages
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
# for crawling
import csv
import schedule
import time
import json


def job():
    # Getting list of company urls=======================
    # Add driver to PATH before starting
    driver = webdriver.Chrome()
    driver.maximize_window()
    driver.get("https://www.thestar.com.my/business/marketwatch/stock-list/")
    delay = 5  # seconds

    # Initialize CSV file in writing mode
    csv_file = open('news.csv', 'a', newline='')
    csv_writer = csv.writer(csv_file)

    # Loop alphabet list and open URL
    url_prefix = "https://www.thestar.com.my/business/marketwatch/stock-list/?alphabet="
    alphabetic = driver.find_elements_by_xpath('//div[@class="btn-group btn-group-sm"]//a[@class="btn btn-default"]')
    alphalist = []
    stock_url_list = []
    for alpha in alphabetic:
        alphalist.append(url_prefix + alpha.text)
    #alphalist = ["https://www.thestar.com.my/business/marketwatch/stock-list/?alphabet=A"]

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
            #print(stock.get_attribute("href"))

    csv_file.close()
    print(f'Total stock urls found:{len(stock_url_list)}.')
    # From every url, load the news url
    for url in stock_url_list:
        driver.get(url)
        try:
            myElem = WebDriverWait(driver, delay).until(EC.presence_of_element_located(
                (By.XPATH,  '//div[@class="story-set-group"]//div[@class="row"]//h2//a[@href]')))
            print("Loading news URL")
        except TimeoutException:
            print("Unable to load news URL, time out error")
            continue
        stock_name = driver.find_element_by_xpath('//*[@id="slcontent_0_ileft_0_compnametxt"]').text.strip().split(':')[0].strip()
        stock_code = driver.find_element_by_xpath('//*[@id="slcontent_0_ileft_0_info"]/ul/div[1]/li[2]').text.strip().split(':')[1].strip()
        print(stock_name)
        print(stock_code)
        news_list =[]
        news = driver.find_elements_by_xpath(
            '//div[@class="story-set-group"]//div[@class="row"]//h2//a[@href]')
        for new in news:
            news_list.append(new.get_attribute("href"))
            #print(new.get_attribute("href"))

        for link in news_list:
            driver.get(link)
            #print ("news")
            try:
                myElem = WebDriverWait(driver, delay).until(EC.presence_of_element_located(
                    (By.CSS_SELECTOR, '#slcontent_0_sleft_0_noticeDiv > a')))
                     #'//*[@id="slcontent_0_sleft_0_noticeDiv"]/a')))
                print("Press continue reading button")
                print(" ")
                javaScript = "document.querySelector('#slcontent_0_sleft_0_noticeDiv > a').click();"
                driver.execute_script(javaScript)
            except TimeoutException:
                print("Can't load continue reading button")
                pass

            title = driver.find_element_by_xpath('//*[@id="wrapper"]/div[1]/main/div[2]/div/div/div/div[1]/div[2]/h1').text
            date = driver.find_element_by_xpath('//*[@id="side-note-article"]/li[2]/p').text
            timestamp = driver.find_element_by_xpath('//*[@id="side-note-article"]/li[2]/time').text
            article = driver.find_element_by_xpath('//*[@id="story-Article"]').text
            # print(title)
            # print(date)
            # print(timestamp)
            # print(article)
            data = {"stock_name":stock_name,
                "stock_code": stock_code,
                "title": title,
                "date": date,
                "time": timestamp,
                "Content": article}


            with open ('news.json', 'a') as json_file:
                json.dump(data, json_file, indent=4)


    driver.close()

job()
schedule.every(5).minutes.do(job)

while True:
    schedule.run_pending()
    time.sleep(1)