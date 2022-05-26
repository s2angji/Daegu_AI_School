# -* coding: utf-8 -*-
from selenium import webdriver
from bs4 import BeautifulSoup as bs
import time
import pandas as pd

# BeautifulSoup: 웹 페이지의 정보를 쉽게 스크랩할 수 있도록 기능을 제공하는 라이브러리입니다.

serarch_keyword = ["감자칩", "수박", "와인"]

for serarch_keywords in serarch_keyword : 

    # 크롤링 도구
    driver = webdriver.Chrome("./chromedriver")
    driver.implicitly_wait(3)
    time.sleep(5)

    # 오픈마켓 접속
    driver.get('http://www.auction.co.kr/')
    time.sleep(5)

    # 상품검색
    driver.find_element_by_xpath("//*[@id='txtKeyword']").send_keys(serarch_keywords)
    time.sleep(2)
    driver.find_element_by_xpath("//*[@id='core_header']/div/div[1]/form/div[1]/input[2]").click()
    time.sleep(2)

    # 상품리스트 가져오기
    html = driver.page_source
    soup = bs(html, 'html.parser')
    itemlist = soup.findAll("div", {"class": "section--itemcard"})
    time.sleep(10)

    # 가져온 상품리스트에서 필요한 상품명, 가격, 상품링크를 출력!!

    # 추가 데이터 -> title price link list 만들어서 타이틀, 가격, 링크 정보를 리스트에 저장 
    title_list = [] 
    price_list = [] 
    link_list  = [] 
    for item in itemlist:
        title = item.find("span", {"class": "text--title"}).text
        price = item.find("strong", {"class": "text--price_seller"}).text
        link = item.find(
            "span", {"class": "text--itemcard_title ellipsis"}).a['href']
        
        # list -> 각 리스트에 정보를 저장 append 를 이용 
        title_list.append(title)
        price_list.append(price)
        link_list.append(link)
        print("상품명 : " + title)
        print("가격 : " + price + "원")
        print("상품 링크 : " + link)
        print("------------------------")

    # 판다스를 이용하여 CSV 필요 저장를 저장 
    df =  pd.DataFrame()
    df['title'] = title_list
    df['price'] = price_list
    df['link'] = link_list

    df.to_csv(f'./{serarch_keywords}.csv', index=False)

    time.sleep(3)
    driver.close()
