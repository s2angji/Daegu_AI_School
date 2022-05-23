# pip install selenium
# pip install beautifulsoup4
from selenium import webdriver
from bs4 import BeautifulSoup as bs
import time
import pandas as pd
# BeautifulSoup : 웹페이지의 정보르 쉽게 스크랩할 수 있도록 기능을 제공
# 하는 라이브러리

# 키워드 input
serarch_keywords = ["감자칩", "그래픽카드", "램"]

for serarch_keyword in serarch_keywords :
    # 크롤링 도구
    driver = webdriver.Chrome("./chromedriver.exe")
    driver.implicitly_wait(3)
    time.sleep(5)

    # 오픈마켓 접속
    driver.get("http://www.auction.co.kr")
    time.sleep(5)

    # 상품 검색 키워드
    # //*[@id="txtKeyword"]
    driver.find_element_by_xpath("//*[@id='txtKeyword']").send_keys(serarch_keyword)
    time.sleep(2)
    # //*[@id="core_header"]/div/div[1]/form/div[1]/input[2]
    # 상품 키워드 검색 버튼 클릭
    driver.find_element_by_xpath(
        "//*[@id='core_header']/div/div[1]/form/div[1]/input[2]").click()

    # 상품 리스트 정보 가져오기
    html = driver.page_source
    soup = bs(html, 'html.parser')
    itemlist = soup.findAll('div', {"class": "section--itemcard"})
    time.sleep(5)

    # 가져온 상품리스트에서 필요한 상품명, 가격, 상품링크를 출력 !!
    csv = [['상품명', '가격', '상품링크']]
    for item in itemlist :
        title = item.find("span", {"class": "text--title"}).text
        price = item.find("strong", {"class": "text--price_seller"}).text
        link = item.find("span", {"class": "text--itemcard_title ellipsis"}).a['href']

        csv.append([title, price, link])

    time.sleep(3)
    pd.DataFrame(csv).to_csv(f"./{serarch_keyword}.csv", index=False)

    driver.close()