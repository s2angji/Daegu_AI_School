from selenium import webdriver
import time
from selenium.webdriver.common.keys import Keys
import pandas as pd
import os

from urllib.request import (urlopen, urlparse, urlretrieve)


# 구글 이미지 URL
chrome_path = "./chromedriver"
# window
# chrome_path = "./chromedriver.exe"
base_url = "https://www.google.co.kr/imghp"

# 구글 검색 옵션
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument("lang=ko_KR")# 한국어
chrome_options.add_argument("window-size=1920x1080")

# driver = webdriver.Chrome(chrome_path, chrome_options=chrome_options)
# driver.get(base_url)
# driver.implicitly_wait(3) # element 로드될 때까지 지정한 시간만큼 대기할 수 있도록 하는 옵션
# driver.get_screenshot_as_file("google_screen.png")
# driver.close()

def selenium_scroll_option() :
    SCROLL_PAUSE_SEC = 1

    # 스크롤 높이 가져옴
    last_height = driver.execute_script(
        "return document.body.scrollHeight")

    while True :
        # 끝까지 스크롤 다운
        driver.execute_script(
            "window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(SCROLL_PAUSE_SEC)
        # 스크롤 다운 후 스크롤 높이 다시 가져옴
        new_height = driver.execute_script(
            "return document.body.scrollHeight")
        if new_height == last_height :
            break
        last_height = new_height

a = "상어"
image_name = "shark"
driver = webdriver.Chrome(chrome_path)
driver.get("http://www.google.co.kr/imghp?hl=ko")
browser = driver.find_element_by_name('q')
browser.send_keys(a)
browser.send_keys(Keys.RETURN)

# 오후 수업 진행 ...
# //*[@id="islmp"]/div/div/div/div[1]/div[2]/div[2]/input
selenium_scroll_option()# 스크롤 하여 이미지 확보
driver.find_element_by_xpath(
    '//*[@id="islmp"]/div/div/div/div[1]/div[2]/div[2]/input').click()
selenium_scroll_option()

# 이미지 저장 src 요소를 리스트업 해서 이미지 url 저장
image = driver.find_elements_by_css_selector(".rg_i.Q4LuWd")
# 클래스 네임에서 공백은 . 을 찍어줌

print(image)
image_url = []
for i in image:
    if i.get_attribute("src") != None :
        image_url.append(i.get_attribute("src"))
    else :
        image_url.append(i.get_attribute("data-src"))

# 전체 이미지 개수
print(f"전체 다운로드한 이미지 개수 : {len(image_url)}")
image_url = pd.DataFrame(image_url)[0].unique()

# 해당하는 파일에 이미지 다운로드
os.makedirs("./shark", exist_ok=True)
shark = "./shark/"
if image_name == 'shark' :
    for t, url in enumerate(image_url, 0) :
        urlretrieve(url, shark + image_name + "_" + str(t) + ".png")

    driver.close()

print("완료")

