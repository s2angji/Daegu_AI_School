import cv2
import numpy as np
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

import pandas as pd
import time
import os
from urllib.request import (urlopen)

from math import (log10, trunc)
from base64 import b64decode
from io import BytesIO
from PIL import Image


class Crawling:
    __driver = None
    __browser = None

    __chrome_path = './chromedriver'
    __base_url = 'https://www.google.co.kr/imghp'

    # 구글 검색 옵션
    __chrome_options = webdriver.ChromeOptions()
    __chrome_options.add_argument('lang=ko_KR')  # 한국어
    __chrome_options.add_argument('window-size=1920x1080')

    def __enter__(self):
        self.__driver = webdriver.Chrome(self.__chrome_path)
        print('Get driver ~!!')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.__driver is None:
            return
        self.__driver.close()
        self.__driver = None
        self.__browser = None
        print('Driver close ~!!')

    __SCROLL_PAUSE_SEC = 1

    def __selenium_scroll(self):
        # 스크롤 높이 가져옴
        last_height = self.__driver.execute_script('return document.body.scrollHeight')

        while True:
            # 끝까지 스크롤 다운
            self.__driver.execute_script('window.scrollTo(0, document.body.scrollHeight);')
            time.sleep(self.__SCROLL_PAUSE_SEC)
            # 스크롤 다운 후 스크롤 높이 다시 가져옴
            new_height = self.__driver.execute_script('return document.body.scrollHeight')
            if new_height == last_height:
                break
            last_height = new_height

    def do_crawling(self, search_keyword, dir_name):
        self.__driver.get('http://www.google.co.kr/imghp?hl=ko')
        self.__browser = self.__driver.find_element(By.NAME, 'q')
        self.__browser.send_keys(search_keyword)
        self.__browser.send_keys(Keys.RETURN)

        # 스크롤 하여 이미지 확보
        self.__selenium_scroll()
        self.__driver.find_element(By.XPATH, '//*[@id="islmp"]/div/div/div/div[1]/div[2]/div[2]/input').click()

        # 이미지 저장 src 요소를 리스트업 해서 이미지 url 저장
        image_url = []
        for i in self.__driver.find_elements(By.CSS_SELECTOR, ".rg_i.Q4LuWd"):
            if i.get_attribute("src") is not None:
                image_url.append(i.get_attribute("src"))
            else:
                image_url.append(i.get_attribute("data-src"))
        image_url = pd.DataFrame(image_url)[0].unique()

        # 전체 이미지 개수
        print(f"전체 다운로드한 이미지 개수 : {len(image_url)}")

        # 해당하는 경로에 이미지 다운로드
        os.makedirs(f"./{dir_name}", exist_ok=True)
        for i, url in enumerate(image_url):
            image = None
            if "base64" in url:
                bytes_image = BytesIO(b64decode(url.split(',')[1]))
                pil_image = Image.open(bytes_image)
                image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_BGR2RGB)
            else:
                image = np.asarray(bytearray(urlopen(url).read()), dtype='uint8')
                image = cv2.imdecode(image, cv2.IMREAD_COLOR)

            # 이미지 비율 그대로 가지면서 리사이즈
            height, width = image.shape[:2]
            if width > height:
                height = int(255 * (height / width))
                width = 255
            else:
                width = int(255 * (width / height))
                height = 255
            image = cv2.resize(image, (width, height))
            aff = np.float32([[1, 0, (255 - width) // 2], [0, 1, (255 - height) // 2]])
            image = cv2.warpAffine(image, aff, (255, 255))
            # cv2.imshow('test', image)
            # cv2.waitKey(0)

            # 해당하는 경로에 이미지 저장
            num_format = str(f'%0{trunc(log10(len(image_url))) + 1}d')
            file_name = f"./{dir_name}/" + dir_name + "_" + (num_format % i) + ".png"
            Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).save(file_name)
