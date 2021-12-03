import urllib3
import ssl
from selenium import webdriver
import os
import time
from utils import HttpRequest
urllib3.disable_warnings()



def get_realPhoto(end_page=None):
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        # Legacy Python that doesn't verify HTTPS certificates by default
        pass
    else:  # Handle target environment that doesn't support HTTPS verification
        ssl._create_default_https_context = _create_unverified_https_context

    n = 1
    result = []
    keyword = ['interracial','double-penetration', 'glamour', 'cum-in-mouth'
               'blowjob','gangbang', 'big-cock', 'facial', 'cowgirl']
    for k in keyword:
        while True:
            try:
                driver.get("https://www.pornpics.com/{}".format(k))
                driver.implicitly_wait(10)
                break
            except:
                time.sleep(3)

        # 스크롤 높이 가져옴
        time.sleep(5)
        last_height = driver.execute_script("return document.body.scrollHeight")
        a=1
        while True:
            if a==20:
                break
            a+=1

            # 끝까지 스크롤 다운
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

            time.sleep(3)
            # 3초 대기


            # 스크롤 다운 후 스크롤 높이 다시 가져옴
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height

        # 이미지 태그 모두 가져옴
        imgs = driver.find_elements_by_tag_name('fake_img')
        count = 0

        # 링크 저장
        for i in imgs:
            src = i.get_attribute('src')
            result.append(src)

    current = os.getcwd()

    # src.txt에 result(링크) 쓰기
    f = open(current + "/src.txt", "w")
    for i in result:
        f.write(i + '\n')
    f.close()

    os.chdir("C:\\Users\\owner\\PycharmProjects\\pythonProject\\real_img")

    # 링크에 접속해서 화면 캡쳐
    try:
        with open(current + "/src.txt", 'r') as f:
            while True:
                line = f.readline()
                line = line.rstrip()
                if not "https://static.pornpics.com" in line:
                    start = line.rfind(".")
                    filetype = line[start:]
                    driver.get(line)
                    driver.save_screenshot("{}{}".format(n, filetype))
                    n += 1
    except:
        pass


if __name__ == "__main__":
    http = HttpRequest()
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    packed_extension_path = "C:\\bihmplhobchoageeokmgbdihknkjbknd.crx"
    options.add_extension(packed_extension_path)
    options.add_argument("user-agent=DN")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    driver = webdriver.Chrome('C:\\Users\\owner\\Downloads\\chromedriver_win32\\chromedriver.exe', options=options)

    get_realPhoto()

