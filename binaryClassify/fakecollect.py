import urllib3
import ssl
from selenium import webdriver
import os
import time
from utils import HttpRequest



def get_fakePhoto(end_page=None):
    # verification issue
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        # Legacy Python that doesn't verify HTTPS certificates by default
        pass
    else:  # Handle target environment that doesn't support HTTPS verification
        ssl._create_default_https_context = _create_unverified_https_context

    n = 1
    result = []
    for i in range(3, 251):
        while True:
            try:
                driver.get("https://cfapfakes.com/page/{}".format(i))
                break
            except:
                time.sleep(3)

        imgs = driver.find_elements_by_tag_name('fake_img')
        print(imgs)
        count = 0
        for img in imgs:
            count += 1
            if count == 1:
                continue
            src = img.get_attribute('src')
            result.append(src)
            if count == 13:
                break

    current = os.getcwd()

    f = open(current + "/src.txt", "w")
    for i in result:
        f.write(i + '\n')
    f.close()

    os.chdir("C:\\Users\\owner\\PycharmProjects\\pythonProject\\fake_img")
    with open(current + "/src.txt", 'r') as f:
        while True:
            line = f.readline()
            if line == '':
                break
            line = line.rstrip()
            start = line.rfind(".")
            size = line.rfind('-')
            filetype = line[start:]
            filetype = filetype.rstrip()

            line = line[:size] + line[start:]
            line = line.rstrip()
            driver.get(line)
            driver.save_screenshot("{}.png".format(n))

            n += 1


if __name__ == "__main__":
    urllib3.disable_warnings()
    http = HttpRequest()
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    packed_extension_path = "C:\\bihmplhobchoageeokmgbdihknkjbknd.crx"
    options.add_extension(packed_extension_path)
    options.add_argument("user-agent=DN")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    driver = webdriver.Chrome('C:\\Users\\owner\\Downloads\\chromedriver_win32\\chromedriver.exe', options=options)

    get_fakePhoto()

