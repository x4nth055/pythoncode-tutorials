# importing required package of webdriver

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from time import sleep
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.opera.options import Options
# Just Run this main.py to execute the below script
from selenium.webdriver.support.wait import WebDriverWait

if __name__ == '__main__':
    # Instantiate the webdriver with the executable location of MS Edge
    browser = webdriver.Edge(r"C:\Users\LenovoE14\Downloads\edgedriver\msedgedriver.exe")
    # Simply just open a new Edge browser and go to lambdatest.com
    browser.maximize_window()
    browser.get('https://www.lambdatest.com')

    try:
        # Get the text box to insert Email using selector ID
        myElem_1 = WebDriverWait(browser, 10).until(EC.presence_of_element_located((By.ID, 'useremail')))
        # Entering the email address
        myElem_1.send_keys("rishabhps@lambdatest.com")
        myElem_1.click()
        # Get the Submit button to click and start free testing using selector CSS_SELECTOR
        myElem_2 = WebDriverWait(browser, 10).until(EC.element_to_be_clickable((By.CSS_SELECTOR, "#testing_form > div")))
        # Starting free testing on LambdaTest
        myElem_2.click()
        sleep(10)
    except TimeoutException:
        print("No element found")

    sleep(10)

    browser.close()