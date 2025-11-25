import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from datetime import datetime,timedelta
import requests
import time
import re

def get_cookie():
    googleTrendsUrl = 'https://google.com'
    response = requests.get(googleTrendsUrl)
    if response.status_code == 200:
        g_cookies = response.cookies.get_dict()
        return g_cookies

def scrape_google_trends(query: list | str = "bitcoin", 
                         start_date: datetime | str = "2025-10-14",
                         end_date: datetime | str = "2025-11-23",
                         headless: bool = True) -> pd.DataFrame:
    """
    Scrape Google Trends for a variable query and date range.
    
    Args:
        query: Search terms (e.g., "bitcoin" or "bitcoin,ethereum").
        date_range: Format "YYYY-MM-DD YYYY-MM-DD" (e.g., "2025-10-14 2025-11-23").
        headless: Run browser invisibly.
    
    Returns:
        DataFrame with interest over time, related queries, topics, and regions.
    """
    # Setup Chrome with anti-detection
    options = webdriver.ChromeOptions()
    if headless:
        options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)

    driver = webdriver.Chrome(options=options)
    
    
    # Build URL with variables
    if isinstance(start_date,datetime): start_date = start_date.strftime("%Y-%m-%d")
    if isinstance(end_date,datetime): end_date = end_date.strftime("%Y-%m-%d")
    if isinstance(query,list): query = '%2C'.join(query)
    url = f"https://trends.google.com/trends/explore?date={start_date}%20{end_date}&q={query}"

    driver.get("https://trends.google.com")
    driver.get(url)
    
    wait = WebDriverWait(driver, 15)
    
    data = {
        "date": [],
        "interest": [],  # For multi-queries, this will be a list per date
        "related_queries_top": [],
        "related_queries_rising": [],
        "related_topics": [],
        "regions": []
    }
    
    try:
        # Step 1: Click the three-dot menu (this opens the dropdown)
        menu_button = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "div.widget-actions")))
        # or more reliable:
        #menu_button = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "div[role='button'][aria-label*='options']")))
        menu_button.click()
        time.sleep(1)  # tiny delay so dropdown animates

        # Step 2: Now click "Download CSV" inside the opened menu
        download_button = wait.until(EC.element_to_be_clickable((
            By.XPATH, "//span[text()='Download CSV']/ancestor::div[@role='menuitem' or @role='button']"
        )))
        download_button.click()

        print("CSV download started!")
    
    except TimeoutException:
        print("Page load timeout – try without headless or check connection.")
        data = {"error": "Scraping failed"}
    
    finally:
        driver.quit()
    
    # Flatten to DataFrame (for multi-day data)
    if "error" not in data:
        df = pd.DataFrame(data)
        # Set local timezone (from previous chat)
        from zoneinfo import ZoneInfo
        local_tz = ZoneInfo("America/New_York")  # Change to your TZ
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize("UTC").dt.tz_convert(local_tz)
        df.set_index("date", inplace=True)
        return df
    else:
        return pd.DataFrame()

# Usage: Make variables easy to change
query = 'bitcoin'
start_date = datetime(2025,6,1)
end_date = start_date + timedelta(90)



headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5)\
            AppleWebKit/537.36 (KHTML, like Gecko) Cafari/537.36'}
if isinstance(start_date,datetime): start_date = start_date.strftime("%Y-%m-%d")
if isinstance(end_date,datetime): end_date = end_date.strftime("%Y-%m-%d")
if isinstance(query,list): query = '%2C'.join(query)
url = f"https://trends.google.com/trends/explore?date={start_date}%20{end_date}&q={query}"

# Source - https://stackoverflow.com/a
# Posted by MadFish.DT
# Retrieved 2025-11-23, License - CC BY-SA 4.0

df = scrape_google_trends(query=query, start_date=start_date, end_date=end_date, headless=False)
print(df.head())  # Preview
#df.to_csv(f"{QUERY}_trends_{DATE_RANGE.replace(' ', '_')}.csv", index=True)  # Export
