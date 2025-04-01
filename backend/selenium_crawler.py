from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def crawl_webpage(url: str) -> str:
    """
    주어진 URL의 웹페이지에서 모든 텍스트를 크롤링합니다.
    
    Args:
        url (str): 크롤링할 웹페이지의 URL
        
    Returns:
        str: 크롤링된 텍스트 내용
    """
    # Chrome 옵션 설정
    chrome_options = Options()
    chrome_options.add_argument('--headless')  # 헤드리스 모드
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    
    try:
        # WebDriver 초기화
        driver = webdriver.Chrome(options=chrome_options)
        
        # 페이지 로드
        driver.get(url)
        
        # 페이지가 완전히 로드될 때까지 대기
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        
        # 모든 텍스트 요소 추출 (script, style 태그 제외)
        text_elements = driver.find_elements(
            By.XPATH, 
            "//*[not(self::script) and not(self::style) and not(self::noscript)]"
        )
        
        # 텍스트 결합 (각 요소마다 줄바꿈 추가)
        all_text = '\n'.join([
            elem.text.strip() 
            for elem in text_elements 
            if elem.text.strip()
        ])
        
        return all_text
        
    except Exception as e:
        print(f"크롤링 중 오류 발생: {str(e)}")
        return f"크롤링 중 오류 발생: {str(e)}"
        
    finally:
        if 'driver' in locals():
            driver.quit()