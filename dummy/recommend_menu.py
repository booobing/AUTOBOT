import os
import glob
import pandas as pd
import streamlit as st
import time
from backend.text_generation import Text_Generation
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import pickle
from io import StringIO
from streamlit_folium import st_folium
import folium  
import re
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By

st.cache_data.clear()

@st.cache_data(ttl=3600)
def load_data_files():
    """데이터 파일들을 로드하는 함수"""
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)

        # 메뉴 데이터 로드
        menu_files = glob.glob(os.path.join(parent_dir, 'data_menu_*.csv'))
        if not menu_files:
            raise FileNotFoundError("메뉴 데이터 파일을 찾을 수 없습니다.")
        latest_menu_file = max(menu_files, key=os.path.getctime)
        menu_data = pd.read_csv(latest_menu_file, encoding='utf-8')

        # final_reviews_updated 데이터 로드
        final_results_file = os.path.join(parent_dir, 'final_reviews_updated.csv')
        if not os.path.exists(final_results_file):
            raise FileNotFoundError("final_reviews_updated.csv 파일을 찾을 수 없습니다.")
        final_results_data = pd.read_csv(final_results_file, encoding='utf-8')

        # DataFrame을 문자열로 변환
        menu_data_str = menu_data.to_string(index=False)
        final_results_data_str = final_results_data.to_string(index=False)

        # 디버깅: 반환되는 값 확인 (디버그 모드에서만)
        if st.session_state.get('debug_mode', False):
            st.write("반환되는 값의 개수:", 2)
            #st.write("menu_data_str 타입:", type(menu_data_str))
            #st.write("final_results_data_str 타입:", type(final_results_data_str))
        #print(menu_data_str[:100])
        return menu_data_str, final_results_data_str
    except Exception as e:
        st.error(f"데이터 파일 로드 중 오류 발생: {e}")
        return "", ""

def get_embeddings_with_retry(texts, embeddings, max_retries=5):
    """임베딩을 생성하는 함수 with 재시도 로직"""
    for attempt in range(max_retries):
        try:
            return embeddings.embed_documents(texts)
        except Exception as e:
            if "rate_limit" in str(e).lower():
                wait_time = min(2 ** attempt * 5, 60)  # 지수 백오프
                st.warning(f"API 레이트 리밋 초과. {wait_time}초 후에 재시도합니다...")
                time.sleep(wait_time)
                continue
            else:
                st.error(f"임베딩 생성 중 오류 발생: {e}")
                raise e
    st.error("임베딩 생성에 실패했습니다.")
    raise Exception("임베딩 생성에 반복 실패했습니다.")

def apply_custom_css():
    st.markdown("""
        <style>
        .stApp {
            background-color: #1a202c;
        }
        .main {
            background-color: #1a202c;
            padding: 2rem;
        }
        .title {
            color: white;
            font-size: 2.5rem;
            font-weight: bold;
            text-align: center;
            margin-bottom: 1rem;
        }
        .subtitle {
            color: #8b5cf6;
            font-size: 2rem;
            font-weight: bold;
            text-align: center;
            margin-bottom: 3rem;
        }
        .restaurant-card {
            background-color: #2d3748;
            padding: 1.5rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
        }
        .rating {
            color: #ef4444;
            font-size: 1.5rem;
            font-weight: bold;
        }
        .review {
            color: #9ca3af;
            font-size: 0.875rem;
        }
        .stTextInput input {
            background-color: #2d3748;
            color: white !important;
            border: 1px solid #4a5568;
            border-radius: 0.5rem;
            padding: 0.5rem 1rem;
            width: 100%;
        }
        .stTextInput input::placeholder {
            color: #9ca3af !important;
            opacity: 1 !important;
        }
        .stButton button {
            background-color: #8b5cf6;
            color: white;
            border: none;
            border-radius: 0.5rem;
            padding: 0.5rem 2rem;
            cursor: pointer;
            white-space: nowrap;
        }
        .stButton button:hover {
            background-color: #7c3aed;
        }
        .search-container {
            display: flex;
            justify-content: center;
            align-items: center;
            width: 100%;
            padding: 0 20px;
        }
        .search-wrapper {
            display: flex;
            justify-content: center;
            align-items: center;
            width: 100%;
            max-width: 800px;
        }
        .column-container {
            width: 100%;
        }
        .cost-info {
            position: fixed;
            bottom: 20px;
            right: 20px;
            background-color: #2d3748;
            padding: 10px;
            border-radius: 5px;
            color: #9ca3af;
            font-size: 0.8rem;
        }
        </style>
    """, unsafe_allow_html=True)

def save_embeddings(embeddings, filepath):
    """임베딩을 파일에 저장하는 함수"""
    with open(filepath, 'wb') as f:
        pickle.dump(embeddings, f)

def load_embeddings(filepath):
    """파일에서 임베딩을 불러오는 함수"""
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    return None

@st.cache_data(ttl=3600)
def load_embeddings_cached(texts, _embeddings):
    """캐싱된 임베딩을 로드하거나 생성하는 함수"""
    embedding_filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'embeddings.pkl')
    all_embeddings = load_embeddings(embedding_filepath)
    
    if all_embeddings is None:
        all_embeddings = []
        batch_size = 50  # 배치 크기 조절

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            try:
                batch_embeddings = get_embeddings_with_retry(batch_texts, _embeddings)
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                st.error(f"임베딩 생성 중 오류 발생: {e}")
                continue
            time.sleep(1)  # 레이트 리밋 방지를 위한 지연
        
        # 생성된 임베딩 저장
        save_embeddings(all_embeddings, embedding_filepath)
    
    return all_embeddings

def setup_driver():
    """Selenium WebDriver 설정"""
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # 헤드리스 모드
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--disable-infobars")
    chrome_options.add_argument("start-maximized")
    chrome_options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.93 Safari/537.36"
    )
    
    return webdriver.Chrome(
        service=ChromeService(ChromeDriverManager().install()),
        options=chrome_options
    )

def get_restaurant_image(restaurant_id):
    """네이버 플레이스에서 식당 이미지를 크롤링하는 함수"""
    if not restaurant_id:  # restaurant_id가 None이거나 빈 문자열인 경우
        return None
        
    try:
        driver = setup_driver()
        url = f"https://pcmap.place.naver.com/restaurant/{restaurant_id}/photo"
        driver.get(url)
        
        # 이미지 로딩 대기
        wait = WebDriverWait(driver, 10)
        img_elements = wait.until(
            EC.presence_of_all_elements_located((By.CLASS_NAME, "K0PDV"))
        )
        
        # 첫 번째 이미지의 src 속성 가져오기
        if img_elements:
            img_url = img_elements[0].get_attribute("src")
            driver.quit()
            return img_url
        
        driver.quit()
        return None
        
    except Exception as e:
        if st.session_state.get('debug_mode', False):
            st.error(f"이미지 크롤링 중 오류 발생: {e}")
        if 'driver' in locals():
            driver.quit()
        return None

def clean_restaurant_name(name):
    """식당 이름에서 특수 문자와 대괄호를 제거하는 함수"""
    # 특수 문자 제거 (공백은 유지)
    name = re.sub(r'[^\w\s가-힣]', '', name)
    # 앞뒤 공백 제거
    name = name.strip()
    return name

def save_faiss_index(docsearch, filepath):
    """FAISS 인덱스를 파일로 저장하는 함수"""
    try:
        docsearch.save_local(filepath)
        if st.session_state.get('debug_mode', False):
            st.write(f"FAISS 인덱스 저장 완료: {filepath}")
        return True
    except Exception as e:
        if st.session_state.get('debug_mode', False):
            st.error(f"FAISS 인덱스 저장 중 오류 발생: {e}")
        return False

def load_faiss_index(filepath, embeddings):
    """저장된 FAISS 인덱스를 로드하는 함수"""
    try:
        if os.path.exists(filepath):
            docsearch = FAISS.load_local(
                filepath, 
                embeddings,
                allow_dangerous_deserialization=True  # 신뢰할 수 있는 로컬 파일이므로 True로 설정
            )
            if st.session_state.get('debug_mode', False):
                st.write(f"FAISS 인덱스 로드 완료: {filepath}")
            return docsearch
        return None
    except Exception as e:
        if st.session_state.get('debug_mode', False):
            st.error(f"FAISS 인덱스 로드 중 오류 발생: {e}")
        return None
    
def menu_data_load():
    """문자열로 변환된 DataFrame을 다시 dict로 변환하는 함수"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)

    # 메뉴 데이터 로드
    menu_files = glob.glob(os.path.join(parent_dir, 'data_menu_*.csv'))
    if not menu_files:
        raise FileNotFoundError("메뉴 데이터 파일을 찾을 수 없습니다.")
    latest_menu_file = max(menu_files, key=os.path.getctime)
    menu_data = pd.read_csv(latest_menu_file, encoding='utf-8')
    #print(menu_data_str)
    
    # DataFrame을 dict로 변환
    #menu_data_dict = menu_data.to_dict(orient='records')
    #print(menu_data)
    return menu_data
    
def find_restaurant_info(menu_data_df, restaurant_name):
    """식당 이름을 바탕으로 식당 번호를 찾는 함수"""
    try:
        # 검색할 식당 이름에서 공백 제거
        restaurant_name_no_space = ''.join(restaurant_name.split())
        
        # DataFrame의 상호명에서 공백을 제거하고 비교
        matching_row = menu_data_df[menu_data_df['상호명'].str.replace(' ', '').str.contains(restaurant_name_no_space, na=False)]
        if not matching_row.empty:
            return (
                matching_row['식당 번호'].iloc[0],
                matching_row['좌표X'].iloc[0],
                matching_row['좌표Y'].iloc[0]
            )
        return None, None, None  # 일치하는 식당이 없을 경우
    except Exception as e:
        if st.session_state.get('debug_mode', False):
            st.error(f"식당 번호 찾기 오류: {e}")
        return None, None, None  # 에러 발생 시에도 None 튜플 반환

def calculate_average_rating(restaurant_name):
    """식당 이름으로 평균 평점을 계산하는 함수"""
    try:
        # 현재 디렉토리 경로 가져오기
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        
        # final_reviews_updated.csv 파일 로드
        final_reviews_file = os.path.join(parent_dir, 'final_reviews_updated.csv')
        final_reviews_updated_df = pd.read_csv(final_reviews_file, encoding='utf-8')
        
        # 검색할 식당 이름에서 공백 제거
        restaurant_name_no_space = ''.join(restaurant_name.split())
        
        # 해당 식당의 리뷰들 찾기
        matching_reviews = final_reviews_updated_df[
            final_reviews_updated_df['상호명'].str.replace(' ', '').str.contains(
                restaurant_name_no_space, 
                na=False, 
                case=False  # 대소문자 구분 없이
            )
        ]
        
        if matching_reviews.empty:
            if st.session_state.get('debug_mode', False):
                st.warning(f"'{restaurant_name}' 식당의 리뷰를 찾을 수 없습니다.")
            return 0.0
        
        # 평균 평점 계산 (긍정 리뷰는 1, 부정 리뷰는 0으로 계산)
        average_rating = matching_reviews['긍부정'].mean() * 10  # 10점 만점으로 변환
        
        if st.session_state.get('debug_mode', False):
            st.write(f"'{restaurant_name}' 식당의 리뷰 수: {len(matching_reviews)}")
            st.write(f"평균 평점: {average_rating:.1f}")
        
        return round(average_rating, 1)  # 소수점 첫째자리까지 반올림
        
    except Exception as e:
        if st.session_state.get('debug_mode', False):
            st.error(f"평점 계산 중 오류 발생: {e}")
        return 0.0


class RestaurantRecommender:
    def __init__(self):
        max_retries = 3
        retry_delay = 5
        
        for attempt in range(max_retries):
            try:
                # secrets.toml에서 API 키 읽기
                openai_api_key = st.secrets["OPENAI_API_KEY"]
                
                # LangChain 컴포넌트 초기화
                self.embeddings = OpenAIEmbeddings(
                    openai_api_key=openai_api_key,
                    model="text-embedding-ada-002",
                    max_retries=5,
                    request_timeout=30
                )
                
                self.llm = ChatOpenAI(
                    model="gpt-4o-mini",
                    openai_api_key=openai_api_key,
                    temperature=0.7,
                    max_tokens=4096
                )
                
                # 메모리 컴포넌트 추가
                self.memory = ConversationBufferMemory(
                    memory_key="chat_history",
                    output_key="answer",  # 출력 키 명시적 지정
                    return_messages=True
                )
                self.text_generation = Text_Generation()
                self.text_generation.gpt_model = "gpt-4o-mini"
                
                self.load_data()
                break  # 성공하면 루프 종료
            except Exception as e:
                if "520" in str(e) and attempt < max_retries - 1:  # Cloudflare 오류
                    st.warning(f"OpenAI API 서버 연결 오류. {retry_delay}초 후 재시도 중... ({attempt + 1}/{max_retries})")
                    time.sleep(retry_delay)
                    continue
                else:
                    st.error(f"추천 시스템 초기화 중 오류 발생: {e}")
                    raise

    def load_data(self):
        try:
            # 데이터 로드
            self.menu_data_str, self.final_results_data_str = load_data_files()
            
            # 텍스트 데이터 준비
            self.restaurant_texts = []
            
            # 메뉴 데이터 처리
            for line in self.menu_data_str.split('\n'):
                if line.strip():  # 빈 줄 제외
                    text = f"메뉴 정보: {line}"
                    self.restaurant_texts.append(text)

            # 리뷰 데이터 처리
            for line in self.final_results_data_str.split('\n'):
                if line.strip():  # 빈 줄 제외
                    text = f"리뷰 정보: {line}"
                    self.restaurant_texts.append(text)

            # 텍스트 분할
            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=1000,
                chunk_overlap=50,
                length_function=len
            )
            
            # 텍스트를 더 작은 청크로 분할
            texts = text_splitter.split_text("\n".join(self.restaurant_texts))
            docs = [Document(page_content=t) for t in texts]
            
            # FAISS 인덱스 파일 경로
            current_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(current_dir)
            self.faiss_index_path = os.path.join(parent_dir, 'faiss_index')
            
            # 저장된 FAISS 인덱스 로드 또는 새로 생성
            self.docsearch = load_faiss_index(self.faiss_index_path, self.embeddings) or FAISS.from_documents(docs, self.embeddings)
            
            # 검색 체인 설정
            self.retriever = self.docsearch.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 500}
            )
            
            # 대화형 검색 체인 설정
            self.chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.retriever,
                memory=self.memory,
                return_source_documents=True,
                verbose=False,
                output_key="answer"  # 출력 키 명시적 지정
            )

        except Exception as e:
            st.error(f"데이터 로딩 중 오류 발생: {e}")
            raise
    
    def search_initial_candidates(self, query):
        """초기 후보군 검색 함수"""
        try:

            menu_data = menu_data_load()
            menu_data_list=""
            for i in range(len(menu_data)):
                menu_data_list+= f"{menu_data['상호명'].iloc[i]}"+":"+f"{menu_data['메뉴'].iloc[i]}"+"\n"

            # 1단계: 초기 후보군 검색
            initial_system_prompt = (
                "당신은 사용자의 선호도에 맞는 맛집을 검색하는 전문가입니다. "
                "식당들 중에서 사용자의 조건에 가장 잘 맞는 열 곳을 선별해주세요. "
                "긍정적인 리뷰는 긍부정이 1이고 부정적인 리뷰는 긍부정이 0입니다. "
                "응답은 반드시 마크다운 형식으로 작성해주세요."
            )

            assistant_content = (
                f"다음은 추천 가능한 식당들의 정보입니다:\n{menu_data_list}\n\n"
                # "1.미미옥\n"
                # "2.디핀신당\n"
                # "3.신당중앙시장 라까예\n"
                # "4.오늘의 초밥\n"
                # "5.중앙감속기\n"
                # "6.광주 가매일식\n"
                # "7.제주 오팬파이어\n"
                # "8.청담동 쵸이닷\n"
                # "9.서울 서교동 진진\n"
                # "10.마마리마켓\n"
            )
            
            # GPT 호출 (한 번만)
            initial_candidates, prompt_tokens, completion_tokens, usd_cost, kor_cost, exchange_rate = (
                self.text_generation.gpt_text_generation(initial_system_prompt, assistant_content, query)
            )
                        
            return (
                initial_candidates,
                prompt_tokens,
                completion_tokens,
                usd_cost,
                kor_cost,
                exchange_rate
            )
            
        except Exception as e:
            st.error(f"초기 후보군 검색 중 오류 발생: {e}")
            raise e

    def generate_final_recommendations(self, query, initial_candidates):
        try:
            menu_data = menu_data_load()
            menu_data_list=""
            for i in range(len(menu_data)):
                menu_data_list+= f"{menu_data['상호명'].iloc[i]}"+":"+f"{menu_data['주소'].iloc[i]}"+"\n"

            # 2단계: 최종 추천 생성
            final_system_prompt = (
                "당신은 사용자의 선호도에 맞는 맛집을 추천해주는 전문가입니다. "
                f"다음은 추천 가능한 식당들의 정보입니다:\n{menu_data_list}\n\n"
                "주어진 정보를 바탕으로 가장 적합한 식당 세 곳을 추천해주세요. "
                "긍정적인 리뷰는 긍부정이 1이고 부정적인 리뷰는 긍부정이 0입니다. "
                "각 식당마다 긍정적인 리뷰 2개와 부정적인 리뷰 1개를 포함해주세요. "
                "응답은 반드시 마크다운 형식으로 작성해주세요."
            )

            final_query = (
                f"{final_system_prompt}\n\n"
                f"사용자 질문: {query}\n\n"
                f"검색된 식당 목록:\n{initial_candidates}\n\n"
                "위 정보들 중에서 가장 적합한 중복없이 식당 세 곳을 추천해주세요. "
                "식당 정보의 변형 없이 그대로 써야합니다. "
                "아래 마크다운 형식으로 작성해주세요:\n\n"
                "### 1. 식당 이름\n"
                "- **위치:** [위치 정보]\n"
                "- **대표 메뉴:** [대표 메뉴 정보]\n"
                "- **분위기:** [분위기 정보]\n"
                "- **가격대:** [가격대 정보]\n"
                "- **평점:** [평점 정보 (1.0~10.0점)]\n"
                "- **리뷰:**\n"
                "  - (긍정) [긍정적인 리뷰 1]\n"
                "  - (긍정) [긍정적인 리뷰 2]\n"
                "  - (부정) [부정적인 리뷰]\n\n"
                "### 2. 식당 이름...\n\n"
                "다음은 한식을 제공하며 2명이 방문하기에 적합하고, 분위기가 좋고 중간 가격대의 식당 세 곳을 추천드립니다."

                "### 1. 신당중앙시장 라까예"
                "- **위치:** 서울 중구 퇴계로85길 42"
                "- **대표 메뉴:** 다양한 한국 전통 요리 (국, 찌개, 전 등)"
                "- **분위기:** 아늑하며 인테리어가 멋지고, 혼밥하기 좋은 자리도 마련되어 있습니다."
                "- **가격대:** 중간 (1인당 8,000원 ~ 15,000원)"
                "- **평점:** 4.5점"
                "- (긍정) [긍정적인 리뷰 1]"
                "- (긍정) [긍정적인 리뷰 2]"
                "- (부정) [부정적인 리뷰]"

                "### 2. 미미옥"
                "- **위치:** 서울 용산구 한강대로15길 27"
                "- **대표 메뉴:** 갈비찜, 비빔밥, 다양한 한식 코스"
                "- **분위기:** 차분하고 고급스러운 인테리어, 친절한 서비스가 특징입니다."
                "- **가격대:** 중간 (1인당 10,000원 ~ 20,000원)"
                "- **평점:** 9.0점"
                "- (긍정) [긍정적인 리뷰 1]"
                "- (긍정) [긍정적인 리뷰 2]"
                "- (부정) [부정적인 리뷰]"

                "### 3. 디핀신당"
                "- **위치:** 서울 중구 퇴계로 411"
                "- **대표 메뉴:** 한정식, 전통 궁중 요리"
                "- **분위기:** 세련된 인테리어로 특별한 날 가기 좋은 분위기입니다."
                "- **가격대:** 중간 (1인당 15,000원 ~ 30,000원)"
                "- **평점:** 8.5점"
                "- (긍정) [긍정적인 리뷰 1]"
                "- (긍정) [긍정적인 리뷰 2]"
                "- (부정) [부정적인 리뷰]"

                "이 세 곳은 맛있는 한식을 다양한 메뉴로 제공하며, 분위기가 좋아 특별한 날이나 기념일에도 적합한"
                "장소입니다. 예약을 하시면 더욱 좋은 자리를 확보할 수 있으니 미리 계획하시기 바랍니다!    "
            )

            final_results = self.chain({
                "question": final_query
            })

            return final_results['answer'], final_results['source_documents']
            
        except Exception as e:
            st.error(f"최종 추천 생성 중 오류 발생: {e}")
            raise e

    def get_recommendations(self, query):
            """메인 추천 함수"""
            try:
                # 1단계: 초기 후보군 검색
                initial_candidates, prompt_tokens, completion_tokens, usd_cost, kor_cost, exchange_rate= self.search_initial_candidates(query)
                
                # 디버그 모드일 때 초기 후보군 표시
                if st.session_state.get('debug_mode', False):
                    st.write("### 초기 검색된 후보군:")
                    st.write(initial_candidates)

                # 2단계: 최종 추천 생성
                recommendation, source_docs = self.generate_final_recommendations(query, initial_candidates)

                # 토큰 사용량 계산
                prompt_tokens = len(query.split())
                completion_tokens += len(recommendation.split())
                
                # 비용 계산
                usd_cost = (prompt_tokens * 0.00001 + completion_tokens * 0.00002)
                kor_cost = usd_cost * exchange_rate

                return (
                    recommendation,
                    source_docs,
                    prompt_tokens,
                    completion_tokens,
                    usd_cost,
                    kor_cost,
                    exchange_rate
                )

            except Exception as e:
                st.error(f"추천 생성 중 오류 발생: {e}")
                raise e

def main():
    st.set_page_config(
        page_title="흑수저부터 백수저까지, 모든 맛의 지도",
        page_icon="🍽️",
        layout="wide"
    )

    # 리뷰 색상 정의
    positive_color = '#10b981'
    negative_color = '#ef4444'
    
    # 디버깅 모드 초기화
    if 'debug_mode' not in st.session_state:
        st.session_state.debug_mode = False

    apply_custom_css()

    # 헤더
    st.markdown('<p class="title">흑수저부터 백수저까지,</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">모든 맛의 지도</p>', unsafe_allow_html=True)

    # 사이드바에 디버깅 모드 토글 추가
    with st.sidebar:
        st.session_state.debug_mode = st.checkbox("디버깅 모드", value=False)

    # 검색 컨테이너 시작
    st.markdown('<div class="search-container">', unsafe_allow_html=True)
    st.markdown('<div class="search-wrapper">', unsafe_allow_html=True)

    # 컬럼 컨테이너 시작
    st.markdown('<div class="column-container">', unsafe_allow_html=True)

    # 검색창과 버튼을 컬럼로 나란히 배치
    col1, col2, col3 = st.columns([1, 3, 1])  # 1:3:1 비율로 분할

    recommender = None
    try:
        recommender = RestaurantRecommender()
    except Exception as e:
        st.error(f"추천 시스템 초기화 중 오류 발생: {e}")

    with col2:  # 중앙 컬럼에 검색창과 버튼 배치
        search_col1, search_col2 = st.columns([4, 1])  # 4:1 비율로 분할
        with search_col1:
            search_text = st.text_input(
                "",
                placeholder="원하시는 조건을 입력해주세요 (예: 분위기 좋은 데이트 식당)",
                label_visibility="collapsed",
                key="search_input"
            )

        with search_col2:
            search_button = st.button("찾기", key="search_button")

    # 컬럼 컨테이너 닫기
    st.markdown('</div>', unsafe_allow_html=True)

    # wrapper와 container 닫기
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # 검색 버튼 클릭 시 동작
    if search_button and search_text:
        if not recommender:
            st.error("추천 시스템이 초기화되지 않았습니다.")
        else:
            with st.spinner("맞춤 식당을 찾는 중..."):
                try:
                    # recommendation 튜플에서 모든 값을 받아옴
                    recommendation, source_docs, prompt_tokens, completion_tokens, usd_cost, kor_cost, exchange_rate = (
                        recommender.get_recommendations(search_text)
                    )
                    
                    # GPT의 원본 응답 표시 (디버그 모드일 때)
                    if st.session_state.debug_mode:
                        st.write("### GPT 원본 응답:")
                        st.write(recommendation)
                        
                    # 추천 결과를 카드 형태로 표시
                    cols = st.columns(3)
                    
                    try:
                        # GPT 응답을 파싱하여 식당 정보 추출
                        restaurants = []
                        current_restaurant = None
                        current_reviews = []
                        lines = recommendation.split('\n')
                        
                        for line in lines:
                            line = line.strip()
                            if not line:
                                continue
                            
                            # 새로운 식당 시작
                            if line.startswith('### '):
                                if current_restaurant and current_reviews:  # 리뷰가 있을 때만 추가
                                    current_restaurant['reviews'] = current_reviews
                                    restaurants.append(current_restaurant)
                                restaurant_name = line.split('.', 1)[1].strip()
                                current_restaurant = {"title": restaurant_name}
                                current_reviews = []
                                average_rating = calculate_average_rating(restaurant_name)
                                rating = f"{average_rating:.1f}"
                                print(f"{average_rating:.1f}")
                                continue
                            
                            # 리뷰 파싱
                            if line.strip().startswith('- ('):  # 들여쓰기 무시하고 파싱
                                try:
                                    review_type = line[line.find('(')+1:line.find(')')]
                                    review_content = line[line.find(')')+1:].strip()
                                    if review_content:  # 리뷰 내용이 있을 때만 추가
                                        current_reviews.append(f"({review_type}) {review_content}")
                                except:
                                    continue
                            
                            # 기본 정보 파싱
                            if line.startswith('- **'):
                                try:
                                    key = line.split('**')[1].split(':')[0].strip()
                                    value = line.split(':**')[1].strip()
                                    
                                    if key == "리뷰":  # 리뷰 섹션 시작 표시
                                        continue
                                        
                                    key_mapping = {
                                        "위치": "위치",
                                        "대표 메뉴": "대표 메뉴",
                                        "분위기": "분위기",
                                        "가격대": "가격대",
                                        #"평점": "평점"
                                    }
                                    
                                    if key in key_mapping:
                                        current_restaurant[key_mapping[key]] = value
                                        current_restaurant['평점'] = rating
                                        
                                except Exception as e:
                                    if st.session_state.debug_mode:
                                        st.warning(f"라인 파싱 실패: {line}, 오류: {e}")
                                    continue
                        
                        # 마지막 식당 추가
                        if current_restaurant and current_reviews:  # 리뷰가 있을 때만 추가
                            current_restaurant['reviews'] = current_reviews
                            restaurants.append(current_restaurant)
                        
                        # 결과가 없으면 기본 메시지 표시
                        if not restaurants:
                            restaurants = [{
                                'title': '파싱 오류',
                                '위치': '정보를 가져오는 중 문제가 발생했습니다',
                                '대표 메뉴': '잠시 후 다시 시도해주세요',
                                '분위기': '-',
                                '가격대': '-',
                                '평점': 0.0
                            }]

                    except Exception as e:
                        if st.session_state.debug_mode:
                            st.error(f"파싱 중 오류 발생: {e}")
                        restaurants = [{
                            'title': '파싱 오류',
                            '위치': '정보를 가져오는 중 문제가 발생했습니다',
                            '대표 메뉴': '잠시 후 다시 시도해주세요',
                            '분위기': '-',
                            '가격대': '-',
                            '평점': 0.0
                        }]

                    # 각 식당 정보를 카드로 표시 (파싱 성공/실패 여부와 관계없이 실행)
                    for idx, restaurant in enumerate(restaurants):
                        with cols[idx % 3]:

                            menu_data = menu_data_load()
                            print(restaurant['title'])
                            restaurant_name_cleaned = clean_restaurant_name(restaurant['title'])
                            restaurant_number, coord_x, coord_y = find_restaurant_info(menu_data, restaurant_name_cleaned)
                            
                            print(restaurant_number)
                            print(coord_x)
                            print(coord_y)  

                            img_url = get_restaurant_image(restaurant_number) if restaurant_number else None
                            
                            
                            #print(img_url)

                            st.markdown(
                                f"""
                                <div class="restaurant-card" style="background-color: #2d3748; padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; min-height: 400px;">
                                    {f'<img src="{img_url}" style="width: 100%; height: 500px; object-fit: cover; border-radius: 0.5rem; margin-bottom: 1rem;">' if img_url else ''}
                                    <div style="padding: 1.5rem;">
                                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                                            <h3 style="color: white; font-size: 1.5rem; font-weight: bold; margin: 0;">
                                                {restaurant.get('title', '이름 없음')}
                                            </h3>
                                            <span class="rating" style="color: #fbbf24; font-size: 1.5rem; font-weight: bold;">
                                                {restaurant.get('평점', '0')}
                                            </span>
                                        </div>
                                        <p style="color: #9ca3af; margin-bottom: 0.5rem;">
                                            {restaurant.get('위치', '위치 정보 없음')}
                                        </p>
                                        <div style="background-color: #374151; padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem;">
                                            <p style="color: #e5e7eb; margin-bottom: 1rem;">
                                                <span style="color: #8b5cf6; font-weight: bold;">대표 메뉴:</span> {restaurant.get('대표 메뉴', '정보 없음')}
                                            </p>
                                            <p style="color: #e5e7eb; margin-bottom: 1rem;">
                                                <span style="color: #8b5cf6; font-weight: bold;">분위기:</span> {restaurant.get('분위기', '정보 없음')}
                                            </p>
                                            <p style="color: #e5e7eb;">
                                                <span style="color: #8b5cf6; font-weight: bold;">가격대:</span> {restaurant.get('가격대', '정보 없음')}
                                            </p>
                                        </div>
                                        <div style="background-color: #374151; padding: 1.5rem; border-radius: 0.5rem;">
                                            <p style="color: #8b5cf6; font-weight: bold; margin-bottom: 0.5rem;">리뷰</p>
                                            {''.join([
                                                f'<p class="review" style="color: #9ca3af; margin-bottom: 0.5rem; padding-left: 0.5rem; border-left: 2px solid {positive_color if "긍정" in review else negative_color};">{review}</p>'
                                                for review in restaurant.get('reviews', [])
                                            ])}
                                        </div>
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                            # 리뷰 섹션 다음에 지도 추가
                            if coord_x is not None and coord_y is not None:
                                try:
                                    st.markdown(
                                        """
                                        <style>
                                        iframe[title="streamlit_folium.st_folium"] {
                                            height: 400px;
                                        }
                                        </style>
                                        """,
                                        unsafe_allow_html=True,
                                    )

                                    # 지도 생성 및 표시
                                    lat = float(coord_y)
                                    lon = float(coord_x)
                                    m = folium.Map(
                                        location=[lat, lon],
                                        zoom_start=15,
                                        scrollWheelZoom=True,
                                        dragging=True
                                    )
                                    
                                    # 마커 추가
                                    folium.Marker(
                                        [lat, lon],
                                        popup=restaurant.get('title', '식당'),
                                        icon=folium.Icon(color='red', icon='info-sign')
                                    ).add_to(m)
                                    
                                    
                                    st_folium(
                                        m, 
                                        height=400,
                                        width="100%",
                                        returned_objects=[]
                                    )
                                                                                                          
                                except Exception as e:
                                    if st.session_state.get('debug_mode', False):
                                        st.error(f"지도 생성 중 오류 발생: {e}")
                                        st.error(f"좌표값: x={coord_x}, y={coord_y}")
                                

                    # 비용 정보를 오른쪽 하단에 표시
                    st.markdown(
                        f"""
                        <div class="cost-info" style="margin-top: 20px; text-align: right; color: #9ca3af;">
                            프롬프트 토큰: {prompt_tokens} | 
                            완성 토큰: {completion_tokens} | 
                            비용: ${usd_cost:.4f} (₩{kor_cost:.2f}) | 
                            환율: ₩{exchange_rate:.2f}/$1
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                except Exception as e:
                    st.error(f"추천받는 중 오류가 발생했습니다: {e}")
                    st.stop()
    elif search_button:
        st.warning("검색어를 입력해주세요.")

if __name__ == "__main__":
    main()