import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import time
import pickle
import pandas as pd
import io
from openai import OpenAI

class Text_Generation:
    def __init__(self):
        self.gpt_model = "gpt-4o-mini"
        self.max_retries = 3
        self.retry_delay = 5
        self.initialize_components()

    def initialize_components(self):
        """LangChain 컴포넌트 초기화"""
        try:
            # .env 파일에서 환경 변수 로드
            load_dotenv()
            self.openai_api_key = os.getenv("OPENAI_API_KEY")
            
            if not self.openai_api_key:
                raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")
            
            self.embeddings = OpenAIEmbeddings(
                openai_api_key=self.openai_api_key,
                model="text-embedding-ada-002",
                max_retries=5,
                request_timeout=30
            )
            
            self.llm = ChatOpenAI(
                model=self.gpt_model,
                openai_api_key=self.openai_api_key,
                temperature=0.7,
                max_tokens=4096
            )
            
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                output_key="answer",
                return_messages=True
            )
            
            # 임베딩 일 경로 설정
            current_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(current_dir)
            self.embeddings_path = os.path.join(parent_dir, 'embeddings.pkl')
            
        except Exception as e:
            raise Exception(f"컴포넌트 초기화 중 오류 발생: {str(e)}")

    def process_and_embed_files(self, uploaded_files):
        """업로드된 파일들을 처리하고 임베딩하는 함수"""
        try:
            # 1. 파일 처리
            all_texts = []
            for file in uploaded_files:
                try:
                    contents = file.file.read()
                    file_extension = file.filename.split('.')[-1].lower()
                    
                    if file_extension == 'txt':
                        text_content = contents.decode('utf-8')
                        all_texts.append(text_content)
                        
                    elif file_extension in ['csv', 'xlsx', 'xls']:
                        if file_extension == 'csv':
                            df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
                        else:
                            df = pd.read_excel(io.BytesIO(contents))
                        
                        # 모든 데이터를 텍스트로 변환
                        text_content = df.to_string(index=False)
                        all_texts.append(text_content)
                        
                except Exception as e:
                    print(f"파일 '{file.filename}' 처리 중 오류 발생: {e}")
                    continue

            # 2. 텍스트 분할
            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=1000,
                chunk_overlap=50,
                length_function=len
            )
            
            split_texts = text_splitter.split_text("\n".join(all_texts))
            
            # 3. 벡터 저장소 설정
            docs = [Document(page_content=t) for t in split_texts]
            docsearch = FAISS.from_documents(docs, self.embeddings)
            
            # 4. FAISS 인덱스 저장
            current_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(current_dir)
            faiss_index_path = os.path.join(parent_dir, 'faiss_index')
            self.save_faiss_index(docsearch, faiss_index_path)
            
            # 5. ConversationalRetrievalChain 설정
            retriever = docsearch.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 500}
            )
            
            self.chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=retriever,
                memory=self.memory,
                return_source_documents=True,
                verbose=False,
                output_key="answer"
            )
            
            return docsearch, len(split_texts)
            
        except Exception as e:
            print(f"파일 처리 및 임베딩 중 오류 발생: {e}")
            raise

    def create_embeddings(self, texts):
        """텍스트를 임베딩으로 변환하고 캐싱하는 함수"""
        try:
            # 캐시 키 생성 (텍스트의 해시값 사용)
            cache_key = hash("".join(texts))
            
            # 캐시된 임베딩 확인
            if os.path.exists(self.embeddings_path):
                with open(self.embeddings_path, 'rb') as f:
                    cached_data = pickle.load(f)
                    if cached_data.get('key') == cache_key:
                        print("캐시된 임베딩 사용")
                        return cached_data.get('embeddings')
            
            # 새로운 임베딩 생성
            all_embeddings = []
            batch_size = 50
            
            total_batches = len(texts) // batch_size + (1 if len(texts) % batch_size else 0)
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                try:
                    batch_embeddings = self.embeddings.embed_documents(batch)
                    all_embeddings.extend(batch_embeddings)
                    
                    # API 레이트 리밋 방지
                    time.sleep(1)
                    
                except Exception as e:
                    print(f"배치 {i//batch_size + 1}/{total_batches} 임베딩 중 오류: {e}")
                    continue
            
            # 임베딩 캐시 저장
            with open(self.embeddings_path, 'wb') as f:
                pickle.dump({
                    'key': cache_key,
                    'embeddings': all_embeddings,
                    'timestamp': time.time()
                }, f)
            
            return all_embeddings
            
        except Exception as e:
            print(f"임베딩 생성 중 오류 발생: {e}")
            raise

    def setup_vector_store(self, texts):
        """벡터 저장소 설정"""
        try:
            # 텍스트 임베딩 생성
            embeddings_vectors = self.create_embeddings(texts)
            
            # 문서 생성
            docs = [Document(page_content=t) for t in texts]
            
            # FAISS 인덱스 파일 경로
            current_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(current_dir)
            faiss_index_path = os.path.join(parent_dir, 'faiss_index')
            
            # 저장된 FAISS 인덱스 로드 또는 새로 생성
            docsearch = self.load_faiss_index(faiss_index_path) or FAISS.from_documents(docs, self.embeddings)
            
            retriever = docsearch.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 500}
            )
            
            self.chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=retriever,
                memory=self.memory,
                return_source_documents=True,
                verbose=False,
                output_key="answer"
            )
            
            return docsearch
            
        except Exception as e:
            print(f"벡터 저장소 설정 중 오류 발생: {e}")
            raise

    def load_faiss_index(self, filepath):
        """FAISS 인덱스 로드"""
        try:
            if os.path.exists(filepath):
                return FAISS.load_local(
                    filepath, 
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
            return None
        except Exception as e:
            print(f"FAISS 인덱스 로드 중 오류 발생: {e}")
            return None

    def save_faiss_index(self, docsearch, filepath):
        """FAISS 인덱스 저장"""
        try:
            docsearch.save_local(filepath)
        except Exception as e:
            print(f"FAISS 인덱스 저장 중 오류 발생: {e}")
            raise

    def gpt_text_generation(self, system_prompt, assistant_content, user_query):
        """텍스트 생성 함수"""
        try:
            if hasattr(self, 'chain'):
                # ConversationalRetrievalChain 사용
                response = self.chain({"question": user_query})
                answer = response["answer"]
            else:
                # 일반 ChatGPT 응답 사용
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "assistant", "content": assistant_content},
                    {"role": "user", "content": user_query}
                ]
                response = self.llm.invoke(messages)
                answer = response.content
            
            # OpenAI API 직접 호출하여 토큰 정보 얻기
            client = OpenAI(api_key=self.openai_api_key)
            api_response = client.chat.completions.create(
                model=self.gpt_model,
                messages=[{"role": "user", "content": user_query}]
            )
            
            # 토큰 수와 비용 계산
            prompt_tokens = api_response.usage.prompt_tokens
            completion_tokens = api_response.usage.completion_tokens
            
            # 비용 계산 (GPT-4 기준)
            prompt_cost = (prompt_tokens * 0.03) / 1000
            completion_cost = (completion_tokens * 0.06) / 1000
            usd_cost = prompt_cost + completion_cost
            
            # 원화 변환
            exchange_rate = 1300
            kor_cost = usd_cost * exchange_rate
            
            return (
                answer,
                prompt_tokens,
                completion_tokens,
                usd_cost,
                kor_cost,
                exchange_rate
            )
            
        except Exception as e:
            print(f"텍스트 생성 중 오류 발생: {e}")
            raise

    def process_uploaded_files(self, uploaded_files):
        """업로드된 파일들을 처리하는 함수"""
        try:
            all_texts = []
            
            for file in uploaded_files:
                try:
                    # 파일 확장자 확인
                    file_extension = file.name.split('.')[-1].lower()
                    
                    if file_extension == 'txt':
                        # 텍스트 파일 처리
                        text_content = file.getvalue().decode('utf-8')
                        all_texts.append(text_content)
                        
                    elif file_extension in ['csv', 'xlsx', 'xls']:
                        # 엑셀/CSV 파일 처리
                        if file_extension == 'csv':
                            df = pd.read_csv(file)
                        else:
                            df = pd.read_excel(file)
                        
                        # DataFrame의 모든 열을 텍스트로 변환
                        text_content = df.to_string(index=False)
                        all_texts.append(text_content)
                        
                    elif file_extension in ['pdf', 'doc', 'docx']:
                        print(f"아직 지원하지 않는 파일 형식입니다: {file_extension}")
                        continue
                        
                    else:
                        print(f"지원하지 않는 파일 형식입니다: {file_extension}")
                        continue
                        
                    print(f"파일 '{file.name}' 처리 완료")
                        
                except Exception as e:
                    print(f"파일 '{file.name}' 처리 중 오류 발생: {e}")
                    continue
                    
            return all_texts
            
        except Exception as e:
            print(f"파일 처리 중 오류 발생: {e}")
            raise