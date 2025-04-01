# Text Generation System Documentation

## 시스템 개요
이 시스템은 파일 업로드부터 텍스트 생성까지의 전체 프로세스를 관리하는 Text_Generation 클래스를 중심으로 구성되어 있습니다.

## 클래스 구조

### 1. 초기화 & 설정 함수
- #### `__init__()`
  - 모델 초기화
  - 기본 설정 구성
- #### `initialize_components()`
  - OpenAI API 키 설정
  - LangChain 컴포넌트 초기화
  - 메모리 설정

### 2. 파일 처리 함수
- #### `process_uploaded_files()`
  - 파일 유효성 검증
  - 텍스트 추출
  - 데이터 정제
- #### `process_and_embed_files()`
  - 파일 처리
  - 텍스트 분할
  - 임베딩 프로세스 시작

### 3. 임베딩 함수
- #### `create_embeddings()`
  - 텍스트 임베딩 생성
  - 배치 처리
  - 캐시 관리
- #### `setup_vector_store()`
  - FAISS 인덱스 설정
  - 벡터 저장소 구성
  - 검색 설정

### 4. GPT 생성 함수
- #### `gpt_text_generation()`
  - 프롬프트 구성
  - GPT 모델 호출
  - 토큰 계산
  - 비용 계산

### 5. 저장 & 로드 함수
- #### `save_faiss_index()`
  - FAISS 인덱스 저장
  - 저장 경로 관리
- #### `save_embeddings()`
  - 임베딩 데이터 저장
  - 캐시 파일 관리

### 6. 유틸리티 함수
- #### `load_faiss_index()`
  - 저장된 인덱스 로드
  - 유효성 검증
- #### `clean_text()`
  - 텍스트 정제
  - 특수문자 처리

### 7. 캐시 관리 함수
- #### `cache_embeddings()`
  - 임베딩 캐시 저장
  - 캐시 키 생성
- #### `clear_cache()`
  - 캐시 초기화
  - 임시 파일 정리

### 8. 에러 처리 함수
- #### `handle_api_error()`
  - API 오류 처리
  - 재시도 로직
- #### `retry_on_error()`
  - 자동 재시도
  - 에러 로깅

## 데이터 처리 흐름도
```mermaid
graph LR
    A[파일 업로드] --> B[텍스트 추출]
    B --> C[텍스트 분할]
    C --> D[임베딩 생성]
    D --> E[벡터 저장소]
    E --> F[GPT 모델]
    F --> G[텍스트 생성]
```

## 의존성
- langchain: LLM 통합 및 체인 관리
- openai: GPT 모델 접근
- faiss-cpu: 벡터 검색 엔진
- streamlit: 웹 인터페이스
- numpy: 수치 연산
- pandas: 데이터 처리

## 캐싱 전략
1. **임베딩 캐시**
   - 생성된 임베딩 저장
   - 재사용으로 API 비용 절감

2. **FAISS 인덱스 캐시**
   - 벡터 검색 인덱스 저장
   - 빠른 검색 성능 유지

3. **API 응답 캐시**
   - API 호출 결과 임시 저장
   - 중복 요청 방지

## 에러 처리 전략
1. **API 레이트 리밋 처리**
   - 자동 재시도
   - 지수 백오프

2. **재시도 메커니즘**
   - 일시적 오류 자동 복구
   - 최대 재시도 횟수 설정

3. **사용자 피드백**
   - 진행 상태 표시
   - 오류 메시지 명확화

## 사용 예시

```python
# Text_Generation 클래스 초기화
text_gen = Text_Generation()

# 파일 처리 및 임베딩
docsearch, num_chunks = text_gen.process_and_embed_files(uploaded_files)

# 텍스트 생성
response = text_gen.gpt_text_generation(system_prompt, assistant_content, user_query)
```

## 주의사항
- API 키는 반드시 안전하게 관리
- 대용량 파일 처리 시 메모리 사용량 주의
- 캐시 저장소 정기적 관리 필요
