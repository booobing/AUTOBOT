from fastapi import FastAPI, UploadFile, File, HTTPException, Security, Depends
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from text_generation import Text_Generation
import asyncio
import json
import os
from dotenv import load_dotenv
from screen_processor import ScreenProcessor
from fastapi.responses import JSONResponse
import logging
from selenium_crawler import crawl_webpage  # 이전에 만든 크롤링 함수 import

# .env 파일에서 API 키 로드
load_dotenv()
API_KEY = os.getenv("API_KEY")

if not API_KEY:
    raise ValueError("API_KEY가 설정되지 않았습니다. .env 파일을 확인해주세요.")

app = FastAPI()

# API 키 헤더 설정
api_key_header = APIKeyHeader(name="X-API-Key")

async def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header != API_KEY:
        raise HTTPException(
            status_code=403,
            detail="잘못된 API 키입니다"
        )
    return api_key_header

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Text_Generation 인스턴스 생성
text_gen = Text_Generation()

# ScreenProcessor 인스턴스 생성
screen_processor = ScreenProcessor()

class QueryRequest(BaseModel):
    system_prompt: str
    assistant_content: str
    user_query: str

class ImageData(BaseModel):
    image: str

class CrawlRequest(BaseModel):
    url: str

@app.post("/api/upload")
async def upload_files(
    files: List[UploadFile] = File(...),
    api_key: str = Depends(get_api_key)
):
    try:
        # 파일 처리 및 임베딩을 비동기로 실행
        docsearch, num_chunks = await asyncio.to_thread(
            text_gen.process_and_embed_files,
            files
        )
        return {"status": "success", "num_chunks": num_chunks}
    except Exception as e:
        print(f"Upload error: {str(e)}")  # 디버깅을 위한 로그 추가
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate")
async def generate_text(
    request: QueryRequest,
    api_key: str = Depends(get_api_key)
):
    try:
        # 벡터 저장소 체크 제거
        response = await asyncio.to_thread(
            text_gen.gpt_text_generation,
            request.system_prompt,
            request.assistant_content,
            request.user_query
        )
        
        if not response:
            raise HTTPException(
                status_code=500,
                detail="텍스트 생성 중 오류가 발생했습니다"
            )
            
        return {
            "text": response[0],
            "prompt_tokens": response[1],
            "completion_tokens": response[2],
            "usd_cost": response[3],
            "kor_cost": response[4],
            "exchange_rate": response[5]
        }
    except Exception as e:
        print(f"Generation error: {str(e)}")  # 디버깅을 위한 로그 추가
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/process-screen")
async def process_screen(data: ImageData):
    try:
        result = await screen_processor.process_image(data.image)
        if 'error' in result:
            return JSONResponse(
                status_code=200,
                content={
                    'success': False,
                    'error': result['error'],
                    'objects': [],
                    'text': '',
                    'should_click': False
                }
            )
        return JSONResponse(
            status_code=200,
            content={
                'success': True,
                **result
            }
        )
    except Exception as e:
        logger.error(f"API 오류: {str(e)}")
        return JSONResponse(
            status_code=200,
            content={
                'success': False,
                'error': str(e),
                'objects': [],
                'text': '',
                'should_click': False
            }
        )

@app.post("/api/crawl")
async def crawl(request: CrawlRequest):
    try:
        text = crawl_webpage(request.url)
        return {"text": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 