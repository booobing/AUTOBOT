import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from openai import OpenAI
from generator.generator import Generator

class TextGenerator:
    def __init__(self, model="gpt-4o", temperature=0.2):
        """TextGenerator 초기화"""
        try:
            # .env 파일에서 환경 변수 로드
            load_dotenv()
            self.openai_api_key = os.getenv("OPENAI_API_KEY")
            
            if not self.openai_api_key:
                raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")
            
            # Generator 인스턴스 생성
            self.generator = Generator(
                model=model,
                temperature=temperature
            )
            
        except Exception as e:
            raise Exception(f"TextGenerator 초기화 중 오류 발생: {str(e)}")

    def generate_content(self, instruction_number, input_data, lang="en"):
        """텍스트 생성 함수"""
        try:
            # generate_with_numbered_instruction 사용
            answer = self.generator.generate_with_numbered_instruction(
                instruction_number=instruction_number,
                input_data=input_data,
                lang=lang
            )
            
            # OpenAI API 직접 호출하여 토큰 정보 얻기
            client = OpenAI(api_key=self.openai_api_key)
            api_response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": str(input_data)}]
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
            
            return {
                "answer": answer,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "usd_cost": usd_cost,
                "kor_cost": kor_cost,
                "exchange_rate": exchange_rate
            }
            
        except Exception as e:
            print(f"텍스트 생성 중 오류 발생: {e}")
            raise