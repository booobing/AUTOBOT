import os
from dotenv import load_dotenv
load_dotenv(verbose=False)

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_anthropic import ChatAnthropic
from langchain_google_vertexai import ChatVertexAI

import openai
import json
from openai import OpenAI

instruction_path = './instructions/'
instruction_file = "instruction.txt"
instruction_upgrade_file = "instruction_upgrade.txt"

GPT4_O_ARGS = {
    "model_name": "gpt-4o",
    "api_key": os.getenv("OPENAI_API_KEY"),
    "max_tokens": 4096,
}

instruction_list_2 = []
instruction_list = os.listdir(instruction_path) 
for insturction in instruction_list:
    if insturction.split('_')[0].isdecimal():
        instruction_list_2.append(insturction)
        
instruction_files = sorted(instruction_list_2, key=lambda x: int(x.split('_')[0]))

# open instruction files to list
instructions = []
for instruction_file in instruction_files:
    with open(instruction_path + instruction_file, 'r', encoding='utf-8') as f:
        instructions.append(f.read())

class Generator:
    def __init__(self, 
                 model:str = "gpt-4o",
                 temperature:float = 0.2,
                 function:int = 0,
                 env_file:str = ".env",
                 base_instruction_file:str = instruction_path+instruction_file,
                 upgrade_instruction_file:str = instruction_path+instruction_upgrade_file):
        """
        Generator 클래스 초기화
        
        Args:
            model: 사용할 모델 (기본값: "gpt-4o")
            temperature: 모델 온도 설정 (기본값: 0.2)
            function: 사용할 함수 인덱스 (기본값: 0)
            env_file: API 키가 저장된 환경 변수 파일 경로
            base_instruction_file: 기본 시스템 프롬프트가 저장된 파일 경로
            upgrade_instruction_file: 업그레이드된 시스템 프롬프트가 저장된 파일 경로
        """
        # set model, llm
        self.model = model
        self.temperature = temperature
        self.function = function
        
        # .env 파일 로드
        load_dotenv(env_file)
        
        # OpenAI API 키 설정
        openai.api_key = os.getenv("OPENAI_API_KEY")
        
        # OpenAI 클라이언트 초기화
        self.client = OpenAI()
        
        # 시스템 프롬프트 파일 경로 저장
        self.base_instruction_file = base_instruction_file
        self.upgrade_instruction_file = upgrade_instruction_file
        
        # 시스템 프롬프트 로드
        self.system_prompt = self._load_instruction(self.base_instruction_file)
        self.system_prompt_upgrade = self._load_instruction(self.upgrade_instruction_file)
        
        # Langchain 모델 설정
        if model == "gpt-4o":
            self.llm = ChatOpenAI(**GPT4_O_ARGS, temperature=temperature)
        # 필요한 경우 다른 모델 설정 추가
    
    def _load_instruction(self, instruction_file, lang="en"):
        """
        시스템 프롬프트 파일을 로드하는 함수
        
        Args:
            instruction_file: 시스템 프롬프트가 저장된 파일 경로
            lang: 언어 코드 (기본값: "en")
            
        Returns:
            str: 로드된 시스템 프롬프트
        """
        try:
            with open(instruction_file, 'r', encoding='utf-8') as f:
                instruction = f.read()
                
            # 언어 코드에 따라 적절한 언어로 변환
            if lang == "ko":
                language = "한국어"
            elif lang == "en":
                language = "English"
            else:
                language = "English"  # 기본값
                
            # {language} 변수 치환
            instruction = instruction.replace("{language}", language)
            
            return instruction
        except Exception as e:
            print(f"Error loading instruction file: {e}")
            return ""
    
    def generate_content(self, input_data, lang="en", instruction_type="default"):
        """
        입력 데이터를 기반으로 콘텐츠를 생성하는 함수
        
        Args:
            input_data: 입력 데이터 (딕셔너리)
            lang: 언어 코드 (기본값: "en")
            instruction_type: 사용할 시스템 프롬프트 유형 (기본값: "default")
            
        Returns:
            str: 생성된 콘텐츠
        """
        try:
            # 사용할 시스템 프롬프트 선택
            if instruction_type == "upgrade":
                system_prompt = self.system_prompt_upgrade
            else:
                system_prompt = self.system_prompt
                
            # {language} 변수가 치환되지 않았다면 여기서 치환
            if "{language}" in system_prompt:
                if lang == "ko":
                    language = "한국어"
                elif lang == "en":
                    language = "English"
                else:
                    language = "English"  # 기본값
                    
                system_prompt = system_prompt.replace("{language}", language)
            
            # 입력 데이터의 각 값에 대해 시스템 프롬프트에서 변수 치환
            for key, value in input_data.items():
                placeholder = "{" + key + "}"
                if placeholder in system_prompt:
                    system_prompt = system_prompt.replace(placeholder, str(value))
            
            # Langchain 모델 사용 여부에 따라 다른 방식으로 콘텐츠 생성
            if hasattr(self, 'llm') and self.model.startswith("chatopenai"):
                # Langchain 사용
                prompt = PromptTemplate.from_template(system_prompt)
                chain = prompt | self.llm
                result = chain.invoke({"input": "Generate content based on the provided instructions."})
                return result.content
            else:
                # OpenAI API 직접 사용
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": "Generate content based on the provided instructions."}
                    ],
                    temperature=0.7,
                    max_tokens=4096
                )
                
                return response.choices[0].message.content
        except Exception as e:
            print(f"Error generating content: {e}")
            return f"Error: Failed to generate content. {e}"
    
    def generate_with_numbered_instruction(self, instruction_number, input_data, lang="en"):
        """
        특정 번호의 지시문을 사용하여 콘텐츠를 생성하는 함수
        
        Args:
            instruction_number: 사용할 지시문 번호 (0부터 시작)
            input_data: 입력 데이터 (딕셔너리)
            lang: 언어 코드 (기본값: "en")
            
        Returns:
            str: 생성된 콘텐츠
        """
        try:
            if instruction_number < 0 or instruction_number >= len(instructions):
                return "Error: Invalid instruction number."
            
            # 지시문 선택
            instruction = instructions[instruction_number]
            
            # 언어 코드에 따라 적절한 언어로 변환
            if lang == "ko":
                language = "한국어"
            elif lang == "en":
                language = "English"
            else:
                language = "English"  # 기본값
                
            # {language} 변수 치환
            instruction = instruction.replace("{language}", language)
            
            # 입력 데이터의 각 값에 대해 지시문에서 변수 치환
            for key, value in input_data.items():
                placeholder = "{" + key + "}"
                if placeholder in instruction:
                    instruction = instruction.replace(placeholder, str(value))
            
            # Langchain 모델 사용 여부에 따라 다른 방식으로 콘텐츠 생성
            if hasattr(self, 'llm') and self.model.startswith("chatopenai"):
                # Langchain 사용
                prompt = PromptTemplate.from_template(instruction)
                chain = prompt | self.llm
                result = chain.invoke({"input": "Generate content based on the provided instructions."})
                return result.content
            else:
                # OpenAI API 직접 사용
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": instruction},
                        {"role": "user", "content": "Generate content based on the provided instructions."}
                    ],
                    temperature=0.7,
                    max_tokens=4096
                )
                
                return response.choices[0].message.content
        except Exception as e:
            print(f"Error generating content: {e}")
            return f"Error: Failed to generate content. {e}"