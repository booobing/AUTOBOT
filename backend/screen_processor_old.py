import cv2
import numpy as np
import pytesseract
import pyautogui
import base64
import torch
from torchvision.models import detection
from PIL import Image
import io
import logging
import os
from datetime import datetime
import easyocr

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ScreenProcessor:
    def __init__(self):
        try:
            # YOLO 모델 로드
            self.model = detection.fasterrcnn_resnet50_fpn(pretrained=True)
            self.model.eval()
            
            # GPU 사용 가능시 GPU 사용
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = self.model.to(self.device)
            
            # Tesseract 설정
            if os.name == 'nt':  # Windows
                tesseract_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
                if os.path.exists(tesseract_path):
                    pytesseract.pytesseract.tesseract_cmd = tesseract_path
                else:
                    raise Exception("Tesseract가 설치되어 있지 않습니다.")
            else:  # Linux/Mac
                if not pytesseract.get_tesseract_version():
                    raise Exception("Tesseract가 설치되어 있지 않습니다.")
                
            # EasyOCR 리더 초기화
            self.reader = easyocr.Reader(['ko', 'en'], gpu=torch.cuda.is_available())
            
            logger.info("ScreenProcessor 초기화 완료")
        except Exception as e:
            logger.error(f"초기화 중 오류 발생: {e}")
            raise
        
    async def process_image(self, image_base64):
        try:
            # base64 문자열 정제
            if ',' in image_base64:
                image_base64 = image_base64.split(',')[1]
            
            # base64 이미지를 numpy 배열로 변환
            image_data = base64.b64decode(image_base64)
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                raise ValueError("이미지 디코딩 실패")
            
            # BGR을 RGB로 변환
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # OCR 텍스트 인식 시도
            try:
                text = await self.recognize_text(image_rgb)
                timestamp = datetime.now().strftime("%H:%M:%S")
                
                # 텍스트가 있을 경우에만 히스토리 업데이트
                if text.strip():
                    self.text_history = [{
                        'timestamp': timestamp,
                        'text': text.strip()
                    }]  # 최근 1개만 저장
                
            except Exception as e:
                logger.error(f"텍스트 인식 실패: {e}")
                text = ""
            
            # "건너뛰기" 텍스트 확인
            should_click = False
            if text and ('건너뛰기' in text.lower() or 'skip' in text.lower()):
                should_click = True
                await self.move_and_click()
            
            return {
                'current_text': text,
                'text_history': self.text_history,
                'should_click': should_click
            }
            
        except Exception as e:
            logger.error(f"이미지 처리 오류: {str(e)}")
            return {
                'current_text': '',
                'text_history': [],
                'should_click': False,
                'error': str(e)
            }
    
    async def detect_objects(self, image):
        try:
            # 이미지 크기 조정
            image = cv2.resize(image, (1280, 960))
            
            # 이미지를 PyTorch 텐서로 변환
            image_tensor = torch.from_numpy(image).float() / 255.0
            image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)
            image_tensor = image_tensor.to(self.device)
            
            with torch.no_grad():
                predictions = self.model(image_tensor)
            
            results = []
            if len(predictions) > 0 and len(predictions[0]['boxes']) > 0:
                for score, label, box in zip(
                    predictions[0]['scores'], 
                    predictions[0]['labels'], 
                    predictions[0]['boxes']
                ):
                    if score > 0.5:  # 신뢰도 임계값
                        results.append({
                            'class': label.item(),
                            'confidence': float(score.item()),
                            'box': box.cpu().tolist()
                        })
            
            return results
            
        except Exception as e:
            logger.error(f"객체 감지 오류: {str(e)}")
            return []
    
    async def recognize_text(self, image):
        try:
            # 이미지 크기 조정 (성능 향상)
            height, width = image.shape[:2]
            new_width = 800
            new_height = int(height * (new_width / width))
            image = cv2.resize(image, (new_width, new_height))
            
            # 이미지 전처리
            # 대비 향상
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            cl = clahe.apply(l)
            enhanced = cv2.merge((cl,a,b))
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
            
            # EasyOCR로 텍스트 인식
            results = self.reader.readtext(enhanced)
            
            # 신뢰도 점수가 0.5 이상인 텍스트만 추출
            texts = []
            for (bbox, text, prob) in results:
                if prob > 0.5:  # 신뢰도 임계값
                    texts.append(text)
            
            return ' '.join(texts)
            
        except Exception as e:
            logger.error(f"텍스트 인식 오류: {str(e)}")
            return ''
    
    async def move_and_click(self):
        try:
            # 화면 크기 확인
            screen_width, screen_height = pyautogui.size()
            
            # 좌표가 화면 내에 있는지 확인
            x, y = 900, 600
            if x < screen_width and y < screen_height:
                pyautogui.moveTo(x, y, duration=0.2)
                pyautogui.click()
                logger.info(f"클릭 수행: ({x}, {y})")
            else:
                logger.warning("클릭 좌표가 화면을 벗어남")
                
        except Exception as e:
            logger.error(f"마우스 제어 오류: {str(e)}") 