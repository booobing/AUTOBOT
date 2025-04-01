import React, { useState, useEffect, useRef } from 'react';
import * as tf from '@tensorflow/tfjs';
import * as cocoSsd from '@tensorflow-models/coco-ssd';
import { createWorker } from 'tesseract.js';

function ScreenCapture() {
  const [isRunning, setIsRunning] = useState(false);
  const [lastDetection, setLastDetection] = useState('');
  const [currentText, setCurrentText] = useState('');
  const [textHistory, setTextHistory] = useState([]);
  const [model, setModel] = useState(null);
  const [ocrWorker, setOcrWorker] = useState(null);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const historyRef = useRef(null);
  const [cropSettings, setCropSettings] = useState({
    x: 25, // percentage
    y: 25,
    width: 50,
    height: 50
  });

  useEffect(() => {
    // COCO-SSD 모델과 Tesseract 워커 로드
    const loadModels = async () => {
      try {
        // COCO-SSD 모델 로드
        const loadedModel = await cocoSsd.load();
        setModel(loadedModel);
        console.log('객체 감지 모델 로드 완료');

        // Tesseract 워커 초기화
        const worker = await createWorker();
        await worker.loadLanguage('kor+eng');
        await worker.initialize('kor+eng');
        setOcrWorker(worker);
        console.log('OCR 워커 초기화 완료');
      } catch (error) {
        console.error('모델 로드 실패:', error);
      }
    };
    loadModels();

    // 화면 해상도에 맞게 캔버스 크기 설정
    const updateCanvasSize = () => {
      if (canvasRef.current && videoRef.current) {
        canvasRef.current.width = videoRef.current.videoWidth || 1920;
        canvasRef.current.height = videoRef.current.videoHeight || 1080;
      }
    };

    window.addEventListener('resize', updateCanvasSize);
    return () => window.removeEventListener('resize', updateCanvasSize);
  }, []);

  useEffect(() => {
    // 새 텍스트가 추가될 때 스크롤을 아래로 이동
    if (historyRef.current) {
      historyRef.current.scrollTop = historyRef.current.scrollHeight;
    }
  }, [textHistory]);

  const startCapture = async () => {
    try {
      const stream = await navigator.mediaDevices.getDisplayMedia({
        video: { 
          cursor: 'always',
          displaySurface: 'monitor',
          logicalSurface: true,
          width: { ideal: 1920 },
          height: { ideal: 1080 },
          frameRate: { ideal: 30 }
        },
        audio: false
      });

      videoRef.current.srcObject = stream;
      streamRef.current = stream;
      
      videoRef.current.onloadedmetadata = () => {
        if (canvasRef.current) {
          canvasRef.current.width = videoRef.current.videoWidth;
          canvasRef.current.height = videoRef.current.videoHeight;
        }
      };
      
      setIsRunning(true);
      setTextHistory([]); // 캡처 시작시 히스토리 초기화
    } catch (error) {
      console.error('화면 공유 오류:', error);
      setLastDetection('화면 공유 권한이 필요합니다.');
    }
  };

  const stopCapture = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
    }
    videoRef.current.srcObject = null;
    setIsRunning(false);
  };

  const processFrame = async () => {
    if (!videoRef.current || !canvasRef.current || !isRunning) return;

    try {
      const context = canvasRef.current.getContext('2d');
      context.imageSmoothingEnabled = true;
      context.imageSmoothingQuality = 'high';
      
      context.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
      context.drawImage(
        videoRef.current,
        0,
        0,
        canvasRef.current.width,
        canvasRef.current.height
      );
      
      // 퍼센트 값을 실제 픽셀 값으로 변환
      const cropX = canvasRef.current.width * (cropSettings.x / 100);
      const cropY = canvasRef.current.height * (cropSettings.y / 100);
      const cropWidth = canvasRef.current.width * (cropSettings.width / 100);
      const cropHeight = canvasRef.current.height * (cropSettings.height / 100);
      
      // 크롭된 영역만 새로운 캔버스에 그리기
      const tempCanvas = document.createElement('canvas');
      tempCanvas.width = cropWidth;
      tempCanvas.height = cropHeight;
      const tempContext = tempCanvas.getContext('2d');
      
      tempContext.drawImage(
        canvasRef.current,
        cropX, cropY, cropWidth, cropHeight,  // 원본에서 크롭할 영역
        0, 0, cropWidth, cropHeight  // 새 캔버스에 그릴 영역
      );
      
      // 크롭된 이미지를 base64로 변환
      const imageData = tempCanvas.toDataURL('image/jpeg', 1.0);
      
      const response = await fetch('http://localhost:8000/api/process-screen', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 
          image: imageData,
          crop: {
            x: cropX,
            y: cropY,
            width: cropWidth,
            height: cropHeight
          }
        })
      });

      if (!response.ok) throw new Error('서버 응답 오류');
      
      const result = await response.json();
      
      // 현재 텍스트 업데이트
      setCurrentText(result.current_text);
      
      // 텍스트 히스토리 업데이트
      if (result.text_history) {
        setTextHistory(result.text_history);
      }

      // 클릭 이벤트 발생 시 표시
      if (result.should_click) {
        setLastDetection('건너뛰기 버튼 클릭됨');
      }

    } catch (error) {
      console.error('프레임 처리 오류:', error);
      setLastDetection('오류 발생');
    }
  };

  useEffect(() => {
    let interval;
    
    if (isRunning) {
      interval = setInterval(processFrame, 5000); // 5초로 변경
    }

    return () => {
      if (interval) {
        clearInterval(interval);
      }
    };
  }, [isRunning]);

  const handleCropChange = (setting, value) => {
    setCropSettings(prev => {
      const newSettings = { ...prev, [setting]: Number(value) };
      
      // 유효성 검사
      if (newSettings.x + newSettings.width > 100) {
        newSettings.width = 100 - newSettings.x;
      }
      if (newSettings.y + newSettings.height > 100) {
        newSettings.height = 100 - newSettings.y;
      }
      
      return newSettings;
    });
  };

  return (
    <div className="screen-capture">
      <h1>화면 캡처 및 텍스트 인식</h1>
      <button 
        onClick={isRunning ? stopCapture : startCapture}
        className={isRunning ? 'running' : ''}
      >
        {isRunning ? '중지' : '시작'}
      </button>
      
      <div className="status">
        <p>상태: {isRunning ? '실행 중' : '중지됨'}</p>
        <p>마지막 이벤트: {lastDetection}</p>
        
        <div className="text-recognition">
          <h3>현재 인식된 텍스트:</h3>
          <pre className="current-text">{currentText}</pre>
          
          <h3>텍스트 히스토리:</h3>
          <div className="text-history" ref={historyRef}>
            {textHistory.map((item, index) => (
              <div key={index} className="history-item">
                <span className="timestamp">{item.timestamp}</span>
                <pre className="history-text">{item.text}</pre>
              </div>
            ))}
          </div>
        </div>
      </div>
      
      <div className="crop-controls" style={{
        margin: '20px 0',
        padding: '15px',
        backgroundColor: '#f5f5f5',
        borderRadius: '8px'
      }}>
        <h3>크롭 영역 설정</h3>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '10px' }}>
          <div>
            <label>X 위치 (%)</label>
            <input
              type="range"
              min="0"
              max="90"
              value={cropSettings.x}
              onChange={(e) => handleCropChange('x', e.target.value)}
            />
            <span>{cropSettings.x}%</span>
          </div>
          <div>
            <label>Y 위치 (%)</label>
            <input
              type="range"
              min="0"
              max="90"
              value={cropSettings.y}
              onChange={(e) => handleCropChange('y', e.target.value)}
            />
            <span>{cropSettings.y}%</span>
          </div>
          <div>
            <label>너비 (%)</label>
            <input
              type="range"
              min="10"
              max={100 - cropSettings.x}
              value={cropSettings.width}
              onChange={(e) => handleCropChange('width', e.target.value)}
            />
            <span>{cropSettings.width}%</span>
          </div>
          <div>
            <label>높이 (%)</label>
            <input
              type="range"
              min="10"
              max={100 - cropSettings.y}
              value={cropSettings.height}
              onChange={(e) => handleCropChange('height', e.target.value)}
            />
            <span>{cropSettings.height}%</span>
          </div>
        </div>
      </div>

      <div className="capture-area">
        <video 
          ref={videoRef} 
          style={{ display: 'none' }} 
          autoPlay 
          playsInline
        />
        <div className="canvas-container" style={{ position: 'relative' }}>
          <canvas 
            ref={canvasRef}
            style={{ 
              maxWidth: '100%',
              height: 'auto',
              border: '1px solid #ccc',
              boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
            }}
          />
          <div 
            className="crop-overlay"
            style={{
              position: 'absolute',
              top: `${cropSettings.y}%`,
              left: `${cropSettings.x}%`,
              width: `${cropSettings.width}%`,
              height: `${cropSettings.height}%`,
              border: '2px dashed red',
              pointerEvents: 'none',
              backgroundColor: 'rgba(255, 0, 0, 0.1)'
            }}
          />
        </div>
      </div>
    </div>
  );
}

export default ScreenCapture;