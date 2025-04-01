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
      
      // 화면 전체를 캡처하도록 보장
      context.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
      context.drawImage(
        videoRef.current,
        0,
        0,
        canvasRef.current.width,
        canvasRef.current.height
      );
      
      // 이미지 품질을 최대로 설정
      const imageData = canvasRef.current.toDataURL('image/jpeg', 1.0);
      
      const response = await fetch('http://localhost:8000/api/process-screen', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ image: imageData })
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
      
      <div className="capture-area">
        <video 
          ref={videoRef} 
          style={{ display: 'none' }} 
          autoPlay 
          playsInline
        />
        <canvas 
          ref={canvasRef}
          style={{ 
            maxWidth: '100%',
            height: 'auto',
            border: '1px solid #ccc',
            boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
          }}
        />
      </div>
    </div>
  );
}

export default ScreenCapture;