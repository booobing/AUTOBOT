import React, { useState } from 'react';

function VoiceRecognition() {
  const [isRecording, setIsRecording] = useState(false);
  const [transcript, setTranscript] = useState('');
  const [showPopup, setShowPopup] = useState(false);

  const startRecording = () => {
    if ('webkitSpeechRecognition' in window) {
      const recognition = new window.webkitSpeechRecognition();
      recognition.lang = 'ko-KR';
      recognition.continuous = false;
      recognition.interimResults = false;

      recognition.onstart = () => {
        setIsRecording(true);
      };

      recognition.onresult = (event) => {
        const text = event.results[0][0].transcript;
        setTranscript(text);
        setShowPopup(true);
      };

      recognition.onerror = (event) => {
        console.error('음성 인식 오류:', event.error);
        setIsRecording(false);
      };

      recognition.onend = () => {
        setIsRecording(false);
      };

      recognition.start();
    } else {
      alert('이 브라우저는 음성 인식을 지원하지 않습니다.');
    }
  };

  return (
    <div className="voice-recognition">
      <h1>음성 인식</h1>
      <button 
        onClick={startRecording}
        disabled={isRecording}
      >
        {isRecording ? '녹음 중...' : '녹음 시작'}
      </button>

      {showPopup && (
        <div className="popup">
          <div className="popup-content">
            <h2>녹음된 텍스트</h2>
            <p>{transcript}</p>
            <button onClick={() => setShowPopup(false)}>닫기</button>
          </div>
        </div>
      )}
    </div>
  );
}

export default VoiceRecognition; 