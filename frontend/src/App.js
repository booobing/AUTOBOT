import React from 'react';
import { HashRouter as Router, Routes, Route, Link } from 'react-router-dom';
import Home from './pages/Home';
import Chat from './pages/Chat';
import TextGeneration from './pages/TextGeneration';
import SimpleChat from './pages/SimpleChat';
import VoiceRecognition from './pages/VoiceRecognition';
import ScreenCapture from './pages/ScreenCapture';
import WebCrawler from './pages/WebCrawler';
import './App.css';

function App() {
  return (
    <Router>
      <div className="App">
        <nav>
          <ul>
            <li><Link to="/">홈</Link></li>
            <li><Link to="/chat">채팅</Link></li>
            <li><Link to="/simple-chat">간단 채팅</Link></li>
            <li><Link to="/text-generation">텍스트 생성</Link></li>
            <li><Link to="/voice-recognition">음성 인식</Link></li>
            <li><Link to="/screen-capture">화면 캡처</Link></li>
            <li><Link to="/web-crawler">웹 크롤러</Link></li>
          </ul>
        </nav>

        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/chat" element={<Chat />} />
          <Route path="/simple-chat" element={<SimpleChat />} />
          <Route path="/text-generation" element={<TextGeneration />} />
          <Route path="/voice-recognition" element={<VoiceRecognition />} />
          <Route path="/screen-capture" element={<ScreenCapture />} />
          <Route path="/web-crawler" element={<WebCrawler />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;