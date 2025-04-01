import React, { useState } from 'react';
import axios from 'axios';

const API_KEY = process.env.REACT_APP_API_KEY;

const api = axios.create({
  headers: {
    'X-API-Key': API_KEY
  }
});

function Chat() {
  const [userQuery, setUserQuery] = useState('');
  const [response, setResponse] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [chatHistory, setChatHistory] = useState([]);
  const [files, setFiles] = useState([]);

  // 시스템과 어시스턴트 프롬프트 미리 정의
  const systemPrompt = "당신은 친절하고 도움이 되는 AI 어시스턴트입니다. 사용자의 질문에 명확하고 정확하게 답변해주세요.";
  const assistantContent = "네, 어떤 도움이 필요하신가요?";

  const handleGenerate = async () => {
    if (!userQuery.trim()) return;

    setLoading(true);
    setError(null);

    try {
      const response = await api.post('http://localhost:8000/api/generate', {
        system_prompt: systemPrompt,
        assistant_content: assistantContent,
        user_query: userQuery,
      });

      setChatHistory(prev => [...prev, 
        { type: 'user', content: userQuery },
        { type: 'assistant', content: response.data.text }
      ]);

      setResponse(response.data);
      setUserQuery('');
    } catch (err) {
      setError(err.response?.data?.detail || '채팅 생성 중 오류가 발생했습니다');
    } finally {
      setLoading(false);
    }
  };

  const handleFileUpload = (event) => {
    const uploadedFiles = Array.from(event.target.files);
    setFiles(prev => [...prev, ...uploadedFiles]);
  };

  const handleFileRemove = (fileName) => {
    setFiles(files.filter(file => file.name !== fileName));
  };

  return (
    <div className="text-generation">
      <section className="upload-section">
        <h2>파일 업로드</h2>
        <div className="input-group">
          <input
            type="file"
            multiple
            onChange={handleFileUpload}
            accept=".txt,.pdf,.doc,.docx"
          />
          {files.length > 0 && (
            <div className="file-list">
              <h3>업로드된 파일</h3>
              <ul>
                {files.map((file, index) => (
                  <li key={index}>
                    {file.name}
                    <button 
                      onClick={() => handleFileRemove(file.name)}
                      style={{ marginLeft: '10px' }}
                    >
                      삭제
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      </section>

      <section className="generate-section">
        <h2>채팅</h2>
        <div className="response-content">
          {chatHistory.map((message, index) => (
            <div key={index} className={`message ${message.type}`}>
              <p>{message.content}</p>
            </div>
          ))}
        </div>

        <div className="input-group">
          <textarea
            placeholder="메시지를 입력하세요..."
            value={userQuery}
            onChange={(e) => setUserQuery(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleGenerate();
              }
            }}
          />
          <button onClick={handleGenerate} disabled={loading}>
            {loading ? '생성 중...' : '전송'}
          </button>
        </div>
      </section>

      {error && <div className="error-message">{error}</div>}

      {response && (
        <section className="response-section">
          <div className="cost-info">
            <p>프롬프트 토큰: {response.prompt_tokens}</p>
            <p>완성 토큰: {response.completion_tokens}</p>
            <p>비용: ${response.usd_cost.toFixed(4)} (₩{response.kor_cost.toFixed(2)})</p>
          </div>
        </section>
      )}
    </div>
  );
}

export default Chat; 