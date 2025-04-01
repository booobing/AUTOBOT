import React, { useState } from 'react';
import axios from 'axios';

const API_KEY = process.env.REACT_APP_API_KEY;

const api = axios.create({
  headers: {
    'X-API-Key': API_KEY
  }
});

function TextGeneration() {
  const [files, setFiles] = useState([]);
  const [systemPrompt, setSystemPrompt] = useState('');
  const [assistantContent, setAssistantContent] = useState('');
  const [userQuery, setUserQuery] = useState('');
  const [response, setResponse] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleFileUpload = async (e) => {
    const formData = new FormData();
    for (let file of e.target.files) {
      formData.append('files', file);
    }

    setLoading(true);
    setError(null);

    try {
      const response = await api.post('http://localhost:8000/api/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      setFiles([...e.target.files]);
      alert(`${response.data.num_chunks} chunks processed successfully!`);
    } catch (err) {
      setError(err.response?.data?.detail || 'Error uploading files');
    } finally {
      setLoading(false);
    }
  };

  const handleGenerate = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await api.post('http://localhost:8000/api/generate', {
        system_prompt: systemPrompt || "You are a helpful assistant.",
        assistant_content: assistantContent || "I am ready to help you.",
        user_query: userQuery,
      });
      setResponse(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || 'Error generating text');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="text-generation">
      <section className="upload-section">
        <h2>파일 업로드</h2>
        <input
          type="file"
          multiple
          onChange={handleFileUpload}
          accept=".txt,.csv,.xlsx,.xls"
        />
        {files.length > 0 && (
          <div className="file-list">
            <h3>업로드된 파일:</h3>
            <ul>
              {Array.from(files).map((file, index) => (
                <li key={index}>{file.name}</li>
              ))}
            </ul>
          </div>
        )}
      </section>

      <section className="generate-section">
        <h2>텍스트 생성</h2>
        <div className="input-group">
          <textarea
            placeholder="시스템 프롬프트"
            value={systemPrompt}
            onChange={(e) => setSystemPrompt(e.target.value)}
          />
          <textarea
            placeholder="어시스턴트 내용"
            value={assistantContent}
            onChange={(e) => setAssistantContent(e.target.value)}
          />
          <textarea
            placeholder="사용자 질문"
            value={userQuery}
            onChange={(e) => setUserQuery(e.target.value)}
          />
          <button onClick={handleGenerate} disabled={loading}>
            {loading ? '생성 중...' : '생성하기'}
          </button>
        </div>
      </section>

      {error && <div className="error-message">{error}</div>}

      {response && (
        <section className="response-section">
          <h2>생성된 응답</h2>
          <div className="response-content">
            <p>{response.text}</p>
            <div className="cost-info">
              <p>프롬프트 토큰: {response.prompt_tokens}</p>
              <p>완성 토큰: {response.completion_tokens}</p>
              <p>비용: ${response.usd_cost.toFixed(4)} (₩{response.kor_cost.toFixed(2)})</p>
            </div>
          </div>
        </section>
      )}
    </div>
  );
}

export default TextGeneration; 