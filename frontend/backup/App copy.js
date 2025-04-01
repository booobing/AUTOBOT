import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

// API 키 설정
const API_KEY = process.env.REACT_APP_API_KEY;

// axios 설정
const api = axios.create({
  headers: {
    'X-API-Key': API_KEY
  }
});

function App() {
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
    <div className="App">
      <header className="App-header">
        <h1>Text Generation System</h1>
      </header>
      
      <main className="App-main">
        <section className="upload-section">
          <h2>Upload Files</h2>
          <input
            type="file"
            multiple
            onChange={handleFileUpload}
            accept=".txt,.csv,.xlsx,.xls"
          />
          {files.length > 0 && (
            <div className="file-list">
              <h3>Uploaded Files:</h3>
              <ul>
                {Array.from(files).map((file, index) => (
                  <li key={index}>{file.name}</li>
                ))}
              </ul>
            </div>
          )}
        </section>

        <section className="generate-section">
          <h2>Generate Text</h2>
          <div className="input-group">
            <textarea
              placeholder="System Prompt"
              value={systemPrompt}
              onChange={(e) => setSystemPrompt(e.target.value)}
            />
            <textarea
              placeholder="Assistant Content"
              value={assistantContent}
              onChange={(e) => setAssistantContent(e.target.value)}
            />
            <textarea
              placeholder="User Query"
              value={userQuery}
              onChange={(e) => setUserQuery(e.target.value)}
            />
            <button onClick={handleGenerate} disabled={loading}>
              {loading ? 'Generating...' : 'Generate'}
            </button>
          </div>
        </section>

        {error && <div className="error-message">{error}</div>}

        {response && (
          <section className="response-section">
            <h2>Generated Response</h2>
            <div className="response-content">
              <p>{response.text}</p>
              <div className="cost-info">
                <p>Prompt Tokens: {response.prompt_tokens}</p>
                <p>Completion Tokens: {response.completion_tokens}</p>
                <p>Cost: ${response.usd_cost.toFixed(4)} (₩{response.kor_cost.toFixed(2)})</p>
              </div>
            </div>
          </section>
        )}
      </main>
    </div>
  );
}

export default App; 