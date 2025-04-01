import React, { useState } from 'react';
import axios from 'axios';

function WebCrawler() {
  const [url, setUrl] = useState('');
  const [crawledText, setCrawledText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  const handleCrawl = async () => {
    if (!url) {
      setError('URL을 입력해주세요.');
      return;
    }

    setIsLoading(true);
    setError('');
    setCrawledText('');

    try {
      const response = await axios.post('http://localhost:8000/api/crawl', { url });
      if (response.data.text) {
        setCrawledText(response.data.text);
      } else {
        setError('크롤링된 텍스트가 없습니다.');
      }
    } catch (err) {
      setError('크롤링 중 오류가 발생했습니다: ' + (err.response?.data?.detail || err.message));
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="web-crawler">
      <div className="crawler-section">
        <h2>웹 크롤러</h2>
        
        <div className="input-group">
          <input
            type="url"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            placeholder="크롤링할 웹사이트 URL을 입력하세요 (예: https://www.example.com)"
            className="url-input"
          />
          <button 
            onClick={handleCrawl}
            disabled={isLoading}
            className={isLoading ? 'loading' : ''}
          >
            {isLoading ? '크롤링 중...' : '크롤링 시작'}
          </button>
        </div>

        {error && <div className="error-message">{error}</div>}

        {crawledText && (
          <div className="result-section">
            <h3>크롤링 결과</h3>
            <div className="crawled-content">
              <pre>{crawledText}</pre>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default WebCrawler; 