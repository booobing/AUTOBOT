#!/bin/bash

# 백엔드 설정
echo "백엔드 설정 시작..."
python -m venv venv
source venv/bin/activate  # Windows의 경우: venv\Scripts\activate
pip install -r backend/requirements.txt

# 프론트엔드 설정
echo "프론트엔드 설정 시작..."
cd frontend
npm install

echo "설정 완료!" 