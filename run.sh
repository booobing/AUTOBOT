#!/bin/bash

# 백엔드 실행
echo "백엔드 서버 시작..."
source deb-env/bin/activate  # Windows의 경우: dev-env\Scripts\activate
cd backend
python main.py &

# 프론트엔드 실행
echo "프론트엔드 서버 시작..."
cd ../frontend
npm start 