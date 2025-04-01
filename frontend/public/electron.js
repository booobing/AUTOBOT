const { app, BrowserWindow, dialog, ipcMain } = require('electron');
const path = require('path');

// isDev 체크 함수
const isDev = process.env.NODE_ENV === 'development';

let mainWindow = null;

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false,
      enableRemoteModule: true,
      webSecurity: !isDev,  // 개발 모드에서는 CORS 비활성화
      sandbox: false        // 샌드박스 비활성화
    }
  });

  // 로드할 URL 설정
  const startUrl = isDev 
    ? 'http://localhost:3000' 
    : `file://${path.join(__dirname, './index.html')}`;

  mainWindow.loadURL(startUrl);

  // IPC 통신 설정
  ipcMain.on('text-input', (event, data) => {
    // 텍스트 입력 처리
    event.reply('text-input-reply', { success: true });
  });

  // 시스템 에러 처리
  ipcMain.on('system-error', (event, error) => {
    dialog.showErrorBox('시스템 오류', error.message);
  });

  // 렌더러 프로세스 충돌 처리
  mainWindow.webContents.on('crashed', () => {
    const options = {
      type: 'error',
      title: '프로세스 충돌',
      message: '애플리케이션이 충돌했습니다.',
      buttons: ['재시작', '종료']
    };

    dialog.showMessageBox(options).then(({ response }) => {
      if (response === 0) {
        app.relaunch();
        app.exit(0);
      } else {
        app.exit(1);
      }
    });
  });

  // 응답 없음 처리
  mainWindow.on('unresponsive', () => {
    const options = {
      type: 'warning',
      title: '응답 없음',
      message: '애플리케이션이 응답하지 않습니다.',
      buttons: ['강제 종료', '대기']
    };

    dialog.showMessageBox(options).then(({ response }) => {
      if (response === 0) {
        mainWindow.destroy();
        createWindow();
      }
    });
  });

  // 창이 닫힐 때
  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

// 앱 초기화
app.whenReady().then(createWindow);

// 모든 창이 닫히면 앱 종료
app.on('window-all-closed', () => {
  app.quit();
});

// 예기치 않은 에러 처리
process.on('uncaughtException', (error) => {
  console.error('Uncaught Exception:', error);
  
  if (mainWindow) {
    const options = {
      type: 'error',
      title: '오류 발생',
      message: '예기치 않은 오류가 발생했습니다.',
      detail: error.toString(),
      buttons: ['재시작', '종료']
    };

    dialog.showMessageBox(options).then(({ response }) => {
      if (response === 0) {
        app.relaunch();
        app.exit(0);
      } else {
        app.exit(1);
      }
    });
  } else {
    app.exit(1);
  }
});

// 정상 종료 처리
app.on('before-quit', () => {
  if (mainWindow && !mainWindow.isDestroyed()) {
    mainWindow.removeAllListeners('close');
    mainWindow.close();
  }
});

// 강제 종료 신호 처리
process.on('SIGTERM', () => app.quit());
process.on('SIGINT', () => app.quit());

module.exports = { createWindow }; 