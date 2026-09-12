# Precious Edu LLM — PHP Chat Frontend

Modern ChatGPT-style web interface built with PHP, HTML5, CSS3, and Vanilla JavaScript. Seamlessly communicates with the FastAPI backend.

---

## 1. Requirements

- **PHP**: PHP 8.0+ (or XAMPP PHP)
- **FastAPI Backend**: Running on `http://127.0.0.1:8000` (Phase 1)
- **MongoDB**: Running locally on `127.0.0.1:27017`

---

## 2. How to Start

### Option A: Using PHP Development Server
From project root:
```bash
cd frontend
C:\xampp\php\php.exe -S 127.0.0.1:8080
```
Or standard PHP:
```bash
cd frontend
php -S 127.0.0.1:8080
```

Open browser at `http://127.0.0.1:8080`.

### Option B: Using XAMPP htdocs
Copy `frontend/` folder into `htdocs/precious_ai/` and access via `http://localhost/precious_ai/`.

---

## 3. Configuration

Backend API base URL is configured in `frontend/config.php`:
```php
define('API_BASE_URL', getenv('FASTAPI_BASE_URL') ?: 'http://127.0.0.1:8000');
```
This is safely exposed to JavaScript as `window.APP_CONFIG.apiBaseUrl`.

---

## 4. Key Architecture & Features

- **No Framework Overheads**: Built cleanly with Vanilla JS for optimal performance.
- **XSS Protection**: User and assistant messages rendered strictly via DOM text nodes (`textContent`).
- **Session Persistence**: Session ID saved in browser `localStorage`. Page refreshes reload history directly from MongoDB via `GET /api/sessions/{session_id}`.
- **Keyboard Shortcuts**:
  - `Enter`: Submit message
  - `Shift + Enter`: Insert newline
- **Robust Error Handling**: Handles backend disconnection, network timeouts, and missing sessions gracefully with animated error toasts.

---

## 5. CORS Configuration

The FastAPI backend has CORS enabled for local frontend access (`http://127.0.0.1:8080`, `http://localhost:8080`, `http://localhost`). Ensure FastAPI backend (`backend/app/main.py`) has CORS middleware configured when running on custom domains.

---

## 6. Manual Testing Steps

1. **Launch Backend & Database**: Ensure MongoDB is active on `127.0.0.1:27017` and launch FastAPI backend on `http://127.0.0.1:8000`.
2. **Launch Frontend Server**: Run `C:\xampp\php\php.exe -S 127.0.0.1:8080` inside `frontend/`.
3. **Initial Load**: Visit `http://127.0.0.1:8080`. Verify a new chat session is automatically initialized and saved in `localStorage`.
4. **Sending Messages**: Type a prompt and hit `Enter` or click `Send`. Verify user message renders immediately, typing indicator displays, and assistant response appears.
5. **Session Persistence**: Refresh the page. Verify existing conversation history is fetched via `GET /api/sessions/{session_id}`.
6. **New Chat**: Click `+ New Chat`. Verify session state resets and a new session ID is generated.
7. **Delete Session**: Click `🗑️ Delete Session`. Verify `DELETE /api/sessions/{session_id}` is executed and UI is reset with a fresh session.
8. **Backend Offline Test**: Stop the FastAPI backend and attempt to send a message. Verify user receives an error toast notification.

