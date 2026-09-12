/**
 * Precious AI — Chatbot Frontend Engine
 *
 * Handles API communication with FastAPI backend, state management,
 * localStorage persistence, XSS-safe markdown rendering, dark/light themes,
 * copy actions, and auto-scrolling controls.
 */

(function () {
    'use strict';

    // --- Configuration & Constants ---
    const API_BASE_URL = (window.APP_CONFIG && window.APP_CONFIG.apiBaseUrl)
        ? window.APP_CONFIG.apiBaseUrl
        : 'http://127.0.0.1:8000';

    const SESSION_STORAGE_KEY = 'precious_ai_active_session_id';
    const THEME_STORAGE_KEY = 'precious_ai_theme';

    // --- 1. API Client Abstraction ---
    class ApiClient {
        static async request(endpoint, options = {}) {
            const url = `${API_BASE_URL}${endpoint}`;
            const defaultHeaders = {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            };

            const config = {
                ...options,
                headers: {
                    ...defaultHeaders,
                    ...options.headers
                }
            };

            try {
                const response = await fetch(url, config);
                const data = await response.json().catch(() => ({}));

                if (!response.ok) {
                    const errorMsg = (data && data.detail)
                        || (data && data.error && data.error.message)
                        || `Server returned HTTP ${response.status}`;
                    throw new Error(errorMsg);
                }

                return data;
            } catch (err) {
                if (err.name === 'TypeError' && err.message.includes('fetch')) {
                    throw new Error('Unable to connect to Precious AI backend server (127.0.0.1:8000). Verify backend is running.');
                }
                throw err;
            }
        }

        static async createSession(title = "New Conversation") {
            return await this.request('/api/sessions', {
                method: 'POST',
                body: JSON.stringify({ title })
            });
        }

        static async getSession(sessionId) {
            return await this.request(`/api/sessions/${encodeURIComponent(sessionId)}`, {
                method: 'GET'
            });
        }

        static async deleteSession(sessionId) {
            return await this.request(`/api/sessions/${encodeURIComponent(sessionId)}`, {
                method: 'DELETE'
            });
        }

        static async sendMessage(sessionId, message) {
            return await this.request('/api/chat', {
                method: 'POST',
                body: JSON.stringify({
                    session_id: sessionId,
                    message: message
                })
            });
        }
    }

    // --- 2. Lightweight XSS-Safe Markdown Renderer ---
    class MarkdownRenderer {
        static escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        static render(text) {
            if (!text) return '';

            // Escape HTML first to prevent XSS
            let safeText = this.escapeHtml(text);

            // 1. Code blocks (```lang ... ```)
            safeText = safeText.replace(/```(\w*)\n([\s\S]*?)```/g, (match, lang, code) => {
                const language = lang || 'code';
                return `
                    <div class="code-block-wrapper">
                        <div class="code-header">
                            <span><i class="fa-solid fa-code"></i> ${language}</span>
                            <button class="copy-code-btn" onclick="window.copyToClipboard(this)">
                                <i class="fa-regular fa-copy"></i> Copy
                            </button>
                        </div>
                        <pre><code>${code.trim()}</code></pre>
                    </div>
                `;
            });

            // 2. Inline code (`code`)
            safeText = safeText.replace(/`([^`]+)`/g, '<code>$1</code>');

            // 3. Bold (**bold**)
            safeText = safeText.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');

            // 4. Italics (*italic*)
            safeText = safeText.replace(/\*([^*]+)\*/g, '<em>$1</em>');

            // 5. Line breaks
            safeText = safeText.replace(/\n/g, '<br>');

            return safeText;
        }
    }

    // Global copy handler for code blocks and responses
    window.copyToClipboard = function (btnElement, customText = null) {
        let textToCopy = customText;
        if (!textToCopy) {
            const wrapper = btnElement.closest('.code-block-wrapper');
            if (wrapper) {
                const codeEl = wrapper.querySelector('pre code');
                if (codeEl) textToCopy = codeEl.innerText;
            }
        }

        if (textToCopy) {
            navigator.clipboard.writeText(textToCopy).then(() => {
                const originalHtml = btnElement.innerHTML;
                btnElement.innerHTML = '<i class="fa-solid fa-check"></i> Copied!';
                setTimeout(() => {
                    btnElement.innerHTML = originalHtml;
                }, 2000);
            }).catch(err => {
                console.error("Copy failed", err);
            });
        }
    };

    // --- 3. Application State ---
    class ChatState {
        constructor() {
            this.sessionId = localStorage.getItem(SESSION_STORAGE_KEY) || null;
            this.theme = localStorage.getItem(THEME_STORAGE_KEY) || 'dark';
            this.messages = [];
            this.isLoading = false;
        }

        setSessionId(id) {
            this.sessionId = id;
            if (id) {
                localStorage.setItem(SESSION_STORAGE_KEY, id);
            } else {
                localStorage.removeItem(SESSION_STORAGE_KEY);
            }
        }

        setTheme(theme) {
            this.theme = theme;
            localStorage.setItem(THEME_STORAGE_KEY, theme);
            document.documentElement.setAttribute('data-theme', theme);
        }

        clearSession() {
            this.setSessionId(null);
            this.messages = [];
        }
    }

    // --- 4. UI Controller ---
    class UIController {
        constructor(state) {
            this.state = state;

            // DOM Elements
            this.messagesContainer = document.getElementById('chat-messages');
            this.inputForm = document.getElementById('chat-form');
            this.messageInput = document.getElementById('chat-input');
            this.sendBtn = document.getElementById('send-btn');
            this.newChatBtn = document.getElementById('new-chat-btn');
            this.deleteChatBtn = document.getElementById('delete-chat-btn');
            this.themeToggleBtn = document.getElementById('theme-toggle-btn');
            this.scrollBottomBtn = document.getElementById('scroll-bottom-btn');
            this.errorToast = document.getElementById('error-toast');
            this.errorMessage = document.getElementById('error-message');
            this.closeErrorBtn = document.getElementById('close-error-btn');

            this.typingElement = null;

            this.initTheme();
            this.initEventListeners();
        }

        initTheme() {
            this.state.setTheme(this.state.theme);
            this.updateThemeIcon();
        }

        updateThemeIcon() {
            const icon = this.themeToggleBtn.querySelector('i');
            if (this.state.theme === 'dark') {
                icon.className = 'fa-solid fa-sun';
            } else {
                icon.className = 'fa-solid fa-moon';
            }
        }

        initEventListeners() {
            // Theme Toggle
            this.themeToggleBtn.addEventListener('click', () => {
                const nextTheme = this.state.theme === 'dark' ? 'light' : 'dark';
                this.state.setTheme(nextTheme);
                this.updateThemeIcon();
            });

            // Form submit
            this.inputForm.addEventListener('submit', (e) => {
                e.preventDefault();
                this.handleSendMessage();
            });

            // Textarea Enter / Shift+Enter keypress
            this.messageInput.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    this.handleSendMessage();
                }
            });

            // Auto-resize textarea
            this.messageInput.addEventListener('input', () => {
                this.messageInput.style.height = 'auto';
                this.messageInput.style.height = `${Math.min(this.messageInput.scrollHeight, 160)}px`;
            });

            // Scroll container listener for float scroll button
            this.messagesContainer.addEventListener('scroll', () => {
                const isNearBottom = this.messagesContainer.scrollHeight - this.messagesContainer.scrollTop - this.messagesContainer.clientHeight < 120;
                this.scrollBottomBtn.style.display = isNearBottom ? 'none' : 'flex';
            });

            this.scrollBottomBtn.addEventListener('click', () => this.scrollToBottom(true));

            // Action buttons
            this.newChatBtn.addEventListener('click', () => this.handleNewChat());
            this.deleteChatBtn.addEventListener('click', () => this.handleDeleteChat());
            this.closeErrorBtn.addEventListener('click', () => this.hideError());
        }

        // --- Render Helpers ---
        renderMessage(role, content, timestamp = null) {
            // Remove empty state if present
            const emptyNotice = this.messagesContainer.querySelector('.empty-state');
            if (emptyNotice) {
                emptyNotice.remove();
            }

            const isUser = role.toLowerCase() === 'user';
            const row = document.createElement('div');
            row.className = `message-row ${isUser ? 'user' : 'assistant'}`;

            // Avatar
            const avatar = document.createElement('div');
            avatar.className = 'avatar';
            avatar.innerHTML = isUser ? '<i class="fa-solid fa-user"></i>' : '<i class="fa-solid fa-bolt"></i>';

            // Wrapper
            const wrapper = document.createElement('div');
            wrapper.className = 'message-content-wrapper';

            // Meta (sender & time)
            const meta = document.createElement('div');
            meta.className = 'message-meta';

            const senderName = document.createElement('span');
            senderName.textContent = isUser ? 'You' : 'Precious AI';
            meta.appendChild(senderName);

            if (timestamp) {
                const timeEl = document.createElement('span');
                const dateObj = new Date(timestamp);
                timeEl.textContent = isNaN(dateObj.getTime())
                    ? timestamp
                    : dateObj.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
                meta.appendChild(document.createTextNode('•'));
                meta.appendChild(timeEl);
            }

            // Bubble
            const bubble = document.createElement('div');
            bubble.className = 'message-bubble';
            
            if (isUser) {
                bubble.textContent = content; // Safe raw text for user
            } else {
                bubble.innerHTML = MarkdownRenderer.render(content); // Rendered markdown
            }

            wrapper.appendChild(meta);
            wrapper.appendChild(bubble);

            // Copy action for assistant responses
            if (!isUser) {
                const actions = document.createElement('div');
                actions.className = 'message-actions';
                
                const copyBtn = document.createElement('button');
                copyBtn.className = 'action-btn';
                copyBtn.innerHTML = '<i class="fa-regular fa-copy"></i> Copy response';
                copyBtn.onclick = function() {
                    window.copyToClipboard(copyBtn, content);
                };
                
                actions.appendChild(copyBtn);
                wrapper.appendChild(actions);
            }

            row.appendChild(avatar);
            row.appendChild(wrapper);

            this.messagesContainer.appendChild(row);
            this.scrollToBottom();
        }

        renderEmptyState() {
            this.messagesContainer.innerHTML = `
                <div class="empty-state">
                    <div class="empty-hero-icon">
                        <i class="fa-solid fa-bolt"></i>
                    </div>
                    <h2>How can I help you today?</h2>
                    <p>Precious AI features persistent session memory and context management.</p>
                </div>
            `;
        }

        showTypingIndicator() {
            this.hideTypingIndicator();

            const indicator = document.createElement('div');
            indicator.className = 'message-row assistant';
            indicator.id = 'typing-row';

            const avatar = document.createElement('div');
            avatar.className = 'avatar';
            avatar.innerHTML = '<i class="fa-solid fa-bolt"></i>';

            const wrapper = document.createElement('div');
            wrapper.className = 'message-content-wrapper';

            const bubble = document.createElement('div');
            bubble.className = 'message-bubble';
            bubble.innerHTML = `
                <div class="typing-dots">
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                </div>
            `;

            wrapper.appendChild(bubble);
            indicator.appendChild(avatar);
            indicator.appendChild(wrapper);

            this.messagesContainer.appendChild(indicator);
            this.scrollToBottom();
            this.typingElement = indicator;
        }

        hideTypingIndicator() {
            if (this.typingElement && this.typingElement.parentNode) {
                this.typingElement.remove();
            }
            this.typingElement = null;
        }

        showError(message) {
            this.errorMessage.textContent = message;
            this.errorToast.style.display = 'flex';
        }

        hideError() {
            this.errorToast.style.display = 'none';
        }

        scrollToBottom(force = false) {
            this.messagesContainer.scrollTo({
                top: this.messagesContainer.scrollHeight,
                behavior: force ? 'smooth' : 'auto'
            });
        }

        setLoading(loading) {
            this.state.isLoading = loading;
            this.sendBtn.disabled = loading;
            this.messageInput.disabled = loading;

            if (loading) {
                this.showTypingIndicator();
            } else {
                this.hideTypingIndicator();
                this.messageInput.focus();
            }
        }

        // --- Core User Workflows ---
        async initSession() {
            this.hideError();

            if (this.state.sessionId) {
                try {
                    const data = await ApiClient.getSession(this.state.sessionId);
                    this.messagesContainer.innerHTML = '';

                    if (data.messages && data.messages.length > 0) {
                        data.messages.forEach(msg => {
                            this.renderMessage(msg.role, msg.content, msg.created_at);
                        });
                    } else {
                        this.renderEmptyState();
                    }
                    return;
                } catch (err) {
                    console.warn("Stored session invalid or expired, creating new session...", err);
                    this.state.clearSession();
                }
            }

            await this.createNewSession();
        }

        async createNewSession() {
            this.hideError();
            try {
                const session = await ApiClient.createSession();
                this.state.setSessionId(session.session_id);
                this.messagesContainer.innerHTML = '';
                this.renderEmptyState();
            } catch (err) {
                this.showError(err.message || 'Failed to create new chat session.');
            }
        }

        async handleSendMessage() {
            const rawText = this.messageInput.value;
            const message = rawText ? rawText.trim() : '';

            if (!message) {
                this.showError('Please enter a message.');
                return;
            }

            this.hideError();

            if (!this.state.sessionId) {
                await this.createNewSession();
                if (!this.state.sessionId) return;
            }

            // Render user message
            this.renderMessage('user', message, new Date().toISOString());

            // Clear input & reset height
            this.messageInput.value = '';
            this.messageInput.style.height = 'auto';

            this.setLoading(true);

            try {
                const response = await ApiClient.sendMessage(this.state.sessionId, message);
                this.renderMessage(response.role || 'assistant', response.response, response.created_at);
            } catch (err) {
                this.showError(err.message || 'Failed to send message.');
            } finally {
                this.setLoading(false);
            }
        }

        async handleNewChat() {
            if (this.state.isLoading) return;
            await this.createNewSession();
        }

        async handleDeleteChat() {
            if (this.state.isLoading || !this.state.sessionId) return;

            const targetId = this.state.sessionId;
            this.hideError();

            try {
                await ApiClient.deleteSession(targetId);
                this.state.clearSession();
                await this.createNewSession();
            } catch (err) {
                this.showError(err.message || 'Failed to delete chat session.');
            }
        }
    }

    // --- App Initialization ---
    document.addEventListener('DOMContentLoaded', () => {
        const state = new ChatState();
        const ui = new UIController(state);
        ui.initSession();
    });
})();
