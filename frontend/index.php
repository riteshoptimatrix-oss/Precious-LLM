<?php
/**
 * Precious Edu LLM — PHP Chat Frontend Presentation Page
 *
 * Premium Glassmorphic ChatGPT-Style Interface.
 */
require_once __DIR__ . '/config.php';
?>
<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Precious AI — Conversational Assistant</title>
    <meta name="description" content="A domain-specific conversational AI chatbot powered by custom LLM engine.">
    <!-- Google Fonts: Inter & Outfit -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Outfit:wght@500;600;700;800&display=swap" rel="stylesheet">
    <!-- FontAwesome / Feather Icons CDN for crisp UI icons -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <!-- Custom CSS -->
    <link rel="stylesheet" href="assets/css/chatbot.css">
</head>
<body>

    <div class="chat-app-wrapper">
        <!-- Chat Main Container -->
        <div class="chat-container">
            <!-- Header -->
            <header class="chat-header">
                <div class="header-brand">
                    <div class="brand-avatar">
                        <i class="fa-solid fa-bolt"></i>
                    </div>
                    <div class="brand-info">
                        <div class="brand-title-row">
                            <h1>Precious AI</h1>
                            <span class="status-badge"><span class="status-dot"></span> Online</span>
                        </div>
                        <p>Domain-Specific Conversational LLM</p>
                    </div>
                </div>
                
                <div class="header-actions">
                    <button id="theme-toggle-btn" class="btn btn-icon" title="Toggle Dark/Light Theme" aria-label="Toggle theme">
                        <i class="fa-solid fa-moon"></i>
                    </button>
                    <button id="new-chat-btn" class="btn btn-primary" title="Create a new conversation session">
                        <i class="fa-solid fa-plus"></i>
                        <span>New Chat</span>
                    </button>
                    <button id="delete-chat-btn" class="btn btn-danger" title="Delete current conversation session">
                        <i class="fa-regular fa-trash-can"></i>
                        <span>Delete</span>
                    </button>
                </div>
            </header>

            <!-- Error Toast Alert -->
            <div id="error-toast" class="error-toast" style="display: none;">
                <i class="fa-solid fa-triangle-exclamation"></i>
                <span id="error-message">Connection error occurred.</span>
                <button id="close-error-btn" aria-label="Close error toast">&times;</button>
            </div>

            <!-- Chat Messages Area -->
            <main id="chat-messages" class="chat-messages" aria-live="polite">
                <!-- Dynamic Messages Rendered by chatbot.js -->
            </main>

            <!-- Scroll To Bottom Floating Button -->
            <button id="scroll-bottom-btn" class="scroll-bottom-btn" style="display: none;" title="Scroll to latest message">
                <i class="fa-solid fa-arrow-down"></i>
            </button>

            <!-- Footer / Input Controls -->
            <footer class="chat-footer">
                <form id="chat-form" class="input-form">
                    <div class="input-wrapper">
                        <textarea 
                            id="chat-input" 
                            class="chat-input" 
                            placeholder="Message Precious AI... (Press Enter to send, Shift+Enter for newline)" 
                            rows="1"
                            required
                        ></textarea>
                        <button type="submit" id="send-btn" class="send-btn" title="Send message">
                            <i class="fa-solid fa-paper-plane"></i>
                        </button>
                    </div>
                </form>
                <div class="footer-note">
                    <span>Precious AI may produce helpful responses leveraging your conversation memory.</span>
                </div>
            </footer>
        </div>
    </div>

    <!-- Client Application JavaScript -->
    <script src="assets/js/chatbot.js"></script>
</body>
</html>
