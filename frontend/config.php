<?php
/**
 * Precious Edu LLM — Frontend Configuration
 *
 * Central configuration file for PHP Chat Frontend.
 * Exposes backend API URL to JavaScript.
 */

// FastAPI Backend URL (configurable via environment variable or default)
define('API_BASE_URL', getenv('FASTAPI_BASE_URL') ?: 'http://127.0.0.1:8000');
?>
<script>
    window.APP_CONFIG = {
        apiBaseUrl: <?php echo json_encode(API_BASE_URL); ?>
    };
</script>
