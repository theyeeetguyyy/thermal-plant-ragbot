// CONFIGURATION: Set your backend API URL here.
// For local testing: 'http://127.0.0.1:8000/api/chat'
// For Hugging Face Spaces: 'https://<your-username>-<your-space-name>.hf.space/api/chat'
const API_URL = 'http://127.0.0.1:8000/api/chat';

document.addEventListener('DOMContentLoaded', () => {
    const chatHistory = document.getElementById('chat-history');
    const userInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');

    function addMessage(text, isUser = false, sources = []) {
        const msgDiv = document.createElement('div');
        msgDiv.className = `message ${isUser ? 'user-message' : 'system-message'}`;
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        
        // Handle basic markdown-like formatting (newlines to <br>)
        let formattedText = text.replace(/\n/g, '<br>');
        contentDiv.innerHTML = formattedText;
        
        if (sources && sources.length > 0) {
            const sourcesDiv = document.createElement('div');
            sourcesDiv.style.marginTop = '8px';
            sources.forEach(source => {
                const span = document.createElement('span');
                span.className = 'source-tag';
                span.textContent = `📄 ${source}`;
                sourcesDiv.appendChild(span);
            });
            contentDiv.appendChild(sourcesDiv);
        }

        msgDiv.appendChild(contentDiv);
        chatHistory.appendChild(msgDiv);
        scrollToBottom();
    }

    function addTypingIndicator() {
        const msgDiv = document.createElement('div');
        msgDiv.className = 'message system-message';
        msgDiv.id = 'typing-indicator';
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'typing-indicator';
        contentDiv.innerHTML = '<span></span><span></span><span></span>';
        
        msgDiv.appendChild(contentDiv);
        chatHistory.appendChild(msgDiv);
        scrollToBottom();
    }

    function removeTypingIndicator() {
        const indicator = document.getElementById('typing-indicator');
        if (indicator) {
            indicator.remove();
        }
    }

    function scrollToBottom() {
        chatHistory.scrollTop = chatHistory.scrollHeight;
    }

    async function sendMessage() {
        const text = userInput.value.trim();
        if (!text) return;

        // Add user message to UI
        addMessage(text, true);
        userInput.value = '';
        
        // Show loading state
        addTypingIndicator();

        try {
            const response = await fetch(API_URL, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ query: text }),
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            removeTypingIndicator();
            addMessage(data.answer, false, data.sources);
            
        } catch (error) {
            removeTypingIndicator();
            console.error('Error:', error);
            addMessage(`Sorry, I encountered an error: ${error.message}. Is the backend running?`, false);
        }
    }

    sendBtn.addEventListener('click', sendMessage);
    
    userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });
});
