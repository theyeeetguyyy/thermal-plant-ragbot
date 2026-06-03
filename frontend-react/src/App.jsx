import React, { useState, useEffect, useRef } from 'react';
import { Send, Upload, PlusCircle, User, Bot, LogOut, FileText, Lock, Trash2 } from 'lucide-react';
import ReactMarkdown from 'react-markdown';

const API_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

function App() {
  const [user, setUser] = useState(null);
  const [chats, setChats] = useState([]);
  const [currentChatId, setCurrentChatId] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [showUpgrade, setShowUpgrade] = useState(false);

  // Load user from local storage
  useEffect(() => {
    const savedUser = localStorage.getItem('user');
    if (savedUser) {
      setUser(JSON.parse(savedUser));
    }
  }, []);

  // Load chats for user
  useEffect(() => {
    if (user?.token) {
      const savedChats = localStorage.getItem(`chats_${user.token}`);
      if (savedChats) {
        const parsedChats = JSON.parse(savedChats);
        setChats(parsedChats);
        if (parsedChats.length > 0) {
          setCurrentChatId(parsedChats[0].id);
        } else {
          createNewChat();
        }
      } else {
        createNewChat();
      }
    }
  }, [user?.token]);

  // Save chats
  useEffect(() => {
    if (user?.token && chats.length > 0) {
      localStorage.setItem(`chats_${user.token}`, JSON.stringify(chats));
    }
  }, [chats, user?.token]);

  const createNewChat = () => {
    const newId = Date.now().toString();
    setChats([{ id: newId, title: 'New Chat', messages: [] }, ...chats]);
    setCurrentChatId(newId);
  };

  const deleteChat = (id) => {
    const newChats = chats.filter(c => c.id !== id);
    setChats(newChats);
    if (currentChatId === id) {
      setCurrentChatId(newChats.length > 0 ? newChats[0].id : null);
    }
  };

  const handleLogout = () => {
    setUser(null);
    setChats([]);
    setCurrentChatId(null);
    localStorage.removeItem('user');
  };

  const activeChat = chats.find(c => c.id === currentChatId);

  return (
    <div className="app-container">
      {!user ? (
        <LoginScreen onLogin={(u) => {
          setUser(u);
          localStorage.setItem('user', JSON.stringify(u));
        }} />
      ) : (
        <>
          <Sidebar 
            chats={chats} 
            currentChatId={currentChatId} 
            onSelectChat={setCurrentChatId}
            onNewChat={createNewChat}
            onDeleteChat={deleteChat}
            onLogout={handleLogout}
            onUpload={() => setIsUploading(true)}
            user={user}
          />
          <ChatArea 
            chat={activeChat}
            user={user}
            setUser={setUser}
            onUpdateChat={(updatedChat) => {
              setChats(chats.map(c => c.id === updatedChat.id ? updatedChat : c));
            }}
            onLimitExceeded={() => setShowUpgrade(true)}
          />
          {isUploading && (
            <UploadModal 
              onClose={() => setIsUploading(false)} 
              user={user}
            />
          )}
          {showUpgrade && (
            <UpgradeModal onClose={() => setShowUpgrade(false)} />
          )}
        </>
      )}
    </div>
  );
}

function LoginScreen({ onLogin }) {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    try {
      const res = await fetch(`${API_URL}/api/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password })
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Login failed');
      onLogin(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="modal-overlay">
      <div className="modal-content">
        <h2 className="modal-title">Thermal RAG Login</h2>
        <form onSubmit={handleSubmit}>
          <div className="form-group">
            <label>Username</label>
            <input 
              type="text" 
              value={username} 
              onChange={e => setUsername(e.target.value)} 
              required 
              placeholder="admin or demo1"
            />
          </div>
          <div className="form-group">
            <label>Password</label>
            <input 
              type="password" 
              value={password} 
              onChange={e => setPassword(e.target.value)} 
              required 
            />
          </div>
          {error && <div className="error-text">{error}</div>}
          <button type="submit" className="btn-primary" disabled={loading}>
            {loading ? 'Logging in...' : 'Sign In'}
          </button>
        </form>
      </div>
    </div>
  );
}

function Sidebar({ chats, currentChatId, onSelectChat, onNewChat, onDeleteChat, onLogout, onUpload, user }) {
  return (
    <div className="sidebar">
      <div className="sidebar-header">
        <div className="sidebar-title">
          <Bot size={24} color="var(--primary)" />
          Thermal RAG
        </div>
      </div>
      <button className="new-chat-btn" onClick={onNewChat}>
        <PlusCircle size={18} /> New Chat
      </button>
      
      <div className="chat-history-list">
        {chats.map(chat => (
          <div 
            key={chat.id} 
            className={`history-item ${chat.id === currentChatId ? 'active' : ''}`}
            onClick={() => onSelectChat(chat.id)}
            style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', overflow: 'hidden' }}>
              <FileText size={16} style={{ flexShrink: 0 }} />
              <span style={{ textOverflow: 'ellipsis', overflow: 'hidden', whiteSpace: 'nowrap' }}>{chat.title}</span>
            </div>
            <div 
              onClick={(e) => { e.stopPropagation(); onDeleteChat(chat.id); }}
              className="delete-btn"
              title="Delete chat"
            >
              <Trash2 size={16} />
            </div>
          </div>
        ))}
      </div>

      <div className="sidebar-footer">
        {user.role === 'admin' && (
          <button className="action-btn" onClick={onUpload}>
            <Upload size={16} /> Manage Documents
          </button>
        )}
        <button className="action-btn" onClick={onLogout} style={{color: '#ef4444', borderColor: '#fee2e2'}}>
          <LogOut size={16} /> Logout
        </button>
      </div>
    </div>
  );
}

function ChatArea({ chat, user, setUser, onUpdateChat, onLimitExceeded }) {
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const endRef = useRef(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chat?.messages, loading]);

  if (!chat) return <div className="main-content" />;

  const handleSend = async () => {
    if (!input.trim() || loading) return;

    const userMessage = { role: 'user', content: input.trim() };
    const updatedMessages = [...chat.messages, userMessage];
    
    // Update title if first message
    let title = chat.title;
    if (chat.messages.length === 0) {
      title = input.slice(0, 30) + '...';
    }

    onUpdateChat({ ...chat, title, messages: updatedMessages });
    setInput('');
    setLoading(true);

    try {
      const res = await fetch(`${API_URL}/api/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${user.token}`
        },
        body: JSON.stringify({ query: userMessage.content })
      });
      
      const data = await res.json();
      
      if (!res.ok) {
        if (res.status === 402) {
          onLimitExceeded();
          // Remove the user's message since it wasn't processed by backend
          onUpdateChat({ ...chat, title, messages: chat.messages });
          return;
        }
        throw new Error(data.detail || 'Failed to fetch response');
      }

      // Update remaining queries for demo user
      if (data.remaining_queries !== undefined && data.remaining_queries >= 0) {
        setUser({ ...user, remaining: data.remaining_queries });
      }

      const aiMessage = { 
        role: 'ai', 
        content: data.answer, 
        sources: data.sources 
      };
      
      onUpdateChat({ ...chat, title, messages: [...updatedMessages, aiMessage] });
      
    } catch (err) {
      console.error(err);
      onUpdateChat({ 
        ...chat, 
        messages: [...updatedMessages, { role: 'ai', content: `Error: ${err.message}` }] 
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="main-content">
      <div className="header">
        <div className="user-info">
          <User size={20} />
          <span>{user.token}</span>
          <span className={`badge ${user.role}`}>{user.role}</span>
          {user.role === 'demo' && (
            <span style={{color: 'var(--text-muted)'}}>
              ({user.remaining} queries left)
            </span>
          )}
        </div>
      </div>
      
      <div className="chat-container">
        {chat.messages.length === 0 && (
          <div style={{textAlign: 'center', color: 'var(--text-muted)', marginTop: '40px'}}>
            <h2>How can I help you today?</h2>
            <p>Ask anything about the thermal power plant.</p>
          </div>
        )}
        
        {chat.messages.map((msg, i) => (
          <div key={i} className={`message-wrapper ${msg.role}`}>
            <div className={`avatar ${msg.role}`}>
              {msg.role === 'user' ? <User size={20} /> : <Bot size={20} />}
            </div>
            <div className="message-content markdown-body">
              <ReactMarkdown>{msg.content}</ReactMarkdown>
              {msg.sources && msg.sources.length > 0 && (
                <div className="sources">
                  {msg.sources.map((s, idx) => (
                    <span key={idx} className="source-tag">📄 {s}</span>
                  ))}
                </div>
              )}
            </div>
          </div>
        ))}
        {loading && (
          <div className="message-wrapper ai">
             <div className="avatar ai"><Bot size={20} /></div>
             <div className="message-content">
               <div className="typing-indicator">
                 <span></span><span></span><span></span>
               </div>
             </div>
          </div>
        )}
        <div ref={endRef} />
      </div>

      <div className="input-area">
        <div className="input-box">
          <input 
            type="text" 
            placeholder="Ask a question..."
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && handleSend()}
            disabled={loading}
          />
          <button className="send-btn" onClick={handleSend} disabled={!input.trim() || loading}>
            <Send size={18} />
          </button>
        </div>
      </div>
    </div>
  );
}

function UploadModal({ onClose, user }) {
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  
  const [documents, setDocuments] = useState([]);
  const [loadingDocs, setLoadingDocs] = useState(true);

  const fetchDocs = async () => {
    try {
      const res = await fetch(`${API_URL}/api/documents`, {
        headers: { 'Authorization': `Bearer ${user.token}` }
      });
      if (res.ok) {
        const data = await res.json();
        setDocuments(data.documents);
      }
    } catch (err) {
      console.error(err);
    } finally {
      setLoadingDocs(false);
    }
  };

  useEffect(() => {
    fetchDocs();
  }, []);

  const handleDelete = async (filename) => {
    try {
      const res = await fetch(`${API_URL}/api/documents/${filename}`, {
        method: 'DELETE',
        headers: { 'Authorization': `Bearer ${user.token}` }
      });
      if (res.ok) {
        fetchDocs();
      }
    } catch (err) {
      console.error(err);
    }
  };

  const handleUpload = async () => {
    if (!file) return;
    setUploading(true);
    setProgress(0);
    setError('');
    setSuccess('');
    
    // Fake progress bar
    const interval = setInterval(() => {
      setProgress(prev => {
        if (prev >= 90) {
          clearInterval(interval);
          return 90;
        }
        return prev + 10;
      });
    }, 500);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch(`${API_URL}/api/upload`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${user.token}`
        },
        body: formData
      });
      
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Upload failed');
      
      clearInterval(interval);
      setProgress(100);
      setSuccess(`Successfully uploaded and ingested: ${data.filename}`);
      setFile(null);
      fetchDocs();
      
      setTimeout(() => {
        setSuccess('');
        setProgress(0);
      }, 3000);
      
    } catch (err) {
      clearInterval(interval);
      setProgress(0);
      setError(err.message);
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="modal-overlay">
      <div className="modal-content" style={{maxWidth: '500px'}}>
        <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px'}}>
          <h2 className="modal-title" style={{margin: 0}}>Manage Documents</h2>
          <button className="action-btn" onClick={onClose} style={{margin: 0, padding: '4px 8px', width: 'auto'}}>Close</button>
        </div>
        
        <div style={{marginBottom: '24px'}}>
          <h3 style={{fontSize: '1rem', marginBottom: '8px'}}>Upload New</h3>
          <p style={{marginBottom: '16px', color: 'var(--text-muted)', fontSize: '0.875rem'}}>Upload a PDF or DOCX file to add it to the RAG knowledge base.</p>
          <div className="form-group">
            <input type="file" accept=".pdf,.docx" onChange={e => setFile(e.target.files[0])} disabled={uploading} />
          </div>
          
          {uploading && (
            <div className="progress-container">
              <div className="progress-bar" style={{ width: `${progress}%` }}></div>
            </div>
          )}
          
          {error && <div className="error-text">{error}</div>}
          {success && <div style={{color: '#16a34a', marginBottom: '16px', textAlign: 'center', fontSize: '0.875rem'}}>{success}</div>}
          
          <button className="btn-primary" onClick={handleUpload} disabled={!file || uploading}>
            {uploading ? 'Processing...' : 'Upload & Ingest'}
          </button>
        </div>

        <div style={{borderTop: '1px solid var(--border)', paddingTop: '24px'}}>
          <h3 style={{fontSize: '1rem', marginBottom: '8px'}}>Existing Documents</h3>
          {loadingDocs ? (
            <p style={{color: 'var(--text-muted)', fontSize: '0.875rem'}}>Loading...</p>
          ) : documents.length === 0 ? (
            <p style={{color: 'var(--text-muted)', fontSize: '0.875rem'}}>No documents uploaded yet.</p>
          ) : (
            <div className="document-list">
              {documents.map(doc => (
                <div key={doc} className="document-item">
                  <span style={{overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', marginRight: '12px'}}>{doc}</span>
                  <button 
                    className="delete-btn" 
                    onClick={() => handleDelete(doc)}
                    style={{border: 'none', background: 'none', padding: '4px', margin: 0}}
                    title="Delete document"
                  >
                    <Trash2 size={16} />
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function UpgradeModal({ onClose }) {
  return (
    <div className="modal-overlay">
      <div className="modal-content" style={{textAlign: 'center'}}>
        <Lock size={48} color="var(--primary)" style={{margin: '0 auto 16px'}} />
        <h2 className="modal-title" style={{marginBottom: '8px'}}>Query Limit Reached</h2>
        <p style={{color: 'var(--text-muted)', marginBottom: '24px'}}>
          You have exhausted your free demo queries. Please upgrade your plan to continue using the Thermal Plant RAG Agent.
        </p>
        <button className="btn-primary" onClick={onClose}>Close</button>
      </div>
    </div>
  );
}

export default App;
