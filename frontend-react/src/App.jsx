import React, { useEffect, useMemo, useRef, useState } from 'react';
import {
  Activity,
  ArrowUp,
  BarChart3,
  Bell,
  Bot,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  Clock,
  Database,
  ExternalLink,
  FileText,
  Globe,
  HelpCircle,
  Layers,
  Loader2,
  LogOut,
  Menu,
  MessageSquare,
  MessageSquarePlus,
  PanelLeftClose,
  Paperclip,
  Search,
  Settings,
  Share2,
  ShieldCheck,
  SlidersHorizontal,
  Trash2,
  Upload,
  User,
  Wrench,
  X,
  Zap,
} from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import logoImg from '../../ID_logo.webp';

const API_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

/* ── Duck SVG mark ───────────────────────────────────────── */
function DuckIcon({ size = 24, className = '' }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="currentColor" className={className} aria-hidden="true">
      <ellipse cx="10" cy="14.5" rx="7" ry="4.5" />
      <circle cx="17.5" cy="9" r="3.5" />
      <path d="M20.8 8.2L24 9.5l-3.2 1.3z" opacity=".9" />
      <circle cx="18.8" cy="7.8" r="0.75" fill="rgba(255,255,255,0.92)" />
      <path d="M5.5 14.5Q9 11.5 13 14.5" stroke="rgba(255,255,255,0.28)" strokeWidth="1.1" fill="none" strokeLinecap="round" />
    </svg>
  );
}

function authHeaders(user) {
  return { Authorization: `Bearer ${user.token}` };
}

/* ── Helpers ─────────────────────────────────────────────── */
const RELEVANCE_MAP = [96, 91, 87, 82, 76, 71];

/* Fix 1: strip "Sources: ..." lines the backend embeds in the answer text */
function parseAIResponse(content) {
  if (!content) return { heading: null, body: '' };
  // Remove any "Sources: ..." or "**Sources**: ..." trailing lines
  let cleaned = content
    .replace(/\*{0,2}Sources?\*{0,2}\s*:\s*[^\n]*/gi, '')
    .replace(/\n{3,}/g, '\n\n')
    .trim();

  const match = cleaned.match(/^#{1,3}\s+(.+)/m);
  if (match) {
    return {
      heading: match[1].trim(),
      body: cleaned.slice(match.index + match[0].length).trim(),
    };
  }
  return { heading: null, body: cleaned };
}

/* Fix 2: rotating suggestion pool */
const ALL_SUGGESTIONS = [
  'What are the boiler startup and shutdown procedures?',
  'Explain the turbine overspeed trip test procedure',
  'What are the PTW requirements before maintenance work?',
  'List the emergency shutdown sequences for the plant',
  'What are the NOx and SOx regulatory emission limits?',
  'How to check bearing temperature thresholds during operation?',
  'What is the steam drum level alarm setpoint?',
  'Describe the turbine warming-up procedure',
  'What confined space entry safety requirements apply?',
  'Explain boiler water chemistry control limits',
  'What is the scheduled PM checklist for rotating equipment?',
  'How to calibrate and verify pressure transmitters?',
];

function getDocType(doc) {
  const name = ((doc?.filename || doc || '')).toLowerCase();
  if (name.endsWith('.pdf'))  return 'PDF';
  if (name.endsWith('.docx')) return 'DOCX';
  return 'DOC';
}

/* ════════════════════════════════════════════════════════════
   App
   ════════════════════════════════════════════════════════════ */

function App() {
  const [user, setUser]               = useState(null);
  const [chats, setChats]             = useState([]);
  const [currentChatId, setCurrentId] = useState(null);
  const [activeChat, setActiveChat]   = useState(null);
  const [loadingChats, setLdChats]    = useState(false);
  const [loadingChat, setLdChat]      = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [showUpgrade, setShowUpgrade] = useState(false);
  const [showHelp, setShowHelp]       = useState(false);
  const [activePage, setActivePage]   = useState('chat');
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [documents, setDocuments]     = useState([]);
  const [kbFilter, setKbFilter]       = useState('All Documents');

  useEffect(() => {
    const saved = localStorage.getItem('user');
    if (saved) setUser(JSON.parse(saved));
  }, []);

  useEffect(() => {
    if (!user?.token) return;
    loadChats();
    loadDocuments();
  }, [user?.token]);

  useEffect(() => {
    if (!user?.token || !currentChatId) { setActiveChat(null); return; }
    loadChat(currentChatId);
  }, [currentChatId, user?.token]);

  const loadDocuments = async () => {
    try {
      const r = await fetch(`${API_URL}/api/documents`, { headers: authHeaders(user) });
      const d = await r.json();
      if (r.ok) setDocuments(d.documents || []);
    } catch (e) { console.error(e); }
  };

  const loadChats = async () => {
    setLdChats(true);
    try {
      const r = await fetch(`${API_URL}/api/chats`, { headers: authHeaders(user) });
      const d = await r.json();
      if (!r.ok) throw new Error(d.detail);
      setChats(d.chats);
      setCurrentId(d.chats[0]?.id || null);
    } catch (e) { console.error(e); } finally { setLdChats(false); }
  };

  const loadChat = async (id) => {
    setLdChat(true);
    try {
      const r = await fetch(`${API_URL}/api/chats/${id}`, { headers: authHeaders(user) });
      const d = await r.json();
      if (!r.ok) throw new Error(d.detail);
      setActiveChat(d);
    } catch (e) { console.error(e); } finally { setLdChat(false); }
  };

  const createNewChat = async () => {
    try {
      const r = await fetch(`${API_URL}/api/chats`, {
        method: 'POST',
        headers: { ...authHeaders(user), 'Content-Type': 'application/json' },
        body: JSON.stringify({ title: 'New Query' }),
      });
      const d = await r.json();
      if (!r.ok) throw new Error(d.detail);
      setChats(prev => [d, ...prev]);
      setCurrentId(d.id);
      setActiveChat({ ...d, messages: [] });
    } catch (e) { console.error(e); }
  };

  const deleteChat = async (id) => {
    try {
      const r = await fetch(`${API_URL}/api/chats/${id}`, { method: 'DELETE', headers: authHeaders(user) });
      if (!r.ok) { const d = await r.json(); throw new Error(d.detail); }
      const next = chats.filter(c => c.id !== id);
      setChats(next);
      if (currentChatId === id) setCurrentId(next[0]?.id || null);
    } catch (e) { console.error(e); }
  };

  const handleLogout = async () => {
    try { if (user?.token) await fetch(`${API_URL}/api/logout`, { method: 'POST', headers: authHeaders(user) }); }
    finally { setUser(null); setChats([]); setCurrentId(null); setActiveChat(null); localStorage.removeItem('user'); }
  };

  const patchChat = (id, patch) => setChats(p => p.map(c => c.id === id ? { ...c, ...patch } : c));

  const openKb = (filter = 'All Documents') => { setKbFilter(filter); setIsUploading(true); };

  return (
    <div className="app-shell">
      {!user ? (
        <LoginScreen onLogin={u => { setUser(u); localStorage.setItem('user', JSON.stringify(u)); }} />
      ) : (
        <>
          <LeftSidebar
            chats={chats} currentChatId={currentChatId} isOpen={sidebarOpen}
            loading={loadingChats} user={user}
            onSelectChat={id => { setCurrentId(id); setActivePage('chat'); if (window.innerWidth < 920) setSidebarOpen(false); }}
            onNewChat={() => { createNewChat(); setActivePage('chat'); }}
            onDeleteChat={deleteChat}
            onLogout={handleLogout} onOpenKb={openKb}
            onOpenHelp={() => setShowHelp(true)}
            onOpenCommunity={() => setActivePage('community')}
            activePage={activePage}
          />
          {sidebarOpen && <div className="sidebar-backdrop" onClick={() => setSidebarOpen(false)} />}
          {activePage === 'community' ? (
            <CommunityPage
              user={user}
              onToggleSidebar={() => setSidebarOpen(v => !v)}
              onBack={() => setActivePage('chat')}
            />
          ) : (
            <ChatArea
              chat={activeChat} user={user} loadingChat={loadingChat}
              documents={documents} chatCount={chats.length}
              sidebarOpen={sidebarOpen}
              onOpenSidebar={() => setSidebarOpen(true)}
              onToggleSidebar={() => setSidebarOpen(v => !v)}
              setUser={setUser}
              onLocalUpdate={setActiveChat}
              onRefreshChats={loadChats}
              onPatchChat={patchChat}
              onCreateChat={createNewChat}
              onLimitExceeded={() => setShowUpgrade(true)}
              onOpenKb={openKb}
              onOpenCommunity={() => setActivePage('community')}
            />
          )}
          {isUploading && (
            <UploadModal onClose={() => { setIsUploading(false); loadDocuments(); }} user={user} filter={kbFilter} />
          )}
          {showUpgrade && <UpgradeModal onClose={() => setShowUpgrade(false)} />}
          {showHelp && <HelpModal onClose={() => setShowHelp(false)} />}
        </>
      )}
    </div>
  );
}

/* ════════════════════════════════════════════════════════════
   Login
   ════════════════════════════════════════════════════════════ */

function LoginScreen({ onLogin }) {
  const [mode, setMode] = useState('login');
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [confirm, setConfirm] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const switchMode = m => { setMode(m); setError(''); setUsername(''); setPassword(''); setConfirm(''); };

  const handleSubmit = async e => {
    e.preventDefault(); setError('');
    if (mode === 'signup') {
      if (password !== confirm)          { setError('Passwords do not match'); return; }
      if (password.length < 6)           { setError('Password must be at least 6 characters'); return; }
      if (!/^[a-zA-Z0-9_]{3,20}$/.test(username)) { setError('Username: 3–20 alphanumeric chars or underscores'); return; }
    }
    setLoading(true);
    try {
      const r = await fetch(`${API_URL}${mode === 'login' ? '/api/login' : '/api/register'}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password }),
      });
      const d = await r.json();
      if (!r.ok) throw new Error(d.detail || (mode === 'login' ? 'Login failed' : 'Registration failed'));
      onLogin(d);
    } catch (e) { setError(e.message); } finally { setLoading(false); }
  };

  return (
    <main className="login-screen">
      <div className="login-brand">
        <div>
          <div className="login-brand-logo"><img src={logoImg} alt="DuckRAG logo" style={{ width: 26, height: 26, objectFit: 'contain', filter: 'brightness(0) invert(1)' }} /></div>
          <div className="login-brand-name">DuckRAG</div>
          <div className="login-brand-sub">Technical Intelligence</div>
          <div className="login-tagline">Engineering Knowledge. Retrieved in Seconds.</div>
          <p className="login-brand-desc">
            Industrial knowledge intelligence for thermal power plant operations, maintenance, and engineering teams.
          </p>
          <div className="login-features">
            <div className="login-feat"><CheckCircle2 size={14} /><span>Answers grounded entirely in your own plant documents</span></div>
            <div className="login-feat"><CheckCircle2 size={14} /><span>Procedures, specs, and safety protocols — structured, not paraphrased</span></div>
            <div className="login-feat"><CheckCircle2 size={14} /><span>Source citations with document reference per response</span></div>
            <div className="login-feat"><CheckCircle2 size={14} /><span>Isolated session context with full query history</span></div>
          </div>
        </div>
        <p className="login-brand-foot">© 2026 DuckRAG · Enterprise Edition</p>
      </div>

      <div className="login-form-panel">
        <div className="login-card">
          <div className="login-mobile-logo">
            <div className="lml-icon"><img src={logoImg} alt="DuckRAG logo" style={{ width: 18, height: 18, objectFit: 'contain', filter: 'brightness(0) invert(1)' }} /></div>
            DuckRAG
          </div>
          <h2 className="login-card-title">
            {mode === 'login' ? 'Sign in to your workspace' : 'Create your account'}
          </h2>
          <div className="auth-tabs">
            <button type="button" className={`auth-tab${mode === 'login' ? ' active' : ''}`} onClick={() => switchMode('login')}>Sign in</button>
            <button type="button" className={`auth-tab${mode === 'signup' ? ' active' : ''}`} onClick={() => switchMode('signup')}>Create account</button>
          </div>
          <form className="login-form" onSubmit={handleSubmit}>
            <label>Username
              <input type="text" value={username} onChange={e => setUsername(e.target.value)} required placeholder={mode === 'login' ? 'Enter your username' : 'Choose a username (3–20 chars)'} autoComplete="username" />
            </label>
            <label>Password
              <input type="password" value={password} onChange={e => setPassword(e.target.value)} required placeholder={mode === 'login' ? 'Enter your password' : 'Minimum 6 characters'} autoComplete={mode === 'login' ? 'current-password' : 'new-password'} />
            </label>
            {mode === 'signup' && (
              <label>Confirm password
                <input type="password" value={confirm} onChange={e => setConfirm(e.target.value)} required placeholder="Re-enter your password" autoComplete="new-password" />
              </label>
            )}
            {error && <div className="form-error">{error}</div>}
            <button type="submit" className="btn-primary full-w" disabled={loading}>
              {loading ? <Loader2 className="spin" size={14} /> : <ShieldCheck size={14} />}
              {loading ? (mode === 'login' ? 'Signing in…' : 'Creating…') : (mode === 'login' ? 'Sign in' : 'Create account')}
            </button>
          </form>
        </div>
      </div>
    </main>
  );
}

/* ════════════════════════════════════════════════════════════
   Left Sidebar
   ════════════════════════════════════════════════════════════ */

function LeftSidebar({ chats, currentChatId, isOpen, loading, user, onSelectChat, onNewChat, onDeleteChat, onLogout, onOpenKb, onOpenHelp, onOpenCommunity, activePage }) {
  const [historyOpen, setHistoryOpen] = useState(true);

  return (
    <aside className={`sidebar ${isOpen ? 'open' : 'closed'}`}>
      {/* Brand */}
      <div className="sb-brand">
        <div className="sb-logo"><img src={logoImg} alt="DuckRAG logo" style={{ width: 18, height: 18, objectFit: 'contain', filter: 'brightness(0) invert(1)' }} /></div>
        <div className="sb-brand-text">
          <div className="sb-name">DuckRAG</div>
          <div className="sb-sub">Technical Intelligence</div>
        </div>
      </div>

      <div className="sb-body">
        {/* New Chat */}
        <button className="sb-new-chat-row" onClick={onNewChat}>
          <MessageSquarePlus size={15} />
          <span>New Chat</span>
        </button>

        {/* Nav + history */}
        <nav className="sb-nav">
          {/* Collapsible Chat History section */}
          <button
            className="sb-section-header"
            onClick={() => setHistoryOpen(v => !v)}
          >
            <div className="sb-section-header-left">
              <MessageSquare size={13} />
              <span>Chat History</span>
              {chats.length > 0 && <span className="sb-badge">{chats.length}</span>}
            </div>
            <ChevronDown size={13} className={`sb-section-chevron${historyOpen ? '' : ' collapsed'}`} />
          </button>

          <div className={`sb-history-list${historyOpen ? ' open' : ''}`}>
            {loading && [0,1,2].map(i => <div key={i} className="sb-skeleton" />)}
            {!loading && chats.length === 0 && (
              <div className="sb-empty-history">
                <Clock size={13} />
                <span>No chats yet</span>
              </div>
            )}
            {!loading && chats.map(c => (
              <button
                key={c.id}
                className={`sb-nav-item${c.id === currentChatId ? ' active' : ''}`}
                onClick={() => onSelectChat(c.id)}
              >
                <MessageSquare size={13} />
                <span>{c.title || 'Untitled Chat'}</span>
                <Trash2
                  size={12} className="sb-del"
                  onClick={e => { e.stopPropagation(); onDeleteChat(c.id); }}
                />
              </button>
            ))}
          </div>

          <div className="sb-sep" />

        </nav>
      </div>

      {/* Footer — Fix 3: Pro button opens localducks.com/contact */}
      <div className="sb-footer">
        <div className="sb-pro-card">
          <div className="sb-pro-label">PRO PLAN</div>
          <button
            className="sb-pro-btn"
            onClick={() => window.open('https://www.localducks.com/contact', '_blank')}
          >
            Upgrade to Pro
          </button>
        </div>
        <nav className="sb-footer-nav">
          <button className="sb-nav-item" onClick={onOpenHelp}>
            <HelpCircle size={15} />
            <span>Help</span>
          </button>
          <button className="sb-nav-item" onClick={onLogout}>
            <LogOut size={15} />
            <span>Sign Out</span>
          </button>
        </nav>
      </div>
    </aside>
  );
}

/* ════════════════════════════════════════════════════════════
   Chat Area
   ════════════════════════════════════════════════════════════ */


function ChatArea({
  chat, user, loadingChat, documents, chatCount,
  sidebarOpen, onOpenSidebar, onToggleSidebar, setUser,
  onLocalUpdate, onRefreshChats, onPatchChat, onCreateChat, onLimitExceeded,
  onOpenKb, onOpenCommunity,
}) {
  const [input, setInput]           = useState('');
  const [sending, setSending]       = useState(false);
  const [selFile, setSelFile]       = useState(null);
  const [showMenu, setShowMenu]       = useState(false);
  const [pendingQ, setPendingQ]       = useState('');
  const [searchVal, setSearchVal]     = useState('');
  const [searchOpen, setSearchOpen]   = useState(false);
  const [recent, setRecent]           = useState(() => {
    try { return JSON.parse(localStorage.getItem('dr_recent') || '[]'); } catch { return []; }
  });

  const menuRef    = useRef(null);
  const searchRef  = useRef(null);
  const inputRef   = useRef(null);
  const endRef     = useRef(null);

  const searchResults = searchVal.trim()
    ? documents.filter(d => (d.filename || d).toLowerCase().includes(searchVal.toLowerCase()))
    : [];

  const messages = chat?.messages || [];

  useEffect(() => {
    const h = e => { if ((e.ctrlKey || e.metaKey) && e.key === 'k') { e.preventDefault(); inputRef.current?.focus(); } };
    document.addEventListener('keydown', h);
    return () => document.removeEventListener('keydown', h);
  }, []);

  useEffect(() => {
    const h = e => { if (menuRef.current && !menuRef.current.contains(e.target)) setShowMenu(false); };
    document.addEventListener('mousedown', h);
    return () => document.removeEventListener('mousedown', h);
  }, []);

  useEffect(() => {
    const h = e => { if (searchRef.current && !searchRef.current.contains(e.target)) setSearchOpen(false); };
    document.addEventListener('mousedown', h);
    return () => document.removeEventListener('mousedown', h);
  }, []);

  useEffect(() => { endRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [messages.length, sending]);

  useEffect(() => { if (chat?.id && pendingQ) { handleSend(pendingQ); setPendingQ(''); } }, [chat?.id]);

  const pairs = useMemo(() => {
    const out = [];
    let i = 0;
    while (i < messages.length) {
      const q = messages[i];
      if (q.role === 'user') {
        const a = messages[i + 1]?.role === 'ai' ? messages[i + 1] : null;
        out.push({ q, a, idx: i });
        i += a ? 2 : 1;
      } else { i++; }
    }
    return out;
  }, [messages]);

  const handleSend = async (value = input) => {
    const query = value.trim();
    if (!query || sending) return;
    if (!chat) { await onCreateChat(); return; }

    const uMsg = { role: 'user', content: query };
    const optimistic = [...messages, uMsg];
    const title = messages.length === 0 ? query.slice(0, 60) : chat.title;

    onLocalUpdate({ ...chat, title, messages: optimistic });
    onPatchChat(chat.id, { title });
    setInput('');
    setSending(true);

    const updR = [query, ...recent.filter(q => q !== query)].slice(0, 6);
    setRecent(updR);
    localStorage.setItem('dr_recent', JSON.stringify(updR));

    try {
      const r = await fetch(`${API_URL}/api/chat`, {
        method: 'POST',
        headers: { ...authHeaders(user), 'Content-Type': 'application/json' },
        body: JSON.stringify({ chat_id: chat.id, query, source_file: selFile || null }),
      });
      const d = await r.json();

      if (!r.ok) {
        if (r.status === 402) { onLimitExceeded(); onLocalUpdate(chat); return; }
        throw new Error(d.detail || 'Request failed');
      }

      if (d.remaining_queries !== undefined && d.remaining_queries >= 0) {
        const u2 = { ...user, remaining: d.remaining_queries };
        setUser(u2); localStorage.setItem('user', JSON.stringify(u2));
      }

      const aMsg = { role: 'ai', content: d.answer, sources: d.sources || [] };
      onLocalUpdate({ ...chat, title, messages: [...optimistic, aMsg] });
      await onRefreshChats();
    } catch (e) {
      onLocalUpdate({ ...chat, messages: [...optimistic, { role: 'ai', content: `Error: ${e.message}`, sources: [] }] });
    } finally { setSending(false); }
  };

  const handleQueryAction = q => {
    if (chat) { handleSend(q); }
    else { setPendingQ(q); onCreateChat(); }
  };

  return (
    <div className="app-content">
      <header className="top-header">
        <div className="header-left">
          <button className="icon-btn" onClick={onToggleSidebar} title="Toggle sidebar">
            <PanelLeftClose size={15} />
          </button>
          <div className="search-wrap" ref={searchRef}>
            <Search size={13} className="search-icon" />
            <input
              className="search-input"
              placeholder="Search documents…"
              value={searchVal}
              onChange={e => { setSearchVal(e.target.value); setSearchOpen(true); }}
              onFocus={() => setSearchOpen(true)}
              onKeyDown={e => {
                if (e.key === 'Enter' && searchVal.trim()) {
                  setSearchOpen(false);
                  setSearchVal('');
                  handleQueryAction(searchVal.trim());
                }
                if (e.key === 'Escape') { setSearchOpen(false); setSearchVal(''); }
              }}
            />
            {searchOpen && searchVal.trim() && (
              <div className="search-dropdown">
                {searchResults.length > 0 ? searchResults.map(doc => {
                  const fn = doc.filename || doc;
                  return (
                    <button
                      key={fn}
                      className="search-dropdown-item"
                      onClick={() => {
                        setSearchOpen(false);
                        setSearchVal('');
                        window.open(`${API_URL}/api/documents/${encodeURIComponent(fn)}/view?token=${user.token}`, '_blank');
                      }}
                    >
                      <FileText size={12} />
                      <span>{fn}</span>
                    </button>
                  );
                }) : (
                  <button
                    className="search-dropdown-item search-query-item"
                    onClick={() => { setSearchOpen(false); setSearchVal(''); handleQueryAction(searchVal.trim()); }}
                  >
                    <Search size={12} />
                    <span>Ask: "{searchVal.trim()}"</span>
                  </button>
                )}
              </div>
            )}
          </div>
        </div>

        <nav className="header-nav">
          <button className="header-nav-btn" onClick={() => onOpenKb('All Documents')}>
            <Database size={14} />
            <span>Knowledge Base</span>
          </button>
          <button className="header-nav-btn" onClick={onOpenCommunity}>
            <Globe size={14} />
            <span>Community</span>
          </button>
          <button className="header-nav-btn" onClick={() => onOpenKb()}>
            <Settings size={14} />
            <span>Settings</span>
          </button>
        </nav>

        <div className="header-actions">
          <div className="user-avatar" title={user.username}>
            {user.username[0].toUpperCase()}
          </div>
        </div>
      </header>

      {/* ── Body — Fix 4: no right panel ── */}
      <div className="app-body">
        <div className="workspace-area">
          <div className="workspace-content">
            {loadingChat ? (
              <div className="center-state"><Loader2 className="spin" size={20} /><span>Loading session…</span></div>
            ) : messages.length === 0 ? (
              /* Fix 2: simple welcome view with rotating suggestions */
              <WelcomeView onSend={handleQueryAction} />
            ) : (
              pairs.map(({ q, a, idx }) => (
                <QueryBlock key={idx} question={q} answer={a} onFollowUp={handleQueryAction} documents={documents} user={user} />
              ))
            )}
            {sending && (
              <div className="ai-response-row fade-in">
                <div className="ai-icon-circle"><Bot size={20} /></div>
                <div className="ai-content">
                  <div className="ai-searching">
                    <Loader2 className="spin" size={13} />
                    <span>Searching knowledge base and retrieving documents…</span>
                  </div>
                </div>
              </div>
            )}
            <div ref={endRef} />
          </div>

          {/* ── Query bar ── */}
          <div className="qbar-outer">
            <div className="qbar-container">
              {/* Chips */}
              <div className="qbar-chips-row">
                {selFile ? (
                  <span className="qbar-chip active">
                    <FileText size={10} />
                    <span>{selFile}</span>
                    <button type="button" className="chip-clear" onClick={() => setSelFile(null)}><X size={9} /></button>
                  </span>
                ) : (
                  <div className="qbar-scope-wrap" ref={menuRef}>
                    <button type="button" className="qbar-chip" onClick={() => documents.length > 0 && setShowMenu(v => !v)}>
                      <SlidersHorizontal size={10} />
                      <span>ALL DOCUMENTS</span>
                      {documents.length > 0 && <ChevronDown size={9} className={showMenu ? 'rot' : ''} />}
                    </button>
                    {showMenu && (
                      <div className="qbar-scope-menu">
                        {documents.map(doc => {
                          const fn = doc.filename || doc;
                          return (
                            <button key={fn} type="button" onClick={() => { setSelFile(fn); setShowMenu(false); }}>
                              <FileText size={10} />{fn}
                            </button>
                          );
                        })}
                      </div>
                    )}
                  </div>
                )}
              </div>

              {/* Input */}
              <form className="qbar-input-row" onSubmit={e => { e.preventDefault(); handleSend(); }}>
                <Paperclip size={14} className="qbar-attach-icon" />
                <input
                  ref={inputRef}
                  className="qbar-input"
                  type="text"
                  placeholder={chat ? 'Ask a follow-up question…' : 'Ask anything about your plant documents…'}
                  value={input}
                  onChange={e => setInput(e.target.value)}
                  disabled={sending}
                />
                <button className="qbar-send-btn" disabled={!input.trim() || sending}>
                  {sending ? <Loader2 className="spin" size={14} /> : <ArrowUp size={14} />}
                </button>
              </form>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

/* ════════════════════════════════════════════════════════════
   Fix 2: Welcome View — simple greeting + rotating suggestions
   ════════════════════════════════════════════════════════════ */

function WelcomeView({ onSend }) {
  /* Pick 4 random suggestions on each mount (changes per new chat) */
  const suggestions = useMemo(() => {
    return [...ALL_SUGGESTIONS].sort(() => Math.random() - 0.5).slice(0, 4);
  }, []);

  return (
    <div className="welcome-view">
      <div className="welcome-duck">
        <Bot size={28} />
      </div>
      <h1 className="welcome-heading">How can I help you today?</h1>
      <p className="welcome-sub">
        Ask anything about your thermal plant manuals, SOPs, safety procedures, equipment specs.
      </p>
      <div className="welcome-suggestions">
        {suggestions.map(q => (
          <button key={q} className="suggestion-btn" onClick={() => onSend(q)}>
            {q}
          </button>
        ))}
      </div>
    </div>
  );
}

/* ════════════════════════════════════════════════════════════
   Query Block — bubble style
   ════════════════════════════════════════════════════════════ */

function QueryBlock({ question, answer, onFollowUp, documents, user }) {
  const parsed = useMemo(() => parseAIResponse(answer?.content || ''), [answer?.content]);
  const [shared, setShared]   = useState(false);
  const [sharing, setSharing] = useState(false);

  const getDocMeta = filename => documents.find(d => (d.filename || d) === filename) || null;

  const handleShare = async () => {
    if (!answer || sharing || shared) return;
    setSharing(true);
    try {
      const r = await fetch(`${API_URL}/api/community`, {
        method: 'POST',
        headers: { ...authHeaders(user), 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: question.content, answer: answer.content, sources: answer.sources || [] }),
      });
      if (r.ok) setShared(true);
    } catch (e) { console.error(e); }
    finally { setSharing(false); }
  };

  return (
    <div className="qa-block fade-in">
      {/* User bubble — right aligned */}
      <div className="user-msg-row">
        <div className="user-bubble">{question.content}</div>
      </div>

      {/* AI response */}
      {answer && (
        <div className="ai-response-row">
          <div className="ai-icon-circle">
            <Bot size={20} />
          </div>
          <div className="ai-content">
            {parsed.heading && (
              <h2 className="ai-response-heading">{parsed.heading}</h2>
            )}
            <div className="markdown-body">
              <ReactMarkdown>{parsed.body || answer.content}</ReactMarkdown>
            </div>

            {answer.sources?.length > 0 && (
              <>
                <hr className="sources-hr" />
                <div className="verified-sources-row">
                  <span className="verified-sources-label">VERIFIED SOURCES</span>
                  <div className="source-chips-row">
                    {answer.sources.map((src) => {
                      const meta = getDocMeta(src);
                      const viewUrl = meta && user
                        ? `${API_URL}/api/documents/${encodeURIComponent(src)}/view?token=${user.token}`
                        : null;
                      return (
                        <button
                          key={src}
                          className="source-chip"
                          onClick={() => viewUrl && window.open(viewUrl, '_blank')}
                          title={src}
                        >
                          <FileText size={11} />
                          {src}
                        </button>
                      );
                    })}
                  </div>
                </div>
              </>
            )}
            <div className="qa-actions-row">
              <button
                className={`share-btn${shared ? ' shared' : ''}`}
                onClick={handleShare}
                disabled={sharing || shared}
                title={shared ? 'Shared to community' : 'Share with community'}
              >
                {sharing ? <Loader2 size={12} className="spin" /> : <Share2 size={12} />}
                <span>{shared ? 'Shared' : 'Share with Community'}</span>
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

/* ════════════════════════════════════════════════════════════
   Upload Modal
   ════════════════════════════════════════════════════════════ */

function UploadModal({ onClose, user, filter = 'All Documents' }) {
  const [file, setFile]         = useState(null);
  const [dragging, setDragging] = useState(false);
  const [uploading, setUp]      = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError]       = useState('');
  const [success, setSuccess]   = useState('');
  const [docs, setDocs]         = useState([]);
  const [ldDocs, setLdDocs]     = useState(true);
  const fileRef = useRef(null);

  const isValid = f => f && ['.pdf', '.docx'].some(e => f.name.toLowerCase().endsWith(e));

  const pickFile = f => {
    setError(''); setSuccess('');
    if (!isValid(f)) { setError('Only PDF and DOCX files are supported.'); return; }
    setFile(f);
  };

  const fetchDocs = async () => {
    setLdDocs(true);
    try {
      const r = await fetch(`${API_URL}/api/documents`, { headers: authHeaders(user) });
      const d = await r.json();
      if (r.ok) setDocs(d.documents || []);
    } catch (e) { console.error(e); } finally { setLdDocs(false); }
  };

  useEffect(() => { fetchDocs(); }, []);

  const handleDelete = async fn => {
    try {
      const r = await fetch(`${API_URL}/api/documents/${encodeURIComponent(fn)}`, { method: 'DELETE', headers: authHeaders(user) });
      if (r.ok) fetchDocs();
    } catch (e) { console.error(e); }
  };

  const handleUpload = async () => {
    if (!file) return;
    setUp(true); setProgress(12); setError(''); setSuccess('');
    const iv = setInterval(() => setProgress(v => Math.min(v + 9, 88)), 420);
    const fd = new FormData(); fd.append('file', file);
    try {
      const r = await fetch(`${API_URL}/api/upload`, { method: 'POST', headers: authHeaders(user), body: fd });
      const d = await r.json();
      if (!r.ok) throw new Error(d.detail || 'Upload failed');
      setProgress(100); setSuccess(`Ingested — ${d.filename}`); setFile(null);
      if (d.remaining_uploads !== undefined) localStorage.setItem('user', JSON.stringify({ ...user, remaining_uploads: d.remaining_uploads }));
      fetchDocs();
    } catch (e) { setProgress(0); setError(e.message); }
    finally { clearInterval(iv); setUp(false); }
  };

  return (
    <div className="modal-overlay">
      <section className="modal-card doc-modal">
        <div className="modal-header">
          <div>
            <h2>{filter}</h2>
            <p>{user.role === 'demo' && user.remaining_uploads >= 0
              ? `${user.remaining_uploads} of 10 uploads remaining on demo plan`
              : 'Upload plant manuals, SOPs, and technical documents for retrieval.'}</p>
          </div>
          <button className="btn-icon" onClick={onClose}><X size={15} /></button>
        </div>

        <div
          className={`upload-zone${dragging ? ' drag-over' : ''}`}
          onDragOver={e => { e.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={e => { e.preventDefault(); setDragging(false); pickFile(e.dataTransfer.files[0]); }}
          onClick={() => !uploading && fileRef.current?.click()}
        >
          <Upload size={16} className="up-icon" />
          <div className="upload-label">
            {file ? <span className="file-sel">{file.name}</span> : <span>Drop a PDF or DOCX here, or <u>browse files</u></span>}
          </div>
          <button className="btn-primary compact" onClick={e => { e.stopPropagation(); handleUpload(); }} disabled={!file || uploading}>
            {uploading ? <Loader2 className="spin" size={14} /> : <Upload size={14} />}
            {uploading ? 'Ingesting…' : 'Upload'}
          </button>
          <input ref={fileRef} type="file" accept=".pdf,.docx" style={{ display: 'none' }} onChange={e => pickFile(e.target.files[0])} disabled={uploading} />
        </div>

        {uploading && <div className="progress-wrap"><div className="progress-bar" style={{ width: `${progress}%` }} /></div>}
        {error   && <div className="msg-err">{error}</div>}
        {success && <div className="msg-ok"><CheckCircle2 size={13} />{success}</div>}

        <div className="doc-list" style={{ marginTop: 16 }}>
          {ldDocs ? (
            <div className="center-row"><Loader2 className="spin" size={15} /> Loading…</div>
          ) : docs.length === 0 ? (
            <div className="center-row">No documents yet</div>
          ) : docs.map(doc => {
            const fn = doc.filename || doc;
            return (
              <div key={fn} className="doc-item">
                <FileText size={12} style={{ color: 'var(--primary)', flexShrink: 0 }} />
                <button className="doc-name-btn" onClick={() => window.open(`${API_URL}/api/documents/${encodeURIComponent(fn)}/view?token=${user.token}`, '_blank')}>{fn}</button>
                <button className="btn-icon danger" onClick={() => handleDelete(fn)} title="Delete"><Trash2 size={12} /></button>
              </div>
            );
          })}
        </div>
      </section>
    </div>
  );
}

/* ════════════════════════════════════════════════════════════
   Upgrade Modal
   ════════════════════════════════════════════════════════════ */

function UpgradeModal({ onClose }) {
  return (
    <div className="modal-overlay">
      <section className="modal-card">
        <div className="upgrade-modal-inner">
          <div className="upgrade-icon"><ShieldCheck size={24} /></div>
          <h2>Query Limit Reached</h2>
          <p>You have used all available queries on the demo plan. Contact your administrator to upgrade access.</p>
          <button className="btn-primary compact" onClick={onClose}>Close</button>
        </div>
      </section>
    </div>
  );
}

/* ════════════════════════════════════════════════════════════
   Community Page
   ════════════════════════════════════════════════════════════ */

function CommunityPage({ user, onToggleSidebar, onBack }) {
  const [posts, setPosts]       = useState([]);
  const [loading, setLoading]   = useState(true);
  const [deleting, setDeleting] = useState(null);

  const fetchPosts = async () => {
    setLoading(true);
    try {
      const r = await fetch(`${API_URL}/api/community`, { headers: authHeaders(user) });
      const d = await r.json();
      if (r.ok) setPosts(d.posts || []);
    } catch (e) { console.error(e); }
    finally { setLoading(false); }
  };

  useEffect(() => { fetchPosts(); }, []);

  const handleDelete = async (id) => {
    setDeleting(id);
    try {
      const r = await fetch(`${API_URL}/api/community/${id}`, { method: 'DELETE', headers: authHeaders(user) });
      if (r.ok) setPosts(p => p.filter(x => x.id !== id));
    } catch (e) { console.error(e); }
    finally { setDeleting(null); }
  };

  const canDelete = (post) => user.role === 'admin' || post.shared_by === user.username;

  return (
    <div className="app-content">
      <header className="top-header">
        <div className="header-left">
          <button className="icon-btn" onClick={onToggleSidebar} title="Toggle sidebar">
            <PanelLeftClose size={15} />
          </button>
          <button className="back-btn" onClick={onBack} title="Back to chat">
            <ChevronRight size={14} style={{ transform: 'rotate(180deg)' }} />
            <span>Back</span>
          </button>
          <div className="community-page-title">
            <Globe size={16} />
            <span>Community Feed</span>
          </div>
        </div>
        <div className="header-actions">
          <div className="user-avatar" title={user.username}>
            {user.username[0].toUpperCase()}
          </div>
        </div>
      </header>

      <div className="community-page-body">
        <div className="community-page-inner">
          <div className="community-page-head">
            <h1>Community Feed</h1>
            <p>Q&amp;A pairs shared by your team — browse plant knowledge contributed from live sessions.</p>
          </div>

          {loading ? (
            <div className="center-state"><Loader2 className="spin" size={20} /><span>Loading posts…</span></div>
          ) : posts.length === 0 ? (
            <div className="community-empty">
              <Globe size={40} />
              <p>No posts yet.<br />Share a response from a chat session to contribute to the community.</p>
            </div>
          ) : (
            <div className="community-feed">
              {posts.map(post => (
                <div key={post.id} className="community-post">
                  <div className="community-post-header">
                    <div className="community-post-meta">
                      <User size={12} />
                      <span>{post.shared_by}</span>
                      <span className="community-dot">·</span>
                      <Clock size={11} />
                      <span>{new Date(post.shared_at).toLocaleDateString(undefined, { day: 'numeric', month: 'short', year: 'numeric' })}</span>
                    </div>
                    {canDelete(post) && (
                      <button
                        className="btn-icon danger"
                        onClick={() => handleDelete(post.id)}
                        disabled={deleting === post.id}
                        title="Delete post"
                      >
                        {deleting === post.id ? <Loader2 size={12} className="spin" /> : <Trash2 size={12} />}
                      </button>
                    )}
                  </div>

                  <div className="community-question">
                    <MessageSquare size={13} />
                    <span>{post.question}</span>
                  </div>

                  <div className="community-answer markdown-body">
                    <ReactMarkdown>{post.answer}</ReactMarkdown>
                  </div>

                  {post.sources?.length > 0 && (
                    <div className="community-sources">
                      {post.sources.map(src => (
                        <span key={src} className="source-chip sm">
                          <FileText size={10} />{src}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

/* ════════════════════════════════════════════════════════════
   Help Modal
   ════════════════════════════════════════════════════════════ */

const HELP_SECTIONS = [
  {
    icon: <MessageSquare size={16} />,
    title: 'Asking Questions',
    steps: [
      'Type your question in the input bar at the bottom and press Enter or click the arrow button.',
      'Use the suggestion cards on the welcome screen to get started quickly.',
      'You can ask follow-up questions inside the same chat — the system keeps context across the conversation.',
      'Press Ctrl + K (or ⌘ K on Mac) to quickly focus the input bar.',
    ],
  },
  {
    icon: <Database size={16} />,
    title: 'Knowledge Base & Documents',
    steps: [
      'Click "Knowledge Base" in the left sidebar to open the document manager.',
      'Upload PDF or DOCX files — plant manuals, SOPs, compliance reports, inspection checklists, etc.',
      'Once uploaded, every new query searches across all your documents automatically.',
      'To search only one document, click the "ALL DOCUMENTS" chip above the input bar and pick a file.',
      'Click any source chip under an answer to open the original document in a new tab.',
    ],
  },
  {
    icon: <MessageSquarePlus size={16} />,
    title: 'Managing Chats',
    steps: [
      'Click "New Chat" in the sidebar to start a fresh session.',
      'All previous chats are listed in the sidebar — click any to reopen it.',
      'Hover over a chat and click the trash icon to delete it permanently.',
    ],
  },
  {
    icon: <ShieldCheck size={16} />,
    title: 'Answers & Sources',
    steps: [
      'Every answer is retrieved directly from your uploaded documents — nothing is made up.',
      '"VERIFIED SOURCES" chips appear below each answer, showing which files the answer came from.',
      'Always verify safety-critical numbers (temperatures, pressures, alarm setpoints) against the original document.',
    ],
  },
  {
    icon: <SlidersHorizontal size={16} />,
    title: 'Header Tabs — Plant Ops, Safety, Compliance',
    steps: [
      'The three tabs at the top (Plant Ops, Safety, Compliance) let you set the context for your session.',
      'Selecting a tab visually flags the mode — use it to keep your team aligned on the query type.',
    ],
  },
  {
    icon: <Upload size={16} />,
    title: 'Demo Plan Limits',
    steps: [
      'Demo accounts have a fixed number of queries and document uploads.',
      'Your remaining queries are tracked automatically after each response.',
      'When you hit the limit a prompt will appear — contact your administrator to upgrade.',
    ],
  },
];

function HelpModal({ onClose }) {
  return (
    <div className="modal-overlay" onClick={onClose}>
      <section className="modal-card help-modal" onClick={e => e.stopPropagation()}>
        <div className="modal-header">
          <div>
            <h2>How DuckRAG Works</h2>
            <p>A quick guide to getting the most out of your plant knowledge assistant.</p>
          </div>
          <button className="btn-icon" onClick={onClose}><X size={15} /></button>
        </div>

        <div className="help-body">
          {HELP_SECTIONS.map(sec => (
            <div key={sec.title} className="help-section">
              <div className="help-section-title">
                {sec.icon}
                <span>{sec.title}</span>
              </div>
              <ul className="help-steps">
                {sec.steps.map((s, i) => (
                  <li key={i}>{s}</li>
                ))}
              </ul>
            </div>
          ))}

          <div className="help-tip">
            <Zap size={13} />
            <span><strong>Tip:</strong> The more specific your question (include equipment names, tag numbers, or document type), the more precise the answer.</span>
          </div>
        </div>
      </section>
    </div>
  );
}

export default App;
