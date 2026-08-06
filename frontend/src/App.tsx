import { useEffect, useState } from "react";
import {
  api,
  getToken,
  setToken,
  type Document,
  type Health,
  type Source,
} from "./api";

type Message =
  | { role: "user"; text: string }
  | { role: "assistant"; text: string; sources: Source[] };

export default function App() {
  const [token, setTokenState] = useState<string | null>(getToken());
  const [health, setHealth] = useState<Health | null>(null);

  useEffect(() => {
    api.health().then(setHealth).catch(() => {});
  }, []);

  if (!token) {
    return <AuthScreen onAuthed={(t) => { setToken(t); setTokenState(t); }} />;
  }
  return (
    <Shell
      health={health}
      onLogout={() => { setToken(null); setTokenState(null); }}
    />
  );
}

// ---- auth ------------------------------------------------------------------

function AuthScreen({ onAuthed }: { onAuthed: (token: string) => void }) {
  const [mode, setMode] = useState<"login" | "register">("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setError("");
    setBusy(true);
    try {
      const { access_token } =
        mode === "login"
          ? await api.login(email, password)
          : await api.register(email, password);
      onAuthed(access_token);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="auth-wrap">
      <form className="card auth" onSubmit={submit}>
        <h1>docchat</h1>
        <p className="sub">Chat with your documents. Offline-friendly RAG.</p>
        <input
          type="email" required placeholder="you@example.com" autoFocus
          value={email} onChange={(e) => setEmail(e.target.value)}
        />
        <input
          type="password" required placeholder="password (8+ chars)" minLength={8}
          value={password} onChange={(e) => setPassword(e.target.value)}
        />
        {error && <div className="error">{error}</div>}
        <button disabled={busy} className="primary">
          {busy ? "Please wait…" : mode === "login" ? "Log in" : "Create account"}
        </button>
        <button type="button" className="link" onClick={() => setMode(mode === "login" ? "register" : "login")}>
          {mode === "login"
            ? "No account? Register"
            : "Have an account? Log in"}
        </button>
      </form>
    </div>
  );
}

// ---- main app ----------------------------------------------------------------

function Shell({ health, onLogout }: { health: Health | null; onLogout: () => void }) {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  async function refreshDocs() {
    try {
      const res = await api.listDocuments();
      setDocuments(res.documents);
    } catch {
      /* keep last list */
    }
  }

  useEffect(() => { refreshDocs(); }, []);

  async function upload(file: File) {
    setError("");
    try {
      await api.uploadDocument(file);
      await refreshDocs();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed");
    }
  }

  async function remove(docId: string) {
    setError("");
    try {
      await api.deleteDocument(docId);
      await refreshDocs();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Delete failed");
    }
  }

  async function send(e?: React.FormEvent) {
    e?.preventDefault();
    const q = input.trim();
    if (!q || busy) return;
    setInput("");
    setError("");
    setMessages((m) => [...m, { role: "user", text: q }]);
    setMessages((m) => [...m, { role: "assistant", text: "", sources: [] }]);
    setBusy(true);
    let acc = "";
    let sources: Source[] = [];
    try {
      await api.chat(q, (ev) => {
        if (ev.type === "sources" && ev.sources) {
          sources = ev.sources;
          setMessages((m) => updateLast(m, (last) => ({ ...last, sources })));
        } else if (ev.type === "delta" && ev.text) {
          acc += ev.text;
          setMessages((m) => updateLast(m, (last) => ({ ...last, text: acc })));
        }
      });
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Chat failed";
      setMessages((m) =>
        updateLast(m, (last) => ({ ...last, text: last.text || msg }))
      );
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="layout">
      <aside className="sidebar card">
        <h1>docchat</h1>
        <div className="providers">
          {health && (
            <span className="pill">{health.embedding_provider} · {health.llm_provider}</span>
          )}
        </div>

        <label className="upload">
          <span>+ Upload .md / .txt</span>
          <input
            type="file" accept=".md,.txt" hidden
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) upload(f);
              e.target.value = "";
            }}
          />
        </label>

        <div className="docs">
          <h2>Documents</h2>
          {documents.length === 0 && <p className="empty">Nothing indexed yet.</p>}
          {documents.map((d) => (
            <div key={d.doc_id} className="doc">
              <span title={d.doc_id}>{d.file_name}</span>
              <small>{d.chunk_count} chunks</small>
              <button className="icon" onClick={() => remove(d.doc_id)}>×</button>
            </div>
          ))}
        </div>

        <button className="link" onClick={onLogout}>Log out</button>
      </aside>

      <main className="chat card">
        <div className="messages">
          {messages.length === 0 && (
            <p className="empty">
              Ask a question about your documents. Responses stream in live.
            </p>
          )}
          {messages.map((m, i) => (
            <MessageBubble key={i} m={m} />
          ))}
        </div>

        {error && <div className="error">{error}</div>}

        <form className="composer" onSubmit={send}>
          <input
            value={input}
            placeholder="Ask anything…"
            onChange={(e) => setInput(e.target.value)}
            disabled={busy}
            autoFocus
          />
          <button className="primary" disabled={busy || !input.trim()}>
            {busy ? "…" : "Send"}
          </button>
        </form>
      </main>
    </div>
  );
}

function MessageBubble({ m }: { m: Message }) {
  if (m.role === "user") {
    return <div className="bubble user"><p>{m.text}</p></div>;
  }
  return (
    <div className="bubble assistant">
      <pre className="answer">{m.text || (m.sources.length === 0 ? "Thinking…" : "")}</pre>
      {m.sources.length > 0 && (
        <details>
          <summary>Sources ({m.sources.length})</summary>
          {m.sources.map((s) => (
            <div key={s.chunk_id} className="source">
              <div className="source-head">
                {s.doc_id} — {Math.round(s.score * 100)}%
              </div>
              <p>{s.text}</p>
            </div>
          ))}
        </details>
      )}
    </div>
  );
}

function updateLast(messages: Message[], fn: (m: Message) => Message): Message[] {
  const copy = [...messages];
  copy[copy.length - 1] = fn(copy[copy.length - 1]);
  return copy;
}
