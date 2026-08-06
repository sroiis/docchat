// Thin typed client for the docchat API.

const TOKEN_KEY = "docchat_token";

export function getToken(): string | null {
  return localStorage.getItem(TOKEN_KEY);
}

export function setToken(token: string | null): void {
  if (token) localStorage.setItem(TOKEN_KEY, token);
  else localStorage.removeItem(TOKEN_KEY);
}

export type User = { id: number; email: string };
export type Document = {
  doc_id: string;
  file_name: string;
  created_at: string;
  chunk_count: number;
};
export type Source = {
  doc_id: string;
  chunk_id: string;
  score: number;
  text: string;
};
export type Health = {
  status: string;
  version: string;
  auth_enabled: boolean;
  embedding_provider: string;
  llm_provider: string;
};

class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    ...(init.headers as Record<string, string>),
  };
  const token = getToken();
  if (token) headers["Authorization"] = `Bearer ${token}`;

  const res = await fetch(`/api${path}`, { ...init, headers });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail ?? detail;
    } catch {
      /* ignore */
    }
    throw new ApiError(res.status, detail);
  }
  return res.json() as Promise<T>;
}

export const api = {
  health: () => request<Health>("/health"),

  register: (email: string, password: string) =>
    request<{ access_token: string }>("/auth/register", {
      method: "POST",
      body: JSON.stringify({ email, password }),
    }),
  login: (email: string, password: string) =>
    request<{ access_token: string }>("/auth/login", {
      method: "POST",
      body: JSON.stringify({ email, password }),
    }),
  me: () => request<User>("/auth/me"),

  listDocuments: () =>
    request<{ documents: Document[] }>("/documents"),
  deleteDocument: (docId: string) =>
    request<{ deleted: string }>(`/documents/${encodeURIComponent(docId)}`, {
      method: "DELETE",
    }),

  ask: (question: string, k = 4) =>
    request<{ answer: string; sources: Source[]; confidence: number }>(
      "/ask",
      { method: "POST", body: JSON.stringify({ question, k }) }
    ),

  uploadDocument: async (file: File) => {
    const token = getToken();
    const headers: Record<string, string> = {};
    if (token) headers["Authorization"] = `Bearer ${token}`;
    const fd = new FormData();
    fd.append("file", file);
    const res = await fetch("/api/documents/upload", {
      method: "POST",
      headers,
      body: fd,
    });
    if (!res.ok) {
      throw new ApiError(res.status, "Upload failed");
    }
    return res.json();
  },

  // Stream a chat answer. onEvent receives {sources} then {delta} then {done}.
  chat: async (
    question: string,
    onEvent: (ev: { type: string; sources?: Source[]; text?: string }) => void,
    k = 4
  ): Promise<void> => {
    const token = getToken();
    const headers: Record<string, string> = { "Content-Type": "application/json" };
    if (token) headers["Authorization"] = `Bearer ${token}`;

    const res = await fetch("/api/chat", {
      method: "POST",
      headers,
      body: JSON.stringify({ question, k }),
    });
    if (!res.ok || !res.body) throw new ApiError(res.status, "Chat failed");

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const parts = buffer.split("\n\n");
      buffer = parts.pop() ?? "";
      for (const part of parts) {
        const line = part.trim();
        if (!line.startsWith("data:")) continue;
        try {
          onEvent(JSON.parse(line.slice(5).trim()));
        } catch {
          /* ignore malformed event */
        }
      }
    }
  },
};
