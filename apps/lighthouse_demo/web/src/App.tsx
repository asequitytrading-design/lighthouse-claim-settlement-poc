import { useCallback, useEffect, useState } from "react";
import { api, Model } from "./api";
import Sidebar, { View } from "./components/Sidebar";
import ChatView from "./components/ChatView";
import Playground from "./components/Playground";
import Dashboard from "./components/Dashboard";
import Evals from "./components/Evals";

function initialTheme(): "light" | "dark" {
  try {
    const saved = localStorage.getItem("lh-theme");
    if (saved === "light" || saved === "dark") return saved;
  } catch { /* storage unavailable */ }
  return window.matchMedia?.("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

const VIEWS: View[] = ["chat", "playground", "dashboard", "evals"];
const viewFromHash = (): View => (VIEWS.find((v) => window.location.hash.split("/")[0] === `#${v}`) ?? "chat");
const sessionFromHash = (): string | null => (window.location.hash.startsWith("#chat/") ? window.location.hash.slice(6) : null);

export default function App() {
  const [view, setViewState] = useState<View>(viewFromHash);
  const setView = (v: View) => { setViewState(v); window.history.replaceState(null, "", `#${v}`); };
  const [theme, setTheme] = useState<"light" | "dark">(initialTheme);
  const [models, setModels] = useState<Model[]>([]);
  const [sessions, setSessions] = useState<{ session_id: string; title: string; updated_at: string }[]>([]);
  const [sessionId, setSessionIdState] = useState<string | null>(sessionFromHash);
  const setSessionId = (id: string | null) => {
    setSessionIdState(id);
    window.history.replaceState(null, "", id ? `#chat/${id}` : "#chat");
  };
  const [health, setHealth] = useState<Record<string, string> | null>(null);

  useEffect(() => {
    document.documentElement.classList.toggle("dark", theme === "dark");
    try { localStorage.setItem("lh-theme", theme); } catch { /* ignore */ }
  }, [theme]);

  const refreshSessions = useCallback(() => { api.sessions().then(setSessions).catch(() => setSessions([])); }, []);
  useEffect(() => {
    api.models().then(setModels).catch(() => setModels([]));
    api.health().then(setHealth).catch(() => setHealth({ workspace_auth: "unreachable" }));
    refreshSessions();
  }, [refreshSessions]);

  return (
    <div className="flex h-full">
      <Sidebar
        view={view} onView={setView} sessions={sessions} activeSession={sessionId}
        onSelect={(id) => { setSessionId(id); setView("chat"); }}
        onNew={() => { setSessionId(null); setView("chat"); }}
        onDelete={async (id) => { await api.deleteSession(id); if (id === sessionId) setSessionId(null); refreshSessions(); }}
        theme={theme} onTheme={() => setTheme(theme === "dark" ? "light" : "dark")} health={health}
      />
      <main className="flex-1 min-w-0 flex flex-col">
        {view === "chat" && (
          <ChatView key={sessionId ?? "new"} sessionId={sessionId} models={models}
                    onSession={(id) => { setSessionId(id); refreshSessions(); }} />
        )}
        {view === "playground" && <Playground models={models} />}
        {view === "dashboard" && <Dashboard />}
        {view === "evals" && <Evals />}
      </main>
    </div>
  );
}
