import { BarChart3, FlaskConical, MessageSquare, Moon, Plus, Sparkles, Sun, Trash2, Wand2 } from "lucide-react";

export type View = "chat" | "playground" | "dashboard" | "evals";

const NAV: { id: View; label: string; icon: any }[] = [
  { id: "chat", label: "Copilot", icon: MessageSquare },
  { id: "playground", label: "Playground", icon: Wand2 },
  { id: "dashboard", label: "Dashboard", icon: BarChart3 },
  { id: "evals", label: "Evaluations", icon: FlaskConical },
];

type Props = {
  view: View; onView: (v: View) => void;
  sessions: { session_id: string; title: string }[]; activeSession: string | null;
  onSelect: (id: string) => void; onNew: () => void; onDelete: (id: string) => void;
  theme: "light" | "dark"; onTheme: () => void; health: Record<string, string> | null;
};

export default function Sidebar(p: Props) {
  const ok = p.health?.workspace_auth === "ok";
  return (
    <aside className="w-64 shrink-0 border-r border-line bg-panel flex flex-col">
      <div className="px-4 pt-4 pb-3 flex items-center gap-2">
        <div className="h-8 w-8 rounded-lg bg-accent/15 text-accent grid place-items-center"><Sparkles size={16} /></div>
        <div>
          <div className="font-semibold leading-tight">Lighthouse</div>
          <div className="text-xs text-muted leading-tight">Claims copilot · synthetic data</div>
        </div>
      </div>
      <div className="px-3">
        <button onClick={p.onNew}
                className="w-full flex items-center gap-2 rounded-lg border border-line px-3 py-2 text-sm hover:bg-sunken transition">
          <Plus size={16} /> New chat
        </button>
      </div>
      <nav className="px-3 mt-4 space-y-0.5">
        {NAV.map(({ id, label, icon: Icon }) => (
          <button key={id} onClick={() => p.onView(id)}
                  className={`w-full flex items-center gap-2.5 rounded-lg px-3 py-2 text-sm transition ${
                    p.view === id ? "bg-sunken font-medium" : "text-muted hover:bg-sunken hover:text-ink"}`}>
            <Icon size={16} /> {label}
          </button>
        ))}
      </nav>
      <div className="px-4 mt-5 mb-1 text-[11px] uppercase tracking-wide text-muted">Conversations</div>
      <div className="flex-1 overflow-y-auto scrollbar-thin px-2">
        {p.sessions.length === 0 && <div className="px-2 py-1 text-sm text-muted">No conversations yet</div>}
        {p.sessions.map((s) => (
          <div key={s.session_id}
               className={`group flex items-center rounded-lg ${p.activeSession === s.session_id && p.view === "chat" ? "bg-sunken" : "hover:bg-sunken"}`}>
            <button onClick={() => p.onSelect(s.session_id)} className="flex-1 min-w-0 text-left px-2 py-1.5 text-sm truncate">
              {s.title}
            </button>
            <button onClick={() => p.onDelete(s.session_id)} title="Delete"
                    className="opacity-0 group-hover:opacity-100 p-1.5 text-muted hover:text-bad"><Trash2 size={14} /></button>
          </div>
        ))}
      </div>
      <div className="border-t border-line px-4 py-3 flex items-center justify-between text-xs text-muted">
        <span className="flex items-center gap-1.5" title={p.health ? JSON.stringify(p.health, null, 1) : "checking"}>
          <span className={`h-2 w-2 rounded-full ${p.health == null ? "bg-muted" : ok ? "bg-ok" : "bg-bad"}`} />
          {p.health == null ? "Checking…" : ok ? "Databricks connected" : "Not connected"}
        </span>
        <button onClick={p.onTheme} className="p-1.5 rounded-md hover:bg-sunken" title="Toggle theme">
          {p.theme === "dark" ? <Sun size={15} /> : <Moon size={15} />}
        </button>
      </div>
    </aside>
  );
}
