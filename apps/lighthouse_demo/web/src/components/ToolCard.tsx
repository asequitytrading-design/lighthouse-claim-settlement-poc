import { useState } from "react";
import { BookOpen, ChevronRight, ClipboardCheck, Database, FileText, Loader2, Wrench } from "lucide-react";
import { fmt } from "../api";

export type Step = {
  id: string; name: string; arguments: any; status?: string; summary?: string; latency_ms?: number; data?: any;
};

const TOOLS: Record<string, { label: string; running: string; via: string; icon: any }> = {
  get_claim_case: { label: "Case file", running: "Looking up the case file…", via: "Unity Catalog function · MCP", icon: FileText },
  search_guidance: { label: "Examiner guidance", running: "Searching the guidance and checking grounding…", via: "Vector Search · MCP · grounding check", icon: BookOpen },
  ask_portfolio: { label: "Portfolio question", running: "Genie is querying Unity Catalog…", via: "Genie · MCP", icon: Database },
  draft_handoff: { label: "Examiner handoff", running: "Drafting with the approved prompt…", via: "Prompt registry @champion", icon: ClipboardCheck },
};

function chip(status?: string) {
  if (!status) return "bg-accent/10 text-accent";
  if (status === "ok") return "bg-ok/10 text-ok";
  if (status === "empty") return "bg-warn/10 text-warn";
  return "bg-bad/10 text-bad";
}

export default function ToolCard({ step }: { step: Step }) {
  const [open, setOpen] = useState(false);
  const meta = TOOLS[step.name] ?? { label: step.name, running: "Running…", via: "tool", icon: Wrench };
  const Icon = meta.icon;
  const running = step.status === undefined;
  const d = step.data ?? {};
  return (
    <div className="rounded-xl border border-line bg-panel">
      <button onClick={() => !running && setOpen(!open)} className="w-full flex items-center gap-3 px-3 py-2 text-left">
        <div className="h-7 w-7 rounded-lg bg-sunken grid place-items-center text-muted"><Icon size={15} /></div>
        <div className="flex-1 min-w-0">
          <div className="text-sm font-medium flex items-center gap-2">
            {meta.label}
            <span className={`text-[11px] px-1.5 py-0.5 rounded-md ${chip(step.status)}`}>{running ? "running" : step.status}</span>
            {d.cache?.hit && <span className="text-[11px] px-1.5 py-0.5 rounded-md bg-accent/10 text-accent" title={`similarity ${d.cache.similarity}`}>cache hit</span>}
          </div>
          <div className="text-xs text-muted truncate">{running ? meta.running : String(step.summary ?? "").replace(" · cache hit", "")} · {d.cache?.hit ? "semantic cache" : meta.via}</div>
        </div>
        {running ? <Loader2 size={15} className="animate-spin text-muted" />
                 : <span className="text-xs text-muted">{fmt.ms(step.latency_ms)}</span>}
        {!running && <ChevronRight size={15} className={`text-muted transition ${open ? "rotate-90" : ""}`} />}
      </button>
      {open && (
        <div className="border-t border-line px-3 py-2 text-xs space-y-2">
          <div><span className="text-muted">Arguments </span><code className="bg-sunken rounded px-1">{JSON.stringify(step.arguments)}</code></div>
          {d.sql && <pre className="bg-sunken rounded-lg p-2 overflow-x-auto whitespace-pre-wrap">{d.sql}</pre>}
          {d.rows?.length > 0 && (
            <table className="text-xs">
              <thead><tr>{(d.columns ?? []).map((c: string) => <th key={c} className="text-left pr-3 text-muted font-medium">{c}</th>)}</tr></thead>
              <tbody>{d.rows.slice(0, 8).map((r: any[], i: number) => <tr key={i}>{r.map((v, j) => <td key={j} className="pr-3">{v}</td>)}</tr>)}</tbody>
            </table>
          )}
          {d.citations?.length > 0 && d.citations.map((c: any) => (
            <div key={c.chunk_id} className="border-l-2 border-accent pl-2">
              <div className="text-muted">{c.doc_id} · similarity {c.score}</div><div>{c.text}</div>
            </div>
          ))}
          {d.grounding_score != null && <div className="text-muted">Grounding score {Number(d.grounding_score).toFixed(2)}{d.refusal_reason ? ` · ${d.refusal_reason}` : ""}</div>}
          {d.prompt_version && <div className="text-muted">Prompt version v{d.prompt_version} (@champion)</div>}
        </div>
      )}
    </div>
  );
}
