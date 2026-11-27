import { useState } from "react";
import { Loader2, Play } from "lucide-react";
import { api, fmt, Model } from "../api";
import Markdown from "./Markdown";
import Page, { Card } from "./Page";

export default function Playground({ models }: { models: Model[] }) {
  const [selected, setSelected] = useState<string[]>(["claims-small", "claims-alt-qwen-next"]);
  const [system, setSystem] = useState("You help a claims examiner. Be concise. Never decide coverage, eligibility or benefit amounts.");
  const [prompt, setPrompt] = useState("In two sentences: what should an examiner do when a police report is referenced in an accident claim but not attached?");
  const [results, setResults] = useState<any[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const toggle = (key: string) => setSelected((s) => s.includes(key) ? s.filter((k) => k !== key) : s.length < 4 ? [...s, key] : s);

  async function run() {
    setBusy(true); setError(null); setResults([]);
    try { setResults(await api.playground(selected, prompt, system)); }
    catch (e: any) { setError(String(e.message ?? e)); }
    finally { setBusy(false); }
  }

  return (
    <Page title="Playground" subtitle="Compare governed models on the same prompt: answer, latency, tokens and cost. Every call goes through Unity Gateway and is tracked.">
      <Card className="p-4 space-y-4">
        <div>
          <div className="text-xs text-muted mb-2">Models (up to 4)</div>
          <div className="flex flex-wrap gap-2">
            {models.map((m) => (
              <button key={m.key} onClick={() => toggle(m.key)}
                      className={`text-sm rounded-lg border px-3 py-1.5 transition ${selected.includes(m.key) ? "border-accent bg-accent/10 text-accent" : "border-line hover:bg-sunken"}`}>
                {m.key}<span className="text-muted ml-1.5 text-xs">{m.model_service.replace("system.ai.", "")}</span>
              </button>
            ))}
          </div>
        </div>
        <label className="block">
          <div className="text-xs text-muted mb-1">System prompt</div>
          <textarea value={system} onChange={(e) => setSystem(e.target.value)} rows={2}
                    className="w-full rounded-lg border border-line bg-bg px-3 py-2 text-sm outline-none focus:border-accent/60" />
        </label>
        <label className="block">
          <div className="text-xs text-muted mb-1">User prompt</div>
          <textarea value={prompt} onChange={(e) => setPrompt(e.target.value)} rows={3}
                    className="w-full rounded-lg border border-line bg-bg px-3 py-2 text-sm outline-none focus:border-accent/60" />
        </label>
        <button onClick={run} disabled={busy || !selected.length || !prompt.trim()}
                className="inline-flex items-center gap-2 rounded-lg bg-accent text-white px-4 py-2 text-sm disabled:opacity-40">
          {busy ? <Loader2 size={15} className="animate-spin" /> : <Play size={15} />} Run on {selected.length} model{selected.length === 1 ? "" : "s"}
        </button>
        {error && <div className="text-sm text-bad">{error}</div>}
      </Card>
      {results.length > 0 && (
        <div className={`grid gap-4 mt-6 ${results.length > 1 ? "md:grid-cols-2" : ""}`}>
          {results.map((r) => (
            <Card key={r.model} className="p-4 flex flex-col">
              <div className="flex items-center justify-between mb-2">
                <div className="font-medium text-sm">{r.model}</div>
                <span className={`text-[11px] px-1.5 py-0.5 rounded-md ${r.status === "ok" ? "bg-ok/10 text-ok" : "bg-bad/10 text-bad"}`}>{r.status}</span>
              </div>
              <div className="flex-1">{r.status === "ok" ? <Markdown text={r.text} /> : <div className="text-sm text-bad">{r.error}</div>}</div>
              <div className="grid grid-cols-3 gap-2 mt-3 pt-3 border-t border-line text-xs">
                <div><div className="text-muted">Latency</div>{fmt.ms(r.latency_ms)}</div>
                <div><div className="text-muted">Tokens in / out</div>{fmt.int(r.input_tokens)} / {fmt.int(r.output_tokens)}</div>
                <div><div className="text-muted">Cost</div>{fmt.usd(r.cost_usd)}</div>
              </div>
            </Card>
          ))}
        </div>
      )}
    </Page>
  );
}
