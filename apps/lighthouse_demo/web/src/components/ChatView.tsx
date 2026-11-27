import { useEffect, useRef, useState } from "react";
import { ArrowUp, BookOpen, Database, FileText, ShieldCheck, Sparkles, Square } from "lucide-react";
import { AgentEvent, api, fmt, Model, streamChat } from "../api";
import Markdown from "./Markdown";
import ToolCard, { Step } from "./ToolCard";

type Turn = {
  role: "user" | "assistant"; content: string; steps?: Step[]; citations?: any[]; usage?: any;
  guardrails?: string[]; thinking?: string | null; error?: string; pending?: boolean;
};

const SUGGESTIONS = [
  { icon: FileText, title: "Prepare a handoff", text: "Prepare the handoff for claim CSET-LTD-002." },
  { icon: BookOpen, title: "Missing documents + guidance", text: "Which documents are missing for CSET-LTD-002 and what does our guidance say about them?" },
  { icon: Database, title: "Portfolio question", text: "How many long-term disability claims are missing a functional capacity evaluation?" },
  { icon: ShieldCheck, title: "Boundary test", text: "Is claim CSET-LIFE-003 fraudulent? Should we deny it?" },
];

export default function ChatView({ sessionId, models, onSession }: {
  sessionId: string | null; models: Model[]; onSession: (id: string) => void;
}) {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [input, setInput] = useState("");
  const [model, setModel] = useState<string>("");
  const [busy, setBusy] = useState(false);
  const currentSession = useRef<string | null>(sessionId);
  const abort = useRef<AbortController | null>(null);
  const bottom = useRef<HTMLDivElement>(null);

  useEffect(() => { if (!model && models.length) setModel((models.find((m) => m.default) ?? models[0]).key); }, [models, model]);
  useEffect(() => {
    if (!sessionId) return;
    api.session(sessionId).then((s) => setTurns(s.messages.map((m) => ({
      role: m.role as Turn["role"], content: m.content, citations: m.meta?.citations, usage: m.meta?.usage,
      guardrails: m.meta?.guardrails,
      steps: (m.meta?.tools ?? []).map((t: any, i: number) => ({ id: `${i}`, name: t.name, arguments: {}, status: t.status, summary: t.summary, latency_ms: t.latency_ms })),
    })))).catch(() => setTurns([]));
  }, [sessionId]);
  useEffect(() => { bottom.current?.scrollIntoView({ behavior: "smooth" }); }, [turns]);

  const update = (fn: (t: Turn) => Turn) => setTurns((all) => [...all.slice(0, -1), fn(all[all.length - 1])]);

  function handle(e: AgentEvent) {
    switch (e.type) {
      case "session": currentSession.current = e.session_id; onSession(e.session_id); break;
      case "model_call":
        update((t) => ({ ...t, thinking: e.status === "started" ? (e.round === 1 ? "Thinking" : "Reading the results")
                                      : e.status === "retry" ? `Retrying (${e.reason === "rate_limit" ? "rate limited" : "empty model reply"})` : null }));
        break;
      case "tool_call": update((t) => ({ ...t, steps: [...(t.steps ?? []), { id: e.id, name: e.name, arguments: e.arguments }] })); break;
      case "tool_result":
        update((t) => ({ ...t, steps: (t.steps ?? []).map((s) => s.id === e.id ? { ...s, status: e.status, summary: e.summary, latency_ms: e.latency_ms, data: e.data } : s) }));
        break;
      case "guardrail": update((t) => ({ ...t, guardrails: [...(t.guardrails ?? []), e.note] })); break;
      case "answer": update((t) => ({ ...t, content: e.text ?? "", citations: e.citations, thinking: null })); break;
      case "usage": update((t) => ({ ...t, usage: e })); break;
      case "error": update((t) => ({ ...t, error: `${e.category}: ${e.message}`, thinking: null })); break;
      case "done": update((t) => ({ ...t, pending: false, thinking: null })); break;
    }
  }

  async function send(text: string) {
    const message = text.trim();
    if (!message || busy) return;
    setInput("");
    setBusy(true);
    setTurns((all) => [...all, { role: "user", content: message }, { role: "assistant", content: "", steps: [], pending: true, thinking: "Thinking" }]);
    abort.current = new AbortController();
    try {
      await streamChat({ message, model, session_id: currentSession.current }, handle, abort.current.signal);
    } catch (err: any) {
      if (err?.name !== "AbortError") update((t) => ({ ...t, error: String(err?.message ?? err) }));
    } finally {
      update((t) => ({ ...t, pending: false, thinking: null }));
      setBusy(false);
    }
  }

  return (
    <div className="flex-1 flex flex-col min-h-0">
      <div className="flex-1 overflow-y-auto scrollbar-thin">
        <div className="max-w-3xl mx-auto px-6 py-8">
          {turns.length === 0 ? (
            <div className="pt-16">
              <div className="h-11 w-11 rounded-xl bg-accent/15 text-accent grid place-items-center mb-4"><Sparkles size={20} /></div>
              <h1 className="text-2xl font-semibold">How can I help with your claims today?</h1>
              <p className="text-muted mt-1">I look up case files in Unity Catalog, search the examiner guidance and ask Genie about the portfolio. I prepare files for a human examiner and never decide claims.</p>
              <div className="grid sm:grid-cols-2 gap-3 mt-8">
                {SUGGESTIONS.map(({ icon: Icon, title, text }) => (
                  <button key={title} onClick={() => send(text)}
                          className="text-left rounded-xl border border-line bg-panel p-4 hover:border-accent/50 hover:shadow-sm transition">
                    <div className="flex items-center gap-2 text-sm font-medium"><Icon size={15} className="text-accent" />{title}</div>
                    <div className="text-sm text-muted mt-1">{text}</div>
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div className="space-y-8">
              {turns.map((t, i) => t.role === "user" ? (
                <div key={i} className="flex justify-end">
                  <div className="max-w-[80%] rounded-2xl bg-sunken px-4 py-2.5 text-[15px] whitespace-pre-wrap">{t.content}</div>
                </div>
              ) : (
                <div key={i} className="flex gap-3">
                  <div className="h-7 w-7 shrink-0 rounded-lg bg-accent/15 text-accent grid place-items-center mt-0.5"><Sparkles size={14} /></div>
                  <div className="flex-1 min-w-0 space-y-3">
                    {(t.steps?.length ?? 0) > 0 && <div className="space-y-2">{t.steps!.map((s) => <ToolCard key={s.id} step={s} />)}</div>}
                    {t.thinking && (
                      <div className="flex items-center gap-2 text-sm text-muted">
                        <span className="flex gap-1"><span className="dot h-1.5 w-1.5 rounded-full bg-muted" /><span className="dot h-1.5 w-1.5 rounded-full bg-muted [animation-delay:.2s]" /><span className="dot h-1.5 w-1.5 rounded-full bg-muted [animation-delay:.4s]" /></span>
                        {t.thinking}…
                      </div>
                    )}
                    {t.content && <Markdown text={t.content} />}
                    {t.error && <div className="text-sm text-bad">Something went wrong: {t.error}</div>}
                    {(t.guardrails?.length ?? 0) > 0 && (
                      <div className="flex flex-wrap gap-1.5">
                        {t.guardrails!.map((g, j) => <span key={j} className="inline-flex items-center gap-1 text-[11px] rounded-md bg-ok/10 text-ok px-1.5 py-0.5"><ShieldCheck size={12} />{g}</span>)}
                      </div>
                    )}
                    {(t.citations?.length ?? 0) > 0 && (
                      <div className="flex flex-wrap gap-1.5">
                        {t.citations!.map((c: any) => (
                          <span key={c.chunk_id ?? c.doc_id} title={c.text} className="text-[11px] rounded-md border border-line px-1.5 py-0.5 text-muted">
                            <BookOpen size={11} className="inline mr-1" />{c.doc_id}
                          </span>
                        ))}
                      </div>
                    )}
                    {t.usage && !t.pending && (
                      <div className="text-[11px] text-muted flex flex-wrap gap-x-3">
                        <span>{t.usage.model}</span><span>{t.usage.model_calls} model · {t.usage.tool_calls} tool calls</span>
                        <span>{fmt.int((t.usage.input_tokens ?? 0) + (t.usage.output_tokens ?? 0))} tokens</span>
                        <span>{fmt.ms(t.usage.latency_ms)}</span><span>{fmt.usd(t.usage.cost_usd)}</span>
                      </div>
                    )}
                  </div>
                </div>
              ))}
              <div ref={bottom} />
            </div>
          )}
        </div>
      </div>
      <div className="px-6 pb-5">
        <div className="max-w-3xl mx-auto rounded-2xl border border-line bg-panel shadow-sm focus-within:border-accent/60 transition">
          <textarea value={input} onChange={(e) => setInput(e.target.value)} rows={1}
                    onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(input); } }}
                    placeholder="Ask about a claim, the guidance or the portfolio…"
                    className="w-full resize-none bg-transparent px-4 pt-3 pb-1 text-[15px] outline-none max-h-40" />
          <div className="flex items-center justify-between px-3 pb-2.5">
            <select value={model} onChange={(e) => setModel(e.target.value)}
                    className="text-xs bg-sunken rounded-lg px-2 py-1.5 border border-line outline-none w-auto max-w-[60%]">
              {models.map((m) => <option key={m.key} value={m.key}>{m.key} · {m.model_service.replace("system.ai.", "")}</option>)}
            </select>
            {busy ? (
              <button onClick={() => abort.current?.abort()} className="h-8 w-8 rounded-full bg-ink text-bg grid place-items-center" title="Stop"><Square size={13} /></button>
            ) : (
              <button onClick={() => send(input)} disabled={!input.trim()} className="h-8 w-8 rounded-full bg-accent text-white grid place-items-center disabled:opacity-30" title="Send"><ArrowUp size={16} /></button>
            )}
          </div>
        </div>
        <div className="max-w-3xl mx-auto text-center text-[11px] text-muted mt-2">Synthetic data · every model call is tracked through Unity Gateway · a human examiner makes all decisions</div>
      </div>
    </div>
  );
}
