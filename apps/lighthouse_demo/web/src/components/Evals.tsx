import { useEffect, useState } from "react";
import { Check, ExternalLink, Loader2, X } from "lucide-react";
import { api } from "../api";
import Page, { Card } from "./Page";

function verdictStyle(v: string) {
  if (v?.startsWith("READY")) return "bg-ok/10 text-ok";
  if (v === "BLOCKED") return "bg-bad/10 text-bad";
  return "bg-warn/10 text-warn";
}

export default function Evals() {
  const [runs, setRuns] = useState<any[]>([]);
  const [selected, setSelected] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.evalRuns().then(setRuns).catch((e) => setError(String(e.message ?? e))).finally(() => setLoading(false));
  }, []);

  async function open(runId: string) {
    setSelected({ loading: true, runId });
    try { setSelected(await api.evalRun(runId)); } catch (e: any) { setError(String(e.message ?? e)); setSelected(null); }
  }

  const cases = selected?.cases ?? [];
  const caseIds: string[] = Array.from(new Set(cases.map((c: any) => c.case_id)));
  const criteria: string[] = Array.from(new Set(cases.map((c: any) => `${c.kind}:${c.criterion}`)));

  return (
    <Page title="Evaluations" subtitle="Prompt evaluation runs (hard gates, rubric, LLM judges) recorded in Unity Catalog; full evidence in MLflow. Verdicts are evidence for a human reviewer, never an approval.">
      {error && <div className="text-sm text-bad mb-4">{error}</div>}
      <Card className="overflow-hidden">
        <table className="w-full text-sm">
          <thead className="text-xs text-muted"><tr>{["Finished", "Prompt", "Model", "Judge", "Cases", "Combined", "Verdict", ""].map((h) => <th key={h} className="text-left font-medium px-4 py-2">{h}</th>)}</tr></thead>
          <tbody>
            {loading && <tr><td className="px-4 py-3 text-muted" colSpan={8}><Loader2 size={14} className="inline animate-spin mr-2" />Loading…</td></tr>}
            {runs.map((r) => (
              <tr key={r.run_id} onClick={() => open(r.run_id)} className={`border-t border-line cursor-pointer hover:bg-sunken ${selected?.run?.run_id === r.run_id ? "bg-sunken" : ""}`}>
                <td className="px-4 py-2 text-muted">{String(r.finished_at).slice(0, 16).replace("T", " ")}</td>
                <td className="px-4 py-2">{/^\d+$/.test(String(r.prompt_version)) ? `v${r.prompt_version}` : r.prompt_version}</td><td className="px-4 py-2">{r.model_key}</td><td className="px-4 py-2 text-muted">{r.judge_model_key ?? "–"}</td>
                <td className="px-4 py-2">{r.cases}</td><td className="px-4 py-2">{(Number(r.combined_score) * 100).toFixed(1)}%</td>
                <td className="px-4 py-2"><span className={`text-[11px] px-1.5 py-0.5 rounded-md ${verdictStyle(r.verdict)}`}>{r.verdict}</span></td>
                <td className="px-4 py-2">{r.mlflow_url && <a href={r.mlflow_url} target="_blank" rel="noreferrer" onClick={(e) => e.stopPropagation()} className="text-accent inline-flex items-center gap-1 text-xs">MLflow <ExternalLink size={12} /></a>}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>
      {selected && (
        <Card className="mt-4 p-4 overflow-x-auto">
          {selected.loading ? <div className="text-sm text-muted"><Loader2 size={14} className="inline animate-spin mr-2" />Loading run…</div> : (
            <>
              <div className="text-sm font-medium mb-3">Per case and criterion · run {selected.run.run_id.slice(0, 8)}</div>
              <table className="text-xs">
                <thead><tr><th className="text-left pr-4 pb-2 text-muted font-medium">Case</th>{criteria.map((c) => <th key={c} className="px-2 pb-2 text-muted font-medium whitespace-nowrap [writing-mode:vertical-rl] rotate-180 text-left">{c.split(":")[1]}</th>)}</tr></thead>
                <tbody>{caseIds.map((id) => (
                  <tr key={id} className="border-t border-line">
                    <td className="pr-4 py-1.5 font-medium whitespace-nowrap">{id}</td>
                    {criteria.map((c) => {
                      const cell = cases.find((x: any) => x.case_id === id && `${x.kind}:${x.criterion}` === c);
                      return <td key={c} className="px-2 py-1.5 text-center" title={cell ? `${c} ${cell.score ?? cell.value}` : ""}>
                        {cell == null ? "" : cell.passed == null ? <span className="text-muted">–</span> : cell.passed ? <Check size={14} className="text-ok inline" /> : <X size={14} className="text-bad inline" />}
                      </td>;
                    })}
                  </tr>))}
                </tbody>
              </table>
            </>
          )}
        </Card>
      )}
    </Page>
  );
}
