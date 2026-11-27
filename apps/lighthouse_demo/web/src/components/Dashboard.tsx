import { useEffect, useMemo, useState } from "react";
import { ExternalLink, Loader2, RefreshCw } from "lucide-react";
import { Bar, BarChart, CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api, fmt } from "../api";
import Page, { Card } from "./Page";

const WINDOWS = [{ h: 1, label: "1 h" }, { h: 6, label: "6 h" }, { h: 24, label: "24 h" }, { h: 168, label: "7 d" }];
const COLORS = ["#6366f1", "#14b8a6", "#f59e0b", "#ec4899", "#0ea5e9", "#84cc16", "#a855f7"];

export default function Dashboard() {
  const [hours, setHours] = useState(24);
  const [data, setData] = useState<any>(null);
  const [recent, setRecent] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  function load() {
    setLoading(true); setError(null);
    Promise.all([api.usage(hours), api.recent(15)])
      .then(([u, r]) => { setData(u); setRecent(r); })
      .catch((e) => setError(String(e.message ?? e)))
      .finally(() => setLoading(false));
  }
  useEffect(load, [hours]);

  const models = data?.by_model ?? [];
  const totals = useMemo(() => models.reduce((a: any, m: any) => ({
    calls: a.calls + (m.calls ?? 0), errors: a.errors + (m.errors ?? 0), cost: a.cost + (m.cost_usd ?? 0),
    tokens: a.tokens + (m.input_tokens ?? 0) + (m.output_tokens ?? 0),
  }), { calls: 0, errors: 0, cost: 0, tokens: 0 }), [models]);
  const timeline = useMemo(() => {
    const byHour: Record<string, any> = {};
    const models = Array.from(new Set((data?.timeline ?? []).map((r: any) => r.model_key)));
    for (const row of data?.timeline ?? []) {
      const hour = String(row.hour).slice(5, 16).replace("T", " ");
      byHour[hour] = byHour[hour] ?? Object.fromEntries([["hour", hour], ...models.map((m) => [m, 0])]);
      byHour[hour][row.model_key] = row.calls;
    }
    return Object.values(byHour);
  }, [data]);
  const keys = models.map((m: any) => m.model_key);

  return (
    <Page title="Usage and cost" subtitle="Every model call of the platform, from the Unity Catalog views (Unity Gateway usage + application events, illustrative rates)."
          actions={
            <div className="flex items-center gap-2">
              <div className="flex rounded-lg border border-line overflow-hidden">
                {WINDOWS.map((w) => (
                  <button key={w.h} onClick={() => setHours(w.h)} className={`px-3 py-1.5 text-sm ${hours === w.h ? "bg-sunken font-medium" : "text-muted hover:bg-sunken"}`}>{w.label}</button>
                ))}
              </div>
              <button onClick={load} className="p-2 rounded-lg border border-line hover:bg-sunken" title="Refresh">
                {loading ? <Loader2 size={15} className="animate-spin" /> : <RefreshCw size={15} />}
              </button>
              {data?.dashboard_url && <a href={data.dashboard_url} target="_blank" rel="noreferrer" className="inline-flex items-center gap-1 text-sm text-accent">Lakeview <ExternalLink size={13} /></a>}
            </div>
          }>
      {error && <div className="text-sm text-bad mb-4">{error}</div>}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {[["Model calls", fmt.int(totals.calls)], ["Tokens", fmt.int(totals.tokens)], ["Cost (illustrative)", fmt.usd(totals.cost)],
          ["Error rate", totals.calls ? `${((totals.errors / totals.calls) * 100).toFixed(1)}%` : "–"]].map(([label, value]) => (
          <Card key={label} className="p-4"><div className="text-xs text-muted">{label}</div><div className="text-2xl font-semibold mt-1">{value}</div></Card>
        ))}
      </div>
      <div className="grid md:grid-cols-2 gap-4 mt-4">
        <Card className="p-4">
          <div className="text-sm font-medium mb-3">Calls per hour by model</div>
          <div className="h-64">
            <ResponsiveContainer>
              <LineChart data={timeline}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgb(var(--line))" />
                <XAxis dataKey="hour" tick={{ fontSize: 11 }} stroke="rgb(var(--muted))" />
                <YAxis tick={{ fontSize: 11 }} stroke="rgb(var(--muted))" allowDecimals={false} />
                <Tooltip contentStyle={{ background: "rgb(var(--panel))", border: "1px solid rgb(var(--line))", fontSize: 12 }} />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                {keys.map((k: string, i: number) => <Line key={k} type="monotone" dataKey={k} stroke={COLORS[i % COLORS.length]} strokeWidth={2} dot={false} isAnimationActive={false} />)}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Card>
        <Card className="p-4">
          <div className="text-sm font-medium mb-3">Cost by model (USD)</div>
          <div className="h-64">
            <ResponsiveContainer>
              <BarChart data={models}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgb(var(--line))" />
                <XAxis dataKey="model_key" tick={{ fontSize: 11 }} stroke="rgb(var(--muted))" />
                <YAxis tick={{ fontSize: 11 }} stroke="rgb(var(--muted))" />
                <Tooltip contentStyle={{ background: "rgb(var(--panel))", border: "1px solid rgb(var(--line))", fontSize: 12 }} formatter={(v: any) => fmt.usd(Number(v))} />
                <Bar dataKey="cost_usd" fill="#6366f1" radius={[6, 6, 0, 0]} isAnimationActive={false} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>
      </div>
      <Card className="mt-4 overflow-hidden">
        <div className="px-4 py-3 text-sm font-medium border-b border-line">Per model</div>
        <table className="w-full text-sm">
          <thead className="text-xs text-muted"><tr>{["Model", "Calls", "Errors", "p50", "p95", "Tokens in", "Tokens out", "Cost"].map((h) => <th key={h} className="text-left font-medium px-4 py-2">{h}</th>)}</tr></thead>
          <tbody>{models.map((m: any) => (
            <tr key={m.model_key} className="border-t border-line">
              <td className="px-4 py-2 font-medium">{m.model_key}</td><td className="px-4 py-2">{fmt.int(m.calls)}</td><td className="px-4 py-2">{fmt.int(m.errors)}</td>
              <td className="px-4 py-2">{fmt.ms(m.p50_ms)}</td><td className="px-4 py-2">{fmt.ms(m.p95_ms)}</td>
              <td className="px-4 py-2">{fmt.int(m.input_tokens)}</td><td className="px-4 py-2">{fmt.int(m.output_tokens)}</td><td className="px-4 py-2">{fmt.usd(m.cost_usd)}</td>
            </tr>))}
          </tbody>
        </table>
      </Card>
      <div className="grid md:grid-cols-2 gap-4 mt-4">
        <Card className="overflow-hidden">
          <div className="px-4 py-3 text-sm font-medium border-b border-line">By activity and purpose</div>
          <table className="w-full text-sm"><tbody>{(data?.by_activity ?? []).map((a: any, i: number) => (
            <tr key={i} className="border-t border-line first:border-0"><td className="px-4 py-2">{a.activity}</td><td className="px-4 py-2 text-muted">{a.purpose}</td><td className="px-4 py-2">{fmt.int(a.calls)}</td><td className="px-4 py-2">{fmt.usd(a.cost_usd)}</td></tr>))}
          </tbody></table>
        </Card>
        <Card className="overflow-hidden">
          <div className="px-4 py-3 text-sm font-medium border-b border-line">Latest calls (no prompt or response text)</div>
          <table className="w-full text-xs"><tbody>{recent.map((c: any, i: number) => (
            <tr key={i} className="border-t border-line first:border-0">
              <td className="px-4 py-1.5 text-muted">{String(c.event_time).slice(11, 19)}</td><td className="px-2 py-1.5">{c.model_key}</td>
              <td className="px-2 py-1.5 text-muted">{c.activity}/{c.purpose}</td><td className="px-2 py-1.5">{fmt.ms(c.latency_ms)}</td>
              <td className={`px-2 py-1.5 ${c.status === "ok" ? "text-ok" : "text-bad"}`}>{c.status}</td></tr>))}
          </tbody></table>
        </Card>
      </div>
    </Page>
  );
}
