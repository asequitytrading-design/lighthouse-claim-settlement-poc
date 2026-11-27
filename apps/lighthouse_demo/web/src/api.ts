// Client for lighthouse-api (/api/*). Chat streams Server-Sent Events from a POST request.
export type AgentEvent = { type: string; [key: string]: any };

export type Model = {
  key: string; model_service: string; family: string | null; status: string;
  input_usd_per_1m: number | null; output_usd_per_1m: number | null; default: boolean;
};

async function json<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, { headers: { "Content-Type": "application/json" }, ...init });
  if (!response.ok) throw new Error(`${response.status} ${await response.text()}`);
  return response.json() as Promise<T>;
}

export const api = {
  health: () => json<Record<string, string>>("/api/health"),
  warmup: () => json<{ steps: Record<string, { status: string; ms: number; detail: any }>; ready: boolean }>("/api/warmup", { method: "POST" }),
  models: () => json<{ models: Model[] }>("/api/models").then((r) => r.models),
  sessions: () => json<{ sessions: { session_id: string; title: string; updated_at: string }[] }>("/api/sessions").then((r) => r.sessions),
  session: (id: string) => json<{ session_id: string; title: string; messages: { role: string; content: string; meta: any }[] }>(`/api/sessions/${id}`),
  deleteSession: (id: string) => json(`/api/sessions/${id}`, { method: "DELETE" }),
  playground: (models: string[], prompt: string, system?: string) =>
    json<{ results: any[] }>("/api/playground", { method: "POST", body: JSON.stringify({ models, prompt, system: system || null }) }).then((r) => r.results),
  usage: (hours: number) => json<any>(`/api/usage/summary?hours=${hours}`),
  recent: (limit = 25) => json<{ calls: any[] }>(`/api/usage/recent?limit=${limit}`).then((r) => r.calls),
  evalRuns: () => json<{ runs: any[] }>("/api/evals/runs").then((r) => r.runs),
  evalRun: (id: string) => json<any>(`/api/evals/runs/${id}`),
};

export async function streamChat(body: { message: string; model?: string; session_id?: string | null },
                                 onEvent: (e: AgentEvent) => void, signal?: AbortSignal) {
  const response = await fetch("/api/chat", {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body), signal,
  });
  if (!response.ok || !response.body) throw new Error(`chat failed: ${response.status}`);
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let index;
    while ((index = buffer.indexOf("\n\n")) >= 0) {
      const chunk = buffer.slice(0, index);
      buffer = buffer.slice(index + 2);
      const data = chunk.split("\n").find((line) => line.startsWith("data: "));
      if (data) onEvent(JSON.parse(data.slice(6)));
    }
  }
}

export const fmt = {
  ms: (v?: number | null) => (v == null ? "–" : v >= 1000 ? `${(v / 1000).toFixed(1)} s` : `${Math.round(v)} ms`),
  usd: (v?: number | null) => (v == null ? "–" : v < 0.01 ? `$${v.toFixed(5)}` : `$${v.toFixed(3)}`),
  int: (v?: number | null) => (v == null ? "–" : Intl.NumberFormat().format(v)),
};
