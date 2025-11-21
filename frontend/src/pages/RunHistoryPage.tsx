import React, { useEffect, useState } from "react";

const API_BASE = "http://localhost:8000";

type BackendChunk = {
  source: string;
  chunk: number;
  doc_id?: string | null;
  distance?: number | null;
  relevance?: string | null;
  text: string;
};

type BackendRun = {
  run_id: string;
  timestamp: string;
  query: string;
  answer: string;
  latency_ms: number;
  trust_score: number;
  top_k: number;
  model?: string | null;
  label?: string | null;
  eval_notes?: string | null;
  retrieved: BackendChunk[];
};

type ChunkMeta = {
  source: string;
  chunk: number;
  distance: number | null;
  text: string;
  relevance: string;
};

type UiRun = {
  run_id: string;
  query: string;
  trust_score: number | null;
  latency_ms: number | null;
  top_k: number | null;
  chunks: ChunkMeta[];
  timestamp: string;
  label?: string | null;
  eval_notes?: string | null;
};

type RunStatusKind = "good" | "mixed" | "off" | "no";

type LabelValue = "good" | "mixed" | "off_topic" | "no_evidence";

/** Status icon + label + kind based on relevance mix. */
function getRunStatus(chunks: ChunkMeta[]): {
  icon: string;
  label: string;
  kind: RunStatusKind;
} {
  if (!chunks.length) return { icon: "⚪", label: "No evidence", kind: "no" };

  const related = chunks.filter((c) => c.relevance === "Related").length;
  const somewhat = chunks.filter((c) => c.relevance === "Somewhat related").length;
  const off = chunks.filter((c) => c.relevance === "Off-topic").length;

  if (related >= somewhat + off) {
    return { icon: "🟢", label: "Mostly related", kind: "good" };
  }
  if (off > related + somewhat) {
    return { icon: "🔴", label: "Mostly off-topic", kind: "off" };
  }
  return { icon: "🟡", label: "Mixed", kind: "mixed" };
}

function formatTrust(score: number | null) {
  if (score == null) return "-";
  return `${Math.round(score)}%`;
}

function formatLatency(latency: number | null) {
  if (latency == null) return "-";
  return `${Math.round(latency)} ms`;
}

function formatDate(iso: string) {
  if (!iso) return "-";
  const t = Date.parse(iso);
  if (Number.isNaN(t)) return "-";
  const d = new Date(t);
  return d.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

const RunHistoryPage: React.FC = () => {
  const [runs, setRuns] = useState<UiRun[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [search, setSearch] = useState("");
  const [statusFilter, setStatusFilter] = useState<RunStatusKind | "all">("all");
  const [sortOrder, setSortOrder] = useState<"newest" | "oldest">("newest");

  const [labelSavingId, setLabelSavingId] = useState<string | null>(null);
  const [exporting, setExporting] = useState(false);
  const [clearing, setClearing] = useState(false);

  // ---------- load runs from backend ----------
  async function fetchRuns() {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/api/runs?limit=200`);
      if (!res.ok) {
        throw new Error(`Backend returned ${res.status}`);
      }
      const data: BackendRun[] = await res.json();

      const uiRuns: UiRun[] = data.map((r) => ({
        run_id: r.run_id,
        query: r.query ?? "",
        trust_score: r.trust_score ?? null,
        latency_ms: r.latency_ms ?? null,
        top_k: r.top_k ?? null,
        timestamp: r.timestamp,
        label: r.label,
        eval_notes: r.eval_notes,
        chunks: (r.retrieved ?? []).map((c) => ({
          source: c.source,
          chunk: c.chunk,
          distance: c.distance ?? null,
          text: c.text,
          relevance: c.relevance ?? "Unknown",
        })),
      }));

      setRuns(uiRuns);
    } catch (err: any) {
      console.error(err);
      setError(err?.message || "Failed to load runs.");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    fetchRuns();
  }, []);

  // ---------- stats ----------
  const totalRuns = runs.length;
  const totalChunks = runs.reduce((sum, r) => sum + r.chunks.length, 0);
  const avgTrust =
    totalRuns > 0
      ? Math.round(
          runs.reduce((sum, r) => sum + (r.trust_score ?? 0), 0) / totalRuns
        )
      : null;

  const goodRuns = runs.filter((r) => getRunStatus(r.chunks).kind === "good").length;
  const offTopicRuns = runs.filter((r) => getRunStatus(r.chunks).kind === "off").length;

  // ---------- actions ----------
  async function clearHistory() {
    const ok = confirm("Clear all backend run logs? This cannot be undone.");
    if (!ok) return;

    setClearing(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/api/runs`, { method: "DELETE" });
      if (!res.ok) {
        throw new Error(`Backend returned ${res.status}`);
      }
      setRuns([]);
    } catch (err: any) {
      console.error(err);
      setError(err?.message || "Failed to clear logs.");
    } finally {
      setClearing(false);
    }
  }

  async function updateLabel(run_id: string, label: LabelValue) {
    setLabelSavingId(run_id);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/api/runs/${run_id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ label }),
      });
      if (!res.ok) {
        const detail = await res.json().catch(() => null);
        const msg = detail?.detail ?? `Backend returned ${res.status}`;
        throw new Error(msg);
      }

      // update local state
      setRuns((prev) =>
        prev.map((r) => (r.run_id === run_id ? { ...r, label } : r))
      );
    } catch (err: any) {
      console.error(err);
      setError(err?.message || "Failed to update label.");
    } finally {
      setLabelSavingId(null);
    }
  }

  async function exportDataset() {
    setExporting(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/api/export-dataset`);
      if (!res.ok) {
        const detail = await res.json().catch(() => null);
        const msg = detail?.detail ?? `Backend returned ${res.status}`;
        throw new Error(msg);
      }

      const text = await res.text();
      const blob = new Blob([text], { type: "text/plain;charset=utf-8" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "fine_tune_dataset.jsonl";
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (err: any) {
      console.error(err);
      setError(err?.message || "Failed to export dataset.");
    } finally {
      setExporting(false);
    }
  }

  // ---------- FILTER + SORT PIPELINE ----------
  const indexedRuns = runs.map((run, index) => ({ run, index }));
  const searchLower = search.trim().toLowerCase();

  const filteredIndexedRuns = indexedRuns
    .filter(({ run }) => {
      if (!searchLower) return true;
      const q = (run.query || "").toLowerCase();
      return q.includes(searchLower);
    })
    .filter(({ run }) => {
      if (statusFilter === "all") return true;
      return getRunStatus(run.chunks).kind === statusFilter;
    });

  const visibleIndexedRuns = [...filteredIndexedRuns].sort((a, b) => {
    const ta = Date.parse(a.run.timestamp) || 0;
    const tb = Date.parse(b.run.timestamp) || 0;
    return sortOrder === "newest" ? tb - ta : ta - tb;
  });

  const visibleRuns = visibleIndexedRuns.map((x) => x.run);

  // ------------------------------------------------------------

  return (
    <main className="min-h-[calc(100vh-64px)] bg-bg px-8 py-6">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <header className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between mb-4">
          <div>
            <h1 className="text-xl font-semibold text-textDark">Run History</h1>
            <p className="text-xs text-textMuted">
              Loaded from backend logs: trust, retrieval mix, labels, and top-k
              settings for each query.
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <button
              type="button"
              onClick={fetchRuns}
              className="text-[11px] px-3 py-1.5 rounded-full border border-slate-200 text-textMuted hover:text-primary hover:border-primary/60 hover:bg-primary/5 transition"
              disabled={loading}
            >
              {loading ? "Refreshing…" : "Refresh"}
            </button>

            <button
              type="button"
              onClick={exportDataset}
              disabled={exporting || runs.length === 0}
              className="text-[11px] px-3 py-1.5 rounded-full border border-emerald-300 bg-emerald-50 text-emerald-700 hover:bg-emerald-100 transition disabled:opacity-60 disabled:cursor-not-allowed"
            >
              {exporting ? "Exporting…" : "Export dataset"}
            </button>

            {runs.length > 0 && (
              <button
                type="button"
                onClick={clearHistory}
                disabled={clearing}
                className="text-[11px] px-3 py-1.5 rounded-full border border-rose-200 bg-rose-50 text-rose-700 hover:bg-rose-100 transition disabled:opacity-60 disabled:cursor-not-allowed"
              >
                {clearing ? "Clearing…" : "Clear logs"}
              </button>
            )}

            {/* sort pills */}
            <div className="flex items-center rounded-full bg-slate-50 border border-slate-200 text-[11px] px-1">
              <button
                type="button"
                onClick={() => setSortOrder("newest")}
                className={[
                  "px-3 py-1 rounded-full transition",
                  sortOrder === "newest"
                    ? "bg-white text-textDark shadow-sm"
                    : "text-textMuted",
                ].join(" ")}
              >
                Newest
              </button>
              <button
                type="button"
                onClick={() => setSortOrder("oldest")}
                className={[
                  "px-3 py-1 rounded-full transition",
                  sortOrder === "oldest"
                    ? "bg-white text-textDark shadow-sm"
                    : "text-textMuted",
                ].join(" ")}
              >
                Oldest
              </button>
            </div>
          </div>
        </header>

        {/* Error banner */}
        {error && (
          <div className="mb-3 rounded-2xl bg-rose-50 border border-rose-200 px-4 py-2 text-[11px] text-rose-700">
            {error}
          </div>
        )}

        {/* Summary cards */}
        <div className="grid grid-cols-1 sm:grid-cols-4 gap-3 mb-4">
          <div className="rounded-2xl bg-white border border-indigo-50 shadow-soft px-4 py-3">
            <div className="text-[11px] text-textMuted">Total runs</div>
            <div className="mt-1 text-lg font-semibold text-textDark">
              {totalRuns}
            </div>
          </div>
          <div className="rounded-2xl bg-white border border-indigo-50 shadow-soft px-4 py-3">
            <div className="text-[11px] text-textMuted">Avg trust</div>
            <div className="mt-1 text-lg font-semibold text-textDark">
              {avgTrust != null ? `${avgTrust}%` : "-"}
            </div>
          </div>
          <div className="rounded-2xl bg-white border border-indigo-50 shadow-soft px-4 py-3">
            <div className="text-[11px] text-textMuted">Total chunks used</div>
            <div className="mt-1 text-lg font-semibold text-textDark">
              {totalChunks}
            </div>
          </div>
          <div className="rounded-2xl bg-white border border-indigo-50 shadow-soft px-4 py-3">
            <div className="text-[11px] text-textMuted">Run quality</div>
            <div className="mt-1 flex items-baseline gap-2 text-xs">
              <span className="inline-flex items-center gap-1 text-emerald-600">
                🟢 {goodRuns}
              </span>
              <span className="inline-flex items-center gap-1 text-rose-500">
                🔴 {offTopicRuns}
              </span>
            </div>
          </div>
        </div>

        {/* Filters */}
        {runs.length > 0 && (
          <div className="mb-3 flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
            <div className="flex-1">
              <div className="relative">
                <input
                  type="text"
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                  placeholder="Search by question text…"
                  className="w-full rounded-full border border-slate-200 bg-white px-3 py-2 pr-8 text-[11px] outline-none focus:border-primary focus:ring-2 focus:ring-primary/15"
                />
                {search && (
                  <button
                    type="button"
                    onClick={() => setSearch("")}
                    className="absolute right-2 top-1/2 -translate-y-1/2 text-[11px] text-slate-400 hover:text-slate-600"
                  >
                    ✕
                  </button>
                )}
              </div>
            </div>

            <div className="flex flex-wrap gap-1.5 text-[11px]">
              {[
                { key: "all" as const, label: "All" },
                { key: "good" as const, label: "Mostly related" },
                { key: "mixed" as const, label: "Mixed" },
                { key: "off" as const, label: "Mostly off-topic" },
                { key: "no" as const, label: "No evidence" },
              ].map((opt) => (
                <button
                  key={opt.key}
                  type="button"
                  onClick={() => setStatusFilter(opt.key)}
                  className={[
                    "px-3 py-1 rounded-full border text-xs transition",
                    statusFilter === opt.key
                      ? "border-primary bg-primary/5 text-primary"
                      : "border-slate-200 text-textMuted hover:border-primary/40 hover:text-primary",
                  ].join(" ")}
                >
                  {opt.label}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Runs list */}
        <div className="rounded-3xl bg-white border border-indigo-50 shadow-soft overflow-hidden">
          {loading && runs.length === 0 ? (
            <div className="px-4 py-6 text-xs text-textMuted">
              Loading runs from backend…
            </div>
          ) : runs.length === 0 ? (
            <div className="px-4 py-6 text-xs text-textMuted">
              No runs logged yet. Ask a question in the Chat Assistant to
              generate history.
            </div>
          ) : visibleRuns.length === 0 ? (
            <div className="px-4 py-6 text-xs text-textMuted">
              No runs match your current filters. Try clearing the search or
              status filters.
            </div>
          ) : (
            <div className="max-h-[60vh] overflow-y-auto">
              <table className="w-full text-[11px]">
                <thead className="bg-slate-50 border-b border-slate-100 sticky top-0 z-10">
                  <tr>
                    <th className="text-left px-3 py-2 font-medium text-textMuted">
                      #
                    </th>
                    <th className="text-left px-3 py-2 font-medium text-textMuted">
                      Question
                    </th>
                    <th className="text-left px-3 py-2 font-medium text-textMuted">
                      Trust
                    </th>
                    <th className="text-left px-3 py-2 font-medium text-textMuted">
                      Latency
                    </th>
                    <th className="text-left px-3 py-2 font-medium text-textMuted">
                      Relevance (R/S/O)
                    </th>
                    <th className="text-left px-3 py-2 font-medium text-textMuted">
                      Status
                    </th>
                    <th className="text-left px-3 py-2 font-medium text-textMuted">
                      Label
                    </th>
                    <th className="text-left px-3 py-2 font-medium text-textMuted">
                      k
                    </th>
                    <th className="text-left px-3 py-2 font-medium text-textMuted">
                      When
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {visibleIndexedRuns.map(({ run }, rowIdx) => {
                    const relatedCount = run.chunks.filter(
                      (c) => c.relevance === "Related"
                    ).length;
                    const somewhatCount = run.chunks.filter(
                      (c) => c.relevance === "Somewhat related"
                    ).length;
                    const offTopicCount = run.chunks.filter(
                      (c) => c.relevance === "Off-topic"
                    ).length;
                    const status = getRunStatus(run.chunks);

                    const statusChipClasses =
                      status.kind === "good"
                        ? "bg-emerald-50 text-emerald-700 border-emerald-200"
                        : status.kind === "off"
                        ? "bg-rose-50 text-rose-700 border-rose-200"
                        : status.kind === "mixed"
                        ? "bg-sky-50 text-sky-700 border-sky-200"
                        : "bg-slate-50 text-slate-500 border-slate-200";

                    const trustPercent = run.trust_score ?? 0;

                    const labelValue = (run.label ?? "") as
                      | LabelValue
                      | ""
                      | undefined;

                    return (
                      <tr
                        key={run.run_id}
                        className="border-b border-slate-50 hover:bg-slate-50/60"
                      >
                        <td className="px-3 py-2 text-textMuted align-top">
                          #{rowIdx + 1}
                        </td>
                        <td className="px-3 py-2 align-top max-w-[320px]">
                          <div
                            className="text-[11px] text-textDark line-clamp-2"
                            title={run.query || "(no question captured)"}
                          >
                            {run.query || "(no question captured)"}
                          </div>
                        </td>
                        <td className="px-3 py-2 align-top text-textDark w-[90px]">
                          <div className="flex items-center gap-2">
                            <span>{formatTrust(run.trust_score)}</span>
                          </div>
                          <div className="mt-1 h-1.5 w-full rounded-full bg-slate-100 overflow-hidden">
                            <div
                              className="h-full rounded-full bg-primary/70"
                              style={{
                                width: `${Math.min(
                                  100,
                                  Math.max(0, trustPercent)
                                )}%`,
                              }}
                            />
                          </div>
                        </td>
                        <td className="px-3 py-2 align-top text-textMuted whitespace-nowrap">
                          {formatLatency(run.latency_ms)}
                        </td>
                        <td className="px-3 py-2 align-top text-textMuted">
                          {relatedCount}/{somewhatCount}/{offTopicCount}
                        </td>
                        <td className="px-3 py-2 align-top">
                          <span
                            className={[
                              "inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[10px] font-medium",
                              statusChipClasses,
                            ].join(" ")}
                          >
                            <span>{status.icon}</span>
                            <span>{status.label}</span>
                          </span>
                        </td>
                        <td className="px-3 py-2 align-top text-textMuted">
                          <select
                            className="border border-slate-200 rounded-full px-2 py-1 text-[10px] bg-white"
                            value={labelValue || ""}
                            onChange={(e) => {
                              const val = e.target.value as LabelValue | "";
                              if (!val) return;
                              updateLabel(run.run_id, val);
                            }}
                            disabled={labelSavingId === run.run_id}
                          >
                            <option value="">(none)</option>
                            <option value="good">good</option>
                            <option value="mixed">mixed</option>
                            <option value="off_topic">off_topic</option>
                            <option value="no_evidence">no_evidence</option>
                          </select>
                        </td>
                        <td className="px-3 py-2 align-top text-textMuted">
                          {run.top_k ?? "-"}
                        </td>
                        <td className="px-3 py-2 align-top text-textMuted whitespace-nowrap">
                          {formatDate(run.timestamp)}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>
    </main>
  );
};

export default RunHistoryPage;
