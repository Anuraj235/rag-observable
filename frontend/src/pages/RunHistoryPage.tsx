import React, { useEffect, useState } from "react";

type ChunkMeta = {
  source: string;
  chunk: number;
  distance: number | null;
  text: string;
  relevance: string;
};

type RunRecord = {
  id: string;
  question: string;
  trust_score: number | null;
  latency_ms: number | null;
  top_k: number | null;
  chunks: ChunkMeta[];
  created_at: number;
};

const RUN_HISTORY_KEY = "rag_run_history";

type RunStatusKind = "good" | "mixed" | "off" | "no";

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

function formatDate(ts: number) {
  if (!ts) return "-";
  const d = new Date(ts);
  return d.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

const RunHistoryPage: React.FC = () => {
  const [runs, setRuns] = useState<RunRecord[]>([]);
  const [search, setSearch] = useState("");
  const [statusFilter, setStatusFilter] = useState<RunStatusKind | "all">("all");
  const [sortOrder, setSortOrder] = useState<"newest" | "oldest">("newest");

  useEffect(() => {
    try {
      const raw = sessionStorage.getItem(RUN_HISTORY_KEY);
      if (!raw) return;
      const parsed: RunRecord[] = JSON.parse(raw);
      setRuns(parsed);
    } catch (err) {
      console.error("Failed to load run history from sessionStorage", err);
    }
  }, []);

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

  function clearHistory() {
    const ok = confirm("Clear all run history for this session?");
    if (!ok) return;

    setRuns([]);
    try {
      sessionStorage.removeItem(RUN_HISTORY_KEY);
    } catch {
      // ignore
    }
  }

  // ---------- FILTER + SORT PIPELINE ----------
  // attach original index so we can sort deterministically regardless of timestamps
  const indexedRuns = runs.map((run, index) => ({ run, index }));
  const searchLower = search.trim().toLowerCase();

  const filteredIndexedRuns = indexedRuns
    // search by question text
    .filter(({ run }) => {
      if (!searchLower) return true;
      const q = (run.question || "").toLowerCase();
      return q.includes(searchLower);
    })
    // filter by status chip
    .filter(({ run }) => {
      if (statusFilter === "all") return true;
      return getRunStatus(run.chunks).kind === statusFilter;
    });

  const visibleIndexedRuns = [...filteredIndexedRuns].sort((a, b) =>
    sortOrder === "newest" ? b.index - a.index : a.index - b.index
  );

  // For length checks
  const visibleRuns = visibleIndexedRuns.map((x) => x.run);
  // -------------------------------------------

  return (
    <main className="min-h-[calc(100vh-64px)] bg-bg px-8 py-6">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <header className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between mb-4">
          <div>
            <h1 className="text-xl font-semibold text-textDark">Run History</h1>
            <p className="text-xs text-textMuted">
              See how each question performed: trust, retrieval mix, and top-k
              settings.
            </p>
          </div>

          <div className="flex items-center gap-2">
            {/* sort pills - always visible */}
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

            {runs.length > 0 && (
              <button
                type="button"
                onClick={clearHistory}
                className="text-[11px] px-3 py-1.5 rounded-full border border-slate-200 text-textMuted hover:text-primary hover:border-primary/60 hover:bg-primary/5 transition"
              >
                Clear history
              </button>
            )}
          </div>
        </header>

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
          {runs.length === 0 ? (
            <div className="px-4 py-6 text-xs text-textMuted">
              No runs recorded yet. Ask a question in the Chat Assistant to see
              history here.
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
                      k
                    </th>
                    <th className="text-left px-3 py-2 font-medium text-textMuted">
                      When
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {visibleIndexedRuns.map(({ run, index }) => {
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

                    // global chronological number, not just row index
                    const globalNumber = index + 1;

                    const statusChipClasses =
                      status.kind === "good"
                        ? "bg-emerald-50 text-emerald-700 border-emerald-200"
                        : status.kind === "off"
                        ? "bg-rose-50 text-rose-700 border-rose-200"
                        : status.kind === "mixed"
                        ? "bg-sky-50 text-sky-700 border-sky-200"
                        : "bg-slate-50 text-slate-500 border-slate-200";

                    const trustPercent = run.trust_score ?? 0;

                    return (
                      <tr
                        key={run.id}
                        className="border-b border-slate-50 hover:bg-slate-50/60"
                      >
                        <td className="px-3 py-2 text-textMuted align-top">
                          #{globalNumber}
                        </td>
                        <td className="px-3 py-2 align-top max-w-[320px]">
                          <div
                            className="text-[11px] text-textDark line-clamp-2"
                            title={run.question || "(no question captured)"}
                          >
                            {run.question || "(no question captured)"}
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
                          {run.top_k ?? "-"}
                        </td>
                        <td className="px-3 py-2 align-top text-textMuted whitespace-nowrap">
                          {formatDate(run.created_at)}
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
