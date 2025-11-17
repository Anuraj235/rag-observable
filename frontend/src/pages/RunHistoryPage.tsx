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

/** Simple status icon based on relevance mix. */
function runStatusIcon(chunks: ChunkMeta[]): { icon: string; label: string } {
  if (!chunks.length) return { icon: "⚪", label: "No evidence" };

  const related = chunks.filter((c) => c.relevance === "Related").length;
  const somewhat = chunks.filter((c) => c.relevance === "Somewhat related").length;
  const off = chunks.filter((c) => c.relevance === "Off-topic").length;

  if (related >= somewhat + off) return { icon: "🟢", label: "Mostly related" };
  if (off > related + somewhat) return { icon: "🔴", label: "Mostly off-topic" };
  return { icon: "🟡", label: "Mixed" };
}

function formatTrust(score: number | null) {
  if (score == null) return "-";
  return `${Math.round(score)}%`;
}

function formatLatency(latency: number | null) {
  if (latency == null) return "-";
  return `${Math.round(latency)} ms`;
}

const RunHistoryPage: React.FC = () => {
  const [runs, setRuns] = useState<RunRecord[]>([]);

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

  return (
    <main className="min-h-[calc(100vh-64px)] bg-bg px-8 py-6">
      <div className="max-w-5xl mx-auto">
        <header className="flex items-baseline justify-between mb-4">
          <div>
            <h1 className="text-xl font-semibold text-textDark">Run History</h1>
            <p className="text-xs text-textMuted">
              Each question/answer pair as an experiment: trust, relevance mix, and
              retrieval settings.
            </p>
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
        </header>

        {/* Summary cards */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 mb-4">
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
        </div>

        {/* Runs table / list */}
        <div className="rounded-3xl bg-white border border-indigo-50 shadow-soft overflow-hidden">
          {runs.length === 0 ? (
            <div className="px-4 py-6 text-xs text-textMuted">
              No runs recorded yet. Ask a question in the Chat Assistant to see
              history here.
            </div>
          ) : (
            <div className="max-h-[60vh] overflow-y-auto">
              <table className="w-full text-[11px]">
                <thead className="bg-slate-50 border-b border-slate-100">
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
                  </tr>
                </thead>
                <tbody>
                  {[...runs]
                    .slice()
                    .reverse()
                    .map((run, idxFromTop) => {
                      const idx = runs.length - idxFromTop; // 1..N
                      const relatedCount = run.chunks.filter(
                        (c) => c.relevance === "Related"
                      ).length;
                      const somewhatCount = run.chunks.filter(
                        (c) => c.relevance === "Somewhat related"
                      ).length;
                      const offTopicCount = run.chunks.filter(
                        (c) => c.relevance === "Off-topic"
                      ).length;
                      const { icon, label } = runStatusIcon(run.chunks);

                      return (
                        <tr
                          key={run.id}
                          className="border-b border-slate-50 hover:bg-slate-50/60"
                        >
                          <td className="px-3 py-2 text-textMuted align-top">
                            #{idx}
                          </td>
                          <td className="px-3 py-2 align-top max-w-[260px]">
                            <div className="text-[11px] text-textDark line-clamp-2">
                              {run.question || "(no question captured)"}
                            </div>
                          </td>
                          <td className="px-3 py-2 align-top text-textDark">
                            {formatTrust(run.trust_score)}
                          </td>
                          <td className="px-3 py-2 align-top text-textMuted">
                            {formatLatency(run.latency_ms)}
                          </td>
                          <td className="px-3 py-2 align-top text-textMuted">
                            {relatedCount}/{somewhatCount}/{offTopicCount}
                          </td>
                          <td className="px-3 py-2 align-top">
                            <span className="inline-flex items-center gap-1 text-[11px] text-textMuted">
                              <span>{icon}</span>
                              <span>{label}</span>
                            </span>
                          </td>
                          <td className="px-3 py-2 align-top text-textMuted">
                            {run.top_k ?? "-"}
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
