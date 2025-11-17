import React, { useEffect, useRef, useState } from "react";
import type { FormEvent } from "react";

/** Remove any trailing “Sources:” block from the model answer text. */
function stripSourcesFromAnswer(text: string): string {
  if (!text) return text;

  const lower = text.toLowerCase();
  const idx = lower.indexOf("sources:");
  if (idx === -1) {
    return text.trim();
  }

  return text.slice(0, idx).trimEnd();
}

type Role = "user" | "assistant";

type ChunkMeta = {
  source: string;
  chunk: number;
  distance: number | null;
  text: string;
  relevance: string;
};

type MessageMeta = {
  trust_score: number | null;
  latency_ms: number | null;
  chunks: ChunkMeta[];
};

type Message = {
  id: string;
  role: Role;
  content: string;
  meta?: MessageMeta;
};

type QueryResponse = {
  answer: string;
  latency_ms: number;
  trust_score: number;
  chunks: {
    source: string;
    chunk: number;
    distance: number | null;
    text: string;
    relevance: string;
  }[];
};

const API_BASE = "http://localhost:8000";

/** Convert distance (smaller = better) to a bar width percentage. */
function distanceWidth(distance: number | null): string {
  if (distance == null) return "0%";
  // Clamp into [0, 1] and treat 0 as best (100% filled), 1 as worst (0%).
  const d = Math.max(0, Math.min(1, distance));
  const closeness = 1 - d;
  return `${Math.round(closeness * 100)}%`;
}

const ChatPage: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [topK, setTopK] = useState<number>(3);
  const [showOffTopic, setShowOffTopic] = useState<boolean>(true);

  const scrollRef = useRef<HTMLDivElement | null>(null);

  // Scroll to bottom when messages change
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  const lastAssistant = [...messages].reverse().find((m) => m.role === "assistant");
  const lastMeta = lastAssistant?.meta;

  async function handleSend(e: FormEvent) {
    e.preventDefault();
    const trimmed = input.trim();
    if (!trimmed || loading) return;

    const userMessage: Message = {
      id: crypto.randomUUID(),
      role: "user",
      content: trimmed,
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setLoading(true);

    try {
      const res = await fetch(`${API_BASE}/api/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: trimmed, top_k: topK }),
      });

      if (!res.ok) {
        throw new Error(`Backend returned ${res.status}`);
      }

      const data: QueryResponse = await res.json();

      const cleanedAnswer = stripSourcesFromAnswer(data.answer);

      const assistantMessage: Message = {
        id: crypto.randomUUID(),
        role: "assistant",
        content: cleanedAnswer,
        meta: {
          trust_score: data.trust_score ?? null,
          latency_ms: data.latency_ms ?? null,
          chunks: (data.chunks ?? []).map((c) => ({
            source: c.source,
            chunk: c.chunk,
            distance: c.distance,
            text: c.text,
            relevance: c.relevance,
          })),
        },
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (err) {
      console.error(err);
      const errorMessage: Message = {
        id: crypto.randomUUID(),
        role: "assistant",
        content:
          "Sorry, something went wrong while contacting the RAG backend. Please try again.",
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  }

  function formatTrust(score: number | null | undefined) {
    if (score == null) return "-";
    return `${Math.round(score)}%`;
  }

  function formatLatency(latency: number | null | undefined) {
    if (latency == null) return "-";
    return `${Math.round(latency)} ms`;
  }

  function relevanceLabel(relevance: string) {
    switch (relevance) {
      case "Related":
        return "Related";
      case "Somewhat related":
        return "Somewhat related";
      case "Off-topic":
        return "Off-topic";
      default:
        return "Unknown";
    }
  }

  function relevanceBadgeClass(relevance: string) {
    switch (relevance) {
      case "Related":
        return "bg-emerald-50 text-emerald-700 border-emerald-200";
      case "Somewhat related":
        return "bg-sky-50 text-sky-700 border-sky-200";
      case "Off-topic":
        return "bg-rose-50 text-rose-700 border-rose-200";
      default:
        return "bg-slate-50 text-slate-500 border-slate-200";
    }
  }

  return (
    <main className="min-h-[calc(100vh-64px)] bg-bg px-8 py-6 flex gap-6">
      {/* LEFT: Chat assistant */}
      <section className="flex-1 flex flex-col">
        {/* Header */}
        <div className="flex items-baseline justify-between">
          <div>
            <h1 className="text-xl font-semibold text-textDark">Chat Assistant</h1>
            <p className="text-xs text-textMuted">
              Ask questions about your documents and get answers backed by retrieved
              chunks.
            </p>
          </div>
        </div>

        {/* Chat container */}
        <div className="mt-4 flex-1 rounded-3xl bg-white shadow-soft border border-indigo-50 flex flex-col">
          {/* Messages */}
          <div
            ref={scrollRef}
            className="flex-1 px-6 pt-6 pb-3 overflow-y-auto space-y-4"
          >
            {messages.length === 0 && !loading && (
              <div className="rounded-2xl bg-indigo-50/60 border border-dashed border-indigo-200 px-4 py-3 text-xs text-textMuted">
                Try asking:
                <span className="ml-2 font-medium text-primary">
                  &ldquo;What is supervised learning?&rdquo;
                </span>
              </div>
            )}

            {messages.map((m) => {
              const isAssistant = m.role === "assistant";
              const allChunks = m.meta?.chunks ?? [];

              const chunksForDisplay = showOffTopic
                ? allChunks
                : allChunks.filter((c) => c.relevance !== "Off-topic");

              // de-dupe sources for the source pills
              const uniqueChunks = Array.from(
                new Map(
                  chunksForDisplay.map((c) => [`${c.source}-${c.chunk}`, c])
                ).values()
              );

              return (
                <div
                  key={m.id}
                  className={`flex ${isAssistant ? "justify-start" : "justify-end"}`}
                >
                  <div
                    className={[
                      "max-w-[70%] rounded-2xl px-4 py-3 text-sm shadow-soft-sm",
                      isAssistant
                        ? "bg-slate-50 text-textDark border border-slate-200 rounded-bl-md"
                        : "bg-primary text-white rounded-br-md",
                    ].join(" ")}
                  >
                    <p className="whitespace-pre-line">{m.content}</p>

                    {/* Source pills directly under assistant message */}
                    {isAssistant && uniqueChunks.length > 0 && (
                      <div className="mt-3 border-t border-slate-200 pt-2">
                        <div className="mb-1 text-[11px] font-medium text-textMuted">
                          Sources
                        </div>
                        <div className="flex flex-wrap gap-1.5">
                          {uniqueChunks.map((c, idx) => (
                            <span
                              key={`${c.source}-${c.chunk}-${idx}`}
                              className={[
                                "inline-flex items-center rounded-full border px-2 py-0.5 text-[11px] max-w-[220px] truncate",
                                relevanceBadgeClass(c.relevance),
                              ].join(" ")}
                            >
                              <span className="mr-1 text-[10px] opacity-80">
                                [{idx + 1}]
                              </span>
                              <span className="truncate">
                                {c.source} · chunk {c.chunk}
                              </span>
                            </span>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              );
            })}

            {loading && (
              <div className="flex items-center gap-2 text-xs text-textMuted">
                <span className="h-2 w-2 rounded-full bg-primary animate-pulse" />
                Thinking…
              </div>
            )}
          </div>

          {/* Input */}
          <form
            onSubmit={handleSend}
            className="border-t border-slate-100 px-6 py-4 flex items-center gap-3"
          >
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask a question about your documents…"
              className="flex-1 rounded-full border border-slate-200 bg-slate-50 px-4 py-2 text-sm outline-none focus:border-primary focus:bg-white focus:ring-2 focus:ring-primary/20"
            />
            <button
              type="submit"
              disabled={loading || !input.trim()}
              className="rounded-full bg-primary px-5 py-2 text-sm font-medium text-white shadow-soft disabled:cursor-not-allowed disabled:bg-primary/40"
            >
              {loading ? "Sending…" : "Send"}
            </button>
          </form>
        </div>
      </section>

      {/* RIGHT: Trust / chunk panel */}
      <aside className="w-[320px] shrink-0">
        <div className="rounded-3xl bg-white shadow-soft border border-indigo-50 p-5">
          <h2 className="text-sm font-semibold text-textDark">Trust Insights</h2>
          <p className="mt-1 text-xs text-textMuted">
            Trust score and retrieved chunk details for the latest answer.
          </p>

          {/* Top-k slider */}
          <div className="mt-4">
            <label className="flex items-center justify-between text-xs font-medium text-textDark">
              <span>Top-k chunks</span>
              <span className="text-textMuted">k = {topK}</span>
            </label>
            <input
              type="range"
              min={1}
              max={8}
              value={topK}
              onChange={(e) => setTopK(Number(e.target.value))}
              className="mt-2 w-full"
            />
            <p className="mt-1 text-[11px] text-textMuted">
              Controls how many chunks the retriever returns for each answer.
            </p>
          </div>

          {/* Rebuild index button */}
          <div className="mt-4">
            <button
              onClick={async () => {
                const ok = confirm(
                  "Rebuild entire index? This will re-embed all files."
                );
                if (!ok) return;

                try {
                  const res = await fetch(`${API_BASE}/api/rebuild`, {
                    method: "POST",
                  });

                  if (!res.ok) throw new Error("Failed to rebuild index");

                  alert("Index rebuilt successfully! 🎉");
                } catch (err) {
                  console.error(err);
                  alert("Error rebuilding index. Check backend logs.");
                }
              }}
              className="text-xs px-3 py-2 rounded-full border border-primary text-primary hover:bg-primary hover:text-white transition"
            >
              Rebuild Index
            </button>
          </div>

          {/* Only show metrics + legend when we have a last answer */}
          {lastMeta ? (
            <>
              {/* Trust ring + stats */}
              <div className="mt-4 flex items-center gap-4">
                <div className="relative h-20 w-20">
                  <div className="absolute inset-0 rounded-full bg-indigo-50" />
                  <div className="absolute inset-1 rounded-full bg-white flex items-center justify-center shadow-inner">
                    <span className="text-lg font-semibold text-primary">
                      {formatTrust(lastMeta.trust_score)}
                    </span>
                  </div>
                </div>
                <div className="flex-1 space-y-1">
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-textMuted">Latency</span>
                    <span className="font-medium text-textDark">
                      {formatLatency(lastMeta.latency_ms)}
                    </span>
                  </div>
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-textMuted">Retrieved chunks</span>
                    <span className="font-medium text-textDark">
                      {lastMeta.chunks.length}
                    </span>
                  </div>
                  <div className="mt-1 inline-flex items-center gap-1 rounded-full bg-slate-50 px-2 py-1 text-[11px] text-textMuted">
                    <span className="h-1.5 w-1.5 rounded-full bg-primary" />
                    Retrieval mode: <span className="font-medium">Strict</span>
                  </div>
                </div>
              </div>

              {/* Relevance legend + toggle */}
              <div className="mt-4 border-t border-slate-100 pt-3">
                <div className="flex items-center justify-between text-[11px] font-medium text-textDark">
                  <span>Relevance breakdown</span>
                  <label className="flex items-center gap-1 text-[11px] text-textMuted cursor-pointer">
                    <input
                      type="checkbox"
                      checked={showOffTopic}
                      onChange={(e) => setShowOffTopic(e.target.checked)}
                      className="h-3 w-3 rounded border-slate-300"
                    />
                    <span>Show off-topic</span>
                  </label>
                </div>

                {(() => {
                  const relatedCount = lastMeta.chunks.filter(
                    (c) => c.relevance === "Related"
                  ).length;
                  const somewhatCount = lastMeta.chunks.filter(
                    (c) => c.relevance === "Somewhat related"
                  ).length;
                  const offTopicCount = lastMeta.chunks.filter(
                    (c) => c.relevance === "Off-topic"
                  ).length;

                  return (
                    <div className="mt-2 space-y-1.5 text-[11px]">
                      <div className="flex items-center justify-between">
                        <span className="inline-flex items-center gap-1 text-textMuted">
                          <span className="h-2 w-2 rounded-full bg-emerald-500" />
                          Related
                        </span>
                        <span className="font-medium text-textDark">
                          {relatedCount}
                        </span>
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="inline-flex items-center gap-1 text-textMuted">
                          <span className="h-2 w-2 rounded-full bg-sky-500" />
                          Somewhat related
                        </span>
                        <span className="font-medium text-textDark">
                          {somewhatCount}
                        </span>
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="inline-flex items-center gap-1 text-textMuted">
                          <span className="h-2 w-2 rounded-full bg-rose-500" />
                          Off-topic
                        </span>
                        <span className="font-medium text-textDark">
                          {offTopicCount}
                        </span>
                      </div>
                    </div>
                  );
                })()}
              </div>

              {/* Chunk list */}
              <div className="mt-5">
                <h3 className="text-xs font-semibold text-textDark">
                  Retrieved chunks
                </h3>
                <ul className="mt-2 space-y-2.5 text-xs">
                  {(showOffTopic
                    ? lastMeta.chunks
                    : lastMeta.chunks.filter((c) => c.relevance !== "Off-topic")
                  ).map((c, i) => (
                    <li
                      key={`${c.source}-${c.chunk}-${i}`}
                      className="rounded-2xl border border-slate-100 bg-slate-50/80 px-3 py-2"
                    >
                      <div className="flex items-center justify-between gap-2">
                        <div className="truncate text-[11px] font-medium text-textDark">
                          {c.source} · chunk {c.chunk}
                        </div>
                        <span
                          className={[
                            "inline-flex items-center rounded-full border px-2 py-0.5 text-[10px] font-medium",
                            relevanceBadgeClass(c.relevance),
                          ].join(" ")}
                        >
                          {relevanceLabel(c.relevance)}
                        </span>
                      </div>
                      <div className="mt-1 line-clamp-2 text-[11px] text-textMuted">
                        {c.text}
                      </div>
                      <div className="mt-1 flex items-center justify-between text-[10px] text-slate-400">
                        <span>
                          dist{" "}
                          {c.distance != null ? c.distance.toFixed(3) : "-"}
                        </span>
                      </div>
                      {/* Mini distance bar */}
                      <div className="mt-1 h-1.5 w-full rounded-full bg-slate-100 overflow-hidden">
                        <div
                          className="h-full rounded-full bg-primary/60"
                          style={{ width: distanceWidth(c.distance) }}
                        />
                      </div>
                    </li>
                  ))}

                  {lastMeta.chunks.length === 0 && (
                    <li className="rounded-2xl bg-slate-50 px-3 py-2 text-[11px] text-textMuted">
                      No chunks recorded for this answer.
                    </li>
                  )}
                </ul>
              </div>
            </>
          ) : (
            <div className="mt-5 rounded-2xl bg-slate-50 px-3 py-3 text-[11px] text-textMuted">
              Ask a question to see trust score and retrieved chunk details here.
            </div>
          )}
        </div>
      </aside>
    </main>
  );
};

export default ChatPage;
