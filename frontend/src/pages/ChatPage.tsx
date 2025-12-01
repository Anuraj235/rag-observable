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

/** Escape regex special chars in a query word. */
function escapeRegExp(str: string): string {
  return str.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/**
 * Highlight query words inside a chunk of text using <mark>.
 */
function highlightText(text: string, query: string): string {
  if (!text || !query) return text;

  const words = Array.from(
    new Set(
      query
        .split(/\s+/)
        .map((w) => w.trim())
        .filter((w) => w.length > 2)
    )
  );

  if (words.length === 0) return text;

  let result = text;
  for (const word of words) {
    const re = new RegExp(`(${escapeRegExp(word)})`, "gi");
    result = result.replace(
      re,
      `<mark class="bg-yellow-100 rounded px-0.5">$1</mark>`
    );
  }
  return result;
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
  /** Original question that produced this answer (for highlighting). */
  question?: string;
  /** Top-k used for this answer (for history page). */
  top_k?: number;
  /** Which model actually produced this answer (base vs fine-tuned). */
  model_used?: string | null;
  /** Whether this answer was requested with the fine-tuned flag. */
  used_finetuned?: boolean | null;
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
  /** Returned by backend: name of model used (ft or base). */
  model_used?: string | null;
};

type Comparison = {
  answer: string;
  model_used?: string | null;
  latency_ms?: number | null;
  used_finetuned?: boolean | null;
};

const API_BASE = "http://localhost:8000";
const CHAT_STORAGE_KEY = "rag_chat_messages";
const RUN_HISTORY_KEY = "rag_run_history";

/** Convert distance (smaller = better) to a bar width percentage. */
function distanceWidth(distance: number | null): string {
  if (distance == null) return "0%";
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
  /** Toggle: should we call the fine-tuned model or the base model? */
  const [useFinetuned, setUseFinetuned] = useState<boolean>(true);

  const [initialized, setInitialized] = useState(false);

  // For hover + click evidence previews on source pills
  const [hoveredEvidenceId, setHoveredEvidenceId] = useState<string | null>(
    null
  );
  const [pinnedEvidenceId, setPinnedEvidenceId] = useState<string | null>(null);

  // A/B comparison results keyed by assistant message id
  const [comparisons, setComparisons] = useState<Record<string, Comparison>>(
    {}
  );
  const [compareLoading, setCompareLoading] = useState<
    Record<string, boolean>
  >({});

  const scrollRef = useRef<HTMLDivElement | null>(null);

  // 🔹 On mount: load chat messages from sessionStorage
  useEffect(() => {
    try {
      const raw = sessionStorage.getItem(CHAT_STORAGE_KEY);
      if (raw) {
        const parsed = JSON.parse(raw) as Message[];
        setMessages(parsed);
      }
    } catch (err) {
      console.error("Failed to load chat messages from storage", err);
    } finally {
      setInitialized(true);
    }
  }, []);

  // ✅ Scroll the inner messages panel when messages length changes
  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
  }, [messages.length]);

  const lastAssistant = [...messages].reverse().find(
    (m) => m.role === "assistant"
  );
  const lastMeta = lastAssistant?.meta;

  // 🔹 Persist chat messages to sessionStorage
  useEffect(() => {
    if (!initialized) return;
    try {
      sessionStorage.setItem(CHAT_STORAGE_KEY, JSON.stringify(messages));
    } catch (err) {
      console.error("Failed to save chat messages to storage", err);
    }
  }, [messages, initialized]);

  // 🔹 Save assistant runs into sessionStorage for the Run History page
  useEffect(() => {
    if (!initialized) return;

    const runs = messages
      .filter((m) => m.role === "assistant" && m.meta)
      .map((m) => ({
        id: m.id,
        question: m.meta?.question ?? "",
        trust_score: m.meta?.trust_score ?? null,
        latency_ms: m.meta?.latency_ms ?? null,
        top_k: m.meta?.top_k ?? null,
        chunks: m.meta?.chunks ?? [],
        model_used: m.meta?.model_used ?? null,
        used_finetuned: m.meta?.used_finetuned ?? null,
        created_at: Date.now(),
      }));

    try {
      sessionStorage.setItem(RUN_HISTORY_KEY, JSON.stringify(runs));
    } catch {
      // ignore if sessionStorage is unavailable
    }
  }, [messages, initialized]);

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
        body: JSON.stringify({
          query: trimmed,
          top_k: topK,
          use_finetuned: useFinetuned,
        }),
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
          question: trimmed,
          top_k: topK,
          model_used: data.model_used ?? null,
          used_finetuned: useFinetuned,
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
      setPinnedEvidenceId(null);
      setHoveredEvidenceId(null);
      setComparisons({});
      setCompareLoading({});
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

  async function handleCompare(message: Message, targetUseFinetuned: boolean) {
    if (!message.meta?.question) return;

    const question = message.meta.question;
    const kForMsg = message.meta.top_k ?? topK;

    setCompareLoading((prev) => ({ ...prev, [message.id]: true }));
    try {
      const res = await fetch(`${API_BASE}/api/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: question,
          top_k: kForMsg,
          use_finetuned: targetUseFinetuned,
        }),
      });

      if (!res.ok) {
        throw new Error(`Backend returned ${res.status}`);
      }

      const data: QueryResponse = await res.json();
      const cleanedAnswer = stripSourcesFromAnswer(data.answer);

      setComparisons((prev) => ({
        ...prev,
        [message.id]: {
          answer: cleanedAnswer,
          model_used: data.model_used ?? null,
          latency_ms: data.latency_ms ?? null,
          used_finetuned: targetUseFinetuned,
        },
      }));
    } catch (err) {
      console.error(err);
      setComparisons((prev) => ({
        ...prev,
        [message.id]: {
          answer:
            "Error while comparing models – check backend logs or try again.",
        },
      }));
    } finally {
      setCompareLoading((prev) => ({ ...prev, [message.id]: false }));
    }
  }

  function clearChat() {
    const ok = confirm("Clear chat and reset run history for this session?");
    if (!ok) return;

    setMessages([]);
    setPinnedEvidenceId(null);
    setHoveredEvidenceId(null);
    setComparisons({});
    setCompareLoading({});

    try {
      sessionStorage.removeItem(CHAT_STORAGE_KEY);
      sessionStorage.removeItem(RUN_HISTORY_KEY);
    } catch {
      // ignore
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

  function formatModelUsed(model?: string | null) {
    if (!model) return "Unknown model";
    return model;
  }

  function modelKindBadge(meta?: MessageMeta) {
    if (!meta) return null;

    const isFt = meta.used_finetuned;
    return (
      <span
        className={[
          "inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[10px] font-medium border",
          isFt
            ? "bg-violet-50 text-violet-700 border-violet-200"
            : "bg-slate-50 text-slate-600 border-slate-200",
        ].join(" ")}
      >
        <span
          className={`h-1.5 w-1.5 rounded-full ${
            isFt ? "bg-violet-500" : "bg-slate-400"
          }`}
        />
        {isFt ? "Fine-tuned" : "Base model"}
      </span>
    );
  }

  return (
    <main className="h-[calc(100vh-64px)] bg-bg px-8 py-6 flex gap-6 overflow-hidden">
      {/* LEFT: Chat assistant */}
      <section className="flex-1 flex flex-col min-h-0">
        {/* Header */}
        <div className="flex items-baseline justify-between">
          <div>
            <h1 className="text-xl font-semibold text-textDark">
              Chat Assistant
            </h1>
            <p className="text-xs text-textMuted">
              Ask questions about your documents and get answers backed by
              retrieved chunks.
            </p>
          </div>

          <div className="flex items-center gap-3">
            {/* Fine-tuned toggle */}
            <div className="flex flex-col items-end gap-0.5">
              <label className="flex items-center gap-2 text-[11px] text-textMuted cursor-pointer">
                <span className="font-medium text-textDark">
                  Use fine-tuned model
                </span>
                <button
                  type="button"
                  onClick={() => setUseFinetuned((prev) => !prev)}
                  className={[
                    "relative inline-flex h-4 w-7 items-center rounded-full border transition",
                    useFinetuned
                      ? "bg-primary border-primary"
                      : "bg-slate-200 border-slate-300",
                  ].join(" ")}
                >
                  <span
                    className={[
                      "inline-block h-3 w-3 transform rounded-full bg-white shadow transition",
                      useFinetuned ? "translate-x-3" : "translate-x-0.5",
                    ].join(" ")}
                  />
                </button>
              </label>
              <span className="text-[10px] text-textMuted">
                {useFinetuned
                  ? "Using instructor tuned on your labeled runs."
                  : "Using base GPT model only."}
              </span>
            </div>

            {/* Clear chat button */}
            <button
              type="button"
              onClick={clearChat}
              className="text-[11px] px-3 py-1.5 rounded-full border border-slate-200 text-textMuted hover:text-primary hover:border-primary/60 hover:bg-primary/5 transition"
            >
              Clear chat
            </button>
          </div>
        </div>

        {/* Chat container */}
        <div className="mt-4 flex-1 min-h-0 rounded-3xl bg-white shadow-soft border border-indigo-50 flex flex-col">
          {/* Messages */}
          <div
            ref={scrollRef}
            className="flex-1 min-h-0 px-6 pt-6 pb-3 overflow-y-auto space-y-4"
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

              const evidenceChunks = allChunks.filter(
                (c) =>
                  c.relevance === "Related" || c.relevance === "Somewhat related"
              );

              const uniqueChunks = Array.from(
                new Map(
                  evidenceChunks.map((c) => [`${c.source}-${c.chunk}`, c])
                ).values()
              );

              const baseIdPrefix = m.id;
              const effectiveEvidenceId = pinnedEvidenceId ?? hoveredEvidenceId;
              const activeEvidenceChunk =
                uniqueChunks.find(
                  (c) =>
                    `${baseIdPrefix}-${c.source}-${c.chunk}` === effectiveEvidenceId
                ) || null;

              const comparison = comparisons[m.id];
              const isCompareLoading = compareLoading[m.id] ?? false;

              const canCompare =
                isAssistant && m.meta?.question && m.meta.used_finetuned != null;

              const compareTargetUseFt = m.meta?.used_finetuned
                ? false
                : true;

              const compareLabel = m.meta?.used_finetuned
                ? "Compare with base model"
                : "Compare with fine-tuned";

              return (
                <div
                  key={m.id}
                  className={`flex ${isAssistant ? "justify-start" : "justify-end"}`}
                >
                  <div
                    className={[
                      "max-w-[70%] rounded-2xl px-4 py-3 text-sm shadow-soft-sm transition-transform duration-150",
                      isAssistant
                        ? "bg-slate-50 text-textDark border border-slate-200 rounded-bl-md hover:-translate-y-0.5"
                        : "bg-primary text-white rounded-br-md hover:-translate-y-0.5",
                    ].join(" ")}
                  >
                    {/* Assistant meta header */}
                    {isAssistant && m.meta && (
                      <div className="mb-2 flex items-center justify-between gap-2 text-[10px] text-textMuted">
                        <div className="flex items-center gap-2 min-w-0">
                          {modelKindBadge(m.meta)}
                          <span className="truncate max-w-[180px]">
                            {formatModelUsed(m.meta.model_used)}
                          </span>
                        </div>
                        <div className="flex items-center gap-3">
                          <span>
                            {m.meta.latency_ms != null
                              ? `${Math.round(m.meta.latency_ms)} ms`
                              : "- ms"}
                          </span>
                          <span>
                            {m.meta.trust_score != null
                              ? `${Math.round(m.meta.trust_score)}%`
                              : "- %"}
                          </span>
                        </div>
                      </div>
                    )}

                    <p className="whitespace-pre-line">{m.content}</p>

                    {/* Model compare row for assistant messages */}
                    {isAssistant && m.meta && (
                      <div className="mt-2 flex items-center justify-between gap-2 text-[10px] text-textMuted">
                        <div className="flex items-center gap-2">
                          <span className="rounded-full bg-slate-100 px-2 py-0.5">
                            k = {m.meta.top_k ?? topK}
                          </span>
                        </div>
                        {canCompare && (
                          <button
                            type="button"
                            onClick={() =>
                              handleCompare(m, compareTargetUseFt)
                            }
                            disabled={isCompareLoading}
                            className="inline-flex items-center gap-1 rounded-full border border-slate-200 px-2 py-0.5 text-[10px] hover:border-primary/60 hover:text-primary disabled:opacity-60 disabled:cursor-not-allowed"
                          >
                            {isCompareLoading && (
                              <span className="h-1.5 w-1.5 rounded-full bg-primary animate-pulse" />
                            )}
                            <span>{compareLabel}</span>
                          </button>
                        )}
                      </div>
                    )}

                    {/* A/B comparison panel */}
                    {isAssistant && comparison && (
                      <div className="mt-3 rounded-2xl border border-indigo-100 bg-indigo-50/50 px-3 py-2 text-[11px] text-textMuted">
                        <div className="flex items-center justify-between mb-1">
                          <span className="text-[11px] font-semibold text-textDark">
                            A/B comparison
                          </span>
                          <span className="text-[10px] text-textMuted truncate max-w-[160px]">
                            {comparison.used_finetuned
                              ? "Comparison: Fine-tuned"
                              : "Comparison: Base model"}
                          </span>
                        </div>
                        <div className="grid grid-cols-2 gap-3 mt-1">
                          <div>
                            <div className="flex items-center justify-between mb-1 text-[10px] text-textMuted">
                              <span className="font-semibold">
                                Original answer
                              </span>
                              <span className="truncate max-w-[110px]">
                                {formatModelUsed(m.meta?.model_used)}
                              </span>
                            </div>
                            <div className="rounded-xl bg-white/80 border border-slate-200 px-2 py-1.5 text-[11px] text-textDark max-h-32 overflow-y-auto whitespace-pre-line">
                              {m.content}
                            </div>
                            <div className="mt-1 text-[10px] text-slate-400">
                              Latency:{" "}
                              {m.meta?.latency_ms != null
                                ? `${Math.round(m.meta.latency_ms)} ms`
                                : "-"}
                            </div>
                          </div>
                          <div>
                            <div className="flex items-center justify-between mb-1 text-[10px] text-textMuted">
                              <span className="font-semibold">
                                Comparison answer
                              </span>
                              <span className="truncate max-w-[110px]">
                                {formatModelUsed(comparison.model_used)}
                              </span>
                            </div>
                            <div className="rounded-xl bg-white/80 border border-slate-200 px-2 py-1.5 text-[11px] text-textDark max-h-32 overflow-y-auto whitespace-pre-line">
                              {comparison.answer}
                            </div>
                            <div className="mt-1 text-[10px] text-slate-400">
                              Latency:{" "}
                              {comparison.latency_ms != null
                                ? `${Math.round(comparison.latency_ms)} ms`
                                : "-"}
                            </div>
                          </div>
                        </div>
                      </div>
                    )}

                    {/* Source pills + evidence preview for assistant messages */}
                    {isAssistant && uniqueChunks.length > 0 && (
                      <div
                        className="mt-3 border-t border-slate-200 pt-2"
                        onMouseLeave={() => {
                          setHoveredEvidenceId(null);
                        }}
                      >
                        <div className="mb-1 flex items-center justify-between text-[11px] text-textMuted">
                          <span className="font-medium">Sources</span>
                          <span className="text-[10px]">
                            {uniqueChunks.length} chunk
                            {uniqueChunks.length === 1 ? "" : "s"}
                          </span>
                        </div>
                        <div className="flex flex-wrap gap-1.5">
                          {uniqueChunks.map((c, idx) => {
                            const chunkId = `${baseIdPrefix}-${c.source}-${c.chunk}`;
                            const isPinned = pinnedEvidenceId === chunkId;

                            return (
                              <button
                                key={`${c.source}-${c.chunk}-${idx}`}
                                type="button"
                                onMouseEnter={() => {
                                  if (!isPinned) {
                                    setHoveredEvidenceId(chunkId);
                                  }
                                }}
                                onClick={() => {
                                  setPinnedEvidenceId((current) =>
                                    current === chunkId ? null : chunkId
                                  );
                                  setHoveredEvidenceId(chunkId);
                                }}
                                className={[
                                  "inline-flex items-center rounded-full border px-2 py-0.5 text-[11px] max-w-[220px] truncate transition",
                                  relevanceBadgeClass(c.relevance),
                                  isPinned ? "ring-1 ring-primary/60" : "",
                                ].join(" ")}
                                title="Click to pin evidence"
                              >
                                <span className="mr-1 text-[10px] opacity-80">
                                  [{idx + 1}]
                                </span>
                                <span className="truncate">
                                  {c.source} · chunk {c.chunk}
                                </span>
                              </button>
                            );
                          })}
                        </div>

                        {/* Evidence preview card (hover or pinned) */}
                        {activeEvidenceChunk && (
                          <div className="mt-3 rounded-2xl border border-slate-200 bg-white/90 px-3 py-2 text-[11px] text-textMuted shadow-soft-sm">
                            <div className="flex items-center justify-between mb-1">
                              <div className="font-medium text-textDark truncate">
                                {activeEvidenceChunk.source} · chunk{" "}
                                {activeEvidenceChunk.chunk}
                              </div>
                              <span
                                className={[
                                  "inline-flex items-center rounded-full border px-2 py-0.5 text-[10px] font-medium",
                                  relevanceBadgeClass(
                                    activeEvidenceChunk.relevance
                                  ),
                                ].join(" ")}
                              >
                                {relevanceLabel(activeEvidenceChunk.relevance)}
                              </span>
                            </div>
                            <div
                              className="mt-1 max-h-40 overflow-y-auto leading-relaxed"
                              dangerouslySetInnerHTML={{
                                __html: highlightText(
                                  activeEvidenceChunk.text,
                                  m.meta?.question ?? ""
                                ),
                              }}
                            />
                            <div className="mt-1 flex items-center justify-between text-[10px] text-slate-400">
                              <span>
                                dist{" "}
                                {activeEvidenceChunk.distance != null
                                  ? activeEvidenceChunk.distance.toFixed(3)
                                  : "-"}
                              </span>
                              {pinnedEvidenceId ===
                                `${baseIdPrefix}-${activeEvidenceChunk.source}-${activeEvidenceChunk.chunk}` && (
                                <button
                                  type="button"
                                  onClick={() => setPinnedEvidenceId(null)}
                                  className="text-[10px] text-primary hover:underline"
                                >
                                  Unpin
                                </button>
                              )}
                            </div>
                          </div>
                        )}
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
              className="rounded-full bg-primary px-5 py-2 text-sm font-medium text-white shadow-soft disabled:cursor-not-allowed disabled:bg-primary/40 transition-transform active:scale-95"
            >
              {loading ? "Sending…" : "Send"}
            </button>
          </form>
        </div>
      </section>

      {/* RIGHT: Trust / chunk panel */}
      <aside className="w-[320px] shrink-0">
        <div className="h-full rounded-3xl bg-white shadow-soft border border-indigo-50 p-5 flex flex-col">
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

          {/* Rebuild index + model used */}
          <div className="mt-4 flex items-center justify-between">
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

            {lastMeta && (
              <div className="text-[10px] text-textMuted text-right max-w-[170px]">
                <div className="flex items-center justify-end gap-2 mb-0.5">
                  <span className="font-medium text-textDark">Model used</span>
                  {modelKindBadge(lastMeta)}
                </div>
                <div className="truncate">
                  {formatModelUsed(lastMeta.model_used)}
                </div>
              </div>
            )}
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
              <div className="mt-5 overflow-y-auto pr-1">
                <h3 className="text-xs font-semibold text-textDark">
                  Retrieved chunks
                </h3>
                <ul className="mt-2 space-y-2.5 text-xs">
                  {(showOffTopic
                    ? lastMeta.chunks
                    : lastMeta.chunks.filter(
                        (c) => c.relevance !== "Off-topic"
                      )
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
              Ask a question to see trust score and retrieved chunk details
              here.
            </div>
          )}
        </div>
      </aside>
    </main>
  );
};

export default ChatPage;
