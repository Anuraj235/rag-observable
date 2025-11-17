// src/App.tsx
import React, { useState } from "react";
import type { KeyboardEvent } from "react";


type Chunk = {
    source: string;
    chunk: number;
    text: string;
    distance?: number | null;
};

type Message = {
    role: "user" | "assistant";
    content: string;
    meta?: {
        trust_score: number;
        latency_ms: number;
        chunks: Chunk[];
    };
};

const API_BASE = "http://localhost:8000";

const App: React.FC = () => {
    const [messages, setMessages] = useState<Message[]>([]);
    const [input, setInput] = useState("");
    const [loading, setLoading] = useState(false);

    const sendMessage = async () => {
        const query = input.trim();
        if (!query || loading) return;

        // add user message
        const userMessage: Message = { role: "user", content: query };
        setMessages((prev) => [...prev, userMessage]);
        setInput("");
        setLoading(true);

        try {
            const res = await fetch(`${API_BASE}/api/query`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ query, top_k: 3 }),
            });

            if (!res.ok) throw new Error(`HTTP ${res.status}`);

            const data = await res.json();

            const assistantMessage: Message = {
                role: "assistant",
                content: data.answer,
                meta: {
                    trust_score: data.trust_score,
                    latency_ms: data.latency_ms,
                    chunks: data.chunks,
                },
            };

            setMessages((prev) => [...prev, assistantMessage]);
        } catch (err) {
            console.error(err);
            const errorMessage: Message = {
                role: "assistant",
                content:
                    "Sorry, something went wrong while contacting the RAG backend.",
            };
            setMessages((prev) => [...prev, errorMessage]);
        } finally {
            setLoading(false);
        }
    };

    const handleKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    };

    const lastAssistant =
        messages.length > 0
            ? [...messages].reverse().find((m) => m.role === "assistant")
            : undefined;

    return (
        <div className="min-h-screen flex flex-col bg-bg">
            {/* Top nav */}
            <header className="w-full border-b border-gray-200 bg-white px-6 py-3 flex items-center justify-between">
                <div>
                    <div className="text-lg font-semibold text-textDark">
                        Faithful RAG
                    </div>
                    <div className="text-xs text-textMuted">Your AI with receipts.</div>
                </div>
                <div className="text-sm text-textMuted">Chat</div>
            </header>

            {/* Main layout */}
            <main className="flex-1 flex gap-6 px-6 py-5">
                {/* Chat column */}
                <section className="flex-[2] flex flex-direction-col">
                    <div className="flex-1 overflow-y-auto pr-3 mb-3 space-y-3">
                        {messages.map((m, idx) =>
                            m.role === "user" ? (
                                <div key={idx} className="flex justify-end">
                                    <div className="max-w-[70%] rounded-2xl bg-primary px-4 py-2 text-sm text-white shadow-soft-sm">
                                        {m.content}
                                    </div>
                                </div>
                            ) : (
                                <div key={idx} className="flex justify-start">
                                    <div className="max-w-[80%] rounded-2xl bg-white px-4 py-3 text-sm text-textDark border border-gray-200 shadow-soft-sm">
                                        <div className="whitespace-pre-wrap">{m.content}</div>

                                        {m.meta && (
                                            <>
                                                {/* trust + latency */}
                                                <div className="mt-3 flex items-center gap-2">
                                                    <span className="inline-flex items-center rounded-full bg-green-100 px-3 py-0.5 text-xs font-semibold text-success">
                                                        {m.meta.trust_score}% Trust
                                                    </span>
                                                    <span className="text-[11px] text-textMuted">
                                                        {m.meta.latency_ms.toFixed(0)} ms
                                                    </span>
                                                </div>

                                                {/* sources */}
                                                {m.meta.chunks?.length > 0 && (
                                                    <div className="mt-2 flex flex-wrap gap-1.5">
                                                        {m.meta.chunks.map((c, i) => (
                                                            <span
                                                                key={i}
                                                                className="inline-flex items-center rounded-full bg-indigo-50 px-3 py-0.5 text-[11px] text-primary"
                                                            >
                                                                {c.source} · chunk {c.chunk}
                                                            </span>
                                                        ))}
                                                    </div>
                                                )}
                                            </>
                                        )}
                                    </div>
                                </div>
                            )
                        )}

                        {loading && (
                            <div className="text-xs text-textMuted">Thinking…</div>
                        )}
                    </div>

                    {/* Input row */}
                    <div className="border-t border-gray-200 pt-3 flex gap-2">
                        <input
                            className="flex-1 rounded-full border border-gray-300 px-4 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary/60"
                            placeholder="Ask a question about your documents…"
                            value={input}
                            onChange={(e) => setInput(e.target.value)}
                            onKeyDown={handleKeyDown}
                        />
                        <button
                            onClick={sendMessage}
                            disabled={loading}
                            className="rounded-full bg-primary px-4 py-2 text-sm font-medium text-white shadow-soft disabled:bg-gray-400"
                        >
                            Send
                        </button>
                    </div>
                </section>

                {/* Insight panel */}
                <section className="flex-1 h-fit rounded-2xl border border-gray-200 bg-white p-4 shadow-soft space-y-3">
                    <h2 className="text-base font-semibold text-textDark">
                        Answer Insight
                    </h2>

                    {lastAssistant && lastAssistant.meta ? (
                        <>
                            <div>
                                <div className="text-[11px] uppercase tracking-[0.2em] text-textMuted">
                                    Trust Score
                                </div>
                                <div className="text-3xl font-bold text-textDark">
                                    {lastAssistant.meta.trust_score}%
                                </div>
                            </div>

                            <div>
                                <div className="text-[11px] uppercase tracking-[0.2em] text-textMuted">
                                    Retrieved Chunks
                                </div>
                                <ul className="mt-1 list-disc pl-4 text-[13px] text-gray-700 space-y-0.5">
                                    {lastAssistant.meta.chunks.map((c, i) => (
                                        <li key={i}>
                                            {c.source} · chunk {c.chunk}
                                            {typeof c.distance === "number" &&
                                                ` · dist ${c.distance.toFixed(3)}`}
                                        </li>
                                    ))}
                                </ul>
                            </div>
                        </>
                    ) : (
                        <p className="text-sm text-textMuted">
                            Ask a question to see trust score and retrieved chunk details.
                        </p>
                    )}
                </section>
            </main>
        </div>
    );
};

export default App;
