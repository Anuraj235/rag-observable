import React from "react";
import { Link } from "react-router-dom";

const LandingPage: React.FC = () => {
  return (
    <div className="px-6 lg:px-12 py-10 space-y-12">
      {/* Hero section */}
      <section className="grid grid-cols-1 lg:grid-cols-2 gap-10 items-center">
        {/* Left: headline + CTA */}
        <div className="space-y-5">
          <div className="inline-flex items-center rounded-full bg-indigo-50 px-3 py-1 text-xs font-medium text-primary shadow-soft-sm">
            🔍 Faithful & Observable RAG
          </div>

          <h1 className="text-3xl sm:text-4xl lg:text-5xl font-semibold tracking-tight text-textDark">
            Your AI <span className="text-primary">With Receipts.</span>
          </h1>

          <p className="text-sm sm:text-base text-textMuted max-w-xl">
            Ask anything about your documents and get answers backed by real
            evidence. Every response is grounded in retrieved chunks, with trust
            scores and citations you can inspect.
          </p>

          {/* Primary CTA buttons */}
          <div className="flex flex-wrap items-center gap-3 pt-2">
            <Link
              to="/chat"
              className="rounded-full bg-primary px-5 py-2.5 text-sm font-medium text-white shadow-soft hover:bg-primary/90"
            >
              Start Chatting
            </Link>

            <button className="rounded-full border border-indigo-200 bg-white px-4 py-2.5 text-sm font-medium text-textDark hover:bg-indigo-50/60">
              View Demo
            </button>
          </div>

          {/* Metrics row */}
          <div className="mt-6 grid grid-cols-3 gap-4 max-w-xs sm:max-w-sm">
            <div>
              <div className="text-xs uppercase tracking-[0.18em] text-textMuted">
                Accuracy
              </div>
              <div className="text-xl font-semibold text-textDark">
                99.9<span className="text-sm">%</span>
              </div>
            </div>
            <div>
              <div className="text-xs uppercase tracking-[0.18em] text-textMuted">
                Documents
              </div>
              <div className="text-xl font-semibold text-textDark">10M+</div>
            </div>
            <div>
              <div className="text-xs uppercase tracking-[0.18em] text-textMuted">
                Response
              </div>
              <div className="text-xl font-semibold text-textDark">
                &lt;100<span className="text-sm">ms</span>
              </div>
            </div>
          </div>
        </div>

        {/* Right: fake preview card */}
        <div className="relative">
          <div className="mx-auto max-w-md rounded-2xl border border-gray-200 bg-white shadow-soft overflow-hidden">
            <div className="flex items-center gap-2 border-b border-gray-100 px-4 py-2 text-xs text-textMuted">
              <span className="h-2.5 w-2.5 rounded-full bg-red-400" />
              <span className="h-2.5 w-2.5 rounded-full bg-yellow-400" />
              <span className="h-2.5 w-2.5 rounded-full bg-green-400" />
              <span className="ml-2">What is supervised learning?</span>
            </div>

            <div className="p-4 space-y-3 text-xs sm:text-[13px]">
              <p className="text-textDark">
                Supervised learning is a type of machine learning where a model
                is trained on labeled examples to predict labels for new inputs.
                It includes tasks such as classification and regression{" "}
                <span className="text-primary font-medium">[1]</span>.
              </p>

              <div className="space-y-1 text-[11px]">
                <div className="font-semibold text-textMuted uppercase tracking-[0.18em]">
                  Sources
                </div>
                <ul className="space-y-0.5">
                  <li>[1] ml_basics.txt (chunk 0)</li>
                  <li>[2] psychology_of_habits.txt (chunk 0)</li>
                  <li>[3] climate_change.txt (chunk 0)</li>
                </ul>
              </div>

              <div className="flex items-center justify-between pt-1">
                <span className="inline-flex items-center rounded-full bg-green-100 px-2.5 py-0.5 text-[11px] font-semibold text-success">
                  90% Trust
                </span>
                <span className="text-[11px] text-textMuted">2435 ms</span>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Features row */}
      <section className="space-y-4">
        <h2 className="text-lg font-semibold text-textDark">
          Trustworthy AI, Every Answer
        </h2>

        <div className="grid gap-4 md:grid-cols-3">
          <div className="rounded-2xl bg-white p-4 shadow-soft border border-gray-200">
            <div className="text-sm font-semibold text-textDark">
              Cited Answers
            </div>
            <p className="mt-1 text-xs text-textMuted">
              Every response includes direct citations to source documents with
              exact chunk references.
            </p>
          </div>

          <div className="rounded-2xl bg-white p-4 shadow-soft border border-gray-200">
            <div className="text-sm font-semibold text-textDark">
              Evidence Highlighter
            </div>
            <p className="mt-1 text-xs text-textMuted">
              See the exact sentences from your documents that support each
              AI-generated answer.
            </p>
          </div>

          <div className="rounded-2xl bg-white p-4 shadow-soft border border-gray-200">
            <div className="text-sm font-semibold text-textDark">
              Trust Score Engine
            </div>
            <p className="mt-1 text-xs text-textMuted">
              Real-time confidence scores based on retrieval quality, relevance,
              and source coverage.
            </p>
          </div>
        </div>
      </section>
    </div>
  );
};

export default LandingPage;
