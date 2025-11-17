import React from "react";
import { NavLink, Route, Routes } from "react-router-dom";
import LandingPage from "./pages/LandingPage";
import ChatPage from "./pages/ChatPage";

const navLinkClass = ({ isActive }: { isActive: boolean }) =>
  [
    "px-3 py-1.5 rounded-full text-sm transition-colors",
    isActive
      ? "bg-primary text-white shadow-soft-sm"
      : "text-textMuted hover:bg-indigo-50",
  ].join(" ");

const App: React.FC = () => {
  return (
    <div className="min-h-screen bg-bg flex flex-col">
      {/* Top nav bar */}
      <header className="w-full border-b border-gray-200 bg-white px-6 py-3 flex items-center justify-between">
        <div>
          <div className="text-lg font-semibold text-textDark">Faithful RAG</div>
          <div className="text-xs text-textMuted">Your AI with receipts.</div>
        </div>

        <nav className="flex items-center gap-2">
          <NavLink to="/" className={navLinkClass} end>
            Home
          </NavLink>
          <NavLink to="/chat" className={navLinkClass}>
            Chat
          </NavLink>
        </nav>
      </header>

      {/* Page content */}
      <div className="flex-1">
        <Routes>
          <Route path="/" element={<LandingPage />} />
          <Route path="/chat" element={<ChatPage />} />
          <Route path="*" element={<LandingPage />} />
        </Routes>
      </div>
    </div>
  );
};

export default App;
