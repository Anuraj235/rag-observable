/** @type {import('tailwindcss').Config} */
export default {
    content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
    theme: {
        extend: {
            colors: {
                primary: "#6366F1",
                secondary: "#8B5CF6",
                success: "#22C55E",
                bg: "#F9FAFB",
                textDark: "#111827",
                textMuted: "#6B7280",
            },
            borderRadius: {
                xl: "14px",
                "2xl": "18px",
            },
            boxShadow: {
                soft: "0 4px 20px rgba(0,0,0,0.06)",
                "soft-sm": "0 4px 16px rgba(0,0,0,0.04)",
            },
        },
    },
    plugins: [],
};
