from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from app.api.chat import router as chat_router
from app.api.knowledge_base import router as kb_router

app = FastAPI()
app.include_router(chat_router)
app.include_router(kb_router)


@app.get("/", response_class=HTMLResponse)
async def home():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Workflow API</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: background 0.4s, color 0.4s;
        }
        body.dark {
            background: linear-gradient(135deg, #0f172a, #1e293b);
            color: #e2e8f0;
        }
        body.light {
            background: linear-gradient(135deg, #f0f4f8, #dbe4ee);
            color: #1e293b;
        }
        .container { text-align: center; padding: 2rem; }
        h1 {
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
            background: linear-gradient(to right, #38bdf8, #818cf8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        p { margin-bottom: 2.5rem; font-size: 1.1rem; }
        body.dark p { color: #94a3b8; }
        body.light p { color: #64748b; }
        .links { display: flex; gap: 1.5rem; justify-content: center; margin-bottom: 2rem; }
        a {
            display: inline-block;
            padding: 0.85rem 2rem;
            border-radius: 0.75rem;
            text-decoration: none;
            font-weight: 600;
            font-size: 1rem;
            transition: transform 0.15s, box-shadow 0.15s;
        }
        a:hover { transform: translateY(-2px); }
        body.dark a:hover { box-shadow: 0 8px 24px rgba(0,0,0,0.3); }
        body.light a:hover { box-shadow: 0 8px 24px rgba(0,0,0,0.1); }
        .docs { background: #38bdf8; color: #0f172a; }
        .redoc { background: #818cf8; color: #0f172a; }
        .toggle {
            background: none;
            border: 2px solid #475569;
            border-radius: 50%;
            width: 2.5rem;
            height: 2.5rem;
            font-size: 1.2rem;
            cursor: pointer;
            transition: border-color 0.3s, transform 0.15s;
            display: inline-flex;
            align-items: center;
            justify-content: center;
        }
        .toggle:hover { transform: scale(1.1); }
        body.dark .toggle { border-color: #94a3b8; color: #e2e8f0; }
        body.light .toggle { border-color: #cbd5e1; }
    </style>
</head>
<body class="dark">
    <div class="container">
        <h1>Workflow API</h1>
        <p>LLM-powered chat API built with FastAPI &amp; LangChain</p>
        <div class="links">
            <a class="docs" href="/docs">Swagger Docs</a>
            <a class="redoc" href="/redoc">ReDoc</a>
        </div>
        <button class="toggle" onclick="toggleTheme()" aria-label="Toggle theme">
            <span id="icon">&#9728;</span>
        </button>
    </div>
    <script>
        function toggleTheme() {
            const body = document.body;
            const icon = document.getElementById("icon");
            if (body.classList.contains("dark")) {
                body.classList.replace("dark", "light");
                icon.textContent = "\u263E";
                localStorage.setItem("theme", "light");
            } else {
                body.classList.replace("light", "dark");
                icon.textContent = "\u2600";
                localStorage.setItem("theme", "dark");
            }
        }
        const saved = localStorage.getItem("theme");
        if (saved === "light") {
            document.body.classList.replace("dark", "light");
            document.getElementById("icon").textContent = "\u263E";
        }
    </script>
</body>
</html>
"""
