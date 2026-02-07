"""
Topic Classifier Test Dashboard

A simple HTML-based test interface for the topic classifier.
Accessible at /dashboard when the service is running.
"""

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

dashboard_router = APIRouter(tags=["dashboard"])

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ModAI Topic Classifier - Test Dashboard</title>
    <style>
        :root {
            --green: #22c55e;
            --yellow: #facc15;
            --orange: #f97316;
            --red: #ef4444;
            --bg: #0f172a;
            --card: #1e293b;
            --text: #f8fafc;
            --muted: #94a3b8;
            --border: #334155;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg);
            color: var(--text);
            min-height: 100vh;
            padding: 2rem;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
        }

        header {
            text-align: center;
            margin-bottom: 2rem;
        }

        h1 {
            font-size: 2rem;
            margin-bottom: 0.5rem;
        }

        .subtitle {
            color: var(--muted);
            font-size: 1rem;
        }

        .grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 2rem;
        }

        @media (max-width: 768px) {
            .grid {
                grid-template-columns: 1fr;
            }
        }

        .card {
            background: var(--card);
            border-radius: 12px;
            padding: 1.5rem;
            border: 1px solid var(--border);
        }

        .card h2 {
            font-size: 1.25rem;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        textarea {
            width: 100%;
            min-height: 120px;
            background: var(--bg);
            border: 1px solid var(--border);
            border-radius: 8px;
            color: var(--text);
            padding: 1rem;
            font-size: 1rem;
            font-family: inherit;
            resize: vertical;
        }

        textarea:focus {
            outline: none;
            border-color: var(--yellow);
        }

        button {
            background: var(--yellow);
            color: #000;
            border: none;
            padding: 0.75rem 1.5rem;
            font-size: 1rem;
            font-weight: 600;
            border-radius: 8px;
            cursor: pointer;
            margin-top: 1rem;
            transition: opacity 0.2s;
        }

        button:hover {
            opacity: 0.9;
        }

        button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }

        .result {
            margin-top: 1.5rem;
        }

        .tier-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.5rem 1rem;
            border-radius: 999px;
            font-weight: 600;
            font-size: 1.1rem;
        }

        .tier-1 { background: var(--green); color: #000; }
        .tier-2 { background: var(--yellow); color: #000; }
        .tier-3 { background: var(--orange); color: #000; }
        .tier-4 { background: var(--red); color: #fff; }

        .topics-list {
            margin-top: 1rem;
        }

        .topic-item {
            background: var(--bg);
            padding: 0.75rem 1rem;
            border-radius: 8px;
            margin-bottom: 0.5rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .topic-name {
            font-weight: 500;
        }

        .topic-confidence {
            color: var(--muted);
            font-size: 0.875rem;
        }

        .mini-tier {
            font-size: 0.75rem;
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
        }

        .stats {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 1rem;
            margin-top: 1rem;
        }

        .stat {
            background: var(--bg);
            padding: 1rem;
            border-radius: 8px;
            text-align: center;
        }

        .stat-value {
            font-size: 1.5rem;
            font-weight: 700;
        }

        .stat-label {
            color: var(--muted);
            font-size: 0.875rem;
        }

        .health-status {
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .health-dot {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: var(--green);
        }

        .health-dot.error {
            background: var(--red);
        }

        .examples {
            margin-top: 1rem;
        }

        .example-btn {
            background: var(--bg);
            color: var(--text);
            border: 1px solid var(--border);
            padding: 0.5rem 0.75rem;
            font-size: 0.875rem;
            margin: 0.25rem;
            border-radius: 6px;
        }

        .example-btn:hover {
            border-color: var(--yellow);
        }

        .json-output {
            background: var(--bg);
            padding: 1rem;
            border-radius: 8px;
            margin-top: 1rem;
            font-family: 'Monaco', 'Menlo', monospace;
            font-size: 0.75rem;
            overflow-x: auto;
            max-height: 300px;
            overflow-y: auto;
        }

        .error-message {
            background: rgba(239, 68, 68, 0.2);
            border: 1px solid var(--red);
            padding: 1rem;
            border-radius: 8px;
            margin-top: 1rem;
        }

        .loading {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            color: var(--muted);
        }

        .spinner {
            width: 20px;
            height: 20px;
            border: 2px solid var(--border);
            border-top-color: var(--yellow);
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>ModAI Topic Classifier</h1>
            <p class="subtitle">Test the 4-tier safety classification system</p>
        </header>

        <div class="grid">
            <!-- Input Panel -->
            <div class="card">
                <h2>Test Classification</h2>
                <textarea id="messageInput" placeholder="Enter a message to classify...">I need help with my math homework</textarea>

                <div class="examples">
                    <p style="color: var(--muted); font-size: 0.875rem; margin-bottom: 0.5rem;">Quick examples:</p>
                    <button class="example-btn" onclick="setExample('I need help with my algebra homework')">Homework Help</button>
                    <button class="example-btn" onclick="setExample('What colleges should I apply to?')">College Advice</button>
                    <button class="example-btn" onclick="setExample('My friend is being mean to me at school')">Peer Issues</button>
                    <button class="example-btn" onclick="setExample('I want to learn about politics and voting')">Politics</button>
                    <button class="example-btn" onclick="setExample('Tell me about dating and relationships')">Dating</button>
                    <button class="example-btn" onclick="setExample('I feel really sad and hopeless')">Mental Health</button>
                </div>

                <button id="classifyBtn" onclick="classify()">Classify Message</button>
            </div>

            <!-- Result Panel -->
            <div class="card">
                <h2>Classification Result</h2>
                <div id="resultArea">
                    <p style="color: var(--muted);">Enter a message and click "Classify Message" to see results.</p>
                </div>
            </div>

            <!-- Health Status -->
            <div class="card">
                <h2>Service Health</h2>
                <div id="healthArea">
                    <div class="loading">
                        <div class="spinner"></div>
                        <span>Checking service health...</span>
                    </div>
                </div>
            </div>

            <!-- Taxonomy Stats -->
            <div class="card">
                <h2>Taxonomy Overview</h2>
                <div id="taxonomyArea">
                    <div class="loading">
                        <div class="spinner"></div>
                        <span>Loading taxonomy...</span>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        const API_BASE = window.location.origin;

        function setExample(text) {
            document.getElementById('messageInput').value = text;
        }

        function getTierClass(tier) {
            return `tier-${tier}`;
        }

        function getTierName(tier) {
            const names = {1: 'GREEN', 2: 'YELLOW', 3: 'ORANGE', 4: 'RED'};
            return names[tier] || 'UNKNOWN';
        }

        function getTierEmoji(tier) {
            const emojis = {1: '✓', 2: '⚠', 3: '🔶', 4: '⛔'};
            return emojis[tier] || '?';
        }

        async function classify() {
            const message = document.getElementById('messageInput').value.trim();
            if (!message) {
                alert('Please enter a message to classify');
                return;
            }

            const btn = document.getElementById('classifyBtn');
            const resultArea = document.getElementById('resultArea');

            btn.disabled = true;
            btn.textContent = 'Classifying...';
            resultArea.innerHTML = '<div class="loading"><div class="spinner"></div><span>Analyzing message...</span></div>';

            try {
                const response = await fetch(`${API_BASE}/api/v1/classify`, {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({message: message})
                });

                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }

                const data = await response.json();

                let topicsHtml = '';
                if (data.topics && data.topics.length > 0) {
                    topicsHtml = '<div class="topics-list">';
                    for (const topic of data.topics) {
                        const confidence = Math.round(topic.confidence * 100);
                        topicsHtml += `
                            <div class="topic-item">
                                <div>
                                    <span class="topic-name">${topic.topic_name}</span>
                                    <span class="topic-confidence">${confidence}% confidence</span>
                                </div>
                                <span class="tier-badge mini-tier ${getTierClass(topic.tier)}">Tier ${topic.tier}</span>
                            </div>
                        `;
                    }
                    topicsHtml += '</div>';
                } else {
                    topicsHtml = '<p style="color: var(--muted); margin-top: 1rem;">No specific topics detected</p>';
                }

                resultArea.innerHTML = `
                    <div class="tier-badge ${getTierClass(data.tier)}">
                        ${getTierEmoji(data.tier)} Tier ${data.tier} - ${data.tier_name}
                    </div>
                    <p style="margin-top: 1rem; color: var(--muted);">Action: <strong style="color: var(--text)">${data.action}</strong></p>
                    <p style="color: var(--muted); font-size: 0.875rem;">Model: ${data.model_used} | Time: ${data.processing_time_ms?.toFixed(0) || 'N/A'}ms | Cached: ${data.cached ? 'Yes' : 'No'}</p>

                    <h3 style="margin-top: 1.5rem; font-size: 1rem;">Detected Topics</h3>
                    ${topicsHtml}

                    <details style="margin-top: 1rem;">
                        <summary style="cursor: pointer; color: var(--muted);">View Raw JSON</summary>
                        <pre class="json-output">${JSON.stringify(data, null, 2)}</pre>
                    </details>
                `;

            } catch (error) {
                resultArea.innerHTML = `
                    <div class="error-message">
                        <strong>Error:</strong> ${error.message}
                    </div>
                `;
            } finally {
                btn.disabled = false;
                btn.textContent = 'Classify Message';
            }
        }

        async function loadHealth() {
            const healthArea = document.getElementById('healthArea');
            try {
                const response = await fetch(`${API_BASE}/api/v1/health`);
                const data = await response.json();

                const isHealthy = data.status === 'healthy' && data.classifier_status === 'ready';

                healthArea.innerHTML = `
                    <div class="health-status">
                        <div class="health-dot ${isHealthy ? '' : 'error'}"></div>
                        <span>${isHealthy ? 'All systems operational' : 'Service degraded'}</span>
                    </div>
                    <div class="stats">
                        <div class="stat">
                            <div class="stat-value">${data.classifier_status === 'ready' ? '✓' : '⏳'}</div>
                            <div class="stat-label">Classifier</div>
                        </div>
                        <div class="stat">
                            <div class="stat-value">${data.redis_connected ? '✓' : '✗'}</div>
                            <div class="stat-label">Redis Cache</div>
                        </div>
                    </div>
                    <p style="margin-top: 1rem; color: var(--muted); font-size: 0.875rem;">
                        Model: ${data.gemini_model}<br>
                        Service: ${data.service} v${data.version}
                    </p>
                `;
            } catch (error) {
                healthArea.innerHTML = `
                    <div class="error-message">
                        <strong>Cannot connect to service:</strong> ${error.message}
                    </div>
                `;
            }
        }

        async function loadTaxonomy() {
            const taxonomyArea = document.getElementById('taxonomyArea');
            try {
                const response = await fetch(`${API_BASE}/api/v1/taxonomy`);
                const data = await response.json();

                taxonomyArea.innerHTML = `
                    <div class="stats">
                        <div class="stat">
                            <div class="stat-value">${data.total_topics}</div>
                            <div class="stat-label">Topics</div>
                        </div>
                        <div class="stat">
                            <div class="stat-value">${data.total_labels}</div>
                            <div class="stat-label">Labels</div>
                        </div>
                    </div>
                    <div style="margin-top: 1rem;">
                        <p style="color: var(--muted); font-size: 0.875rem; margin-bottom: 0.5rem;">Topics by Tier:</p>
                        <div style="display: flex; gap: 0.5rem; flex-wrap: wrap;">
                            <span class="tier-badge mini-tier tier-1">Tier 1: ${data.tiers?.['1'] || 0}</span>
                            <span class="tier-badge mini-tier tier-2">Tier 2: ${data.tiers?.['2'] || 0}</span>
                            <span class="tier-badge mini-tier tier-3">Tier 3: ${data.tiers?.['3'] || 0}</span>
                            <span class="tier-badge mini-tier tier-4">Tier 4: ${data.tiers?.['4'] || 0}</span>
                        </div>
                    </div>
                `;
            } catch (error) {
                taxonomyArea.innerHTML = `
                    <div class="error-message">
                        <strong>Error loading taxonomy:</strong> ${error.message}
                    </div>
                `;
            }
        }

        // Load health and taxonomy on page load
        document.addEventListener('DOMContentLoaded', () => {
            loadHealth();
            loadTaxonomy();
        });

        // Allow Enter key to submit
        document.getElementById('messageInput').addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && e.ctrlKey) {
                classify();
            }
        });
    </script>
</body>
</html>
"""


@dashboard_router.get("/dashboard", response_class=HTMLResponse)
async def get_dashboard():
    """
    Test Dashboard for Topic Classifier

    A simple HTML interface to test the topic classifier interactively.
    Displays health status, taxonomy overview, and allows testing classification.
    """
    return HTMLResponse(content=DASHBOARD_HTML)


@dashboard_router.get("/test", response_class=HTMLResponse)
async def get_test_page():
    """Alias for dashboard at /test"""
    return HTMLResponse(content=DASHBOARD_HTML)
