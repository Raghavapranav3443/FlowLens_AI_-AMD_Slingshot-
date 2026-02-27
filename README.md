<div align="center">

# ⚡ FlowLens AI
### AI-Powered Business Process Intelligence Platform
**AMD Slingshot Hackathon 2026**

[![License](https://img.shields.io/badge/License-FlowLens%20Custom-blue.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18+-61DAFB?logo=react&logoColor=black)](https://reactjs.org)
[![AMD ROCm](https://img.shields.io/badge/AMD-ROCm%206.0-ED1C24?logo=amd&logoColor=white)](https://rocm.docs.amd.com)
[![Ollama](https://img.shields.io/badge/Ollama-LLaMA%203.2-black?logo=ollama)](https://ollama.ai)

*Transforming raw WhatsApp-style workflow chat logs into actionable process intelligence — locally, privately, and AMD GPU-accelerated.*

</div>

---

## 🎯 What is FlowLens AI?

Most businesses run critical workflows — invoice approvals, payment processing, refunds — through informal chat channels. The data is there, but it's invisible. **FlowLens AI turns that chaos into clarity.**

Upload a plain-text workflow chat log. FlowLens instantly parses every event, computes cycle times, detects SLA breaches, identifies bottlenecks, calculates financial impact, and deploys dual AI engines — **Google Gemini 2.5 Flash** for cloud inference and **LLaMA 3.2:3B via Ollama** for fully local, privacy-first, AMD GPU-accelerated inference — to deliver boardroom-ready insights in seconds.

**No database. No complex setup. No data ever leaving your machine if you choose local mode.**

---

## 🏆 AMD Slingshot Relevance

FlowLens AI is purpose-built to leverage AMD hardware at every layer:

| AMD Technology | How FlowLens Uses It |
|---|---|
| **AMD GPU (RX 6000/7000, Instinct MI)** | Runs LLaMA 3.2 via Ollama + ROCm for fully local LLM inference |
| **PyTorch + ROCm 6.0** | GPU-accelerated Pearson correlation matrix computation across workflow stages |
| **Ryzen AI NPU (XDNA)** | Detected and reported in the live hardware dashboard |
| **ROCm Backend** | Real-time tokens/sec benchmark with CPU baseline comparison and speedup factor |

The application detects AMD hardware at startup, reports GPU name, compute units, VRAM, and ROCm version live in the UI, and shows a real-time inference performance panel (tokens/sec, GPU speedup vs CPU baseline, total inference time) with every analysis. On non-AMD hardware it gracefully falls back to CPU — **nothing breaks**.

---

## ✨ Core Features

### 📊 Process Mining Engine
- Parses WhatsApp-style, bracket-format, and multiple regional date format chat logs automatically
- Extracts invoice IDs, amounts (₹), actors, timestamps, and action types from natural language
- Computes per-case cycle times, stage-level average durations, and standard deviations
- Identifies the bottleneck stage using comparative stage duration analysis
- Tracks SLA compliance per stage (Approval ≤ 2h, Payment ≤ 4h, Refund ≤ 6h) with breach counts

### 🤖 Dual AI Inference Modes
| Mode | Engine | Privacy | Speed |
|---|---|---|---|
| ☁️ **Cloud** | Gemini 2.5 Flash | Data sent to Google | Fastest |
| ⚡ **Local AMD** | LLaMA 3.2:3B via Ollama | 100% on-device | GPU-dependent |
| 📡 **Local Stream** | LLaMA 3.2:3B (SSE) | 100% on-device | Real-time token stream |

Both modes generate identical structured insights: operational risks, bottleneck analysis, SLA improvement suggestions, and staffing recommendations.

### 💰 Financial Impact Analysis
- Calculates monthly labor cost (based on stage hours × hourly rate)
- Quantifies SLA breach penalties per breach event
- Computes cash-flow opportunity cost using Working Capital theory (WIP × daily capital rate)
- Displays total value processed, average/median/min/max invoice values, and cost per case

### 🔮 What-If Simulation Engine
Four simulation scenarios powered by formal mathematical models:

| Scenario | Model Used |
|---|---|
| **Add Approvers** | M/M/c Queuing Theory with diminishing returns exponent |
| **Auto-Approval** | Linear invoice distribution model capped at 70% auto-approval |
| **Smart Routing** | Little's Law bottleneck optimisation (25% improvement factor) |
| **Custom Target** | Direct percentage reduction with proportional cost projection |

Each simulation returns: new cycle time, new SLA breach count, gross savings, net savings, annual projection, and payback period in months.

### 📄 AI-Generated SOP Documents
- Scaffold-first architecture: all structural data (SLAs, actors, KPIs, stage metrics) is computed deterministically in Python — the AI only writes professional prose
- Generates complete Standard Operating Procedures with title, objective, scope, prerequisites, role responsibilities, per-stage action descriptions, decision points, escalation paths, and exception handling
- Available via both Gemini (structured JSON schema output) and local LLaMA
- Includes version history, document ID, review date, and KPI targets auto-populated from live metrics

### 💬 AI Copilot Chat
- Conversational process intelligence interface with full chat history
- Context-aware: every message is grounded in the live uploaded process data
- Available in both Gemini (cloud) and LLaMA (local) modes
- Answers questions like *"Which actor is causing the most SLA breaches?"* or *"What would happen if we added two approvers?"*

### 📋 Implementation Plans & PDF Export
- One-click 8-week phased implementation plans for each optimisation scenario
- Role-assigned tasks with owner accountability per phase
- Export full simulation analysis reports as branded PDF (via jsPDF) with financial tables, recommendations, and process context

### ⚡ AMD Hardware Dashboard
- Live detection of AMD GPU model, compute units, VRAM, and ROCm version via `rocminfo`
- Ryzen AI NPU (XDNA) detection
- Real-time inference benchmark: tokens/sec, GPU speedup vs 8 TPS CPU baseline, total inference time
- GPU-accelerated Pearson correlation matrix between workflow stages (computed via PyTorch tensors on AMD GPU)

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     FlowLens AI                             │
│                                                             │
│  ┌──────────────┐         ┌──────────────────────────────┐  │
│  │   React 18   │ ←HTTP→  │    FastAPI Backend           │  │
│  │   Frontend   │         │                              │  │
│  │              │         │  ┌─────────────────────────┐ │  │
│  │  • Dashboard │         │  │  Process Mining Engine  │ │  │
│  │  • What-If   │         │  │  (Pure Python)          │ │  │
│  │  • SOP Gen   │         │  └─────────────────────────┘ │  │
│  │  • Copilot   │         │                              │  │
│  │  • PDF Export│         │  ┌─────────┐  ┌──────────┐  │  │
│  └──────────────┘         │  │ Gemini  │  │  Ollama  │  │  │
│                           │  │  2.5F   │  │ LLaMA3.2 │  │  │
│                           │  └─────────┘  └──────────┘  │  │
│                           │                    ↑          │  │
│                           │             ┌──────────────┐  │  │
│                           │             │  AMD GPU     │  │  │
│                           │             │  ROCm 6.0    │  │  │
│                           │             └──────────────┘  │  │
│                           └──────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

**Backend:** Python 3.10+ · FastAPI · Uvicorn · httpx · PyTorch (ROCm) · python-dotenv  
**Frontend:** React 18 · Recharts · react-router-dom · jsPDF  
**AI:** Google Gemini 2.5 Flash (cloud) · LLaMA 3.2:3B via Ollama (local)  
**AMD Stack:** ROCm 6.0 · PyTorch ROCm build · rocminfo

---

## 🚀 Getting Started

### Prerequisites

Install these manually before anything else:

| Requirement | Download | Notes |
|---|---|---|
| **Python 3.10+** | [python.org](https://www.python.org/downloads/) | Required for backend |
| **Node.js 18+** | [nodejs.org](https://nodejs.org) | Required for frontend |
| **Ollama** | [ollama.ai/download](https://ollama.ai/download) | Required for local AI mode |

> **AMD GPU users:** For GPU-accelerated inference, additionally install [ROCm 6.0](https://rocm.docs.amd.com/) and the ROCm PyTorch build (see Step 4).

---

### Installation

**Step 1 — Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/flowlens-ai.git
cd flowlens-ai
```

**Step 2 — Set up your environment variables**
```bash
cp .env.example .env
```
Open `.env` and add your Gemini API key (free at [aistudio.google.com](https://aistudio.google.com/app/apikey)):
```
GEMINI_API_KEY=your_gemini_api_key_here
```
> Local AMD mode works without a Gemini key — it's only needed for cloud inference.

**Step 3 — Install Python dependencies**
```bash
pip install -r requirements.txt
```

**Step 4 — (Optional) AMD GPU acceleration**

For AMD GPU with ROCm 6.0:
```bash
pip install torch --index-url https://download.pytorch.org/whl/rocm6.0
```
For CPU-only PyTorch:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**Step 5 — Launch**
```bash
python start.py
```

That's it. `start.py` automatically handles:
- ✅ Running `npm install` if `node_modules` is missing or outdated
- ✅ Starting the Ollama server if not already running
- ✅ Pulling `llama3.2:3b` if not already downloaded (~2GB, first run only)
- ✅ Starting the FastAPI backend on `http://localhost:8000`
- ✅ Starting the React frontend on `http://localhost:3000`
- ✅ Opening your browser automatically

---

### Services After Launch

| Service | URL |
|---|---|
| **App** | http://localhost:3000 |
| **Backend API** | http://localhost:8000 |
| **API Docs (Swagger)** | http://localhost:8000/docs |

Press `Ctrl+C` in the terminal to stop all services.

---

## 🧪 Demo Logs

Three ready-to-use demo chat logs are included in `/demo_logs/` to immediately showcase the full range of FlowLens AI's capabilities:

| File | Scenario | Cases | Actors | Efficiency | Highlight |
|---|---|---|---|---|---|
| `demo_log_1_critical.txt` | 🔴 Critical Process | 50 | 6 | Very Low | 100% SLA breach rate, all invoices breached — maximum AI risk alerts and savings potential |
| `demo_log_2_moderate.txt` | 🟡 Moderate Process | 70 | 6 | Medium | Heavy PAYMENT bottleneck, high actor variance — showcases bottleneck detection |
| `demo_log_3_efficient.txt` | 🟢 Efficient Process | 70 | 6 | High (≈94) | Only 1 SLA breach, fast cycle times — great contrast benchmark |

**Upload any of these on the dashboard to instantly see the full analysis pipeline in action.**

---

## 📁 Project Structure

```
flowlens-ai/
│
├── backend/
│   └── main.py              # FastAPI app — process mining, AI inference, simulation
│
├── frontend/
│   ├── src/
│   │   ├── App.js           # Full React application (single-file architecture)
│   │   └── App.css          # Design system — dark theme, typography, components
│   └── package.json
│
├── demo_logs/
│   ├── demo_log_1_critical.txt
│   ├── demo_log_2_moderate.txt
│   └── demo_log_3_efficient.txt
│
├── start.py                 # One-click launcher — boots all services automatically
├── requirements.txt         # Python dependencies
├── .env.example             # Environment variable template
├── .gitignore
└── LICENSE
```

---

## 📋 Log Format

FlowLens AI parses chat logs in the following formats:

```
DD/MM/YYYY, HH:MM - ActorName: message text #invoiceNumber ₹amount
```

**Supported action keywords** (case-insensitive):

| Keyword in message | Maps to stage |
|---|---|
| `sent invoice` | INVOICE_SENT |
| `approved` | APPROVAL |
| `payment received` | PAYMENT |
| `refund initiated` | REFUND_INITIATED |
| `refund completed` | REFUND_COMPLETED |

**Example:**
```
03/01/2025, 08:05 - Priya: Sent invoice #1042 for ₹1,20,000 to Apex Corp
03/01/2025, 09:55 - Arjun: Approved invoice #1042 for ₹1,20,000
03/01/2025, 13:10 - Meena: Payment received for invoice #1042 for ₹1,20,000
```

Multiple date formats, 12h/24h time, and bracket-style WhatsApp exports are all supported.

---

## 🔌 API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Full system health check (Gemini, Ollama, AMD GPU) |
| `GET` | `/health/ollama` | Ollama + model availability check |
| `GET` | `/hardware/amd` | AMD GPU/NPU specs via rocminfo |
| `POST` | `/analyze` | Cloud analysis via Gemini 2.5 Flash |
| `POST` | `/analyze-local` | Local analysis via LLaMA 3.2 (blocking) |
| `POST` | `/analyze-stream` | Local analysis via LLaMA 3.2 (SSE streaming) |
| `POST` | `/simulate` | What-if scenario simulation |
| `POST` | `/chat` | Local copilot chat (LLaMA) |
| `POST` | `/chat-gemini` | Cloud copilot chat (Gemini) |
| `POST` | `/sop` | SOP generation via LLaMA |
| `POST` | `/sop-gemini` | SOP generation via Gemini (structured schema) |

Full interactive documentation available at `http://localhost:8000/docs` when the server is running.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Backend Framework** | FastAPI 0.111+ with async/await throughout |
| **AI — Cloud** | Google Gemini 2.5 Flash (structured JSON schema output) |
| **AI — Local** | LLaMA 3.2:3B via Ollama (blocking + SSE streaming) |
| **AMD Acceleration** | PyTorch ROCm 6.0, rocminfo hardware detection |
| **Async HTTP** | httpx (Gemini API + Ollama communication) |
| **Frontend** | React 18, react-router-dom, Recharts |
| **PDF Export** | jsPDF (client-side, no server needed) |
| **Process Models** | M/M/c Queuing Theory, Little's Law, Working Capital theory |
| **Launcher** | Python subprocess orchestration (cross-platform) |

---

## 👥 Team

**FlowLens AI** — built for AMD Slingshot Hackathon 2026

| Name | Role |
|---|---|
| **Juluru Raghava Pranav** | Co-owner & Developer |
| **Kaarthikeyan Ganesh** | Co-owner & Developer |
| **Ashish Sheelam** | Co-owner & Developer |

---

## 📄 License

Copyright © 2026 FlowLens AI — Juluru Raghava Pranav, Kaarthikeyan Ganesh, Ashish Sheelam.

This project is protected under a custom license. Direct copying or misrepresentation of this work is strictly prohibited. Derivative works must credit the original authors. See [LICENSE](./LICENSE) for full terms.

---

<div align="center">
  <sub>Built with ⚡ for AMD Slingshot 2026 · FlowLens AI · All rights reserved</sub>
</div>
