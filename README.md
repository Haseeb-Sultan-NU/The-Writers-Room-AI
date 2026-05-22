# 🎬 Project Montage — Autonomous Multimodal Story Generation System

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-000000?style=for-the-badge&logo=chainlink&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)

A two-phase autonomous creative system that transforms raw human intent into fully synchronized multimodal audiovisual content — from screenplay to lip-synced video — using a stateful, hierarchical multi-agent architecture.

---

## System Overview

| Phase | Name | Role |
|-------|------|------|
| Phase 1 | The Writer's Room | Autonomous story generation, character extraction, and image synthesis |
| Phase 2 | The Studio Floor | Parallel audio/video synthesis with lip-sync and face mapping |

---

## Phase 1 — The Writer's Room
### Autonomous Story & Image Generation Layer

Phase 1 transforms a raw human prompt into a structured, machine-interpretable narrative with character art, using a Supervisor-Worker hierarchical model orchestrated via LangGraph.

### Key Features

- **Dynamic MCP Tool Discovery:** Zero hardcoded API functions. All agents query a custom Model Context Protocol (MCP) registry at runtime to discover tools (`generate_script_segment`, `extract_character_profiles`, etc.) via structured Pydantic schemas
- **Dual-Mode Intake:**
  - *Auto Mode* — Autonomously expands a raw prompt into a multi-scene screenplay using Groq (LLaMA 3.1)
  - *Manual Mode* — Validates uploaded scripts for structural integrity (scene headings, dialogue) and parses them into standardized JSON
- **Stateful Memory (ChromaDB):** Character identities and traits are embedded and persisted to a local vector database ensuring cross-scene continuity
- **Human-in-the-Loop (HitL) Checkpoint:** Graph execution pauses after the writing phase via `MemorySaver`, freezing state and awaiting manual approval before proceeding to art generation
- **Multimodal Art Synthesis:** Dynamically generates character reference images from LLM-extracted visual profiles
- **Interactive Streamlit UI:** Visualizes the state graph, review checkpoints, and final JSON/image deliverables

### Phase 1 Architecture

Raw Prompt / Uploaded Script
↓
Intake & Validation Agent
↓
Scriptwriter Agent (Groq LLaMA 3.1)
↓
── HitL Checkpoint (Human Approval) ──
↓
Character Extraction Agent
↓
ChromaDB Memory Commit
↓
Image Synthesis Agent (Pollinations.ai)
↓
OUTPUT: scene_manifest.json + character art

### Phase 1 Tech Stack

| Layer | Tools |
|-------|-------|
| Orchestration | LangGraph, LangChain |
| LLM Engine | Groq API (`llama-3.1-8b-instant`) |
| Vector Memory | ChromaDB, Sentence Transformers |
| Image Synthesis | Pollinations.ai Cloud API |
| Frontend | Streamlit |

---

## Phase 2 — The Studio Floor
### Video & Audio Synthesis Layer

Phase 2 consumes `scene_manifest.json` from Phase 1 and produces fully synchronized audiovisual scenes through a parallel multi-agent execution architecture.

### Key Features

- **Parallel Audio/Video Pipelines:** Audio and video branches execute concurrently per scene, converging at the Lip Sync Agent
- **Task Graph-based Execution:** All scenes are decomposed into independent, parallelizable workloads via the `get_task_graph` MCP tool
- **Stateful Resumability:** Intermediate outputs are committed via `commit_memory`, supporting full pipeline recovery after failures or interruptions
- **Temporal Alignment:** Frame-by-frame lip sync ensures speech timing matches facial geometry across all output scenes

### Phase 2 Architecture

INPUT: scene_manifest.json
↓
Scene Parser Agent (get_task_graph)
↓
┌─────────────────┬──────────────────┐
│  AUDIO PIPELINE │  VIDEO PIPELINE  │
│                 │                  │
│ Voice Synth     │ Video Generation │
│ Agent           │ Agent            │
│ (voice cloning, │ (scene visuals   │
│  emotion-aware) │  from references)│
│                 │                  │
│                 │ Face Swap Agent  │
│                 │ (identity-       │
│                 │  validated)      │
└────────┬────────┴────────┬─────────┘
↓                 ↓
Lip Sync Agent (Fusion Layer)
↓
OUTPUT: raw_scenes/*.mp4

### Agent Definitions

| Agent | Role | MCP Tools |
|-------|------|-----------|
| Scene Parser | Segments `scene_manifest.json` into parallelizable tasks | `get_task_graph`, `commit_memory` |
| Voice Synth | Clones character voices and generates emotion-aware speech | `voice_cloning_synthesizer` |
| Video Generation | Generates scene visuals from character refs and descriptions | `query_stock_footage` |
| Face Swap | Maps generated characters onto video frames with identity validation | `face_swapper`, `identity_validator` |
| Lip Sync | Frame-by-frame alignment of audio waveform with facial movements | `lip_sync_aligner` |

### Phase 2 Tech Stack

| Layer | Tools |
|-------|-------|
| Orchestration | LangGraph (`Send()` API for parallel branching) |
| Voice Synthesis | TTS / Voice Cloning |
| Video Generation | Frame-based generative pipeline |
| Lip Sync | Frame alignment models |
| Memory | MCP `commit_memory` |

---

## Repository Structure

Project-Montage/
├── .env                      # API keys (gitignored)
├── requirements.txt
├── image_assets/             # Generated character art
├── chroma_data/              # Persistent vector memory
├── raw_scenes/               # Phase 2 output videos
└── src/
├── app.py                # Streamlit frontend
├── graph.py              # Phase 1 LangGraph orchestrator
├── graph_phase2.py       # Phase 2 parallel LangGraph orchestrator
├── state.py              # Shared TypedDict state
├── memory.py             # ChromaDB initialization
├── mcp_registry.py       # Dynamic MCP tool registry & Pydantic schemas
├── agents_writers.py     # Scriptwriter & Validator nodes
├── agents_artists.py     # Character Designer & Image Synthesizer nodes
└── comfyui_backup.py     # Local SDXL integration backup

---

## Setup

```bash
git clone https://github.com/your-username/project-montage.git
cd project-montage
pip install -r requirements.txt
```

Add your API keys to `.env`:

---

## Running

**Phase 1 — Writer's Room (Streamlit UI):**
```bash
streamlit run src/app.py
```

**Phase 2 — Studio Floor (parallel pipeline):**
```bash
python -m src.graph_phase2
```

---

## LangGraph Nodes

| Node | Phase |
|------|-------|
| `scene_parser_node` | 2 |
| `voice_synth_node` | 2 |
| `video_gen_node` | 2 |
| `face_swap_node` | 2 |
| `lip_sync_node` | 2 |
| `scriptwriter_node` | 1 |
| `character_extractor_node` | 1 |
| `image_synthesizer_node` | 1 |
