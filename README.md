# 🎬 The Writer's Room: Project Montage (Phase 1)

**Autonomous Story and Image Generation Layer**

Phase 1 establishes the foundation of the PROJECT MONTAGE system. It transforms raw human intent into a structured, machine-interpretable narrative representation using a stateful, multi-agent creative system.

## 🚀 Key Features & Architecture

This system strictly adheres to a Supervisor-Worker hierarchical model orchestrated via LangGraph, featuring:

* **Dynamic MCP Tool Discovery (Strict Constraint):** Zero hardcoded API functions. All agents query a custom Model Context Protocol (MCP) registry at runtime to discover their tools (`generate_script_segment`, `extract_character_profiles`, etc.) via structured Pydantic schemas.
* **Dual-Mode Intake Logic:** * *Auto Mode:* Autonomously expands a raw prompt into a multi-scene screenplay using Groq (LLaMA 3.1).
  * *Manual Mode:* Validates uploaded scripts for structural integrity (scene headings, dialogue) and parses them into standardized JSON.
* **Stateful Memory (ChromaDB):** All extracted character identities and traits are embedded and saved to a local persistent vector database to ensure cross-scene continuity.
* **Human-in-the-Loop (HitL) Checkpoint:** Graph execution pauses after the writing phase, utilizing `MemorySaver` to freeze state and await manual human approval before proceeding to the Art Department.
* **Multimodal Art Synthesis:** Dynamically generates character reference images based on LLM-extracted visual profiles.
* **Interactive UI:** A complete Streamlit frontend visualizing the state graph, review checkpoints, and final JSON/Image deliverables.

## 🛠️ Tech Stack

* **Orchestration:** LangGraph, LangChain
* **LLM Engine:** Groq API (`llama-3.1-8b-instant`)
* **Vector Memory:** ChromaDB, Sentence Transformers
* **Image Synthesis:** Pollinations.ai Cloud API *(Note: Local ComfyUI integration is fully coded and backed up in `src/comfyui_backup.py` to accommodate hardware constraints during rapid testing).*
* **Frontend:** Streamlit

## 📂 Project Structure

```text
The-Writers-Room-AI/
├── .env                    # API Keys (Ignored)
├── requirements.txt        # Python dependencies
├── image_assets/           # Generated character art
├── chroma_data/            # Persistent vector memory
└── src/
    ├── app.py              # Streamlit GUI frontend
    ├── graph.py            # LangGraph StateGraph orchestrator
    ├── state.py            # TypedDict central shared state
    ├── memory.py           # ChromaDB initialization
    ├── mcp_registry.py     # Dynamic tool definitions & schemas
    ├── agents_writers.py   # Scriptwriter & Validator nodes
    ├── agents_artists.py   # Character Designer & Image Synthesizer nodes
    └── comfyui_backup.py   # Local SDXL integration backup