import streamlit as st
import os
import uuid
from src.graph import app

# --- 1. PAGE SETUP ---
st.set_page_config(page_title="The Writer's Room", layout="wide", page_icon="🎬")

st.title("🎬 The Writer's Room: Project Montage")
st.markdown("### Phase 1: Autonomous Story and Image Generation Layer")
st.markdown("---")

# --- 2. SESSION STATE MANAGEMENT ---
# We need a unique thread ID for LangGraph's memory, and a phase tracker for the UI
if "thread_id" not in st.session_state:
    # Use a random UUID so hitting "Start Over" creates a truly fresh memory state
    st.session_state.thread_id = str(uuid.uuid4())
if "phase" not in st.session_state:
    st.session_state.phase = "input" # States: 'input', 'reviewing', 'finished'

# The config dictionary required by LangGraph's checkpointer
config = {"configurable": {"thread_id": st.session_state.thread_id}}

# --- 3. SIDEBAR: MODE SELECTOR ---
st.sidebar.header("⚙️ Mode Selector")
input_mode = st.sidebar.radio("Select Input Mode:", ["auto", "manual"], format_func=lambda x: x.capitalize())
st.sidebar.markdown("*(Maps to LangGraph routing edges)*")

# --- 4. PHASE: INPUT ---
if st.session_state.phase == "input":
    st.subheader("1. Script Intake")
    
    if input_mode == "auto":
        prompt = st.text_area("Enter your story prompt:", placeholder="e.g., A cyberpunk detective investigates a neon city...", height=150)
        button_text = "✨ Generate Script"
    else:
        prompt = st.text_area("Paste your manual script here:", placeholder="INT. NEON CITY STREET - NIGHT\n\nDETECTIVE\nWhere is the data disk?", height=150)
        button_text = "🔍 Validate Script"
        
    if st.button(button_text):
        if not prompt.strip():
            st.error("Please enter a prompt or script to begin.")
        else:
            with st.spinner("Agents are writing..."):
                initial_state = {
                    "input_mode": input_mode,
                    "raw_prompt": prompt,
                    "script": {}, "characters": [], "images": [], "status": "starting", "errors": []
                }
                
                # Stream the graph up to the interrupt (hitl_checkpoint)
                for event in app.stream(initial_state, config):
                    pass 
                
                # Advance UI state to trigger the Human-in-the-Loop review
                st.session_state.phase = "reviewing"
                st.rerun()

# --- 5. PHASE: HUMAN-IN-THE-LOOP REVIEW ---
if st.session_state.phase in ["reviewing", "finished"]:
    # Retrieve the paused state from LangGraph's memory
    snapshot = app.get_state(config)
    current_state = snapshot.values
    
    st.subheader("2. Script Draft (Review)")
    
    # Handle Validation Errors (Manual Mode)
    if current_state.get("errors"):
        st.error(f"❌ Validation Failed: {current_state['errors']}")
        if st.button("Reset"):
            st.session_state.phase = "input"
            st.rerun()
    else:
        # Display the JSON output
        st.json(current_state.get("script", {}))
        
        # Checkpoint Controls
        if st.session_state.phase == "reviewing":
            st.warning("⚠️ **Human-in-the-Loop Checkpoint:** Do you approve this script for production?")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Approve & Send to Art Dept"):
                    with st.spinner("Synthesizing character art..."):
                        # Passing None resumes the graph from where it paused
                        for event in app.stream(None, config):
                            pass
                        st.session_state.phase = "finished"
                        st.rerun()
            with col2:
                if st.button("❌ Reject & Start Over"):
                    st.session_state.phase = "input"
                    st.session_state.thread_id = str(uuid.uuid4()) # Clear memory
                    st.rerun()

# --- 6. PHASE: ART DEPARTMENT (FINAL) ---
if st.session_state.phase == "finished":
    st.markdown("---")
    st.subheader("3. Final Deliverables (Art Department)")
    
    snapshot = app.get_state(config)
    final_state = snapshot.values
    
    chars = final_state.get("characters", [])
    images = final_state.get("images", [])
    
    if not images:
        st.info("No images were generated.")
    else:
        # Create dynamic columns based on number of generated images
        cols = st.columns(len(images))
        for idx, img_path in enumerate(images):
            with cols[idx]:
                if os.path.exists(img_path):
                    char_name = chars[idx]['name'] if idx < len(chars) else "Character"
                    st.image(img_path, caption=char_name, use_container_width=True)
                    if idx < len(chars):
                        st.caption(f"**Traits:** {chars[idx]['personality']}")
                else:
                    st.error(f"Image file missing at {img_path}")
                    
    if st.button("🔄 Start New Project"):
        st.session_state.phase = "input"
        st.session_state.thread_id = str(uuid.uuid4()) # Wipe thread memory for a fresh start
        st.rerun()