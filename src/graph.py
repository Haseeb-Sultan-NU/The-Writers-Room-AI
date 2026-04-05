from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from src.state import GraphState
from src.agents_writers import script_validator_node, scriptwriter_node
from src.agents_artists import character_designer_node, image_synthesizer_node

# --- 1. HUMAN IN THE LOOP NODE ---
def hitl_node(state: GraphState) -> GraphState:
    """
    This acts as a checkpoint. The graph will pause BEFORE executing this node,
    allowing the user to review the generated script in the state.
    """
    print("\n[SYSTEM] Human-in-the-Loop review complete. Proceeding to Art Department...")
    return state

# --- 2. ROUTING LOGIC ---
def route_input_mode(state: GraphState) -> str:
    """Determines whether to route to the Validator or the Scriptwriter."""
    mode = state.get("input_mode", "auto")
    if mode == "manual":
        return "validate"
    return "write"

# --- 3. BUILD THE GRAPH ---
builder = StateGraph(GraphState)

# Add all our agent nodes
builder.add_node("script_validator", script_validator_node)
builder.add_node("scriptwriter", scriptwriter_node)
builder.add_node("hitl_checkpoint", hitl_node)
builder.add_node("character_designer", character_designer_node)
builder.add_node("image_synthesizer", image_synthesizer_node)

# Define the flow (Edges)
builder.add_conditional_edges(
    START,
    route_input_mode,
    {
        "validate": "script_validator",
        "write": "scriptwriter"
    }
)

# Both writing paths lead to the human review checkpoint
builder.add_edge("script_validator", "hitl_checkpoint")
builder.add_edge("scriptwriter", "hitl_checkpoint")

# After review, it goes to the artists
builder.add_edge("hitl_checkpoint", "character_designer")
builder.add_edge("character_designer", "image_synthesizer")
builder.add_edge("image_synthesizer", END)

# Compile the graph with memory (required for the HitL pause feature)
memory = MemorySaver()
# We interrupt *before* the hitl_checkpoint node executes
app = builder.compile(checkpointer=memory, interrupt_before=["hitl_checkpoint"])

# =====================================================================
# TESTING THE FULL WORKFLOW
# =====================================================================
if __name__ == "__main__":
    print("--- STARTING THE WRITER'S ROOM WORKFLOW ---")
    
    initial_state = {
        "input_mode": "auto",
        "raw_prompt": "A cyberpunk detective investigates a neon city",
        "script": {}, "characters": [], "images": [], "status": "starting", "errors": []
    }
    
    # We need a thread ID to track the stateful memory
    config = {"configurable": {"thread_id": "production_run_001"}}

    # RUN 1: From START to the HitL Checkpoint
    print("\n>>> STAGE 1: WRITING PHASE")
    for event in app.stream(initial_state, config):
        pass # The nodes will print their own logs

    # Check if we are paused at the checkpoint
    snapshot = app.get_state(config)
    if snapshot.next and snapshot.next[0] == "hitl_checkpoint":
        print(f"\n[PAUSED] Workflow stopped for Human Review.")
        print(f"Current Script Location: {snapshot.values['script']['scenes'][0]['location']}")
        
        user_input = input("\nDo you approve this script? (y/n): ")
        
        if user_input.lower() == 'y':
            print("\n>>> STAGE 2: ART PHASE")
            # Resume the graph by passing None as the input state
            for event in app.stream(None, config):
                 pass
        else:
            print("\n[ABORTED] Script rejected by user.")