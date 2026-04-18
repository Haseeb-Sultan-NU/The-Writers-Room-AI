import json
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from src.state import GraphState, SceneTaskState

# --- MOCK AGENTS (We will hook up the real tools in Phases B, C, D) ---

def scene_parser_node(state: GraphState):
    print("\n[AGENT: SCENE PARSER] Reading scene_manifest.json...")
    # For now, we mock reading the JSON that Phase 1 would have produced.
    try:
        with open("scene_manifest.json", "r") as f:
            manifest = json.load(f)
        print(f"-> Parsed {len(manifest.get('scenes', []))} scenes.")
        return {"script": manifest, "characters": manifest.get("characters", []), "status": "parsed"}
    except Exception as e:
        print("[ERROR] Could not find scene_manifest.json.")
        return {"errors": ["Missing manifest"], "status": "failed"}

def voice_synth_node(state: SceneTaskState):
    from src.mcp_registry import registry
    
    scene_id = state['scene_id']
    print(f"  [AUDIO PIPELINE] Processing Scene {scene_id}...")
    
    audio_results = []
    dialogues = state['scene_data'].get('dialogue', [])
    
    # If there is dialogue, generate audio for each line
    for i, line_data in enumerate(dialogues):
        speaker = line_data.get("speaker", "Unknown")
        text = line_data.get("line", "")
        
        if text:
            # Dynamically call the MCP tool
            result = registry.execute_tool(
                "voice_cloning_synthesizer", 
                text=text, 
                character_name=speaker, 
                scene_id=scene_id
            )
            
            if result["status"] == "success":
                audio_results.append({
                    "scene_id": scene_id,
                    "character": speaker,
                    "line_index": i,
                    "audio_path": result["audio_path"]
                })
                
    return {"audio_tracks": audio_results}

def video_gen_node(state: SceneTaskState):
    from src.mcp_registry import registry
    
    scene_id = state['scene_id']
    location = state['scene_data'].get('location', 'cinematic background')
    print(f"  [VIDEO PIPELINE] Processing Scene {scene_id}...")
    
    # Query Pexels based on the scene's location description
    result = registry.execute_tool(
        "query_stock_footage", 
        query=location, 
        scene_id=scene_id
    )
    
    if result["status"] == "success":
        return {"base_videos": [{"scene_id": scene_id, "video_path": result["video_path"]}]}
    else:
        return {"errors": [f"Scene {scene_id} video generation failed."]}

def lip_sync_node(state: GraphState):
    print("\n[AGENT: LIP SYNC (FUSION)] Merging parallel audio and video streams...")
    # This node automatically waits for BOTH parallel branches to finish
    print(f"-> Received {len(state.get('audio_tracks', []))} audio tracks.")
    print(f"-> Received {len(state.get('base_videos', []))} video tracks.")
    return {"status": "fusion_complete"}

# --- PARALLEL ROUTING LOGIC (The Core Requirement) ---

def dispatch_parallel_tasks(state: GraphState):
    """
    Reads the parsed scenes and uses the Send() API 
    to trigger the Audio and Video branches concurrently for each scene.
    """
    if state.get("status") == "failed":
        return END
        
    dispatches = []
    scenes = state.get("script", {}).get("scenes", [])
    
    for scene in scenes:
        task_data = {
            "scene_id": scene["scene_id"], 
            "scene_data": scene, 
            "character_profiles": state.get("characters", [])
        }
        
        # Fork the process: Send task to Audio branch
        dispatches.append(Send("voice_synth_node", task_data))
        # Fork the process: Send task to Video branch
        dispatches.append(Send("video_gen_node", task_data))
        
    return dispatches

# --- GRAPH CONSTRUCTION ---
workflow = StateGraph(GraphState)

# Add Nodes
workflow.add_node("scene_parser", scene_parser_node)
workflow.add_node("voice_synth_node", voice_synth_node)
workflow.add_node("video_gen_node", video_gen_node)
workflow.add_node("lip_sync_node", lip_sync_node)

# Add Edges
workflow.add_edge(START, "scene_parser")

# Use conditional edges with the Send() API to fork the process
workflow.add_conditional_edges("scene_parser", dispatch_parallel_tasks, ["voice_synth_node", "video_gen_node"])

# Both branches converge at the Lip Sync Agent
workflow.add_edge("voice_synth_node", "lip_sync_node")
workflow.add_edge("video_gen_node", "lip_sync_node")
workflow.add_edge("lip_sync_node", END)

app_phase2 = workflow.compile()

# --- EXECUTION TEST ---
if __name__ == "__main__":
    print("--- STARTING THE STUDIO FLOOR (PHASE 2) ---")
    
    # Generate a dummy manifest to test the routing
    dummy_manifest = {
        "scenes": [
            {
                "scene_id": 1, 
                "location": "Dark Alley", 
                "dialogue": [{"speaker": "DETECTIVE", "line": "It's quiet tonight. Too quiet."}]
            }
        ],
        "characters": [{"name": "DETECTIVE"}]
    }
    with open("scene_manifest.json", "w") as f:
        json.dump(dummy_manifest, f)
        
    initial_state = {"manifest_path": "scene_manifest.json"}
    
    for event in app_phase2.stream(initial_state):
        pass
    print("\n[SUCCESS] Phase 2 Parallel Execution Complete.")