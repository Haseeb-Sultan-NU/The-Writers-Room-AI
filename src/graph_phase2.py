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
    character_profiles = state.get('character_profiles', [])
    
    print(f"  [VIDEO PIPELINE] Processing Scene {scene_id}...")
    
    # 1. Fetch Stock Footage
    vid_result = registry.execute_tool("query_stock_footage", query=location, scene_id=scene_id)
    if vid_result["status"] != "success":
        return {"errors": [f"Scene {scene_id} video generation failed."]}
        
    base_video_path = vid_result["video_path"]
    final_video_path = base_video_path
    
    # 2. Extract Character Identity Path
    image_path = None
    if character_profiles:
        image_path = character_profiles[0].get("image_path")
        
    if image_path:
        # 3. Validate Identity (Critical Constraint)
        val_result = registry.execute_tool("identity_validator", image_path=image_path, video_path=base_video_path)
        
        if val_result.get("is_valid"):
            # 4. Swap Face
            swap_result = registry.execute_tool(
                "face_swapper", 
                image_path=image_path, 
                video_path=base_video_path, 
                scene_id=scene_id
            )
            if swap_result["status"] == "success":
                final_video_path = swap_result["video_path"]
                
    return {
        "base_videos": [{"scene_id": scene_id, "video_path": base_video_path}],
        "swapped_videos": [{"scene_id": scene_id, "video_path": final_video_path}]
    }

def lip_sync_node(state: GraphState):
    from src.mcp_registry import registry
    
    print("\n[AGENT: LIP SYNC (FUSION)] Merging parallel audio and video streams...")
    
    # Grab the completed assets from the shared state
    audio_tracks = state.get("audio_tracks", [])
    swapped_videos = state.get("swapped_videos", [])
    
    final_scenes = []
    
    # Match the audio and video by scene_id
    for audio in audio_tracks:
        scene_id = audio["scene_id"]
        # Find the matching video for this scene
        matching_video = next((v for v in swapped_videos if v["scene_id"] == scene_id), None)
        
        if matching_video:
            # Execute the fusion tool
            result = registry.execute_tool(
                "lip_sync_aligner",
                video_path=matching_video["video_path"],
                audio_path=audio["audio_path"],
                scene_id=scene_id
            )
            
            if result["status"] == "success":
                final_scenes.append({
                    "scene_id": scene_id,
                    "final_path": result["final_video_path"]
                })
                
    return {"final_scenes": final_scenes, "status": "completed"}

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
                # Changed query so Pexels gives us a video of a person to swap faces with
                "location": "close up portrait man looking at camera dark night", 
                "dialogue": [{"speaker": "DETECTIVE", "line": "It's quiet tonight. Too quiet."}]
            }
        ],
        "characters": [
            {
                "name": "DETECTIVE", 
                # Tell Phase 2 exactly where the Phase 1 image is sitting
                "image_path": "image_assets/char_86582b.jpg" 
            }
        ]
    }
    with open("scene_manifest.json", "w") as f:
        json.dump(dummy_manifest, f)
        
    initial_state = {"manifest_path": "scene_manifest.json"}
    
    for event in app_phase2.stream(initial_state):
        pass
    print("\n[SUCCESS] Phase 2 Parallel Execution Complete.")