import json
from src.state import GraphState
from src.mcp_registry import registry

def character_designer_node(state: GraphState) -> GraphState:
    """
    Extracts character identities from the generated script and saves them to memory.
    """
    print("\n[AGENT: CHARACTER DESIGNER] Analyzing script for characters...")
    
    # Safely get the script data
    script_data = state.get("script", {})
    if not script_data:
        print("[ERROR] No script found to analyze.")
        state["status"] = "failed"
        return state

    # Convert the script dict back to a string so our tool can "read" it
    script_text = json.dumps(script_data)
    
    # Dynamically call the extraction tool
    print("-> Calling MCP Tool 'extract_character_profiles'")
    result = registry.execute_tool("extract_character_profiles", script_text=script_text)
    extracted_chars = result.get("characters", [])
    
    # Save to our shared LangGraph state
    state["characters"] = extracted_chars
    
    # Dynamically call the memory tool to save identities to ChromaDB
    print("-> Calling MCP Tool 'commit_memory' to save character identities")
    for char in extracted_chars:
        registry.execute_tool("commit_memory", collection_name="character_metadata", data=char)

    print(f"[SUCCESS] Extracted and memorized {len(extracted_chars)} character(s).")
    return state


def image_synthesizer_node(state: GraphState) -> GraphState:
    """
    Takes character profiles and generates visual assets via local API.
    """
    print("\n[AGENT: IMAGE SYNTHESIZER] Preparing to generate character art...")
    
    characters = state.get("characters", [])
    if not characters:
        print("[ERROR] No characters found to draw.")
        state["status"] = "failed"
        return state

    image_paths = []
    
    for char in characters:
        # Build the visual prompt
        visual_prompt = f"{char['name']}, {char['appearance']}, {char['reference_style']}"
        print(f"-> Calling MCP Tool 'generate_character_image' for: {char['name']}")
        
        # Dynamically execute the image generation
        result = registry.execute_tool("generate_character_image", prompt=visual_prompt)
        
        if result["status"] == "success":
            image_paths.append(result["image_path"])
            print(f"      Saved asset to: {result['image_path']}")

    # Update state with the final image paths
    state["images"] = image_paths
    print("[SUCCESS] Image synthesis complete.")
    
    return state

# =====================================================================
# TESTING THE NODES
# =====================================================================
if __name__ == "__main__":
    # We simulate a state that has already passed through the Scriptwriter
    test_state: GraphState = {
        "input_mode": "auto",
        "raw_prompt": "",
        "script": {"scenes": [{"scene_id": 1, "location": "Neon City Street"}]}, # Mock script
        "characters": [], 
        "images": [], 
        "status": "processing", 
        "errors": []
    }
    
    print("--- PIPELINE TEST: ART DEPARTMENT ---")
    state_after_designer = character_designer_node(test_state)
    final_state = image_synthesizer_node(state_after_designer)