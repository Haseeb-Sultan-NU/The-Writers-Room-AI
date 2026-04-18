import json
from typing import Callable, Dict, Any, List
from pydantic import BaseModel, Field

class ToolRegistry:
    """
    Simulates the Model Context Protocol (MCP) tool discovery mechanism.
    Agents query this registry at runtime to find and execute tools dynamically.
    """
    def __init__(self):
        self._tools: Dict[str, Dict[str, Any]] = {}

    def register_tool(self, name: str, description: str, schema: type[BaseModel], func: Callable):
        """Registers a tool along with its Pydantic schema for dynamic discovery."""
        # We extract the JSON schema from the Pydantic model
        self._tools[name] = {
            "description": description,
            "schema": schema.model_json_schema(),
            "func": func
        }

    def discover_tools(self) -> List[Dict[str, Any]]:
        """
        Agents call this to find out what they can do. 
        Returns a list of available tools and their required JSON schemas.
        """
        return [{"name": name, "description": data["description"], "schema": data["schema"]} 
                for name, data in self._tools.items()]

    def execute_tool(self, tool_name: str, **kwargs) -> Any:
        """Executes a tool dynamically by name, preventing hardcoded API calls."""
        if tool_name not in self._tools:
            raise ValueError(f"Tool '{tool_name}' not found in MCP registry.")
        return self._tools[tool_name]["func"](**kwargs)

# =====================================================================
# TOOL SCHEMAS (Defining the structured JSON inputs agents must provide)
# =====================================================================

class ParseScriptInput(BaseModel):
    raw_text: str = Field(..., description="The raw manual script text to parse.")

class GenerateScriptInput(BaseModel):
    prompt: str = Field(..., description="The user's prompt or story idea.")
    num_scenes: int = Field(default=5, description="Number of scenes to generate.")

class CommitMemoryInput(BaseModel):
    collection_name: str = Field(..., description="The ChromaDB collection to save to (e.g., 'script_history').")
    data: Dict[str, Any] = Field(..., description="The JSON data or metadata to save.")

class VoiceSynthesisInput(BaseModel):
    text: str = Field(..., description="The dialogue text to synthesize into speech.")
    character_name: str = Field(..., description="The name of the character speaking.")
    scene_id: int = Field(..., description="The ID of the scene.")

# =====================================================================
# TOOL IMPLEMENTATIONS (Dummy functions for now, we will connect LLMs later)
# =====================================================================

def generate_script_segment(prompt: str, num_scenes: int = 1) -> Dict[str, Any]:
    """Uses Groq to autonomously generate a structured script segment."""
    import os
    import json
    from langchain_groq import ChatGroq
    from dotenv import load_dotenv
    load_dotenv()

    print(f"      [SYSTEM] Calling Groq LLM for script generation...")
    
    # Initialize the fast Groq model
    llm = ChatGroq(
        temperature=0.7, 
        model_name="llama-3.1-8b-instant", 
        groq_api_key=os.getenv("GROQ_API_KEY")
    )
    
    # We ask the LLM to output pure JSON matching our required schema
    system_prompt = f"""You are an expert screenwriter. Generate a {num_scenes}-scene script based on this prompt: "{prompt}". 
    You MUST output valid JSON ONLY. No markdown, no intro text.
    Format required:
    {{
      "scenes": [
        {{
          "scene_id": 1,
          "location": "Brief Location Description",
          "characters": ["CHAR1", "CHAR2"],
          "dialogue": [
            {{"speaker": "CHAR1", "line": "Dialogue here", "visual_cue": "Visual instruction"}}
          ]
        }}
      ]
    }}"""
    
    try:
        response = llm.invoke(system_prompt)
        # Parse the JSON string returned by the LLM into a Python dictionary
        script_data = json.loads(response.content)
        return {"status": "success", "message": "Script generated.", "script": script_data}
    except Exception as e:
        print(f"      [ERROR] Groq script generation failed: {e}")
        # Fallback to avoid crashing the pipeline
        fallback = {"scenes": [{"scene_id": 1, "location": "Fallback Location", "characters": ["ERROR"], "dialogue": [{"speaker": "ERROR", "line": "Failed to generate.", "visual_cue": "none"}]}]}
        return {"status": "failed", "message": str(e), "script": fallback}

def commit_memory(collection_name: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """Placeholder logic for interacting with ChromaDB."""
    return {"status": "success", "message": f"Saved agent data to {collection_name}"}

# --- NEW SCHEMAS FOR PHASE 4 ---

class ExtractCharacterInput(BaseModel):
    script_text: str = Field(..., description="The script text to analyze for characters.")

class GenerateImageInput(BaseModel):
    prompt: str = Field(..., description="The visual description of the character.")
    negative_prompt: str = Field(default="ugly, deformed, low resolution", description="What to avoid in the image.")

# --- NEW FUNCTIONS FOR PHASE 4 ---

def extract_character_profiles(script_text: str) -> Dict[str, Any]:
    """Uses Groq to analyze the script and extract character identities."""
    import os
    import json
    from langchain_groq import ChatGroq
    from dotenv import load_dotenv
    load_dotenv()

    print(f"      [SYSTEM] Calling Groq LLM for character extraction...")
    
    llm = ChatGroq(
        temperature=0.3, # Lower temperature for analytical tasks
        model_name="llama-3.1-8b-instant", 
        groq_api_key=os.getenv("GROQ_API_KEY")
    )
    
    system_prompt = f"""Read the following script and extract the main characters.
    You MUST output valid JSON ONLY. No markdown, no intro text.
    Format required:
    {{
      "characters": [
        {{
          "name": "CHARACTER NAME",
          "personality": "3 comma separated traits",
          "appearance": "Physical description and clothing",
          "reference_style": "Cinematic, fantasy illustration, highly detailed, 8k"
        }}
      ]
    }}
    
    Script:
    {script_text}"""
    
    try:
        response = llm.invoke(system_prompt)
        character_data = json.loads(response.content)
        return {"status": "success", "characters": character_data.get("characters", [])}
    except Exception as e:
        print(f"      [ERROR] Groq character extraction failed: {e}")
        return {"status": "failed", "characters": []}

def generate_character_image(prompt: str, negative_prompt: str = "") -> Dict[str, Any]:
    """
    Generates a character reference image using a free, keyless cloud API (Pollinations.ai).
    Zero local storage or GPU required.
    """
    import os
    import urllib.parse
    import urllib.request
    import uuid
    
    print("      [SYSTEM] Requesting image from lightweight cloud API...")
    
    # 1. Ensure our output folder exists
    os.makedirs("./image_assets", exist_ok=True)
    
    # 2. Format the URL with the visual prompt
    encoded_prompt = urllib.parse.quote(prompt)
    url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?nologo=true&model=flux"
    
    # 3. Create a unique filename
    filename = f"./image_assets/char_{uuid.uuid4().hex[:6]}.jpg"
    
    try:
        # 4. Create a request with a standard browser User-Agent to prevent 403 Forbidden errors
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        )
        
        # 5. Download and save the image
        with urllib.request.urlopen(req) as response:
            with open(filename, 'wb') as out_file:
                out_file.write(response.read())
                
        print(f"      [SUCCESS] Image successfully downloaded and saved to: {filename}")
        return {"status": "success", "image_path": filename}
        
    except Exception as e:
        print(f"      [ERROR] Image generation failed: {e}")
        return {"status": "failed", "image_path": None}
    
def parse_manual_script(raw_text: str) -> Dict[str, Any]:
    """Uses Groq to convert a raw text script into the standardized JSON format."""
    import os
    import json
    from langchain_groq import ChatGroq
    from dotenv import load_dotenv
    load_dotenv()

    print(f"      [SYSTEM] Calling Groq LLM to parse manual script...")
    
    llm = ChatGroq(
        temperature=0.1, # Very low temp because we just want extraction, not creativity
        model_name="llama-3.1-8b-instant", 
        groq_api_key=os.getenv("GROQ_API_KEY")
    )
    
    system_prompt = f"""You are a data extractor. Convert this raw screenplay into structured JSON.
    You MUST output valid JSON ONLY. No markdown, no intro text.
    Format required:
    {{
      "scenes": [
        {{
          "scene_id": 1,
          "location": "Location extracted from INT/EXT",
          "characters": ["CHAR1", "CHAR2"],
          "dialogue": [
            {{"speaker": "CHAR1", "line": "Dialogue here", "visual_cue": "Action lines or context"}}
          ]
        }}
      ]
    }}
    
    Raw Script:
    {raw_text}"""
    
    try:
        response = llm.invoke(system_prompt)
        script_data = json.loads(response.content)
        return {"status": "success", "script": script_data}
    except Exception as e:
        print(f"      [ERROR] Groq parsing failed: {e}")
        return {"status": "failed", "script": {}}
    
def voice_cloning_synthesizer(text: str, character_name: str, scene_id: int) -> dict:
    """Uses Edge-TTS to generate high-quality voice audio for characters."""
    import asyncio
    import edge_tts
    import os

    print(f"      [MCP] Synthesizing voice for {character_name}...")
    
    # Ensure our output directory exists as required by Phase 2
    os.makedirs("raw_scenes", exist_ok=True)
    
    # Assign a distinct voice based on the character name
    # Christopher is a deep male voice, Aria is a clear female voice
    voice = "en-US-ChristopherNeural" if character_name.upper() in ["DETECTIVE", "KING ARIN", "WIZARD"] else "en-US-AriaNeural"
    
    # Format the file path
    safe_name = character_name.replace(" ", "_").lower()
    file_path = f"raw_scenes/audio_scene{scene_id}_{safe_name}.wav"
    
    # Edge-TTS runs asynchronously, so we wrap it
    async def _generate():
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(file_path)
        
    try:
        asyncio.run(_generate())
        print(f"      [SUCCESS] Audio saved to {file_path}")
        return {"status": "success", "audio_path": file_path}
    except Exception as e:
        print(f"      [ERROR] Voice synthesis failed: {e}")
        return {"status": "failed", "audio_path": None}
# =====================================================================
# INITIALIZE & REGISTER
# =====================================================================

# Create the global registry instance
registry = ToolRegistry()

# Dynamically register the tools
registry.register_tool(
    name="generate_script_segment", 
    description="Generates a multi-scene screenplay segment based on user intent.", 
    schema=GenerateScriptInput, 
    func=generate_script_segment
)

registry.register_tool(
    name="commit_memory", 
    description="Saves state or metadata to the persistent Vector DB.", 
    schema=CommitMemoryInput, 
    func=commit_memory
)
registry.register_tool(
    name="extract_character_profiles",
    description="Analyzes a script to extract character identities and visual descriptions.",
    schema=ExtractCharacterInput,
    func=extract_character_profiles
)

registry.register_tool(
    name="generate_character_image",
    description="Generates a character reference image using local stable diffusion.",
    schema=GenerateImageInput,
    func=generate_character_image
)
registry.register_tool(
    name="parse_manual_script",
    description="Parses raw screenplay format into standardized JSON.",
    schema=ParseScriptInput,
    func=parse_manual_script
)
registry.register_tool(
    name="voice_cloning_synthesizer",
    description="Generates spoken audio waveforms from dialogue text.",
    schema=VoiceSynthesisInput,
    func=voice_cloning_synthesizer
)

if __name__ == "__main__":
    # Test 1: Can we discover tools dynamically?
    print("--- MCP TOOL DISCOVERY ---")
    available_tools = registry.discover_tools()
    print(json.dumps(available_tools, indent=2))
    
    print("\n--- MCP TOOL EXECUTION ---")
    # Test 2: Can we execute a tool without hardcoding the function call?
    # This simulates an agent sending a JSON payload.
    agent_json_payload = {
        "tool": "generate_script_segment",
        "input": {
            "prompt": "A cyberpunk detective investigates a neon city",
            "num_scenes": 3
        }
    }
    
    result = registry.execute_tool(agent_json_payload["tool"], **agent_json_payload["input"])
    print(f"Agent Execution Result: {result}")