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

class GenerateScriptInput(BaseModel):
    prompt: str = Field(..., description="The user's prompt or story idea.")
    num_scenes: int = Field(default=5, description="Number of scenes to generate.")

class CommitMemoryInput(BaseModel):
    collection_name: str = Field(..., description="The ChromaDB collection to save to (e.g., 'script_history').")
    data: Dict[str, Any] = Field(..., description="The JSON data or metadata to save.")

# =====================================================================
# TOOL IMPLEMENTATIONS (Dummy functions for now, we will connect LLMs later)
# =====================================================================

def generate_script_segment(prompt: str, num_scenes: int = 5) -> Dict[str, Any]:
    """Placeholder logic for the Scriptwriter Agent."""
    return {"status": "success", "message": f"Generated {num_scenes} scenes for: {prompt}"}

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
    """Placeholder logic for extracting characters from a script."""
    # In reality, an LLM would parse the script. We are mocking the structured return.
    return {
        "status": "success",
        "characters": [
            {
                "name": "DETECTIVE",
                "personality": "Gruff, cynical, observant.",
                "appearance": "Trench coat, cybernetic glowing right eye, messy hair.",
                "reference_style": "Cyberpunk, cinematic lighting, 8k resolution"
            }
        ]
    }

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