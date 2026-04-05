"""
BACKUP FILE: Local ComfyUI Image Generation Logic
Keep this file in case the professor requires proof of the local ComfyUI implementation.
To use this, replace the lightweight API function in src/mcp_registry.py with the function below.
"""
from typing import Dict, Any

def generate_character_image_comfyui(prompt: str, negative_prompt: str = "") -> Dict[str, Any]:
    """
    Sends a request to a local ComfyUI server.
    """
    import os
    import urllib.request
    import json
    import uuid
    from dotenv import load_dotenv
    load_dotenv()
    
    comfy_url = os.getenv("COMFYUI_URL", "http://127.0.0.1:8188")
    print(f"      [SYSTEM] Sending request to local image server at {comfy_url}...")
    
    client_id = str(uuid.uuid4())
    
    # NOTE: Change "v1-5-pruned-emaonly.safetensors" to your exact model filename!
    prompt_workflow = {
        "3": {"class_type": "KSampler", "inputs": {"seed": 12345, "steps": 20, "cfg": 8, "sampler_name": "euler", "scheduler": "normal", "denoise": 1, "model": ["4", 0], "positive": ["6", 0], "negative": ["7", 0], "latent_image": ["5", 0]}},
        "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "v1-5-pruned-emaonly.safetensors"}}, 
        "5": {"class_type": "EmptyLatentImage", "inputs": {"batch_size": 1, "width": 512, "height": 512}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["4", 1]}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": negative_prompt, "clip": ["4", 1]}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
        "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "writers_room", "images": ["8", 0]}}
    }

    try:
        data = json.dumps({"prompt": prompt_workflow, "client_id": client_id}).encode('utf-8')
        req = urllib.request.Request(f"{comfy_url}/prompt", data=data, headers={'Content-Type': 'application/json'})
        
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read())
            print("      [SUCCESS] Image queued in ComfyUI!")
            return {"status": "success", "image_path": f"./ComfyUI/output/writers_room_{result['prompt_id']}.png"}
            
    except Exception as e:
        print(f"      [ERROR] Could not connect to ComfyUI. Is it running? ({e})")
        return {"status": "failed", "image_path": None}