import os
import subprocess
from pathlib import Path

def convert_mermaid_to_image():
    # File paths
    md_file = "/home/paramesh/.gemini/antigravity/brain/571f732f-98d6-41c3-8897-bcab3ff85958/database_schema.md"
    temp_mmd_file = "/tmp/database_schema.mmd"
    output_image = "/home/paramesh/Documents/ai-deploy-azure/database_schema.jpg"
    
    # Extract Mermaid content
    with open(md_file, "r") as f:
        content = f.read()
    
    start_tag = "```mermaid\n"
    end_tag = "```"
    start_idx = content.find(start_tag) + len(start_tag)
    end_idx = content.find(end_tag, start_idx)
    
    mermaid_content = content[start_idx:end_idx].strip()
    
    # Save the raw Mermaid content to a temp file
    with open(temp_mmd_file, "w") as f:
        f.write(mermaid_content)
        
    print(f"Extracted Mermaid content to {temp_mmd_file}")
    print("Now installing and running mmdc (Mermaid CLI) to convert to image...")
    
    # Using npx to run mermaid-cli
    cmd = [
        "npx", 
        "-y",
        "@mermaid-js/mermaid-cli", 
        "-i", temp_mmd_file, 
        "-o", output_image,
        "-b", "white"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"Successfully generated image: {output_image}")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error generating image: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        
        # Cleanup
        if os.path.exists(temp_mmd_file):
            os.remove(temp_mmd_file)
        raise

if __name__ == "__main__":
    convert_mermaid_to_image()
