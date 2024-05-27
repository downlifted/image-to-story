import os
import subprocess
import json
import socket
import sys

COMFYUI_DIR = r"C:\Users\bewiz\OneDrive\Desktop\AI\ComfyUI_windows_portable"
COMFYUI_EXECUTABLE = os.path.join(COMFYUI_DIR, "run_nvidia_gpu.bat")
WORKFLOW_PATH = os.path.join(COMFYUI_DIR, "ComfyUI", "workflow.json")

def ensure_dependencies():
    """Ensure that all necessary dependencies are downloaded."""
    try:
        os.makedirs(os.path.join(COMFYUI_DIR, "models", "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(COMFYUI_DIR, "models", "clip_vision"), exist_ok=True)
        os.makedirs(os.path.join(COMFYUI_DIR, "models", "ipadapter"), exist_ok=True)
        os.makedirs(os.path.join(COMFYUI_DIR, "models", "controlnet"), exist_ok=True)

        dependencies = [
            ("https://huggingface.co/gingerlollipopdx/ModelsXL/resolve/main/dreamshaperXL_lightningDPMSDE.safetensors", "models/checkpoints/dreamshaperXL_lightningDPMSDE.safetensors"),
            ("https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors", "models/clip_vision/CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors"),
            ("https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors", "models/ipadapter/ip-adapter-plus_sdxl_vit-h.safetensors"),
            ("https://huggingface.co/SargeZT/controlnet-sd-xl-1.0-depth-16bit-zoe/resolve/main/depth-zoe-xl-v1.0-controlnet.safetensors", "models/controlnet/depth-zoe-xl-v1.0-controlnet.safetensors")
        ]

        for url, path in dependencies:
            full_path = os.path.join(COMFYUI_DIR, path)
            if not os.path.exists(full_path):
                print(f"Downloading {url}...")
                subprocess.run(["curl", "-L", url, "-o", full_path])
                print(f"Downloaded {url} to {full_path}")
            else:
                print(f"Dependency {path} already exists.")

    except Exception as e:
        print(f"Error ensuring dependencies: {str(e)}")

def start_comfyui():
    """Start the ComfyUI instance."""
    try:
        print("Starting ComfyUI...")
        process = subprocess.Popen(
            [COMFYUI_EXECUTABLE], cwd=COMFYUI_DIR,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1
        )
        return process
    except Exception as e:
        print(f"Error starting ComfyUI: {str(e)}")
        return str(e)

def update_workflow(style_image_path, structure_image_path, prompt):
    """Update the workflow with the new images and prompt."""
    try:
        with open(WORKFLOW_PATH, 'r') as file:
            workflow = json.load(file)

        for node in workflow['nodes']:
            if node['type'] == 'LoadImage':
                if 'style' in node['widgets_values'][0].lower():
                    node['widgets_values'][0] = style_image_path
                elif 'structure' in node['widgets_values'][0].lower():
                    node['widgets_values'][0] = structure_image_path
            if node['type'] == 'CLIPTextEncode':
                node['widgets_values'][0] = prompt

        with open(WORKFLOW_PATH, 'w') as file:
            json.dump(workflow, file, indent=4)
        print("Workflow updated successfully.")
        return None
    except Exception as e:
        return str(e)

def process_images(style_image_path, structure_image_paths, prompt):
    """Process images using the workflow."""
    error = update_workflow(style_image_path, structure_image_paths[0], prompt)
    if error:
        return error

    process = start_comfyui()
    if isinstance(process, str):
        return process  # Return the error message

    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())

    process.wait()
    output_image_path = os.path.join(COMFYUI_DIR, "output", f"transformed_{os.path.basename(structure_image_paths[0])}")
    if os.path.exists(output_image_path):
        return [output_image_path]
    else:
        return f"Error processing image: {os.path.basename(structure_image_paths[0])}"

def main():
    print("Starting ComfyUI server...")
    ensure_dependencies()
    print("Dependencies ensured.")

    # Set up a socket server to receive data from the Streamlit application
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('localhost', 65432))
    server_socket.listen(1)
    print("ComfyUI server listening on port 65432...")
    
    while True:
        print("Waiting for a connection...")
        conn, addr = server_socket.accept()
        print(f"Connected by {addr}")
        
        try:
            data = conn.recv(4096)
            print(f"Received data: {data}")
            if not data:
                break

            received_data = json.loads(data.decode('utf-8'))
            style_image_path = received_data['style_image_path']
            structure_image_paths = received_data['structure_image_paths']
            prompt = received_data['prompt']
            
            print(f"Processing images with style: {style_image_path}, structure: {structure_image_paths}, prompt: {prompt}")
            result = process_images(style_image_path, structure_image_paths, prompt)
            print(f"Result: {result}")

            response = json.dumps(result)
            print(f"Sending response: {response}")
            conn.sendall(response.encode('utf-8'))
        except Exception as e:
            print(f"Error: {str(e)}")
            conn.sendall(json.dumps(str(e)).encode('utf-8'))
        finally:
            conn.close()
            print("Connection closed.")

if __name__ == "__main__":
    main()
