import os
import json
import random
import requests
import streamlit as st
from PIL import Image
from io import BytesIO
import zipfile
import socket
import subprocess
import time

# HuggingFace API token
API_KEY = "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"
if not API_KEY:
    st.error("API_KEY environment variable not set. Please set it in the Streamlit Cloud settings.")
headers = {"Authorization": f"Bearer {API_KEY}"}

# Ensure set_page_config is called first
st.set_page_config(page_title="Photo to Style Transfer", page_icon="🎨", layout="wide")

# List of artists and modifiers (trimmed for brevity)
artists = ["Takashi Murakami", "Tyler Edlin"]
modifiers = ['4K', 'unreal engine']

def start_comfyui_server():
    """Start the ComfyUI server process."""
    server_script = "run_comfyui.py"
    return subprocess.Popen(["python", server_script], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def image_to_text(image_path):
    salesforce_blip = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-base"
    API_URL = salesforce_blip

    with open(image_path, "rb") as f:
        data = f.read()

    response = requests.post(API_URL, headers=headers, data=data)
    response = response.json()

    try:
        return response[0]["generated_text"]
    except Exception as e:
        return f"Error: {str(e)}"

def generate_prompt(input_text, artists, modifiers, custom_text, define_artist, no_artist):
    if input_text.startswith("Error"):
        return "There was an error processing the image. Please try again."

    falcon_7b = "https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct"
    API_URL = falcon_7b

    prompt = f"Create a unique and high-quality prompt for AI art based on the context of the photo: {input_text}."
    if custom_text:
        prompt += f" Theme: {custom_text}."
    if not no_artist:
        if define_artist:
            artist_count = random.choice([0, 1, 2])
            if artist_count > 0:
                artist_list = ', '.join(random.sample(artists, min(artist_count, len(artists))))
                prompt += f" Style by {artist_list}."
        else:
            artist = ', '.join(random.sample(artists, 1))
            prompt += f" Style by {artist}."
    if modifiers:
        modifier = ', '.join(random.sample(modifiers, min(10, len(modifiers))))
        prompt += f" Modifier: {modifier}."

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 250,
            "do_sample": True,
            "top_k": 10,
            "temperature": 1,
            "return_full_text": False,
        },
        "options": {
            "wait_for_model": True
        }
    }

    response = requests.post(API_URL, headers=headers, json=payload)
    response = response.json()

    try:
        generated_text = response[0]["generated_text"]
        if "As the AI language model" not in generated_text and "I am unable to render visual data" not in generated_text:
            return generated_text
    except Exception as e:
        return f"Error: {str(e)}"

    return "Error: Failed to generate a valid prompt after multiple attempts."

def zip_images(image_paths):
    zip_filename = "transformed_images.zip"
    with zipfile.ZipFile(zip_filename, 'w') as zipf:
        for image_path in image_paths:
            zipf.write(image_path, os.path.basename(image_path))
    return zip_filename

def main_ui():
    st.title("ComfyUI Workflow Automation")
    
    # Start the ComfyUI server
    comfyui_server_process = start_comfyui_server()
    time.sleep(5)  # Wait for the server to start
    
    # User inputs for style and structure images
    st.header("Upload Images")
    uploaded_style_image = st.file_uploader("Choose a style image...", type=["jpg", "png"], key="style")
    uploaded_structure_images = st.file_uploader("Choose structure images...", type=["jpg", "png"], accept_multiple_files=True, key="structure")
    
    # User input for custom prompt
    custom_prompt = st.text_input("Add Custom Text (Optional)")
    define_artist = st.radio("Artist Selection", ["Let AI define artist", "Use my own artist", "No artist"])
    selected_artists = st.multiselect("Select Artists (Optional)", artists) if define_artist == "Use my own artist" else []
    selected_modifiers = st.multiselect("Select Modifiers (Optional)", modifiers)

    if st.button("Start ComfyUI and Update Workflow"):
        if uploaded_style_image and uploaded_structure_images:
            # Save uploaded style image
            style_image_path = os.path.join("images", uploaded_style_image.name)
            os.makedirs("images", exist_ok=True)
            with open(style_image_path, "wb") as f:
                f.write(uploaded_style_image.getvalue())

            # Save uploaded structure images
            structure_image_paths = []
            for image in uploaded_structure_images:
                image_path = os.path.join("images", image.name)
                with open(image_path, "wb") as f:
                    f.write(image.getvalue())
                structure_image_paths.append(image_path)
            
            # Generate caption from the style image
            caption = image_to_text(style_image_path)
            st.write(f"**Generated Caption:** {caption}")
            
            # Generate prompt
            prompt = generate_prompt(caption, selected_artists, selected_modifiers, custom_prompt, define_artist == "Let AI define artist", define_artist == "No artist")
            st.write(f"**Generated Prompt:** {prompt}")

            # Prepare data to send to run_comfyui.py
            data = {
                'style_image_path': style_image_path,
                'structure_image_paths': structure_image_paths,
                'prompt': prompt
            }

            # Send data to run_comfyui.py
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.connect(('localhost', 65432))
                    s.sendall(json.dumps(data).encode('utf-8'))
                    response = s.recv(4096)
                    print(f"Received response: {response}")
            except ConnectionRefusedError:
                st.error("Failed to connect to ComfyUI server. Make sure it is running.")
                return

            result = json.loads(response.decode('utf-8'))

            if isinstance(result, str):
                st.error(result)
                return

            output_images = result

            # Display output images
            for output_image_path in output_images:
                st.image(output_image_path, caption=f"Transformed Image: {os.path.basename(output_image_path)}", use_column_width=True)

            if len(output_images) > 1:
                zip_filename = zip_images(output_images)
                st.success("Batch processing completed. Download the transformed images:")
                st.download_button(label="Download ZIP", data=open(zip_filename, "rb").read(), file_name=zip_filename, mime="application/zip")
        else:
            st.error("Please upload a style image and structure images.")

    # Ensure the ComfyUI server is terminated when done
    if comfyui_server_process:
        comfyui_server_process.terminate()

if __name__ == "__main__":
    main_ui()
