import os
import subprocess
import json
import random
import requests
import streamlit as st
from PIL import Image
from io import BytesIO
import zipfile
import time

# HuggingFace API token
API_KEY = "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"
headers = {"Authorization": f"Bearer {API_KEY}"}

# Define the ComfyUI directory and executable
COMFYUI_DIR = r"C:\Users\bewiz\OneDrive\Desktop\AI\ComfyUI_windows_portable"
COMFYUI_EXECUTABLE = os.path.join(COMFYUI_DIR, "run_nvidia_gpu.bat")

# Ensure set_page_config is called first
st.set_page_config(page_title="Photo to Style Transfer", page_icon="🎨", layout="wide")

# List of artists and modifiers (trimmed for brevity)
artists = ["Takashi Murakami", "Tyler Edlin"]
modifiers = ['4K', 'unreal engine']

def start_comfyui(port):
    """Start the ComfyUI instance and capture its output."""
    env = os.environ.copy()
    env['COMFYUI_PORT'] = str(port)
    process = subprocess.Popen(
        [COMFYUI_EXECUTABLE], cwd=COMFYUI_DIR,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1, env=env
    )
    return process, port

def read_workflow(file_path):
    """Read the workflow.json file."""
    with open(file_path, 'r') as file:
        workflow = json.load(file)
    return workflow

def update_workflow(workflow, style_image_path, structure_image_path, prompt):
    """Update the workflow with the new images and prompt."""
    st.write(f"Updating workflow with style image: {style_image_path} and structure image: {structure_image_path}")
    for node in workflow['nodes']:
        if node['type'] == 'LoadImage':
            if 'title' in node and node['title'] == 'Style':
                st.write(f"Updating style image node with path: {style_image_path}")
                node['widgets_values'][0] = style_image_path
            elif 'title' in node and node['title'] == 'Structure':
                st.write(f"Updating structure image node with path: {structure_image_path}")
                node['widgets_values'][0] = structure_image_path
        if node['type'] == 'CLIPTextEncode':
            st.write(f"Updating CLIPTextEncode node with prompt: {prompt}")
            node['widgets_values'][0] = prompt
    return workflow

def write_workflow(workflow, file_path):
    """Write the updated workflow back to workflow.json."""
    st.write(f"Writing updated workflow to: {file_path}")
    with open(file_path, 'w') as file:
        json.dump(workflow, file, indent=4)

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
            style_image_path = os.path.join(os.getcwd(), "images", uploaded_style_image.name)
            os.makedirs(os.path.dirname(style_image_path), exist_ok=True)
            with open(style_image_path, "wb") as f:
                f.write(uploaded_style_image.getvalue())

            # Save uploaded structure images
            structure_image_paths = []
            for image in uploaded_structure_images:
                image_path = os.path.join(os.getcwd(), "images", image.name)
                with open(image_path, "wb") as f:
                    f.write(image.getvalue())
                structure_image_paths.append(image_path)
            
            # Generate caption from the style image
            caption = image_to_text(style_image_path)
            prompt = generate_prompt(caption, selected_artists, selected_modifiers, custom_prompt, define_artist == "Let AI define artist", define_artist == "No artist")

            # Display the generated prompt
            st.write("**Generated Prompt:**", prompt)

            # Read the workflow
            workflow_path = "workflow.json"  # Path to your uploaded workflow file
            workflow = read_workflow(workflow_path)

            # Process each structure image
            output_images = []
            for structure_image_path in structure_image_paths:
                updated_workflow = update_workflow(workflow, style_image_path, structure_image_path, prompt)
                write_workflow(updated_workflow, workflow_path)

                # Start ComfyUI for each updated workflow with a new port each time
                port = random.randint(5000, 6000)
                process, port = start_comfyui(port)

                # Display IP and port
                st.write(f"ComfyUI running at: http://localhost:{port}")

                # Show progress and terminal output
                st.header("ComfyUI Output")
                output_placeholder = st.empty()
                output_text = ""

                while True:
                    output = process.stdout.readline()
                    if output == '' and process.poll() is not None:
                        break
                    if output:
                        output_text += output.strip() + '\n'
                        output_placeholder.text(output_text)
                        if "Saving image" in output:
                            output_images.append(structure_image_path)

                # Wait for the process to finish
                process.wait()

                # Save the transformed image
                output_image_path = os.path.join(os.getcwd(), "output", f"transformed_{os.path.basename(structure_image_path)}")
                if os.path.exists(output_image_path):
                    output_images.append(output_image_path)
                    st.image(output_image_path, caption=f"Transformed Image: {os.path.basename(structure_image_path)}", use_column_width=True)
                else:
                    st.error(f"Error processing image: {os.path.basename(structure_image_path)}")

                st.success(f"ComfyUI started and workflow updated successfully for structure image {os.path.basename(structure_image_path)}.")
            
            if len(output_images) > 1:
                zip_filename = zip_images(output_images)
                st.success("Batch processing completed. Download the transformed images:")
                st.download_button(label="Download ZIP", data=open(zip_filename, "rb").read(), file_name=zip_filename, mime="application/zip")
        else:
            st.error("Please upload a style image and structure images.")

if __name__ == "__main__":
    main_ui()
