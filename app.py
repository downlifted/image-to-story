import os
import random
import requests
import streamlit as st
import pandas as pd

# Directly setting the API key
API_KEY = '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'
headers = {"Authorization": f"Bearer {API_KEY}"}

# List of artists and modifiers
artists = [
    "Takashi Murakami", "Tyler Edlin", "Tom Thomson", "Thomas Kinkade", "Terry Oakes", 
    "Ted Nasmith", "Cassius Marcellus Coolidge", "Clyle Caldwell", "Dave Dorman", 
    "Earl Norem", "James Gurney", "James Paick", "John Blanch", "John Martin", 
    "Justin Gerard", "Jakub Różalski", "Jean Delville", "Jeff Easley", "Agnes Lawrence Pelton", 
    "Andrew Robinson", "Alan Lee", "Anton Fadeev", "Bob Byerley", "Brothers Hildebrandt", 
    "Bob Eggleton", "Chris Labrooy", "Chriss Foss", "Chris Moore", "Dan Mumford", 
    "Christopher Balaskas", "Eiichiro Oda", "Beeple", "Jeremey Smith", "Jeremiah Ketner", 
    "Michael Whelan", "Michelangelo", "Mike Winkelmann", "Noah Bradley"
]

modifiers = [
    '4K', 'unreal engine', 'octane render', '8k octane render', 'photorealistic', 
    'mandelbulb fractal', 'Highly detailed carvings', 'Atmosphere', 'Dramatic lighting', 
    'Sakura blossoms', 'magical atmosphere', 'muted colors', 'Highly detailed', 
    'Epic composition', 'incomparable reality', 'ultra detailed', 'unreal 5', 
    'concept art', 'smooth', 'sharp focus', 'illustration', 'evocative', 'mysterious', 
    'epic scene', 'intricate details', 'Pop Surrealism', 'sharp photography', 
    'hyper realistic', 'maximum detail', 'ray tracing', 'volumetric lighting', 
    'cinematic', 'realistic lighting', 'high resolution render', 'hyper realism', 
    'insanely detailed', 'intricate', 'volumetric light', 'light rays', 'shock art', 
    'dystopian art', 'cgsociety', 'fantasy art', 'matte drawing', 'speed painting', 
    'darksynth', 'redshift', 'color field', 'rendered in cinema4d', 'imax', '#vfxfriday', 
    'oil on canvas', 'figurative art', 'detailed painting', 'soft mist', 'daz3d', 
    'zbrush', 'anime', 'behance hd', 'panfuturism', 'futuristic', 'pixiv', 
    'auto-destructive art', 'apocalypse art', 'afrofuturism', 
    'reimagined by industrial light and magic', 'metaphysical painting', 'wiccan', 
    'grotesque', 'whimsical', 'psychedelic art', 'digital art', 'fractalism', 
    'anime aesthetic', 'chiaroscuro', 'mystical', 'majestic', 'digital painting', 
    'psychedelic', 'synthwave', 'cosmic horror', 'lovecraftian', 'vanitas', 'macabre', 
    'toonami', 'hologram', 'magic realism', 'impressionism', 'neo-fauvism', 'fauvism', 
    'synchromism'
]

def image_to_text(image_source):
    salesforce_blip = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-base"
    API_URL = salesforce_blip

    with open(image_source, "rb") as f:
        data = f.read()

    response = requests.post(API_URL, headers=headers, data=data)
    response = response.json()

    try:
        return response[0]["generated_text"]
    except Exception as e:
        return f"Error: {str(e)}"

def generatePrompt(inputText, add_artist, add_modifier):
    if inputText.startswith("Error"):
        return "There was an error processing the image. Please try again."

    falcon_7b = "https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct"
    API_URL = falcon_7b

    prompt = f"Create a unique and high-quality prompt for AI art based on the context of the photo: {inputText}."
    if add_artist:
        artist = random.choice(artists)
        prompt += f" Style by {artist}."
    if add_modifier:
        modifier = random.choice(modifiers)
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
        return response[0]["generated_text"]
    except Exception as e:
        return f"Error: {str(e)}"

def generate_csv(folder_path, add_artist, add_modifier):
    data = []
    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_name)
        if image_name.endswith(('jpg', 'png')):
            caption = image_to_text(image_path)
            prompt = generatePrompt(caption, add_artist, add_modifier)
            data.append([image_name, prompt])
    
    df = pd.DataFrame(data, columns=["Image Name", "Prompt"])
    csv_path = os.path.join(folder_path, "prompts.csv")
    df.to_csv(csv_path, index=False)
    return csv_path

def single_image_ui():
    st.header("Single Image to Prompt")
    uploaded_file = st.file_uploader("Choose a photo...", type=["jpg", "png"])

    add_artist = st.checkbox("Add artist style", key="single_artist")
    add_modifier = st.checkbox("Add additional modifier", key="single_modifier")

    if uploaded_file is not None:
        image_path = f"images/{uploaded_file.name}"
        os.makedirs("images", exist_ok=True)

        with open(image_path, "wb") as file:
            file.write(uploaded_file.getvalue())
        st.image(uploaded_file, caption="Photo successfully uploaded", use_column_width=True)

        if st.button("Generate Prompt", key="single_generate"):
            with st.spinner('Generating Prompt...'):
                caption = image_to_text(image_path)
                prompt = generatePrompt(caption, add_artist, add_modifier)

            # Deleting the images
            os.remove(image_path)

            st.write("**Caption:**", caption)
            st.write("**Prompt:**", prompt)

def batch_image_ui():
    st.header("Batch Process Images to Generate Prompts")
    uploaded_folder = st.file_uploader("Upload a folder of images...", type=None, accept_multiple_files=True)

    add_artist = st.checkbox("Add artist style", key="batch_artist")
    add_modifier = st.checkbox("Add additional modifier", key="batch_modifier")

    if uploaded_folder:
        folder_path = "batch_images"
        os.makedirs(folder_path, exist_ok=True)

        for uploaded_file in uploaded_folder:
            image_path = os.path.join(folder_path, uploaded_file.name)
            with open(image_path, "wb") as file:
                file.write(uploaded_file.getvalue())

        if st.button("Generate CSV", key="batch_generate"):
            with st.spinner('Generating CSV...'):
                csv_path = generate_csv(folder_path, add_artist, add_modifier)
            st.success(f"CSV generated successfully: {csv_path}")
            st.download_button(label="Download CSV", data=open(csv_path, "rb").read(), file_name="prompts.csv", mime="text/csv")

def main_ui():
    st.set_page_config(page_title="Photo to Prompt", page_icon="🎨", layout="wide")

    hide_default_format = """
    <style>
    #MainMenu {visibility: hidden; }
    footer {visibility: hidden;}
    .stApp {max-width: 1200px; margin: auto; background: url('https://images.unsplash.com/photo-1580072000227-b0e1d4c9b94e') no-repeat center center fixed; background-size: cover;}
    </style>
    """
    st.markdown(hide_default_format, unsafe_allow_html=True)

    st.title("Photo to AI Art Prompt")
    st.subheader("Turn your photos into stunning AI art prompts")

    mode = st.sidebar.radio("Choose Mode", ["Single Image", "Batch Processing"])
    
    with st.sidebar.expander("Creator Info"):
        st.write("Created by BeWiZ")
        st.write("[Twitter](https://x.com/AiAnarchist)")
        st.write("Modified Version of Photo to Story by Priyansh Bhardwaj")
        st.write("Special thanks to the original artist for inspiration.")
        st.image('/workspaces/image-to-story/logotrans.png', use_column_width=True)

    if mode == "Single Image":
        single_image_ui()
    else:
        batch_image_ui()

    st.markdown("---")
    st.markdown("### Additional Information")
    
    with st.expander("**Tech stack**"):
        st.write('''
        - **LLM : Falcon-7B-Instruct**
        - **HuggingFace**
        - **Langchain**
        ''')

    with st.expander("**App Working**"):
        st.write('''
        - **Upload Your Image(s)**: Upload an image or multiple images you want to generate prompts for.
        - **Generate Prompt**: Click the generate button to create AI art prompts based on the uploaded images.
        - **Download CSV**: If you uploaded multiple images, download the generated prompts as a CSV file.
        ''')

if __name__ == "__main__":
    main_ui()
