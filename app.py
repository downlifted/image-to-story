import os
import random
import requests
import streamlit as st
import pandas as pd
from PIL import Image
from io import BytesIO
from zipfile import ZipFile

# Directly setting the API key
API_KEY = os.getenv('API_KEY')
if not API_KEY:
    st.error("API_KEY environment variable not set. Please set it in the Streamlit Cloud settings.")

# Set page configuration
st.set_page_config(page_title="Photo to Prompt", page_icon="🎨", layout="wide")

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

# Example affiliate URLs
affiliate_links = {
    "MidJourney": "https://www.midjourney.com/",
    "DreamStudio": "https://www.dreamstudio.ai/",
    "DALL·E": "https://www.openai.com/dall-e",
    "DeepAI": "https://deepai.org/",
    "Craiyon": "https://www.craiyon.com/"
}

affiliate_images = {
    "MidJourney": "https://getsby.com/wp-content/uploads/2023/09/Midjourney-logo.png",
    "DreamStudio": "https://assets-global.website-files.com/6508cd9252452cdcc016ad1d/6553707eb4424680593f8f14_logo.webp",
    "DALL·E": "https://static.thenounproject.com/png/2486994-200.png",
    "DeepAI": "https://deepai.org/static/images/flops-highlighted.svg",
    "Craiyon": "https://assets.eweek.com/uploads/2024/01/craiyon-icon.png"
}

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

def generatePrompt(inputText, artists, modifiers, custom_text, define_artist, no_artist, retries=3):
    if inputText.startswith("Error"):
        return "There was an error processing the image. Please try again."

    falcon_7b = "https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct"
    API_URL = falcon_7b

    for attempt in range(retries):
        prompt = f"Create a unique and high-quality prompt for AI art based on the context of the photo: {inputText}."
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
            continue

    return "Error: Failed to generate a valid prompt after multiple attempts."

def generate_image(prompt):
    stable_diffusion_api = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2"
    payload = {"inputs": prompt}
    response = requests.post(stable_diffusion_api, headers=headers, json=payload)
    
    if response.status_code == 200:
        return response.content
    else:
        st.error("Error generating image. Status code: {}".format(response.status_code))
        return None

def save_file(data, filename, file_format):
    if file_format == "csv":
        data.to_csv(filename, index=False)
    elif file_format == "txt":
        with open(filename, "w") as file:
            for index, row in data.iterrows():
                file.write(f"{row['Prompt']}\n")
    elif file_format == "doc":
        from docx import Document
        doc = Document()
        for index, row in data.iterrows():
            doc.add_paragraph(row['Prompt'])
        doc.save(filename)

def create_zip_file(folder_path, artists, modifiers, custom_text, define_artist, no_artist):
    zip_filename = "prompts.zip"
    with ZipFile(zip_filename, 'w') as zipf:
        for image_name in os.listdir(folder_path):
            if image_name.endswith(('jpg', 'png')):
                image_path = os.path.join(folder_path, image_name)
                caption = image_to_text(image_path)
                prompt = generatePrompt(caption, artists, modifiers, custom_text, define_artist, no_artist)
                doc_filename = f"{os.path.splitext(image_name)[0]}.txt"
                with open(doc_filename, "w") as file:
                    file.write(prompt)
                zipf.write(doc_filename)
                os.remove(doc_filename)
    return zip_filename

def single_image_ui():
    st.header("Single Image to Prompt")
    uploaded_file = st.file_uploader("Choose a photo...", type=["jpg", "png"], help="Upload a photo to generate AI art prompt")

    custom_text = st.text_input("Add Custom Text (Optional)")
    define_artist = st.radio("Artist Selection", ["Let AI define artist", "Use my own artist", "No artist"])
    selected_artists = st.multiselect("Select Artists (Optional)", artists) if define_artist == "Use my own artist" else []
    selected_modifiers = st.multiselect("Select Modifiers (Optional)", modifiers)
    generate_image_checkbox = st.checkbox("Generate Image from Prompt")

    if uploaded_file is not None:
        image_path = f"images/{uploaded_file.name}"
        os.makedirs("images", exist_ok=True)

        with open(image_path, "wb") as file:
            file.write(uploaded_file.getvalue())
        st.image(uploaded_file, caption="Photo successfully uploaded", use_column_width=True)

        if st.button("Generate Prompt and Image", key="single_generate"):
            with st.spinner('Generating Prompt and Image...'):
                caption = image_to_text(image_path)
                prompt = generatePrompt(caption, selected_artists, selected_modifiers, custom_text, define_artist == "Let AI define artist", define_artist == "No artist")
                
                st.write("**Caption:**", caption)
                st.write("**Prompt:**", prompt)
                
                # Generate and display image if checkbox is selected
                if generate_image_checkbox:
                    image_data = generate_image(prompt)
                    if image_data:
                        image = Image.open(BytesIO(image_data))
                        st.image(image, caption="Generated AI Art", use_column_width=True)
                    else:
                        st.error("Error generating image. Please try again.")

                # Provide link to generate image on Bing
                bing_url = f"https://www.bing.com/images/create?q={prompt.replace(' ', '+')}"
                st.markdown(f"[Generate this image on Bing]({bing_url})", unsafe_allow_html=True)
                
                # Provide affiliate links
                st.write("If you would like to generate better art based on your prompt, click here:")
                st.markdown("<div class='affiliate-logos'>"
                            f"<a href='{affiliate_links['MidJourney']}' target='_blank'><img src='{affiliate_images['MidJourney']}'></a>"
                            f"<a href='{affiliate_links['DreamStudio']}' target='_blank'><img src='{affiliate_images['DreamStudio']}'></a>"
                            f"<a href='{affiliate_links['DALL·E']}' target='_blank'><img src='{affiliate_images['DALL·E']}'></a>"
                            f"<a href='{affiliate_links['DeepAI']}' target='_blank'><img src='{affiliate_images['DeepAI']}'></a>"
                            f"<a href='{affiliate_links['Craiyon']}' target='_blank'><img src='{affiliate_images['Craiyon']}'></a>"
                            "</div>", unsafe_allow_html=True)
                
                # Deleting the images
                os.remove(image_path)

def generate_csv(folder_path, artists, modifiers, custom_text, include_image_name, output_format, define_artist, no_artist):
    data = []
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('jpg', 'png'))]
    total_files = len(image_files)

    progress_bar = st.progress(0)

    for idx, image_name in enumerate(image_files):
        image_path = os.path.join(folder_path, image_name)
        caption = image_to_text(image_path)
        prompt = generatePrompt(caption, artists, modifiers, custom_text, define_artist, no_artist)
        if include_image_name:
            data.append([image_name, prompt])
        else:
            data.append([prompt])
        
        # Update progress
        progress_bar.progress((idx + 1) / total_files)

    columns = ["Image Name", "Prompt"] if include_image_name else ["Prompt"]
    df = pd.DataFrame(data, columns=columns)
    filename = f"prompts.{output_format}"
    save_file(df, filename, output_format)
    
    return filename

def batch_image_ui():
    st.header("Batch Process Images to Generate Prompts")
    uploaded_folder = st.file_uploader("Upload a folder of images...", type=None, accept_multiple_files=True, help="Upload multiple images to generate prompts in batch")

    custom_text = st.text_input("Add Custom Text (Optional)")
    define_artist = st.radio("Artist Selection", ["Let AI define artist", "Use my own artist", "No artist"], key="batch_define_artist")
    selected_artists = st.multiselect("Select Artists (Optional)", artists, key="batch_artist") if define_artist == "Use my own artist" else []
    selected_modifiers = st.multiselect("Select Modifiers (Optional)", modifiers, key="batch_modifier")
    include_image_name = st.checkbox("Include Image Name in Output")
    output_format = st.selectbox("Select Output Format", ["csv", "txt", "doc"])
    zip_output_checkbox = st.checkbox("Output as ZIP file with prompts in separate documents")

    if uploaded_folder:
        folder_path = "batch_images"
        os.makedirs(folder_path, exist_ok=True)

        for uploaded_file in uploaded_folder:
            image_path = os.path.join(folder_path, uploaded_file.name)
            with open(image_path, "wb") as file:
                file.write(uploaded_file.getvalue())

        if st.button("Generate File", key="batch_generate"):
            with st.spinner('Generating File...'):
                if zip_output_checkbox:
                    zip_filename = create_zip_file(folder_path, selected_artists, selected_modifiers, custom_text, define_artist == "Let AI define artist", define_artist == "No artist")
                    st.success(f"ZIP file generated successfully: {zip_filename}")
                    st.download_button(label="Download ZIP File", data=open(zip_filename, "rb").read(), file_name=zip_filename, mime="application/zip")
                else:
                    output_file = generate_csv(folder_path, selected_artists, selected_modifiers, custom_text, include_image_name, output_format, define_artist == "Let AI define artist", define_artist == "No artist")
                    st.success(f"File generated successfully: {output_file}")
                    st.download_button(label="Download File", data=open(output_file, "rb").read(), file_name=output_file, mime="text/csv" if output_format == "csv" else "text/plain")

        # Clear cache after completion
        for file in os.listdir(folder_path):
            os.remove(os.path.join(folder_path, file))
        os.rmdir(folder_path)

def main_ui():
    hide_default_format = """
    <style>
    #MainMenu {visibility: hidden; }
    footer {visibility: hidden;}
    .stApp {max-width: 1200px; margin: auto; background: url('https://images.unsplash.com/photo-1580072000227-b0e1d4c9b94e') no-repeat center center fixed; background-size: cover;}
    .sidebar .sidebar-content {text-align: center; width: 350px;}
    .affiliate-logos {display: flex; justify-content: center; gap: 10px;}
    .affiliate-logos img {width: 60px; height: auto;}
    .creator-info {display: flex; justify-content: center; gap: 10px; align-items: center;}
    .creator-info img {width: 30px; height: auto;}
    </style>
    """
    st.markdown(hide_default_format, unsafe_allow_html=True)

    st.title("Photo to AI Art Prompt")
    st.subheader("Turn your photos into stunning AI art prompts")

    mode = st.sidebar.radio("Choose Mode", ["Single Image", "Batch Processing"])
    
    with st.sidebar.expander("Creator Info"):
        st.write("Created by BeWiZ")
        st.markdown(
            "<div class='creator-info'>"
            "<a href='https://x.com/AiAnarchist' target='_blank'><img src='https://static.vecteezy.com/system/resources/thumbnails/027/395/710/small/twitter-brand-new-logo-3-d-with-new-x-shaped-graphic-of-the-world-s-most-popular-social-media-free-png.png' alt='X'></a>"
            "<a href='mailto:downlifted@gmail.com'><img src='https://upload.wikimedia.org/wikipedia/commons/7/7e/Gmail_icon_(2020).svg' alt='Email'></a>"
            "</div>", unsafe_allow_html=True
        )
        st.write("Modified Version of Photo to Story by Priyansh Bhardwaj")
        st.write("Special thanks to the original artist for inspiration.")
        st.image('https://raw.githubusercontent.com/downlifted/image-to-story/master/logotrans.png', use_column_width=True)

    st.sidebar.markdown("### Generate Art Online")
    st.sidebar.markdown("<div class='affiliate-logos'>"
                        f"<a href='{affiliate_links['MidJourney']}' target='_blank'><img src='{affiliate_images['MidJourney']}'></a>"
                        f"<a href='{affiliate_links['DreamStudio']}' target='_blank'><img src='{affiliate_images['DreamStudio']}'></a>"
                        f"<a href='{affiliate_links['DALL·E']}' target='_blank'><img src='{affiliate_images['DALL·E']}'></a>"
                        f"<a href='{affiliate_links['DeepAI']}' target='_blank'><img src='{affiliate_images['DeepAI']}'></a>"
                        f"<a href='{affiliate_links['Craiyon']}' target='_blank'><img src='{affiliate_images['Craiyon']}'></a>"
                        "</div>", unsafe_allow_html=True)

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
        - **Upload Your Image(s)**: Upload a single image or multiple images to generate AI art prompts.
        - **Custom Text**: Add optional custom text to include in your prompts.
        - **Artist Selection**: Choose whether to let the AI define the artist, use your own artist selection, or no artist at all.
        - **Modifiers**: Select optional modifiers to refine the style and details of your prompts.
        - **Output**: Download the generated prompts as a CSV, TXT, or DOC file, or get them in a ZIP file if you choose to output prompts in separate documents.
        - **Generate Image**: Check the box to generate AI art from the prompt using Stable Diffusion 2.
        ''')

if __name__ == "__main__":
    main_ui()
