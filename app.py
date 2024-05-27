import os
import random
import requests
import streamlit as st
from PIL import Image
from io import BytesIO

# Ensure all required packages are installed
try:
    import replicate
except ImportError as e:
    st.error("The 'replicate' module is not installed. Please install it using 'pip install replicate' in your terminal.")
    raise e

# Set API keys
API_KEY = os.getenv('API_KEY')
REPLICATE_API_TOKEN = os.getenv('REPLICATE_API_TOKEN')
if not API_KEY or not REPLICATE_API_TOKEN:
    st.error("API_KEY or REPLICATE_API_TOKEN environment variable not set. Please set them in the Streamlit Cloud settings.")

# Set page configuration
st.set_page_config(page_title="Photo to Style Transfer", page_icon="🎨", layout="wide")

headers = {"Authorization": f"Bearer {API_KEY}"}
replicate_client = replicate.Client(api_token=REPLICATE_API_TOKEN)

# List of artists and modifiers
artists = [
    "Takashi Murakami", "Beeple", "Tyler Edlin", "Andrew Robinson", "Anton Fadeev", 
    "Chris Labrooy", "Dan Mumford", "Michelangelo", "Noah Bradley", "Cassius Marcellus Coolidge",
    "Ted Nasmith", "James Gurney", "James Paick", "Justin Gerard", "Jakub Różalski", 
    "Jeff Easley", "Agnes Lawrence Pelton", "Alan Lee", "Bob Byerley", "Bob Eggleton"
]

modifiers = [
    '4K', 'unreal engine', 'octane render', '8k octane render', 'photorealistic', 'mandelbulb fractal', 
    'Highly detailed carvings', 'Atmosphere', 'Dramatic lighting', 'Sakura blossoms', 'magical atmosphere', 
    'muted colors', 'Highly detailed', 'Epic composition', 'incomparable reality', 'ultra detailed', 
    'unreal 5', 'concept art', 'smooth', 'sharp focus', 'illustration', 'evocative', 'mysterious', 
    'epic scene', 'intricate details', 'Pop Surrealism', 'sharp photography', 'hyper realistic', 
    'maximum detail', 'ray tracing', 'volumetric lighting', 'cinematic', 'realistic lighting', 
    'high resolution render', 'hyper realism', 'insanely detailed', 'volumetric light', 'light rays', 
    'shock art', 'dystopian art', 'cgsociety', 'fantasy art', 'matte drawing', 'speed painting', 
    'darksynth', 'redshift', 'color field', 'rendered in cinema4d', 'imax', '#vfxfriday', 'oil on canvas', 
    'figurative art', 'detailed painting', 'soft mist', 'daz3d', 'zbrush', 'anime', 'behance hd', 
    'panfuturism', 'futuristic', 'pixiv', 'auto-destructive art', 'apocalypse art', 'afrofuturism', 
    'reimagined by industrial light and magic', 'metaphysical painting', 'wiccan', 'grotesque', 'whimsical', 
    'psychedelic art', 'digital art', 'fractalism', 'anime aesthetic', 'chiaroscuro', 'mystical', 'majestic', 
    'digital painting', 'psychedelic', 'synthwave', 'cosmic horror', 'lovecraftian', 'vanitas', 'macabre', 
    'toonami', 'hologram', 'magic realism', 'impressionism', 'neo-fauvism', 'fauvism', 'synchromism'
]

base_image_url = "https://raw.githubusercontent.com/downlifted/aiart/main/stripe/"

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
        pass

    return "Error: Failed to generate a valid prompt after multiple attempts."

def run_style_transfer(structure_image_url, style_image_path, prompt, denoise_strength=0.64):
    with open(style_image_path, "rb") as f:
        style_image = f.read()

    output = replicate_client.run(
        "fofr/style-transfer:f1023890703bc0a5a3a2c21b5e498833be5f6ef6e70e9daf6b9b3a4fd8309cf0",
        input={
            "style_image": style_image,
            "structure_image": structure_image_url,
            "model": "high-quality",
            "width": 1024,
            "height": 1024,
            "prompt": prompt,
            "output_format": "png",
            "output_quality": 100,
            "negative_prompt": "bird, feathers, female, woman, dress, face, eyes, animal, man, human, hands, eyes, face, mouth, nose, human, man, woman, animal, hair, cloth, sheets",
            "number_of_images": 1,
            "structure_depth_strength": 1.2,
            "structure_denoising_strength": denoise_strength
        }
    )
    return output

def get_valid_image_url(base_url, start, end, extension="jpg"):
    for _ in range(10):
        image_number = random.randint(start, end)
        image_url = f"{base_url}{image_number}.{extension}"
        if validate_url(image_url):
            return image_url
    return None

def validate_url(url):
    try:
        response = requests.head(url, timeout=10)
        return response.status_code == 200
    except requests.RequestException:
        return False

def single_image_ui():
    st.header("Single Image to Prompt and Style Transfer")
    uploaded_file = st.file_uploader("Choose a photo...", type=["jpg", "png"], help="Upload a photo to generate AI art prompt and perform style transfer")

    custom_text = st.text_input("Add Custom Text (Optional)")
    define_artist = st.radio("Artist Selection", ["Let AI define artist", "Use my own artist", "No artist"])
    selected_artists = st.multiselect("Select Artists (Optional)", artists) if define_artist == "Use my own artist" else []
    selected_modifiers = st.multiselect("Select Modifiers (Optional)", modifiers)

    if uploaded_file is not None:
        style_image_path = f"style_images/{uploaded_file.name}"
        os.makedirs("style_images", exist_ok=True)

        with open(style_image_path, "wb") as file:
            file.write(uploaded_file.getvalue())
        st.image(uploaded_file, caption="Photo successfully uploaded", use_column_width=True)

        if st.button("Generate Prompt and Perform Style Transfer", key="single_generate"):
            with st.spinner('Generating Prompt and Performing Style Transfer...'):
                caption = image_to_text(style_image_path)
                prompt = generate_prompt(caption, selected_artists, selected_modifiers, custom_text, define_artist == "Let AI define artist", define_artist == "No artist")

                st.write("**Caption:**", caption)
                st.write("**Prompt:**", prompt)

                # Perform style transfer
                structure_image_url = get_valid_image_url(base_image_url, 1, 40)
                if structure_image_url:
                    progress_bar = st.progress(0)
                    output = run_style_transfer(structure_image_url, style_image_path, prompt)
                    progress_bar.progress(100)
                    if output:
                        st.image(output, caption="Generated AI Art", use_column_width=True)
                    else:
                        st.error("Error performing style transfer. Please try again.")
                else:
                    st.error("Failed to fetch a valid structure image. Please try again.")

def main_ui():
    st.title("Pic-To-Prompt: Photo to AI Art Prompt and Style Transfer")

    st.subheader("Turn your photos into stunning AI art prompts and perform style transfer")

    mode = st.sidebar.radio("Choose Mode", ["Single Image", "Batch Processing"])
    
    if mode == "Single Image":
        single_image_ui()
    else:
        st.error("Batch Processing not implemented yet")

    st.markdown("---")
    st.markdown("### Additional Information")
    
    with st.expander("**Tech stack**"):
        st.write('''
        - **LLM : Falcon-7B-Instruct**
        - **HuggingFace**
        - **Replicate**
        ''')

    with st.expander("**App Working**"):
        st.write('''
        - **Upload Your Image(s)**: Upload a single image to generate AI art prompts and perform style transfer.
        - **Custom Text**: Add optional custom text to include in your prompts.
        - **Artist Selection**: Choose whether to let the AI define the artist, use your own artist selection, or no artist at all.
        - **Modifiers**: Select optional modifiers to refine the style and details of your prompts.
        - **Style Transfer**: The uploaded image is used as the style image for the style transfer using Replicate API.
        ''')

if __name__ == "__main__":
    main_ui()
