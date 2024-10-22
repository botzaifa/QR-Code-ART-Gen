import streamlit as st
from PIL import Image
import torch
from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel, DDIMScheduler
from diffusers.utils import load_image

# Load the ControlNet and Stable Diffusion models
controlnet = ControlNetModel.from_pretrained(
    "DionTimmer/controlnet_qrcode-control_v11p_sd21",
    torch_dtype=torch.float16
)

pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1",
    controlnet=controlnet,
    safety_checker=None,
    torch_dtype=torch.float16
)

pipe.enable_xformers_memory_efficient_attention()
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()

# Helper function to resize images
def resize_for_condition_image(input_image: Image, resolution: int):
    input_image = input_image.convert("RGB")
    W, H = input_image.size
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H = int(round(H / 64.0)) * 64
    W = int(round(W / 64.0)) * 64
    img = input_image.resize((W, H), resample=Image.LANCZOS)
    return img

# Streamlit UI
st.title("AI ART QR Code Generator")

# File uploader for the QR code image
qr_code_image_file = st.file_uploader("Upload QR Code Image (e.g., apple1.png)", type=["png", "jpg", "jpeg"])
if qr_code_image_file:
    qr_code_image = Image.open(qr_code_image_file)
    qr_code_image = resize_for_condition_image(qr_code_image, 768)
    st.image(qr_code_image, caption="QR Code Image", use_column_width=True)

# File uploader for the base image
base_image_file = st.file_uploader("Upload Base Image (e.g., apple2.png)", type=["png", "jpg", "jpeg"])
if base_image_file:
    base_image = Image.open(base_image_file)
    base_image = resize_for_condition_image(base_image, 768)
    st.image(base_image, caption="Base Image", use_column_width=True)

# Text inputs for prompt and negative prompt
prompt = st.text_input("Prompt", "iPhone, realistic, Black and White, futuristic")
negative_prompt = st.text_input("Negative Prompt", "ugly, disfigured, low quality, blurry, nsfw")

# Sliders for guidance scale, conditioning scale, and strength
guidance_scale = st.slider("Guidance Scale", min_value=1.0, max_value=50.0, value=30.0)
conditioning_scale = st.slider("ControlNet Conditioning Scale", min_value=0.5, max_value=5.0, value=2.5)
strength = st.slider("Strength", min_value=0.1, max_value=1.0, value=0.9)

# Generate the image
if st.button("Generate Image") and qr_code_image_file and base_image_file:
    generator = torch.manual_seed(123121231)
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=base_image,
        control_image=qr_code_image,
        width=768,
        height=768,
        guidance_scale=guidance_scale,
        controlnet_conditioning_scale=conditioning_scale,
        generator=generator,
        strength=strength,
        num_inference_steps=150
    )
    st.image(image.images[0], caption="Generated Image", use_column_width=True)

# to run type this in terminal:
# python -m streamlit run app.py