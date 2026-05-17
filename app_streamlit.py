import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import streamlit as st
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionControlNetImg2ImgPipeline, ControlNetModel, UniPCMultistepScheduler
import cv2
import numpy as np
import io
import time
import random
import insightface
from insightface.app import FaceAnalysis
from pathlib import Path
from huggingface_hub import hf_hub_download
from controlnet_aux import OpenposeDetector

STYLES = {
    "Anime":        "anime style, studio ghibli, makoto shinkai, beautiful highly detailed face, large expressive sparkling anime eyes, detailed iris and pupil, clean sharp lineart, soft cel shading, vibrant colors, perfect anatomy, perfect hands, detailed fingers, flowing hair",
    "Cyberpunk":    "cyberpunk portrait, neon lights, futuristic city, blade runner, ultra detailed, cinematic, perfect hands, detailed fingers, sharp eyes, neon iris, dramatic shadows",
    "Comic Hero":   "comic book superhero portrait, marvel style, dynamic lighting, bold colors, ink outlines, highly detailed face, expressive eyes, perfect muscular anatomy, perfect hands, strong fingers",
    "Oil Painting": "oil painting portrait, renaissance style, classical art, rembrandt lighting, textured canvas, highly detailed face, realistic eyes, perfect hands, detailed fingers, masterful brushwork",
    "Pro Headshot": "professional headshot, studio lighting, sharp focus, 8k, photorealistic, clean background, highly detailed face, natural eyes, perfect skin, professional attire, perfect hands",
    "Samurai":      "samurai warrior portrait, feudal japan, dramatic lighting, ultra detailed, epic, highly detailed face, intense eyes, perfect hands gripping weapon, detailed armor, flowing robes",
    "Movie Poster": "cinematic movie poster portrait, dramatic lighting, epic composition, 8k, volumetric light, highly detailed face, intense cinematic eyes, perfect hands, heroic pose",
    "Watercolor":   "watercolor portrait, soft colors, artistic, dreamy, beautiful illustration, flowing paint, highly detailed face, beautiful expressive eyes, soft perfect hands, delicate fingers",
}

MODEL_DIR     = Path(os.environ.get("MODEL_DIR", "./models"))
SWAPPER_MODEL = MODEL_DIR / "inswapper_128.onnx"
GFPGAN_MODEL  = MODEL_DIR / "GFPGANv1.4.pth"
SD_MODEL      = "Lykon/DreamShaper"

def download_models():
    os.makedirs("models", exist_ok=True)
    try:
        if not SWAPPER_MODEL.exists():
            print("Downloading inswapper_128.onnx...")
            hf_hub_download(
                repo_id="Tejaswi2006/avatar-models",
                filename="inswapper_128.onnx",
                local_dir="./models",
                local_dir_use_symlinks=False
            )
        if not GFPGAN_MODEL.exists():
            print("Downloading GFPGANv1.4.pth...")
            hf_hub_download(
                repo_id="Tejaswi2006/avatar-models",
                filename="GFPGANv1.4.pth",
                local_dir="./models",
                local_dir_use_symlinks=False
            )
    except Exception as e:
        print(f"DOWNLOAD FAILED: {e}")

download_models()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def load_base_model():
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        SD_MODEL,
        torch_dtype=torch.float16,
        safety_checker=None,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_attention_slicing()
    return pipe.to(DEVICE)

@st.cache_resource
def load_controlnet_model():
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-openpose",
        torch_dtype=torch.float16
    )
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        SD_MODEL,
        controlnet=controlnet,
        torch_dtype=torch.float16,
        safety_checker=None,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_attention_slicing()
    return pipe.to(DEVICE)

@st.cache_resource
def load_face_swapper():
    app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    swapper = insightface.model_zoo.get_model(str(SWAPPER_MODEL), download=False)
    return app, swapper

@st.cache_resource
def load_gfpgan():
    from gfpgan import GFPGANer
    return GFPGANer(
        model_path=str(GFPGAN_MODEL),
        upscale=1,
        arch="clean",
        channel_multiplier=2,
        bg_upsampler=None
    )

@st.cache_resource
def load_openpose():
    return OpenposeDetector.from_pretrained("lllyasviel/ControlNet")

def prepare_image(uploaded_file, size=(512, 512)) -> Image.Image:
    img = Image.open(uploaded_file).convert("RGB")
    if img.size != size:
        img = img.resize(size, Image.LANCZOS)
    return img

def swap_face(source_img, target_img, app, swapper):
    src = cv2.cvtColor(np.array(source_img), cv2.COLOR_RGB2BGR)
    tgt = cv2.cvtColor(np.array(target_img), cv2.COLOR_RGB2BGR)
    src_faces = app.get(src)
    tgt_faces = app.get(tgt)
    if len(src_faces) == 0:
        st.warning("No face detected in your uploaded photo.")
        return target_img
    if len(tgt_faces) == 0:
        st.warning("No face detected in generated image - skipping swap.")
        return target_img
    result = tgt.copy()
    for tgt_face in tgt_faces:
        result = swapper.get(result, tgt_face, src_faces[0], paste_back=True)
    tgt_faces2 = app.get(result)
    if len(tgt_faces2) > 0:
        for tgt_face in tgt_faces2:
            result = swapper.get(result, tgt_face, src_faces[0], paste_back=True)
    return Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))

def restore_face(image: Image.Image, restorer) -> Image.Image:
    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    _, _, restored = restorer.enhance(
        img_bgr,
        has_aligned=False,
        only_center_face=False,
        paste_back=True
    )
    return Image.fromarray(cv2.cvtColor(restored, cv2.COLOR_BGR2RGB))

def make_openpose(image: Image.Image, detector) -> Image.Image:
    return detector(image)

def run_pipe(pipe, prompt, negative, image, strength, steps, guidance, seed):
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    return pipe(
        prompt=prompt, negative_prompt=negative,
        image=image, strength=strength,
        num_inference_steps=int(steps),
        guidance_scale=guidance,
        generator=gen,
        num_images_per_prompt=1
    ).images[0]

def run_controlnet_pipe(pipe, prompt, negative, image, pose, strength, steps, guidance, seed, cn_scale=0.5):
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    return pipe(
        prompt=prompt, negative_prompt=negative,
        image=image,
        control_image=pose,
        strength=strength,
        num_inference_steps=int(steps),
        guidance_scale=guidance,
        controlnet_conditioning_scale=cn_scale,
        generator=gen,
        num_images_per_prompt=1
    ).images[0]

st.title("AI Avatar Generator")
st.markdown("Your face. Your outfit. Any art style!")

uploaded = st.file_uploader("Upload your photo", type=["jpg", "jpeg", "png"])

if uploaded:
    st.image(prepare_image(uploaded), width=200, caption="Your photo")

col1, col2 = st.columns(2)
with col1:
    style = st.selectbox("Choose style", list(STYLES.keys()))
with col2:
    gender = st.selectbox("Choose gender", ["Woman", "Man"])

steps      = st.slider("Quality steps", 15, 50, 25, 5)
variations = st.slider("Number of variations", 1, 4, 1, 1)

with st.expander("Advanced"):
    base_strength  = st.slider("Base strength (Step 1)", 0.3, 0.6, 0.50, 0.05)
    style_strength = st.slider("Style strength (Step 3)", 0.2, 0.5, 0.25, 0.05)
    guidance       = st.slider("Guidance scale", 5.0, 12.0, 8.5, 0.5)
    cn_scale       = st.slider("ControlNet strength", 0.3, 0.9, 0.5, 0.05)

if st.button("Generate Avatar") and uploaded:
    image             = prepare_image(uploaded)
    pipe              = load_base_model()
    cn_pipe           = load_controlnet_model()
    openpose_detector = load_openpose()
    app, swapper      = load_face_swapper()
    restorer          = load_gfpgan()

    negative = (
        "blurry, low quality, ugly, watermark, distorted face, bad anatomy, deformed, disfigured, "
        "extra limbs, fused fingers, bad hands, overexposed, washed out, mutilated, extra fingers, "
        "nsfw, malformed hands, poorly drawn hands, mutated hands, clipping, missing fingers, "
        "extra digits, fewer digits, cropped, worst quality, jpeg artifacts, signature, username, "
        "artist name, bad proportions, gross proportions, duplicate, error, out of frame, "
        "ugly eyes, crossed eyes, lazy eye, asymmetric eyes, bad eyes, "
        "full body, wide shot, far away, small face, tiny face, distant,"
        "asymmetric eyes, uneven eyes, different sized eyes, one eye bigger, mismatched eyes"
    )
    base_prompt = (
    f"realistic portrait of a {gender.lower()}, natural skin tone, sharp focus, "
    f"studio lighting, 8k, highly detailed face, beautiful eyes, "
    f"professional photo, masterpiece, best quality, close up portrait, "
    f"preserve natural hair, keep original hairstyle, preserve outfit, preserve background"
    )

    style_prompt = (
    f"{STYLES[style]}, portrait of a {gender.lower()}, natural skin tone, "
    f"masterpiece, best quality, 8k, highly detailed face, beautiful expressive eyes, "
    f"looking at viewer, close up portrait, upper body shot, "
    f"preserve natural hair, keep original hairstyle, same outfit, same background"
    )

    num_var     = int(variations)
    total_steps = num_var * 5
    completed   = 0
    progress    = st.progress(0.0, text="Starting generation...")
    start       = time.time()
    all_results = []

    for i in range(num_var):
        seed = random.randint(0, 2**32 - 1)

        completed += 1
        progress.progress(completed / total_steps,
                          text=f"[{i+1}/{num_var}] Step 1: Generating realistic base...")
        realistic = run_pipe(pipe, base_prompt, negative,
                             image, base_strength, steps, guidance, seed)

        completed += 1
        progress.progress(completed / total_steps,
                          text=f"[{i+1}/{num_var}] Step 2: Swapping your face in...")
        face_swapped = swap_face(image, realistic, app, swapper)

        completed += 1
        progress.progress(completed / total_steps,
                          text=f"[{i+1}/{num_var}] Step 3: Extracting pose structure...")
        pose = make_openpose(face_swapped, openpose_detector)

        completed += 1
        progress.progress(completed / total_steps,
                          text=f"[{i+1}/{num_var}] Step 4: Applying {style} style with ControlNet...")
        final = run_controlnet_pipe(cn_pipe, style_prompt, negative,
                                    face_swapped, pose,
                                    style_strength, steps, guidance, seed, cn_scale)

        completed += 1
        progress.progress(completed / total_steps,
                          text=f"[{i+1}/{num_var}] Step 5: Restoring face details...")
        final = restore_face(final, restorer)

        all_results.append(final)
        torch.cuda.empty_cache()

    progress.progress(1.0, text="Done!")
    elapsed = time.time() - start

    st.markdown("### Original")
    st.image(image, width=250)
    st.markdown(f"### Generated Avatars - {style} style")
    st.caption("Your face. Original outfit. New art style.")

    cols = st.columns(num_var)
    for i, (col, result) in enumerate(zip(cols, all_results)):
        with col:
            st.image(result, caption=f"Variation {i+1}")
            buf = io.BytesIO()
            result.save(buf, format="PNG")
            st.download_button(f"Download {i+1}", buf.getvalue(),
                               f"{style}_avatar_{i+1}.png", "image/png",
                               key=f"dl_{i}")

    st.success(f"Done in {elapsed:.1f}s!")