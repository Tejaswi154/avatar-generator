# AI Avatar Generator

Transform your photo into stunning art styles using AI.

## Features
- 8 art styles: Anime, Cyberpunk, Comic Hero, Oil Painting, Pro Headshot, Samurai, Movie Poster, Watercolor
- Face swapping using InsightFace
- Pose preservation using ControlNet + OpenPose
- Face restoration using GFPGAN
- Up to 4 variations per generation

## Tech Stack
- Stable Diffusion (DreamShaper8)
- ControlNet (OpenPose)
- InsightFace (inswapper_128)
- GFPGAN v1.4
- Streamlit

## Live Demo
[Try it on Hugging Face](https://huggingface.co/spaces/Tejaswi2006/avatar-generator)

## Local Setup
```bash
pip install -r requirements.txt
streamlit run app_streamlit.py
```

## Models Required
- DreamShaper8 — place in `models/dreamshaper8`
- inswapper_128.onnx — place in `models/`
- GFPGANv1.4.pth — place in `models/`
