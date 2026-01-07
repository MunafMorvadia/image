import os
import csv
import time
import gc
from typing import List, Optional, Tuple
from PIL import Image
import torch
from fastapi import FastAPI
import gradio as gr
from huggingface_hub import HfFolder, login
from diffusers import FluxPipeline, FluxImg2ImgPipeline


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

HF_TOKEN = os.environ.get("HUGGINGFACE_TOKEN") or HfFolder.get_token()

t2i_pipe = None
i2i_pipe = None

OUT_DIR = os.path.join(os.getcwd(), "outputs")
os.makedirs(OUT_DIR, exist_ok=True)


def unload(kind: Optional[str] = None):
    global t2i_pipe, i2i_pipe
    if kind == "t2i" and i2i_pipe is not None:
        del i2i_pipe
        i2i_pipe = None
    if kind == "i2i" and t2i_pipe is not None:
        del t2i_pipe
        t2i_pipe = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def ensure_auth():
    if HF_TOKEN:
        try:
            login(token=HF_TOKEN, add_to_git_credential=False)
            HfFolder.save_token(HF_TOKEN)
        except Exception:
            pass


def ensure_t2i():
    global t2i_pipe
    unload("t2i")
    if t2i_pipe is None:
        ensure_auth()
        t2i_pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",
            torch_dtype=DTYPE,
            token=HF_TOKEN,
        )
        try:
            t2i_pipe = t2i_pipe.to(DEVICE)
        except Exception:
            t2i_pipe.enable_model_cpu_offload()


def ensure_i2i():
    global i2i_pipe
    unload("i2i")
    if i2i_pipe is None:
        ensure_auth()
        i2i_pipe = FluxImg2ImgPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",
            torch_dtype=DTYPE,
            token=HF_TOKEN,
        )
        try:
            i2i_pipe = i2i_pipe.to(DEVICE)
        except Exception:
            i2i_pipe.enable_model_cpu_offload()


def coerce_size(w: int, h: int) -> Tuple[int, int]:
    w = max(256, int(w))
    h = max(256, int(h))
    return (w // 32) * 32, (h // 32) * 32


def make_generator(seed: Optional[str]):
    if not seed:
        return None
    try:
        s = int(seed)
    except ValueError:
        s = int(time.time())
    return torch.Generator(DEVICE).manual_seed(s)


def preset_to_wh(preset: str):
    if "768x432" in preset:
        return 768, 432
    if "1024x576" in preset:
        return 1024, 576
    if "1280x720" in preset:
        return 1280, 720
    if "512x512" in preset:
        return 512, 512
    return None


def apply_preset(preset, w, h):
    wh = preset_to_wh(preset)
    return (wh[0], wh[1]) if wh else (w, h)


def text_to_image(prompt, width, height, steps, max_seq_len, seed, counter):
    try:
        ensure_t2i()
        width, height = coerce_size(width, height)
        gen = make_generator(seed)
        img = t2i_pipe(
            prompt=str(prompt),
            guidance_scale=0.0,
            height=int(height),
            width=int(width),
            num_inference_steps=int(steps),
            max_sequence_length=int(max_seq_len),
            generator=gen,
        ).images[0]
        name = f"character_{counter}.png"
        path = os.path.join(OUT_DIR, name)
        img.save(path)
        return path, f"Saved {name}", counter + 1
    except Exception as e:
        return None, f"Error: {type(e).__name__}: {e}", counter


def image_to_image(prompt, init_img, width, height, steps, strength, max_seq_len, seed, counter):
    try:
        ensure_i2i()
        width, height = coerce_size(width, height)
        gen = make_generator(seed)
        img_in = init_img.convert("RGB").resize((width, height), Image.LANCZOS)
        img = i2i_pipe(
            prompt=str(prompt),
            image=img_in,
            guidance_scale=0.0,
            height=int(height),
            width=int(width),
            num_inference_steps=int(steps),
            strength=float(strength),
            max_sequence_length=int(max_seq_len),
            generator=gen,
        ).images[0]
        name = f"character_{counter}.png"
        path = os.path.join(OUT_DIR, name)
        img.save(path)
        return path, f"Saved {name}", counter + 1
    except Exception as e:
        return None, f"Error: {type(e).__name__}: {e}", counter


def run_bulk(mode, prompts_file, images_files, preset, width, height, steps, strength, max_seq_len, seed, bulk_counter):
    try:
        wh = preset_to_wh(preset)
        width, height = coerce_size(*(wh if wh else (width, height)))
        gen = make_generator(seed)

        if prompts_file is None:
            return [], "Upload a TXT or CSV prompts file.", bulk_counter

        pfname = prompts_file.name if hasattr(prompts_file, "name") else str(prompts_file)
        prompts: List = []
        if pfname.lower().endswith(".txt"):
            with open(pfname, "r", encoding="utf-8") as f:
                prompts = [ln.strip() for ln in f if ln.strip()]
        else:
            with open(pfname, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                for row in reader:
                    if not row:
                        continue
                    if mode == "t2i":
                        prompts.append(row[0])
                    else:
                        if len(row) >= 2 and os.path.isfile(row[0]):
                            prompts.append((row[0], row[1]))
                        else:
                            prompts.append(row[0])

        gallery_paths = []
        logs = []

        if mode == "t2i":
            ensure_t2i()
            for idx, p in enumerate(prompts, start=1):
                img = t2i_pipe(
                    prompt=str(p),
                    guidance_scale=0.0,
                    height=int(height),
                    width=int(width),
                    num_inference_steps=int(steps),
                    max_sequence_length=int(max_seq_len),
                    generator=gen,
                ).images[0]
                name = f"image_{bulk_counter + idx}.png"
                path = os.path.join(OUT_DIR, name)
                img.save(path)
                gallery_paths.append(path)
                logs.append(f"[t2i] {name} | {p}")
            bulk_counter += len(prompts)
        else:
            ensure_i2i()
            imgs_list = images_files or []
            if prompts and isinstance(prompts[0], tuple):
                items = prompts
            else:
                if not imgs_list:
                    return [], "Upload images or provide image paths via CSV for i2i.", bulk_counter
                paths = [getattr(x, "name", str(x)) for x in imgs_list]
                items = [(paths[i % len(paths)], p) for i, p in enumerate(prompts)]
            for idx, (imgp, p) in enumerate(items, start=1):
                img_in = Image.open(imgp).convert("RGB").resize((width, height), Image.LANCZOS)
                img = i2i_pipe(
                    prompt=str(p),
                    image=img_in,
                    guidance_scale=0.0,
                    height=int(height),
                    width=int(width),
                    num_inference_steps=int(steps),
                    strength=float(strength),
                    max_sequence_length=int(max_seq_len),
                    generator=gen,
                ).images[0]
                name = f"image_{bulk_counter + idx}.png"
                path = os.path.join(OUT_DIR, name)
                img.save(path)
                gallery_paths.append(path)
                logs.append(f"[i2i] {name} | {imgp} | {p}")
            bulk_counter += len(items)

        return gallery_paths, "\n".join(logs), bulk_counter
    except Exception as e:
        return [], f"Error: {type(e).__name__}: {e}", bulk_counter


app = FastAPI()

with gr.Blocks(title="FLUX.1 schnell - Render UI") as demo:
    gr.Markdown("## FLUX.1 schnell - Image Generator (Render)")
    gr.Markdown(f"Device: {DEVICE}, dtype: {str(DTYPE).split('.')[-1]}")

    with gr.Tab("Text to Image"):
        t2i_prompt = gr.Textbox(label="Prompt", lines=6)
        t2i_preset = gr.Dropdown(
            choices=[
                "Mobile 16:9 (768x432)",
                "Mobile 16:9 (1024x576)",
                "Mobile 16:9 (1280x720)",
                "Square 1:1 (512x512)",
                "Custom",
            ],
            value="Mobile 16:9 (1024x576)",
            label="Preset",
        )
        t2i_width = gr.Number(label="Width", value=1024)
        t2i_height = gr.Number(label="Height", value=576)
        t2i_steps = gr.Slider(label="Steps", minimum=1, maximum=15, value=4, step=1)
        t2i_msl = gr.Slider(label="Max Seq Len", minimum=64, maximum=256, value=256, step=8)
        t2i_seed = gr.Textbox(label="Seed", value="")
        t2i_counter = gr.State(1)
        t2i_out = gr.Image(label="Output", type="filepath")
        t2i_status = gr.Markdown()
        t2i_preset.change(apply_preset, [t2i_preset, t2i_width, t2i_height], [t2i_width, t2i_height])
        t2i_run = gr.Button("Generate")
        t2i_run.click(
            text_to_image,
            inputs=[t2i_prompt, t2i_width, t2i_height, t2i_steps, t2i_msl, t2i_seed, t2i_counter],
            outputs=[t2i_out, t2i_status, t2i_counter],
            concurrency_limit=1,
        )

    with gr.Tab("Image to Image"):
        i2i_prompt = gr.Textbox(label="Prompt", lines=6)
        i2i_image = gr.Image(label="Input Image", type="pil")
        i2i_preset = gr.Dropdown(
            choices=[
                "Mobile 16:9 (768x432)",
                "Mobile 16:9 (1024x576)",
                "Mobile 16:9 (1280x720)",
                "Square 1:1 (512x512)",
                "Custom",
            ],
            value="Mobile 16:9 (1024x576)",
            label="Preset",
        )
        i2i_width = gr.Number(label="Width", value=1024)
        i2i_height = gr.Number(label="Height", value=576)
        i2i_steps = gr.Slider(label="Steps", minimum=1, maximum=15, value=4, step=1)
        i2i_strength = gr.Slider(label="Strength", minimum=0.1, maximum=1.0, value=0.6, step=0.05)
        i2i_msl = gr.Slider(label="Max Seq Len", minimum=64, maximum=256, value=256, step=8)
        i2i_seed = gr.Textbox(label="Seed", value="")
        i2i_counter = gr.State(1)
        i2i_out = gr.Image(label="Output", type="filepath")
        i2i_status = gr.Markdown()
        i2i_preset.change(apply_preset, [i2i_preset, i2i_width, i2i_height], [i2i_width, i2i_height])
        i2i_run = gr.Button("Generate")
        i2i_run.click(
            image_to_image,
            inputs=[i2i_prompt, i2i_image, i2i_width, i2i_height, i2i_steps, i2i_strength, i2i_msl, i2i_seed, i2i_counter],
            outputs=[i2i_out, i2i_status, i2i_counter],
            concurrency_limit=1,
        )

    with gr.Tab("Bulk"):
        bulk_mode = gr.Radio(choices=["t2i", "i2i"], value="t2i", label="Mode")
        bulk_prompts_file = gr.File(label="Prompts File (TXT or CSV)", file_types=[".txt", ".csv"])
        bulk_images_files = gr.File(
            label="Images for i2i (optional, multiple)",
            file_count="multiple",
            type="filepath",
            file_types=["image"],
        )
        bulk_preset = gr.Dropdown(
            choices=[
                "Mobile 16:9 (768x432)",
                "Mobile 16:9 (1024x576)",
                "Mobile 16:9 (1280x720)",
                "Square 1:1 (512x512)",
                "Custom",
            ],
            value="Mobile 16:9 (1024x576)",
            label="Preset",
        )
        bulk_width = gr.Number(label="Width", value=1024)
        bulk_height = gr.Number(label="Height", value=576)
        bulk_steps = gr.Slider(label="Steps", minimum=1, maximum=15, value=4, step=1)
        bulk_strength = gr.Slider(label="Strength (i2i)", minimum=0.1, maximum=1.0, value=0.6, step=0.05)
        bulk_msl = gr.Slider(label="Max Seq Len", minimum=64, maximum=256, value=256, step=8)
        bulk_seed = gr.Textbox(label="Seed", value="")
        bulk_counter = gr.State(1)
        bulk_gallery = gr.Gallery(label="Generated", columns=4, height=300)
        bulk_log = gr.Textbox(label="Log", lines=10)
        bulk_preset.change(apply_preset, [bulk_preset, bulk_width, bulk_height], [bulk_width, bulk_height])
        bulk_run = gr.Button("Run Bulk")
        bulk_run.click(
            run_bulk,
            inputs=[
                bulk_mode,
                bulk_prompts_file,
                bulk_images_files,
                bulk_preset,
                bulk_width,
                bulk_height,
                bulk_steps,
                bulk_strength,
                bulk_msl,
                bulk_seed,
                bulk_counter,
            ],
            outputs=[bulk_gallery, bulk_log, bulk_counter],
            concurrency_limit=1,
        )

app = gr.mount_gradio_app(app, demo, path="/")
