"""Comic: Download Models - fetch the models this pack needs straight onto the machine ComfyUI runs on.

Useful on a remote / shared server where you can't copy files: add this node, pick what you need, Run.
Files that already exist (in any configured model folder) are skipped. Downloads go to a .part file first,
so an interrupted download never leaves a broken model behind."""
import os
import time
import urllib.request

import folder_paths

HF = "https://huggingface.co/"
MODELS = {
    "image model fp8 (24 GB+ VRAM)": ("diffusion_models", "qwen_image_edit_2511_fp8mixed.safetensors",
                                      HF + "Comfy-Org/Qwen-Image-Edit_ComfyUI/resolve/main/split_files/diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors"),
    "image model GGUF Q4 (8-16 GB VRAM)": ("unet", "qwen-image-edit-2511-Q4_K_M.gguf",
                                           HF + "unsloth/Qwen-Image-Edit-2511-GGUF/resolve/main/qwen-image-edit-2511-Q4_K_M.gguf"),
    "text encoder": ("text_encoders", "qwen_2.5_vl_7b_fp8_scaled.safetensors",
                     HF + "Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors"),
    "vae": ("vae", "qwen_image_vae.safetensors",
            HF + "Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors"),
    "lightning lora": ("loras", "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
                       HF + "lightx2v/Qwen-Image-Edit-2511-Lightning/resolve/main/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"),
    "vision model 2B": ("text_encoders", "qwen3.5_2b_bf16.safetensors",
                        HF + "Comfy-Org/Qwen3.5/resolve/main/text_encoders/qwen3.5_2b_bf16.safetensors"),
    "vision model 4B": ("text_encoders", "qwen3.5_4b_bf16.safetensors",
                        HF + "Comfy-Org/Qwen3.5/resolve/main/text_encoders/qwen3.5_4b_bf16.safetensors"),
}


def _exists(folder, name):
    try:
        return folder_paths.get_full_path(folder, name) is not None
    except Exception:
        return False


def _download(url, dest):
    tmp = dest + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": "ComfyUI-QwenComic"})
    t0, last = time.time(), 0
    with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while True:
            buf = r.read(8 << 20)
            if not buf:
                break
            f.write(buf)
            done += len(buf)
            if time.time() - last > 10:
                last = time.time()
                pct = f"{100 * done / total:.0f}%" if total else ""
                print(f"[QwenComic] downloading {os.path.basename(dest)}: {done / 1e9:.1f} GB {pct} "
                      f"({done / 1e6 / max(1e-3, time.time() - t0):.0f} MB/s)")
    os.replace(tmp, dest)
    return done


class ComicDownloadModels:
    """Download the models for the comic workflow onto the ComfyUI machine (skips files that exist)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image_model": (["fp8 (24 GB+ VRAM, servers)", "GGUF Q4 (8-16 GB VRAM)", "both", "none"],),
            "vision_model": (["2B (fast)", "4B (more accurate)", "both", "none"],),
            "lightning_lora": ("BOOLEAN", {"default": True, "tooltip": "needed for fast mode (4 steps)"}),
            "really_download": ("BOOLEAN", {"default": False, "label_on": "download now",
                                            "label_off": "only check what is missing"}),
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "QwenComic"

    @classmethod
    def IS_CHANGED(cls, **kw):
        return float("nan")

    def run(self, image_model, vision_model, lightning_lora, really_download):
        want = ["text encoder", "vae"]
        if image_model.startswith(("fp8", "both")):
            want.append("image model fp8 (24 GB+ VRAM)")
        if image_model.startswith(("GGUF", "both")):
            want.append("image model GGUF Q4 (8-16 GB VRAM)")
        if vision_model.startswith(("2B", "both")):
            want.append("vision model 2B")
        if vision_model.startswith(("4B", "both")):
            want.append("vision model 4B")
        if lightning_lora:
            want.append("lightning lora")
        lines = []
        for key in want:
            folder, name, url = MODELS[key]
            if _exists(folder, name):
                lines.append(f"OK       {key}: {name}")
                continue
            if not really_download:
                lines.append(f"MISSING  {key}: {name}  -> models/{folder}/")
                continue
            dest_dir = folder_paths.get_folder_paths(folder)[0]
            os.makedirs(dest_dir, exist_ok=True)
            dest = os.path.join(dest_dir, name)
            try:
                n = _download(url, dest)
                lines.append(f"DONE     {key}: {name} ({n / 1e9:.1f} GB) -> {dest_dir}")
            except Exception as e:
                lines.append(f"FAILED   {key}: {name}: {e}")
        if any(l.startswith(("DONE", "MISSING")) for l in lines):
            lines.append("")
            lines.append("After downloading: press R in ComfyUI (or reload the page) so the model lists update."
                         if really_download else "Turn 'really_download' on and Run to fetch the missing files.")
        report = "\n".join(lines)
        print("[QwenComic] model check:\n" + report)
        return {"ui": {"text": [report]}, "result": (report,)}


NODE_CLASS_MAPPINGS = {"ComicDownloadModels": ComicDownloadModels}
NODE_DISPLAY_NAME_MAPPINGS = {"ComicDownloadModels": "Comic: Download Models (for servers)"}
