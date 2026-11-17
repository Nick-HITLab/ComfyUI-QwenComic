import copy
import math

import numpy as np
import torch
from PIL import Image

from . import comic_render as cr

CAT = "Qwen Comic"


# ---------------------------------------------------------------- helpers
def tensor_to_pils(t):
    if t is None:
        return []
    arr = (t.detach().cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
    return [Image.fromarray(a[..., :3]) for a in arr]


def pil_to_tensor(img):
    return torch.from_numpy(np.array(img.convert("RGB")).astype(np.float32) / 255.0)[None, ]


def pils_to_tensor(imgs):
    return torch.cat([pil_to_tensor(i) for i in imgs], dim=0)


def round16(v):
    return max(256, int(round(v / 16.0)) * 16)


def gen_size(w, h, megapixels):
    total = megapixels * 1024 * 1024
    s = math.sqrt(total / float(w * h))
    return round16(w * s), round16(h * s)


def _dialogue_prompt(lines):
    parts = []
    for l in lines:
        t = l["text"].replace('"', "'")
        if l["style"] == "caption":
            parts.append(f'a rectangular narration caption box reading "{t}"')
        elif l["style"] == "sfx":
            parts.append(f'big stylised sound-effect lettering "{t}"')
        elif l["style"] == "thought":
            parts.append(f'a cloud-shaped thought bubble from {l["speaker"]} reading "{t}"')
        elif l["style"] == "shout":
            parts.append(f'a jagged shout balloon from {l["speaker"]} reading "{t}"')
        else:
            parts.append(f'a speech balloon from {l["speaker"]} reading "{t}"')
    return "; ".join(parts)


# ---------------------------------------------------------------- nodes
class ComicPageLayout:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "template": (list(cr.TEMPLATES.keys()), {"default": "classic 1-2-1"}),
            "page_width": ("INT", {"default": 1600, "min": 256, "max": 8192, "step": 16}),
            "page_height": ("INT", {"default": 2400, "min": 256, "max": 8192, "step": 16}),
            "margin": ("INT", {"default": 48, "min": 0, "max": 1024}),
            "gutter": ("INT", {"default": 28, "min": 0, "max": 512}),
            "border": ("INT", {"default": 6, "min": 0, "max": 64}),
            "reading_order": (["left-to-right", "right-to-left (manga)"],),
            "custom_rows": ("STRING", {"default": "1, 2:1, 1:1:1",
                                       "tooltip": "Only for 'custom'. One entry per row: a number = N equal panels, ratios like 2:1 = unequal widths."}),
            "custom_row_heights": ("STRING", {"default": "",
                                              "tooltip": "Only for 'custom'. Relative row heights, e.g. '2, 1, 1'. Empty = equal."}),
        }}

    RETURN_TYPES = ("COMIC_LAYOUT", "IMAGE", "INT")
    RETURN_NAMES = ("layout", "layout_guide", "panel_count")
    FUNCTION = "run"
    CATEGORY = CAT

    def run(self, template, page_width, page_height, margin, gutter, border, reading_order,
            custom_rows, custom_row_heights):
        lay = cr.build_layout(template, page_width, page_height, margin, gutter, border, reading_order,
                              custom_rows, custom_row_heights)
        guide = cr.draw_layout_guide(lay)
        return (lay, pil_to_tensor(guide), len(lay["panels"]))


class ComicControlCanvas:
    """Blank page with the panel borders - place references and text on it."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "layout": ("COMIC_LAYOUT",),
            "show_panel_numbers": ("BOOLEAN", {"default": False}),
        }}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("canvas",)
    FUNCTION = "run"
    CATEGORY = CAT

    def run(self, layout, show_panel_numbers):
        return (pil_to_tensor(cr.draw_layout_guide(layout, numbers=show_panel_numbers)),)


class ComicPlaceImage:
    """Paste a reference image at a position inside a panel (0 = whole page)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "canvas": ("IMAGE",),
            "layout": ("COMIC_LAYOUT",),
            "image": ("IMAGE",),
            "panel": ("INT", {"default": 1, "min": 0, "max": 64, "tooltip": "0 = whole page coordinates"}),
            "x": ("FLOAT", {"default": 0.5, "min": -0.5, "max": 1.5, "step": 0.01, "tooltip": "center, 0=left 1=right of the panel"}),
            "y": ("FLOAT", {"default": 0.6, "min": -0.5, "max": 1.5, "step": 0.01, "tooltip": "center, 0=top 1=bottom of the panel"}),
            "width": ("FLOAT", {"default": 0.4, "min": 0.02, "max": 2.0, "step": 0.01, "tooltip": "width relative to the panel width"}),
            "remove_white_background": ("BOOLEAN", {"default": True}),
        }, "optional": {
            "mask": ("MASK", {"tooltip": "Connect the MASK of a Load Image with a transparent PNG"}),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = CAT

    def run(self, canvas, layout, image, panel, x, y, width, remove_white_background, mask=None):
        el = tensor_to_pils(image)[0]
        alpha = None
        if mask is not None and mask.shape[-1] > 1 and mask.shape[-2] > 1 and float(mask.max()) > 0:
            m = (1.0 - mask[0].detach().cpu().numpy()).clip(0, 1)
            alpha = Image.fromarray((m * 255).astype(np.uint8)).resize(el.size)
        out = []
        for c in tensor_to_pils(canvas):
            rect = cr.panel_rect(layout, panel, c.size)
            out.append(cr.paste_element(c, el, alpha, rect, x, y, width, remove_white_background))
        return (pils_to_tensor(out),)


class ComicTextBubble:
    """Draw a speech bubble / caption / SFX at a position. Use it on the control canvas
    (AI will redraw it) or on the finished page (crisp final lettering)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "layout": ("COMIC_LAYOUT",),
            "text": ("STRING", {"multiline": True, "default": "Hello there!"}),
            "style": (cr.BUBBLE_STYLES,),
            "panel": ("INT", {"default": 1, "min": 0, "max": 64, "tooltip": "0 = whole page coordinates"}),
            "x": ("FLOAT", {"default": 0.3, "min": -0.5, "max": 1.5, "step": 0.01}),
            "y": ("FLOAT", {"default": 0.2, "min": -0.5, "max": 1.5, "step": 0.01}),
            "max_width": ("FLOAT", {"default": 0.45, "min": 0.05, "max": 1.0, "step": 0.01, "tooltip": "max text width relative to panel width"}),
            "font": (cr.list_fonts(), {"default": "ComicNeue-Bold.ttf"}),
            "font_size": ("INT", {"default": 30, "min": 6, "max": 400}),
            "tail_x": ("FLOAT", {"default": 0.45, "min": -1.0, "max": 1.5, "step": 0.01, "tooltip": "where the tail points (panel coords). -1 = no tail"}),
            "tail_y": ("FLOAT", {"default": 0.55, "min": -1.0, "max": 1.5, "step": 0.01}),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = CAT

    def run(self, image, layout, text, style, panel, x, y, max_width, font, font_size, tail_x, tail_y):
        out = []
        for img in tensor_to_pils(image):
            img = img.copy()
            rx, ry, rw, rh = cr.panel_rect(layout, panel, img.size)
            scale = img.size[0] / float(layout["width"])
            tail = None
            if tail_x >= 0 and tail_y >= 0 and style not in ("caption", "sfx", "plain text"):
                tail = (rx + tail_x * rw, ry + tail_y * rh)
            cr.draw_bubble(img, text, rx + x * rw, ry + y * rh, max_width * rw, font,
                           max(6, int(font_size * scale)), style, tail=tail)
            out.append(img)
        return (pils_to_tensor(out),)


class ComicDirector:
    """Turns layout + script + references into per-generation prompts/images.
    Outputs are LISTS: one entry per panel (per-panel mode) or one entry (whole-page mode)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "layout": ("COMIC_LAYOUT",),
            "style_image": ("IMAGE",),
            "script": ("STRING", {"multiline": True, "default":
                "Panel 1: Wide establishing shot of a rainy neon city street at night.\n"
                "Caption: Kortrijk, 2089.\n\n"
                "Panel 2: A girl in a yellow raincoat looks up at a giant hologram.\n"
                "Mia: Where did everyone go?\n\n"
                "Panel 3: Close-up of a small robot tugging at her sleeve.\n"
                "Bolt: Follow me. Quickly!\n\n"
                "Panel 4: They run into a glowing alley, puddles splashing.\n"
                "SFX: SPLASH\n"}),
            "whole_page": ("BOOLEAN", {"default": False, "label_on": "whole page at once",
                                       "label_off": "per panel -> assemble"}),
            "lettering": (["overlay bubbles (crisp text)", "AI draws the text", "both", "no text"],),
            "style_notes": ("STRING", {"multiline": False, "default": "comic book art, clean ink lines",
                                       "tooltip": "Extra style words added to every prompt"}),
            "camera": (cr.CAMERA_MODES, {"tooltip": "Auto-pick varied shot types / camera angles for panels without one. "
                                                  "Write your own per panel as 'Panel 2 (low angle, close-up): ...'"}),
            "staging": (["auto: place speakers left/right", "off"], {
                "tooltip": "Tell the model where each character stands, so bubbles can point at the right person"}),
            "megapixels": ("FLOAT", {"default": 1.0, "min": 0.25, "max": 2.5, "step": 0.05,
                                     "tooltip": "Generation size per image. ~1.0 per panel, 1.3-1.6 for whole page"}),
            "use_control_canvas": ("BOOLEAN", {"default": False, "label_on": "control canvas ON",
                                               "label_off": "control canvas OFF"}),
            "use_character_sheet": ("BOOLEAN", {"default": True, "label_on": "character sheet ON",
                                                "label_off": "character sheet OFF"}),
            "cast": ("STRING", {"multiline": True, "default":
                "Mia: 12-year-old girl, short messy red hair, freckles, green eyes, bright yellow raincoat, red rubber boots\n"
                "Bolt: small round robot, light-blue metal body, big yellow glowing eyes, red antenna bulb",
                "tooltip": "One character per line: 'Name: looks'. The description is added to every panel that mentions the name."}),
        }, "optional": {
            "control_canvas": ("IMAGE", {"lazy": True, "tooltip": "Page with placed references/text (Comic Control Canvas chain) or your own sketch"}),
            "character_ref": ("IMAGE", {"lazy": True, "tooltip": "Character sheet (generated or your own) used as a pixel reference in every panel with a cast member"}),
        }}

    RETURN_TYPES = ("COMIC_PLAN", "STRING", "IMAGE", "IMAGE", "IMAGE", "INT", "INT", "STRING", "STRING")
    RETURN_NAMES = ("plan", "prompts", "image1", "image2", "image3", "width", "height", "info", "pixel_refs")
    OUTPUT_IS_LIST = (False, True, True, True, True, True, True, False, True)
    FUNCTION = "run"
    CATEGORY = CAT

    def check_lazy_status(self, use_control_canvas=False, control_canvas=None, use_character_sheet=False,
                          character_ref=None, **kwargs):
        need = []
        if use_control_canvas and control_canvas is None:
            need.append("control_canvas")
        if use_character_sheet and character_ref is None:
            need.append("character_ref")
        return need

    def run(self, layout, style_image, script, whole_page, lettering, style_notes, camera, staging, megapixels,
            use_control_canvas, use_character_sheet, cast="", control_canvas=None, character_ref=None):
        n = len(layout["panels"])
        panels, warnings, _pre = cr.parse_script(script, n)
        panels = cr.assign_shots(layout, panels, camera)
        cast_list = cr.parse_cast(cast)
        ai_text = lettering in ("AI draws the text", "both")
        style = style_image[:1]
        char = character_ref[:1] if (use_character_sheet and character_ref is not None) else None
        canvas_pil = tensor_to_pils(control_canvas[:1])[0] if (use_control_canvas and control_canvas is not None) else None
        notes = style_notes.strip().rstrip(".")
        order = "right to left (manga order)" if layout["reading_order"].startswith("right") else "left to right, top to bottom"
        do_stage = staging.startswith("auto")

        prompts, im1, im2, im3, ws, hs, flags_out = [], [], [], [], [], [], []

        def pack(imgs, flags):
            imgs = list(imgs) + [None, None, None]
            return imgs[0], imgs[1], imgs[2], ",".join(flags)

        style_rule = ("is ONLY a style reference: copy its drawing technique (line work, colours, shading, "
                      "rendering) but do NOT copy its people, animals, vehicles, objects, setting or composition, "
                      "and do not draw its frame or speech balloons.")

        figs = cr.sheet_figures(cast_list)
        sheet_pil = tensor_to_pils(char)[0] if char is not None else None
        fig_boxes = cr.sheet_figure_boxes(sheet_pil, len(figs)) if (sheet_pil is not None and figs) else None
        for i, pan in enumerate(panels):
            for l in pan["lines"]:
                l["speaker_raw"] = l["speaker"]
                l["speaker"], _v = cr.canonical_speaker(cast_list, l["speaker"])
            pan["variants"] = cr.panel_variants(cast_list, pan)
            pan["description"] = cr.strip_variant_tags(pan["description"])
            who = cr.cast_in_panel(cast_list, pan)
            pan["cast"] = [c["name"] for c in who]
            pan["stage"], pan["stage_text"] = (cr.stage_panel(pan, who, i) if do_stage else ({}, ""))

        if whole_page:
            guide = canvas_pil if canvas_pil is not None else cr.draw_layout_guide(layout)
            imgs, flags = [pil_to_tensor(guide), style], ["1", "0"]
            s_idx, c_idx = 2, None
            if char is not None:
                imgs.append(char); flags.append("1"); c_idx = 3
            p = [f"Create one complete comic book page. Picture 1 is the page layout: draw exactly {n} panels "
                 f"inside the black panel borders shown in Picture 1, keeping the white gutters between them."]
            if layout.get("dynamic"):
                p.append("The panel borders are slanted and some panels overlap as insets: follow their exact "
                         "shapes and angles from Picture 1.")
            if canvas_pil is not None:
                p.append("Picture 1 also contains rough placed reference images and text: keep them at the same "
                         "position and size, but redraw everything as finished artwork.")
            p.append(f"Picture {s_idx} {style_rule}")
            if c_idx:
                p.append(f"Picture {c_idx} is the character sheet: every character must look exactly as drawn there "
                         f"(same face, hair, clothing, colours and proportions) in every panel.")
            if notes:
                p.append(f"Style: {notes}.")
            all_members = []
            for pan in panels:
                for c in cr.cast_in_panel(cast_list, pan):
                    if c not in all_members:
                        all_members.append(c)
            if all_members:
                p.append("Characters (keep them identical in every panel): " + cr.cast_prompt(all_members) + ".")
                for pan_i, pan in enumerate(panels):
                    for nm, v in pan.get("variants", {}).items():
                        p.append(f"In panel {pan_i + 1}, {nm} looks different: {v['description'].rstrip('.')}"
                                 + (f" ({nm} has NO {v['without']} there)." if v["without"] else "."))
            p.append(f"Panels in reading order ({order}):")
            for i, pan in enumerate(panels):
                desc = pan["description"] or "continue the story"
                shot = f", {pan['shot']}" if pan.get("shot") else ""
                seg = f"Panel {i + 1} ({cr.describe_position(layout, i)}{shot}): {desc.rstrip('.')}."
                if pan["stage_text"]:
                    seg += " " + pan["stage_text"]
                elif pan["cast"]:
                    seg += " Shows " + ", ".join(pan["cast"]) + "."
                if ai_text and pan["lines"]:
                    seg += " Lettering: " + _dialogue_prompt(pan["lines"]) + "."
                p.append(seg)
            if not ai_text:
                p.append("Do not draw any speech balloons, captions or lettering; leave some empty space at the top of each panel for text.")
            else:
                p.append("All lettering must be spelled exactly as given, clean and legible.")
            prompts.append(" ".join(p))
            a, b, c, f = pack(imgs, flags)
            im1.append(a); im2.append(b); im3.append(c); flags_out.append(f)
            w, h = gen_size(layout["width"], layout["height"], megapixels)
            ws.append(w); hs.append(h)
        else:
            for i, pan in enumerate(panels):
                px, py, pw, ph = layout["panels"][i]
                crop_t = None
                if canvas_pil is not None:
                    r = cr.panel_rect(layout, i + 1, canvas_pil.size)
                    b = max(1, int(layout["border"] * canvas_pil.size[0] / layout["width"])) + 1
                    crop = canvas_pil.crop((r[0] + b, r[1] + b, r[0] + r[2] - b, r[1] + r[3] - b))
                    if not cr.is_blank(crop):
                        crop_t = pil_to_tensor(crop)
                imgs, flags = [], []
                if crop_t is not None:
                    imgs.append(crop_t); flags.append("1")
                imgs.append(style); flags.append("0")
                s_idx = len(imgs)
                c_idx = None
                cropped = False
                if char is not None and pan["cast"]:
                    ref = char
                    if fig_boxes:
                        # only the figures of THIS panel (and the right version), so nobody else leaks in
                        picks = []
                        for nm in pan["cast"]:
                            v = pan.get("variants", {}).get(nm)
                            k = next((j for j, f in enumerate(figs)
                                      if f[0] == nm and f[1] == (v["label"] if v else None)), None)
                            if k is not None:
                                picks.append(k)
                        if picks:
                            ref = pil_to_tensor(cr.sheet_subset(sheet_pil, fig_boxes, picks))
                            cropped = True
                    imgs.append(ref); flags.append("1")
                    c_idx = len(imgs)
                p = [f"Create a single comic book panel illustration (panel {i + 1} of {n})."]
                if crop_t is not None:
                    p.append("Picture 1 is a rough layout for this panel: keep the placed elements and any text "
                             "at the same positions and sizes, and redraw it all as finished, coherent artwork.")
                p.append(f"Picture {s_idx} {style_rule}")
                if c_idx and cropped:
                    names = " and ".join(pan["cast"])
                    p.append(f"Picture {c_idx} is the character reference for this panel: it shows {names}"
                             f"{' (left to right)' if len(pan['cast']) > 1 else ''}. Draw "
                             f"{'them' if len(pan['cast']) > 1 else 'this character'} exactly as they look there "
                             f"(same face, hair, clothing, colours and proportions), in a new pose that fits the scene; "
                             f"do not copy the reference pose or white background.")
                elif c_idx:
                    names = " and ".join(pan["cast"])
                    p.append(f"Picture {c_idx} is the character sheet: draw {names} exactly as they look there "
                             f"(same face, hair, clothing, colours and proportions). From the sheet, only {names} "
                             f"may appear in this panel; do not add the other characters.")
                    for nm, v in pan.get("variants", {}).items():
                        k = next((j for j, f in enumerate(figs) if f[0] == nm and f[1] == v["label"]), None)
                        b = next((j for j, f in enumerate(figs) if f[0] == nm and f[1] is None), None)
                        if k is not None and len(figs) > 1:
                            p.append(f"In this panel {nm} must look like figure {k + 1} from the left on the sheet "
                                     f"({v['label']}), not like figure {b + 1}.")
                    if len(figs) > len(cast_list):
                        p.append("Some characters appear twice on the sheet (two versions); draw each character only once.")
                who = cr.cast_in_panel(cast_list, pan)
                if who:
                    p.append("Characters in this panel: " + cr.cast_prompt(who, pan.get("variants")) + ".")
                for nm, v in pan.get("variants", {}).items():
                    if v["without"]:
                        p.append(f"Important: {nm} has NO {v['without']} on its head or body in this panel "
                                 f"(it looks like the reference: {v['description'].rstrip('.')}).")
                others = [c["name"] for c in cast_list if c["name"] not in pan["cast"]]
                if others and cast_list:
                    p.append(f"{' and '.join(others)} {'is' if len(others) == 1 else 'are'} NOT in this panel; do not draw "
                             f"{'them' if len(others) > 1 else 'this character'} and do not give anyone their clothes or colours.")
                if pan.get("shot"):
                    p.append(f"Camera / composition: {pan['shot'].rstrip('.')}.")
                p.append(f"Scene: {(pan['description'] or 'continue the story').rstrip('.')}.")
                if pan["cast"]:
                    cnt = len(pan["cast"])
                    p.append(f"There {'is exactly ONE character' if cnt == 1 else f'are exactly {cnt} characters'} "
                             f"from the sheet in this panel ({', '.join(pan['cast'])}); each of them appears only once, "
                             f"never twice.")
                if pan["lines"] and not ai_text:
                    p.append("Compose the shot so the characters' heads are in the middle or lower part of the image "
                             "and the top quarter of the image is open background (sky, wall, ceiling).")
                if pan["stage_text"]:
                    p.append(pan["stage_text"])
                if layout.get("dynamic"):
                    p.append("Keep the main subject and faces well inside the centre of the image, with some "
                             "extra background around the edges. Do not draw any borders, frames or panel lines.")
                if notes:
                    p.append(f"Style: {notes}.")
                if ai_text and pan["lines"]:
                    p.append("Lettering: " + _dialogue_prompt(pan["lines"]) + ". Spell all text exactly, clean and legible.")
                else:
                    p.append("Do not draw any speech balloons, bubbles, captions, text or signs with writing.")
                p.append("Full-bleed artwork, no panel border, no frame, no watermark.")
                prompts.append(" ".join(p))
                a, b, c, f = pack(imgs, flags)
                im1.append(a); im2.append(b); im3.append(c); flags_out.append(f)
                w, h = gen_size(pw, ph, megapixels)
                ws.append(w); hs.append(h)

        plan = {"layout": copy.deepcopy(layout), "panels": panels, "whole_page": bool(whole_page),
                "lettering": lettering, "canvas_used": canvas_pil is not None,
                "cast_names": [c["name"] for c in cast_list],
                "cast_desc": {c["name"]: c["description"] for c in cast_list},
                "sheet_owners": [f[0] for f in figs]}
        info = [f"mode: {'whole page' if whole_page else 'per panel'} | generations: {len(prompts)}"]
        info += [f"[{i + 1}] {w}x{h} refs({fl}): {p}" for i, (w, h, p, fl) in enumerate(zip(ws, hs, prompts, flags_out))]
        info += ["WARNING: " + w for w in warnings]
        return (plan, prompts, im1, im2, im3, ws, hs, "\n\n".join(info), flags_out)


class ComicQwenEncode:
    """Text Encode Qwen Image Edit Plus with per-image reference control.

    pixel_refs: comma list per image ('1' = pixel reference via VAE + vision, '0' = vision only).
    The Director fills it in: style image = vision only (no content copying), character sheet and
    layouts = pixel reference (so characters stay identical)."""

    MODES = ["auto (use Director's choice)", "loose: nothing as pixel reference",
             "strict: everything as pixel reference"]

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "clip": ("CLIP",),
            "prompt": ("STRING", {"multiline": True, "default": ""}),
            "reference_mode": (cls.MODES,),
            "pixel_refs": ("STRING", {"default": "", "tooltip": "e.g. '0,1'. Empty = every image is a pixel reference"}),
        }, "optional": {
            "vae": ("VAE",),
            "image1": ("IMAGE",),
            "image2": ("IMAGE",),
            "image3": ("IMAGE",),
        }}

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "run"
    CATEGORY = CAT

    TEMPLATE = ("<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, "
                "objects, background), then explain how the user's text instruction should alter or modify the "
                "image. Generate a new image that meets the user's requirements while maintaining consistency "
                "with the original input where appropriate.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n"
                "<|im_start|>assistant\n")

    def run(self, clip, prompt, reference_mode, pixel_refs, vae=None, image1=None, image2=None, image3=None):
        import comfy.utils
        import node_helpers
        images = [image1, image2, image3]
        flags = [f.strip() for f in str(pixel_refs).split(",")] if str(pixel_refs).strip() else []
        ref_latents, images_vl, image_prompt = [], [], ""
        for i, image in enumerate(images):
            if image is None:
                continue
            samples = image.movedim(-1, 1)
            scale_by = math.sqrt(384 * 384 / (samples.shape[3] * samples.shape[2]))
            s = comfy.utils.common_upscale(samples, round(samples.shape[3] * scale_by),
                                           round(samples.shape[2] * scale_by), "area", "disabled")
            images_vl.append(s.movedim(1, -1))
            if reference_mode.startswith("strict"):
                use_px = True
            elif reference_mode.startswith("loose"):
                use_px = False
            else:
                use_px = (flags[i] != "0") if i < len(flags) else True
            if vae is not None and use_px:
                scale_by = math.sqrt(1024 * 1024 / (samples.shape[3] * samples.shape[2]))
                w = round(samples.shape[3] * scale_by / 8.0) * 8
                h = round(samples.shape[2] * scale_by / 8.0) * 8
                s = comfy.utils.common_upscale(samples, w, h, "area", "disabled")
                ref_latents.append(vae.encode(s.movedim(1, -1)[:, :, :, :3]))
            image_prompt += "Picture {}: <|vision_start|><|image_pad|><|vision_end|>".format(i + 1)
        tokens = clip.tokenize(image_prompt + prompt, images=images_vl, llama_template=self.TEMPLATE)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        if ref_latents:
            conditioning = node_helpers.conditioning_set_values(conditioning, {"reference_latents": ref_latents},
                                                                append=True)
        return (conditioning,)


class ComicCharacterSheet:
    """Prompt + size for a one-time character reference sheet made from the CAST list."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "cast": ("STRING", {"multiline": True, "default": ""}),
            "style_notes": ("STRING", {"default": "comic book art, clean ink lines"}),
            "pose": (["front view, neutral standing pose", "three-quarter view, relaxed pose",
                      "front and side view (turnaround)"],),
        }}

    RETURN_TYPES = ("STRING", "INT", "INT")
    RETURN_NAMES = ("prompt", "width", "height")
    FUNCTION = "run"
    CATEGORY = CAT

    def run(self, cast, style_notes, pose):
        figs = cr.sheet_figures(cr.parse_cast(cast))
        n = max(1, len(figs))
        who = "; ".join(f"{i + 1}) {nm}: {d.rstrip('.')}" if lab is None else
                        f"{i + 1}) {nm} again, the version '{lab}': {d.rstrip('.')}"
                        for i, (nm, lab, d) in enumerate(figs)) or "the main character"
        p = (f"A clean character reference sheet for a comic book. Draw {'these ' + str(n) + ' characters side by side' if n > 1 else 'this character'}, "
             f"full body from head to feet, {pose}, evenly spaced and not overlapping, all at a clear readable size, "
             f"on a plain white background: {who}. "
             f"Picture 1 is ONLY a style reference: copy its drawing technique (line work, colours, shading) but not "
             f"its content. Style: {style_notes.strip().rstrip('.')}. No text, no names, no labels, no frames, no background scenery.")
        w, h = {1: (832, 1216), 2: (1344, 768), 3: (1536, 704), 4: (1664, 640)}.get(n, (1792, 576))
        return (p, w, h)
class ComicVisionLoader:
    """Loads Qwen3.5 (vision-language) for the panel check / story writer. 'fp8' halves the memory so it
    fits next to nothing else on an 8 GB GPU and generates many times faster than bf16 with offloading."""

    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        return {"required": {
            "clip_name": (folder_paths.get_filename_list("text_encoders"), {"default": "qwen3.5_2b_bf16.safetensors",
                          "tooltip": "qwen3.5_2b = fast check on 8 GB GPUs, qwen3.5_4b = more accurate"}),
            "weight_dtype": (["default", "fp8_e4m3fn"],),
        }}

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "run"
    CATEGORY = CAT

    def run(self, clip_name, weight_dtype):
        import folder_paths
        import comfy.sd
        path = folder_paths.get_full_path_or_raise("text_encoders", clip_name)
        opts = {}
        if weight_dtype.startswith("fp8"):   # note: only lowers memory if ComfyUI casts on load
            opts["dtype"] = torch.float8_e4m3fn
        clip = comfy.sd.load_clip(ckpt_paths=[path], embedding_directory=folder_paths.get_folder_paths("embeddings"),
                                  clip_type=comfy.sd.CLIPType.STABLE_DIFFUSION, model_options=opts)
        return (clip,)


class ComicPanelSampler:
    """KSampler + VAE decode for every panel, with an optional VISION CHECK (Qwen3.5 VL via a CLIP loader):
    each panel is inspected, and panels where a character is duplicated, missing or doesn't belong are
    redrawn with a new seed (up to max_redraws). The character/face boxes are passed on to the lettering."""

    INPUT_IS_LIST = True

    @classmethod
    def INPUT_TYPES(cls):
        import comfy.samplers
        return {"required": {
            "model": ("MODEL",),
            "vae": ("VAE",),
            "positive": ("CONDITIONING",),
            "negative": ("CONDITIONING",),
            "latent_image": ("LATENT",),
            "plan": ("COMIC_PLAN",),
            "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
            "steps": ("INT", {"default": 4, "min": 1, "max": 200}),
            "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 30.0, "step": 0.1}),
            "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler"}),
            "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "simple"}),
            "check_panels": ("BOOLEAN", {"default": True, "label_on": "vision check ON", "label_off": "vision check OFF"}),
            "max_redraws": ("INT", {"default": 2, "min": 0, "max": 6, "tooltip": "extra attempts for a bad panel"}),
        }, "optional": {
            "checker": ("CLIP", {"tooltip": "Comic Vision Model Loader (Qwen3.5 2B/4B)"}),
            "clip": ("CLIP", {"tooltip": "Qwen-Image text encoder. Connected = AUTO-FIX: duplicates, strangers and "
                                         "forbidden items are removed from a panel with an edit pass"}),
        }}

    RETURN_TYPES = ("IMAGE", "COMIC_DETECT", "STRING")
    RETURN_NAMES = ("images", "detections", "report")
    OUTPUT_IS_LIST = (True, False, False)
    FUNCTION = "run"
    CATEGORY = CAT

    @staticmethod
    def _sample(model, vae, seed, steps, cfg, sampler_name, scheduler, pos, neg, latent):
        import nodes as comfy_nodes
        out = comfy_nodes.common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, pos, neg, latent,
                                          denoise=1.0)[0]
        img = vae.decode(out["samples"])
        if img.ndim == 5:
            img = img.reshape(-1, *img.shape[-3:])
        return img[:1]

    @staticmethod
    def _square(pil, S=768):
        # Always the SAME input size (letterboxed square): ComfyUI's LLM cache is sized by the first call,
        # a longer image token sequence on a later call crashes CUDA.
        sc = S / float(max(pil.size))
        w, h = max(1, int(pil.width * sc)), max(1, int(pil.height * sc))
        box = Image.new("RGB", (S, S), (255, 255, 255))
        ox, oy = (S - w) // 2, (S - h) // 2
        box.paste(pil.resize((w, h), Image.LANCZOS), (ox, oy))
        return box, ox, oy, w, h

    @staticmethod
    def _ask(checker, prompt, box, max_length):
        tokens = checker.tokenize(prompt, image=pil_to_tensor(box), min_length=1, thinking=False)
        kw = dict(do_sample=False, max_length=max_length, temperature=1.0, top_k=50, top_p=1.0, min_p=0.0,
                  repetition_penalty=1.05, seed=0)

        def _reset():
            # ComfyUI keeps a per-thread CUDA allocation graph between LLM calls; several generate() calls
            # inside ONE node execution break it ("aimdo memory compile error"). Reset it around each call.
            try:
                import comfy.model_prefetch as mp
                if hasattr(mp, "cleanup_malloc_graph"):
                    mp.cleanup_malloc_graph()
            except Exception:
                pass

        def _gen():
            try:
                return checker.generate(tokens, presence_penalty=0.0, mtp=False, **kw)
            except TypeError:
                return checker.generate(tokens, **kw)

        _reset()
        try:
            ids = _gen()
        except RuntimeError as e:
            print(f"[QwenComic] vision check retry after: {e}")
            _reset()
            ids = _gen()
        _reset()
        return checker.decode(ids)

    @classmethod
    def _inspect(cls, checker, img, plan, i):
        pan = plan["panels"][i]
        names = plan.get("cast_names", [])
        pil = tensor_to_pils(img)[0]
        S = 768
        box, ox, oy, w, h = cls._square(pil, S)
        desc = dict(plan.get("cast_desc", {}))
        for nm, v in pan.get("variants", {}).items():
            desc[nm] = v["description"]
        text = cls._ask(checker, cr.vlm_check_prompt(desc, pan.get("cast", [])), box, 320)
        dets = cr.parse_vlm_boxes(text, S, S, names)
        unreliable = cr.vlm_output_unreliable(text, dets)
        ftext = cls._ask(checker, cr.vlm_face_prompt(), box, 260)
        faces = [d["box"] for d in cr.parse_vlm_boxes(ftext, S, S, [])]

        # undo the letterbox: square coords -> panel coords
        def unbox(b):
            if b is None:
                return None
            x0, y0, x1, y1 = [(b[0] * S - ox) / w, (b[1] * S - oy) / h, (b[2] * S - ox) / w, (b[3] * S - oy) / h]
            x0, x1 = max(0.0, min(1.0, x0)), max(0.0, min(1.0, x1))
            y0, y1 = max(0.0, min(1.0, y0)), max(0.0, min(1.0, y1))
            return (x0, y0, x1, y1) if x1 - x0 > 0.01 and y1 - y0 > 0.01 else None
        dets = [dict(d, box=unbox(d["box"]), face=None) for d in dets]
        dets = cr.dedupe_dets([d for d in dets if d["box"]])
        faces = [f for f in (unbox(f) for f in faces) if f]
        faces = [f for k, f in enumerate(faces) if not any(cr._iou(f, g) > 0.7 for g in faces[:k])]
        dets, extra = cr.attach_faces(dets, faces)

        def crop_of(bx, m=0.2):
            x0, y0, x1, y1 = bx
            mx, my = (x1 - x0) * m, (y1 - y0) * m
            return pil.crop((int(max(0, x0 - mx) * pil.width), int(max(0, y0 - my) * pil.height),
                             int(min(1, x1 + mx) * pil.width), int(min(1, y1 + my) * pil.height)))

        # a character that should NOT be here: double-check that crop (small models mislabel birds, strangers)
        expected = pan.get("cast", [])
        for d in dets:
            if d["label"] in names and d["label"] not in expected:
                q = cr.vlm_identity_prompt(d["label"], desc.get(d["label"], ""))
                ans = cls._ask(checker, q, cls._square(crop_of(d["box"]), S)[0], 20)
                text += f"\n[is it {d['label']}? {ans.strip()[:30]}]"
                if not cr.parse_yes_no(ans):
                    d["label"] = "other"
        # an unknown PERSON that the scene doesn't ask for (random bystander, a hand from nowhere)
        if not cr.scene_allows_people(pan.get("description", "")):
            for d in dets:
                if d["label"] == "other":
                    ans = cls._ask(checker, cr.vlm_person_prompt(), cls._square(crop_of(d["box"], 0.1), S)[0], 20)
                    text += f"\n[stranger? {ans.strip()[:30]}]"
                    if cr.parse_yes_no(ans):
                        d["stranger"] = True
        dets += [{"label": "other", "box": f, "face": f} for f in extra]
        err, notes = cr.judge_panel(dets, expected, names, unreliable)
        n_str = sum(1 for d in dets if d.get("stranger"))
        if n_str and not unreliable:
            err += 1.0 * n_str
            notes.append(f"{n_str} unknown person{'s' if n_str > 1 else ''}")
        fixes = [] if unreliable else cr.fix_instructions(dets, pan, plan.get("cast_desc", {}), names)
        text += "\n[heads] " + ftext.strip()[:200]

        # variants: 'Bolt [no bulb]' -> ask whether the forbidden feature is still there
        for nm, v in pan.get("variants", {}).items():
            if not v.get("without"):
                continue
            mine = [d["box"] for d in dets if d["label"] == nm]
            crop = pil
            if mine:
                x0, y0, x1, y1 = max(mine, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
                mx, my = (x1 - x0) * 0.08, (y1 - y0) * 0.08
                crop = pil.crop((int(max(0, x0 - mx) * pil.width), int(max(0, y0 - my) * pil.height),
                                 int(min(1, x1 + mx) * pil.width), int(min(1, y1 + my) * pil.height)))
            q = cr.vlm_feature_prompt(nm, plan.get("cast_desc", {}).get(nm, ""), v["without"])
            ans = cls._ask(checker, q, cls._square(crop, S)[0], 40)
            has = cr.parse_yes_no(ans)
            text += f"\n[{nm} has {v['without']}? {ans.strip()[:40]}]"
            if has:
                err += 1.5
                notes.append(f"{nm} still has {v['without']}")
                fixes.append(f"remove the {v['without']} from the top of {nm} ({cr.short_desc(desc.get(nm, ''))}), "
                             f"so that it looks like this: {v['description'].rstrip('.')}")
        # is there free space for this panel's balloons?
        talk = [l for l in pan.get("lines", []) if l["style"] != "sfx"]
        if any(l["style"] != "caption" for l in talk) and not unreliable \
                and plan.get("lettering", "").startswith(("overlay", "both")):
            try:
                st, hd, bd = cr.stage_from_vlm(dets)
                lay = plan["layout"]
                bx, by, bw, bh = lay["panels"][i]
                sxp, syp = pil.width / float(bw), pil.height / float(bh)
                poly = [((x - bx) * sxp, (y - by) * syp) for x, y in cr.panel_poly(lay, i)]
                blocked = [[((x - bx) * sxp, (y - by) * syp) for x, y in cr.panel_poly(lay, j)]
                           for j in range(len(lay["panels"])) if j != i]
                fs = int(plan.get("font_size_hint", 30) * sxp)
                fit = cr.balloon_fit(pil, talk, "ComicNeue-Bold.ttf", fs, st, hd, bd, poly, blocked)
                text += f"\n[balloon room: {fit:.2f}]"
                if fit > 0.35:
                    err += 0.4
                    notes.append("no free space for the balloons")
            except Exception as e:
                text += f"\n[balloon room check failed: {e}]"
        return dets, err, notes, text, fixes

    @staticmethod
    def _edit(clip, model, vae, img, instruction, seed, steps, cfg, sampler_name, scheduler):
        """Fix a panel with Qwen-Image-Edit itself: the panel is the pixel reference, the prompt says what to remove."""
        import node_helpers
        import comfy.model_management
        prompt = ("Edit this comic panel: " + instruction.rstrip(".") + ". Keep everything else exactly the same: "
                  "the same characters, poses, faces, background, colours, line art and drawing style. "
                  "Do not add anything new, no text, no speech balloons.")
        enc = ComicQwenEncode()
        pos = enc.run(clip, prompt, ComicQwenEncode.MODES[0], "1", vae=vae, image1=img)[0]
        neg = enc.run(clip, "", ComicQwenEncode.MODES[0], "1", vae=vae, image1=img)[0]
        pos = node_helpers.conditioning_set_values(pos, {"reference_latents_method": "index_timestep_zero"})
        neg = node_helpers.conditioning_set_values(neg, {"reference_latents_method": "index_timestep_zero"})
        h, w = img.shape[1], img.shape[2]
        lat = {"samples": torch.zeros([1, 16, h // 8, w // 8], device=comfy.model_management.intermediate_device())}
        out = ComicPanelSampler._sample(model, vae, seed, steps, cfg, sampler_name, scheduler, pos, neg, lat)
        if out.shape[1:3] != img.shape[1:3]:
            out = torch.nn.functional.interpolate(out.movedim(-1, 1), size=(h, w), mode="bilinear").movedim(1, -1)
        return out

    def run(self, model, vae, positive, negative, latent_image, plan, seed, steps, cfg, sampler_name, scheduler,
            check_panels, max_redraws, checker=None, clip=None):
        model, vae, plan = model[0], vae[0], plan[0]
        clip = clip[0] if clip else None
        seed, steps, cfg = seed[0], steps[0], cfg[0]
        sampler_name, scheduler = sampler_name[0], scheduler[0]
        check, redraws = check_panels[0], max_redraws[0]
        checker = checker[0] if checker else None
        print(f"[QwenComic] panel sampler: {len(positive)} panels, check={check}, checker={'yes' if checker else 'no'}, "
              f"cast={plan.get('cast_names')}")
        n = len(positive)
        L = lambda lst, i: lst[min(i, len(lst) - 1)]
        args = lambda i, k: (model, vae, seed + k * 7919, steps, cfg, sampler_name, scheduler,
                             L(positive, i), L(negative, i), L(latent_image, i))
        imgs = [self._sample(*args(i, 0)) for i in range(n)]
        if not check or checker is None or plan.get("whole_page") or not plan.get("cast_names"):
            why = ("check_panels is off" if not check else "no vision model connected" if checker is None else
                   "whole-page mode" if plan.get("whole_page") else "no cast")
            return (imgs, {}, f"vision check skipped: {why}")
        report, best = [], {}
        cands = {i: imgs[i] for i in range(n)}
        for attempt in range(redraws + 1):
            if attempt > 0:
                failed = [i for i in best if best[i]["err"] > 0]
                if not failed:
                    break
                # redraw all failed panels first, then check them (fewer model swaps in VRAM)
                cands = {i: self._sample(*args(i, attempt)) for i in failed}
            for i, cand in cands.items():
                try:
                    dets, err, notes, raw, fixes = self._inspect(checker, cand, plan, i)
                except Exception as e:   # never lose the page because the checker failed
                    dets, err, notes, raw, fixes = [], 0.0, [], f"check failed: {e}", []
                report.append(f"panel {i + 1} try {attempt + 1}: {'OK' if err == 0 else ', '.join(notes)}"
                              f" | {raw.strip()[:300]}")
                if i not in best or err < best[i]["err"]:
                    best[i] = {"err": err, "dets": dets, "img": cand}
                # AUTO-FIX: let Qwen-Image-Edit remove what doesn't belong, then check again
                if clip is not None and fixes and err > 0:
                    instr = "; ".join(fixes)
                    try:
                        fixed = self._edit(clip, model, vae, cand, instr, seed + 101 + attempt, steps, cfg,
                                           sampler_name, scheduler)
                        dets2, err2, notes2, raw2, _f = self._inspect(checker, fixed, plan, i)
                        report.append(f"panel {i + 1} try {attempt + 1} auto-fix ({instr[:160]}): "
                                      f"{'OK' if err2 == 0 else ', '.join(notes2)}")
                        if err2 < best[i]["err"]:
                            best[i] = {"err": err2, "dets": dets2, "img": fixed}
                    except Exception as e:
                        report.append(f"panel {i + 1} auto-fix failed: {e}")
        out = [best[i]["img"] if i in best else imgs[i] for i in range(n)]
        detections = {i: best[i]["dets"] for i in best}
        return (out, detections, "\n".join(report))


class ComicAssemblePage:
    INPUT_IS_LIST = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "plan": ("COMIC_PLAN",),
            "images": ("IMAGE",),
            "redraw_borders_whole_page": ("BOOLEAN", {"default": False,
                                                      "tooltip": "Whole-page mode only: draw the exact layout borders over the AI page"}),
        }}

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("page", "panels")
    FUNCTION = "run"
    CATEGORY = CAT

    def run(self, plan, images, redraw_borders_whole_page):
        plan = plan[0]
        redraw = redraw_borders_whole_page[0]
        pils = []
        for t in images:
            pils.extend(tensor_to_pils(t))
        lay = plan["layout"]
        if plan["whole_page"]:
            page = cr.fit_whole_page(lay, pils[0], redraw)
            panels = [page]
        else:
            page = cr.assemble_panels(lay, pils)
            panels = pils
        # panels may have different sizes -> resize to first for batch output
        ref = panels[0].size
        panels_t = pils_to_tensor([p.resize(ref) for p in panels])
        return (pil_to_tensor(page), panels_t)


class ComicAutoLetter:
    """Draws the script dialogue as bubbles on the finished page (when lettering uses overlay)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "page": ("IMAGE",),
            "plan": ("COMIC_PLAN",),
            "font": (cr.list_fonts(), {"default": "ComicNeue-Bold.ttf"}),
            "font_size": ("INT", {"default": 30, "min": 6, "max": 300}),
            "placement": (["smart: near the speaker", "classic: along the top"], {
                "tooltip": "smart = balloon above/next to whoever speaks, in a calm area, tail pointing at them"}),
        }, "optional": {
            "detections": ("COMIC_DETECT", {"tooltip": "From Comic Panel Sampler (vision check): exact character and face boxes"}),
            "character_sheet": ("IMAGE", {"tooltip": "Connect the character sheet: characters are then FOUND in each "
                                                     "panel by their colours, so tails point at the right person and "
                                                     "balloons avoid faces"}),
            "adjust": ("STRING", {"multiline": True, "default": "",
                                  "tooltip": "Move a balloon by hand, one per line: 'panel.balloon: x=0.8 y=0.15' "
                                             "(centre, 0-1 inside the panel). Also size=0.9 (text size), width=0.5, "
                                             "tail_x / tail_y. See the balloon list output for the numbers."}),
        }}

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("page", "balloon list")
    FUNCTION = "run"
    CATEGORY = CAT

    def run(self, page, plan, font, font_size, placement="smart: near the speaker", character_sheet=None,
            detections=None, adjust=""):
        if plan["lettering"] not in ("overlay bubbles (crisp text)", "both"):
            return (page, "lettering is not 'overlay': no balloons drawn")
        adj_all = cr.parse_balloon_adjust(adjust)
        listing = []
        lay = plan["layout"]
        names = plan.get("cast_names", [])
        owners = plan.get("sheet_owners") or names
        palettes = cr.character_palettes(tensor_to_pils(character_sheet[:1])[0], len(owners), owners)[:len(names)] \
            if (character_sheet is not None and names) else []
        out = []
        for img in tensor_to_pils(page):
            img = img.copy()
            scale = img.size[0] / float(lay["width"])
            fs = max(6, int(font_size * scale))
            for i, pan in enumerate(plan["panels"]):
                if not pan["lines"]:
                    continue
                safe = cr.panel_safe_rect(lay, i + 1, img.size)
                if placement.startswith("smart"):
                    rect = cr.panel_rect(lay, i + 1, img.size)
                    stage, heads = (pan.get("stage") or {}) if not plan.get("whole_page") else {}, None
                    bodies = None
                    det = (detections or {}).get(i) if not plan.get("whole_page") else None
                    if det:
                        stage, heads, bodies = cr.stage_from_vlm(det)
                    elif palettes and pan.get("cast"):
                        crop = img.crop((rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3]))
                        idx = [names.index(n) for n in pan["cast"] if n in names]
                        found = cr.locate_characters(crop, [names[k] for k in idx], [palettes[k] for k in idx])
                        if found:
                            det_stage, heads = cr.stage_from_detection(found)
                            stage = {**stage, **det_stage}
                    print(f"[QwenComic] letter panel {i + 1}: {'vision boxes' if det else 'NO vision boxes'} "
                          f"(detections for panels {sorted((detections or {}).keys())}), stage={stage}")
                    sxp, syp = img.size[0] / float(lay["width"]), img.size[1] / float(lay["height"])
                    poly = [(x * sxp, y * syp) for x, y in cr.panel_poly(lay, i)]
                    blocked = [[(x * sxp, y * syp) for x, y in cr.panel_poly(lay, j)]
                               for j in range(len(lay["panels"])) if j != i]
                    adj = {b: v for (p_, b), v in adj_all.items() if p_ == i + 1}
                    plans = cr.smart_letter_panel(img, rect, safe, pan["lines"], font, fs, stage, heads, bodies, poly,
                                                  blocked, adj) or []
                    if len(out) == 0:
                        for pl in plans:
                            rx_ = (pl["cx"] - rect[0]) / max(1, rect[2]); ry_ = (pl["cy"] - rect[1]) / max(1, rect[3])
                            note = " (moved by hand)" if pl.get("manual") else (
                                " - covers a character a little" if pl["head_cov"] + pl["body_cov"] > 0.15 else "")
                            listing.append(f"{i + 1}.{pl['number']}  {pl['line']['speaker']}: "
                                           f"\"{pl['line']['text'][:40]}\"  x={rx_:.2f} y={ry_:.2f} "
                                           f"size={pl['fs'] / float(fs):.1f}{note}")
                else:
                    cr.auto_letter_panel(img, safe, pan["lines"], font, fs)
            out.append(img)
        listing = ["Balloon numbers (panel.balloon). To move one, type e.g. '" +
                   (listing[0].split()[0] if listing else "3.1") + ": x=0.7 y=0.15' in 'adjust'.", ""] + listing
        return (pils_to_tensor(out), "\n".join(listing))

class ComicReferenceMode:
    """Decides how strongly Qwen looks at the reference images.

    With the VAE connected, Qwen-Image-Edit gets the reference *pixels* and tends to copy the style image's
    people/objects into every panel. Without it, it only 'sees' the images through the vision encoder:
    the style carries over, the content does not. Layouts (whole page / control canvas) need the pixels."""

    MODES = ["auto (recommended)", "loose: style only, no copying", "strict: pixel reference (may copy content)"]

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "vae": ("VAE",),
            "plan": ("COMIC_PLAN",),
            "mode": (cls.MODES,),
        }}

    RETURN_TYPES = ("VAE",)
    RETURN_NAMES = ("vae_or_none",)
    FUNCTION = "run"
    CATEGORY = CAT

    def run(self, vae, plan, mode):
        if mode.startswith("strict"):
            return (vae,)
        if mode.startswith("loose"):
            return (None,)
        needs_layout = plan.get("whole_page") or plan.get("canvas_used")
        return (vae if needs_layout else None,)


STORY_SYSTEM = """You are an experienced comic book writer and storyboard artist.
You turn a story into a script for ONE comic page. Output ONLY the script, no introduction, no markdown, no bold.

Format (follow exactly):
Panel 1 (<shot type / camera angle>): <visual description>
Name: <dialogue>
Panel 2 (<shot type / camera angle>): <visual description>
Caption: <narration>
Name (thought): <inner thought>
Name (shout): <shouted line>
SFX: <sound effect>

Rules:
- Write EXACTLY {n} panels, numbered 1 to {n}, in reading order.
- In the brackets, choose a varied, cinematic shot for every panel (e.g. wide establishing shot, low angle, bird's-eye view, over-the-shoulder, extreme close-up, dutch angle). Never use the same shot twice in a row.
- The visual description is ONE line and must stand on its own (an image generator reads each panel separately): which characters are visible (use their names), what they do, facial expression, setting, time of day and lighting.
- Always refer to characters by the same name. Do not describe their looks; that is handled separately.
- Match the shot to the panel shape: wide panels suit establishing and action shots, tall panels suit full-body or dramatic moments, small panels suit close-ups and reactions.
- Pace the story so it has a beginning, a turn and an ending (or cliffhanger) on this page.
- Dialogue and captions: at most {max_lines} lines per panel, each at most {max_words} words. Some panels may have no text. Write all dialogue, captions and SFX in {language}.
- Descriptions are always in English."""


class ComicStoryWriter:
    """Builds the instruction for a local LLM (core 'Generate Text' node) that turns a story into a panel script."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "layout": ("COMIC_LAYOUT",),
            "story": ("STRING", {"multiline": True, "default":
                "In a rainy future Kortrijk, Mia finds the streets suddenly empty. A small robot called Bolt "
                "appears and tells her to follow him. They run through a glowing alley and discover that "
                "everyone is hiding from a giant hologram that has come to life."}),
            "dialogue_language": ("STRING", {"default": "English"}),
            "max_lines_per_panel": ("INT", {"default": 2, "min": 0, "max": 6}),
            "max_words_per_line": ("INT", {"default": 14, "min": 3, "max": 40}),
            "extra_instructions": ("STRING", {"multiline": True, "default": "",
                                              "tooltip": "e.g. 'funny tone', 'end on a cliffhanger', 'no dialogue in panel 1'"}),
        }, "optional": {
            "cast": ("STRING", {"forceInput": True, "tooltip": "Same cast text as the Director, so the writer uses the right names"}),
        }}

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("prompt", "system_prompt")
    FUNCTION = "run"
    CATEGORY = CAT

    def run(self, layout, story, dialogue_language, max_lines_per_panel, max_words_per_line,
            extra_instructions, cast=""):
        n = len(layout["panels"])
        system = STORY_SYSTEM.format(n=n, max_lines=max_lines_per_panel, max_words=max_words_per_line,
                                     language=dialogue_language.strip() or "English")
        shapes = "\n".join(f"Panel {i + 1}: {cr.describe_position(layout, i)}"
                           f"{'' if (w > 1.6 * h or h > 1.6 * w) else ', square-ish'}"
                           f" ({round(100 * w * h / (layout['width'] * layout['height']))}% of the page)"
                           for i, (x, y, w, h) in enumerate(layout["panels"]))
        parts = [f"Page layout ({n} panels):\n{shapes}"]
        members = cr.parse_cast(cast)
        if members:
            parts.append("Characters (use exactly these names):\n" + "\n".join(f"- {c['name']}" for c in members))
        parts.append("Story:\n" + story.strip())
        if extra_instructions.strip():
            parts.append("Extra instructions: " + extra_instructions.strip())
        parts.append(f"Now write the script with exactly {n} panels.")
        return ("\n\n".join(parts), system)


class ComicScriptSelect:
    """Choose between your own script and the AI-written one. 'review first' stops after writing
    the script so you can read it before any image is generated."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "source": (["my script (below)", "AI script from story"],),
            "review_first": ("BOOLEAN", {"default": False, "label_on": "stop after script (review)",
                                         "label_off": "continue to images"}),
            "my_script": ("STRING", {"multiline": True, "default":
                "Panel 1: Wide establishing shot of a rainy neon city street at night, empty and quiet.\n"
                "Caption: Kortrijk, 2089.\n\n"
                "Panel 2: Medium shot, Mia looks up at a giant flickering hologram, worried.\n"
                "Mia: Where did everyone go?\n\n"
                "Panel 3: Close-up, Bolt tugs at Mia's sleeve, eyes glowing.\n"
                "Bolt: Follow me. Quickly!\n\n"
                "Panel 4: Mia and Bolt run into a glowing alley, puddles splashing.\n"
                "SFX: SPLASH\n"}),
        }, "optional": {
            "ai_script": ("STRING", {"forceInput": True, "lazy": True}),
        }}

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("script", "script_preview")
    FUNCTION = "run"
    CATEGORY = CAT

    def check_lazy_status(self, source, ai_script=None, **kwargs):
        if source.startswith("AI") and ai_script is None:
            return ["ai_script"]
        return []

    def run(self, source, review_first, my_script, ai_script=None):
        if source.startswith("AI"):
            text = ai_script or ""
            text = text.split("</think>")[-1]
            lines = [cr.clean_line(l) if cr.PANEL_RE.match(cr.clean_line(l)) else l.rstrip()
                     for l in text.strip().splitlines()]
            script = "\n".join(l.replace("**", "") for l in lines).strip()
            # drop chatter before the first panel ("Sure! Here is the script:")
            out = script.splitlines()
            first = next((i for i, l in enumerate(out) if cr.PANEL_RE.match(cr.clean_line(l))), 0)
            script = "\n".join(out[first:]).strip()
        else:
            script = my_script
        if review_first:
            from comfy_execution.graph_utils import ExecutionBlocker
            return (ExecutionBlocker(None), script)
        return (script, script)


NODE_CLASS_MAPPINGS = {
    "ComicPageLayout": ComicPageLayout,
    "ComicControlCanvas": ComicControlCanvas,
    "ComicPlaceImage": ComicPlaceImage,
    "ComicTextBubble": ComicTextBubble,
    "ComicDirector": ComicDirector,
    "ComicAssemblePage": ComicAssemblePage,
    "ComicAutoLetter": ComicAutoLetter,
    "ComicStoryWriter": ComicStoryWriter,
    "ComicScriptSelect": ComicScriptSelect,
    "ComicReferenceMode": ComicReferenceMode,
    "ComicQwenEncode": ComicQwenEncode,
    "ComicCharacterSheet": ComicCharacterSheet,
    "ComicPanelSampler": ComicPanelSampler,
    "ComicVisionLoader": ComicVisionLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ComicPageLayout": "Comic Page Layout",
    "ComicControlCanvas": "Comic Control Canvas",
    "ComicPlaceImage": "Comic Place Image (at position)",
    "ComicTextBubble": "Comic Text / Bubble (at position)",
    "ComicDirector": "Comic Director (script -> prompts)",
    "ComicAssemblePage": "Comic Assemble Page",
    "ComicAutoLetter": "Comic Auto Letter (script bubbles)",
    "ComicStoryWriter": "Comic Story Writer (story -> LLM instruction)",
    "ComicScriptSelect": "Comic Script Select (mine / AI + review)",
    "ComicReferenceMode": "Comic Reference Mode (style only / strict)",
    "ComicQwenEncode": "Comic Qwen Encode (per-image references)",
    "ComicCharacterSheet": "Comic Character Sheet (from cast)",
    "ComicPanelSampler": "Comic Panel Sampler (+ vision check & redraw)",
    "ComicVisionLoader": "Comic Vision Model Loader (Qwen3.5, fp8)",
}
