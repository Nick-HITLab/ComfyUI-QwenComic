"""Pure-PIL helpers for comic layouts, speech bubbles and page assembly.

No ComfyUI imports here so the module can be unit-tested on its own.
"""
import math
import os
import re

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

FONT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")

# --------------------------------------------------------------------------
# Fonts
# --------------------------------------------------------------------------

def list_fonts():
    names = []
    if os.path.isdir(FONT_DIR):
        for f in sorted(os.listdir(FONT_DIR)):
            if f.lower().endswith((".ttf", ".otf")):
                names.append(f)
    return names or ["default"]


def load_font(name, size):
    size = max(6, int(size))
    if name and name != "default":
        path = name if os.path.isabs(name) else os.path.join(FONT_DIR, name)
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    for fallback in ("DejaVuSans-Bold.ttf", "Arial Bold.ttf", "arialbd.ttf"):
        try:
            return ImageFont.truetype(fallback, size)
        except Exception:
            continue
    try:
        return ImageFont.load_default(size)
    except TypeError:  # very old Pillow
        return ImageFont.load_default()


def _text_w(draw, text, font):
    l, t, r, b = draw.textbbox((0, 0), text, font=font)
    return r - l


def wrap_text(text, font, max_w):
    """Greedy word wrap. Respects explicit newlines."""
    dummy = ImageDraw.Draw(Image.new("L", (4, 4)))
    lines = []
    for para in str(text).split("\n"):
        words = para.split()
        if not words:
            lines.append("")
            continue
        cur = words[0]
        for w in words[1:]:
            trial = cur + " " + w
            if _text_w(dummy, trial, font) <= max_w:
                cur = trial
            else:
                lines.append(cur)
                cur = w
        lines.append(cur)
    return lines


def text_block(text, font, max_w, spacing=0.18):
    dummy = ImageDraw.Draw(Image.new("L", (4, 4)))
    lines = wrap_text(text, font, max_w)
    asc, desc = font.getmetrics() if hasattr(font, "getmetrics") else (font.size, 0)
    line_h = int((asc + desc) * (1 + spacing))
    w = max([_text_w(dummy, ln, font) for ln in lines] + [1])
    h = line_h * len(lines)
    return lines, w, h, line_h


def draw_text_lines(draw, lines, cx, top, line_h, font, fill, stroke_width=0, stroke_fill=None, align="center"):
    y = top
    for ln in lines:
        w = _text_w(draw, ln, font)
        x = cx - w / 2 if align == "center" else cx
        draw.text((x, y), ln, font=font, fill=fill, stroke_width=stroke_width, stroke_fill=stroke_fill)
        y += line_h


# --------------------------------------------------------------------------
# Layouts
# --------------------------------------------------------------------------

TEMPLATES = {
    "1 panel (splash)": {"rows": ["1"]},
    "2 rows": {"rows": ["1", "1"]},
    "3 rows": {"rows": ["1", "1", "1"]},
    "2 columns": {"rects": [(0, 0, .5, 1), (.5, 0, .5, 1)]},
    "2x2 grid": {"rows": ["2", "2"]},
    "2x3 grid (6)": {"rows": ["2", "2", "2"]},
    "3x3 grid (9)": {"rows": ["3", "3", "3"]},
    "classic 1-2-1": {"rows": ["1", "2", "1"]},
    "2-1-2": {"rows": ["2", "1", "2"]},
    "big top + 2 bottom": {"rows": ["1", "2"], "heights": [0.6, 0.4]},
    "2 top + big bottom": {"rows": ["2", "1"], "heights": [0.4, 0.6]},
    "big left + 2 right": {"rects": [(0, 0, .6, 1), (.6, 0, .4, .5), (.6, .5, .4, .5)]},
    "uneven 5 (2:1 / 1 / 1:2)": {"rows": ["2:1", "1", "1:2"], "heights": [0.3, 0.4, 0.3]},
    "strip 3 in a row": {"rows": ["3"]},
    "strip 4 in a row": {"rows": ["4"]},
    # ---- dynamic layouts: slanted gutters, insets and bleed (split trees, see _tree_polys)
    "dynamic: slanted 1-2-1": {"tree": ("h", .33, .07, "L", ("h", .68, -.06, ("v", .52, .10, "L", "L"), "L"))},
    "dynamic: zigzag 5": {"tree": ("h", .30, .08, ("v", .56, -.12, "L", "L"),
                                   ("h", .64, -.08, "L", ("v", .42, .14, "L", "L")))},
    "dynamic: action 6": {"tree": ("h", .24, .05, ("v", .40, .08, "L", "L"),
                                   ("h", .58, -.07, ("v", .62, -.10, "L", "L"), ("v", .35, .12, "L", "L")))},
    "dynamic: diagonal duo": {"tree": ("h", .50, .30, "L", "L")},
    "dynamic: tall hero + 3": {"tree": ("v", .55, .08, "L", ("h", .33, -.04, "L", ("h", .66, .04, "L", "L")))},
    "dynamic: bleed top + 3 slanted": {"tree": ("h", .46, .08, "B", ("v", .33, .07, "L", ("v", .67, -.07, "L", "L")))},
    "dynamic: bleed splash + 2 insets": {"tree": "B", "insets": [(.05, .05, .44, .25), (.51, .71, .44, .24)]},
    "custom (use custom_rows)": None,
}


# --------------------------------------------------------------------------
# Polygon helpers (dynamic panels)
# --------------------------------------------------------------------------

def _clip_halfplane(poly, a, b, keep_left, offset):
    """Sutherland-Hodgman clip of a convex polygon against the line a->b.
    keep_left: keep points with signed distance >= offset (left of a->b), else <= -offset."""
    ax, ay = a
    bx, by = b
    L = math.hypot(bx - ax, by - ay) or 1.0

    def dist(p):
        d = ((bx - ax) * (p[1] - ay) - (by - ay) * (p[0] - ax)) / L
        return d - offset if keep_left else -d - offset

    out = []
    n = len(poly)
    for i in range(n):
        p, q = poly[i], poly[(i + 1) % n]
        dp, dq = dist(p), dist(q)
        if dp >= 0:
            out.append(p)
        if (dp >= 0) != (dq >= 0):
            t = dp / (dp - dq)
            out.append((p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t))
    return out


def _tree_polys(node, poly, cw, ch, gutter, out):
    if isinstance(node, str):
        out.append((poly, node == "B"))
        return
    kind, t, sl, first, second = node
    if kind == "h":  # horizontal-ish cut; first = above
        a, b = (0.0, (t - sl / 2) * ch), (cw, (t + sl / 2) * ch)
        top = _clip_halfplane(poly, a, b, keep_left=False, offset=gutter / 2)
        bot = _clip_halfplane(poly, a, b, keep_left=True, offset=gutter / 2)
        _tree_polys(first, top, cw, ch, gutter, out)
        _tree_polys(second, bot, cw, ch, gutter, out)
    else:  # vertical-ish cut; first = left
        a, b = ((t - sl / 2) * cw, 0.0), ((t + sl / 2) * cw, ch)
        left = _clip_halfplane(poly, a, b, keep_left=True, offset=gutter / 2)
        right = _clip_halfplane(poly, a, b, keep_left=False, offset=gutter / 2)
        _tree_polys(first, left, cw, ch, gutter, out)
        _tree_polys(second, right, cw, ch, gutter, out)


def _point_in_poly(x, y, poly):
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            xi = x1 + (y - y1) * (x2 - x1) / ((y2 - y1) or 1e-9)
            if x < xi:
                inside = not inside
    return inside


def _x_range_at(poly, y):
    xs = []
    n = len(poly)
    for i in range(n):
        (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % n]
        if (y1 <= y <= y2) or (y2 <= y <= y1):
            if abs(y2 - y1) < 1e-9:
                xs += [x1, x2]
            else:
                xs.append(x1 + (y - y1) * (x2 - x1) / (y2 - y1))
    return (min(xs), max(xs)) if xs else None


def _safe_rect(poly, bbox):
    """Largest axis-aligned rectangle inside a convex polygon (grid search over top/bottom edges)."""
    x, y, w, h = bbox
    best, best_a = None, -1
    steps = 24
    for i in range(steps):
        yt = y + h * i / steps
        for j in range(i + 4, steps + 1):
            yb = y + h * j / steps
            r1, r2 = _x_range_at(poly, yt + 0.5), _x_range_at(poly, yb - 0.5)
            if not r1 or not r2:
                continue
            xl, xr = max(r1[0], r2[0]), min(r1[1], r2[1])
            a = (xr - xl) * (yb - yt) * (1 - 0.6 * (yt - y) / h)   # prefer rects that reach the top
            if xr > xl and a > best_a:
                best, best_a = [int(xl) + 4, int(yt) + 4, int(xr - xl) - 8, int(yb - yt) - 8], a
    return best or [int(x + w * .2), int(y + h * .2), int(w * .6), int(h * .6)]


def _bbox(poly):
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
    return [int(round(x0)), int(round(y0)), int(round(x1 - x0)), int(round(y1 - y0))]


def panel_poly(layout, idx):
    polys = layout.get("polys")
    if polys:
        return [tuple(p) for p in polys[idx]]
    x, y, w, h = layout["panels"][idx]
    return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]


def _closed(poly):
    pts = [tuple(p) for p in poly]
    return pts + [pts[0]]


def _parse_row(entry):
    entry = entry.strip()
    if not entry:
        return [1.0]
    if ":" in entry:
        vals = [float(v) for v in entry.split(":") if v.strip()]
        return vals or [1.0]
    n = max(1, int(float(entry)))
    return [1.0] * n


def rows_to_rects(rows, heights=None):
    rows = [_parse_row(r) for r in rows]
    if not heights or len(heights) != len(rows):
        heights = [1.0] * len(rows)
    tot_h = float(sum(heights))
    rects, y = [], 0.0
    for ratios, hw in zip(rows, heights):
        h = hw / tot_h
        tot_w, x = float(sum(ratios)), 0.0
        for r in ratios:
            w = r / tot_w
            rects.append((x, y, w, h))
            x += w
        y += h
    return rects


def build_layout(template, width, height, margin, gutter, border, reading_order="left-to-right",
                 custom_rows="1, 2, 1", custom_heights=""):
    spec = TEMPLATES.get(template)
    if spec is None:
        rows = [r for r in re.split(r"[,;\n]", custom_rows) if r.strip()] or ["1"]
        heights = None
        if custom_heights.strip():
            try:
                heights = [float(h) for h in re.split(r"[,;\s]+", custom_heights.strip()) if h]
            except ValueError:
                heights = None
        norm = rows_to_rects(rows, heights)
    elif "tree" in spec:
        return _build_tree_layout(spec, width, height, margin, gutter, border, reading_order)
    elif "rects" in spec:
        norm = list(spec["rects"])
    else:
        norm = rows_to_rects(spec["rows"], spec.get("heights"))

    cw, ch = width - 2 * margin, height - 2 * margin
    eps = 1e-6
    panels = []
    for (x, y, w, h) in norm:
        x0 = margin + x * cw + (gutter / 2 if x > eps else 0)
        y0 = margin + y * ch + (gutter / 2 if y > eps else 0)
        x1 = margin + (x + w) * cw - (gutter / 2 if x + w < 1 - eps else 0)
        y1 = margin + (y + h) * ch - (gutter / 2 if y + h < 1 - eps else 0)
        px, py, pw, ph = int(round(x0)), int(round(y0)), int(round(x1 - x0)), int(round(y1 - y0))
        if reading_order == "right-to-left (manga)":
            px = width - px - pw
        panels.append([px, py, pw, ph])
    return {
        "width": int(width), "height": int(height), "margin": int(margin),
        "gutter": int(gutter), "border": int(border), "panels": panels,
        "reading_order": reading_order, "dynamic": False,
        "safe": [list(p) for p in panels], "insets": [False] * len(panels),
    }


def _build_tree_layout(spec, width, height, margin, gutter, border, reading_order):
    cw, ch = width - 2 * margin, height - 2 * margin
    leaves = []
    _tree_polys(spec["tree"], [(0, 0), (cw, 0), (cw, ch), (0, ch)], cw, ch, gutter, leaves)
    polys, insets, eps = [], [], 0.5
    for poly, bleed in leaves:
        pts = []
        for (px, py) in poly:
            if bleed:  # push points on the content border out to the page edge
                px = -margin if px < eps else (cw + margin if px > cw - eps else px)
                py = -margin if py < eps else (ch + margin if py > ch - eps else py)
            pts.append((px + margin, py + margin))
        polys.append(pts)
        insets.append(False)
    for (ix, iy, iw, ih) in spec.get("insets", []):
        x0, y0 = margin + ix * cw, margin + iy * ch
        polys.append([(x0, y0), (x0 + iw * cw, y0), (x0 + iw * cw, y0 + ih * ch), (x0, y0 + ih * ch)])
        insets.append(True)
    if reading_order == "right-to-left (manga)":
        polys = [[(width - x, y) for (x, y) in p] for p in polys]
    polys = [[(min(max(x, 0), width), min(max(y, 0), height)) for (x, y) in p] for p in polys]
    panels = [_bbox(p) for p in polys]
    return {
        "width": int(width), "height": int(height), "margin": int(margin),
        "gutter": int(gutter), "border": int(border), "panels": panels,
        "polys": [[[round(x, 1), round(y, 1)] for (x, y) in p] for p in polys],
        "reading_order": reading_order, "dynamic": True,
        "safe": _safe_rects_avoiding_insets(polys, panels, insets), "insets": insets,
    }


def _safe_rects_avoiding_insets(polys, panels, insets):
    safe = []
    ins = [b for b, f in zip(panels, insets) if f]
    for poly, b, is_inset in zip(polys, panels, insets):
        r = _safe_rect(poly, b)
        if not is_inset and ins:
            x, y, w, h = r
            covered = sorted((max(y, iy - 20), min(y + h, iy + ih + 20)) for ix, iy, iw, ih in ins
                             if ix < x + w and ix + iw > x and iy < y + h and iy + ih > y)
            gaps, cur = [], y
            for a, bb in covered:
                if a > cur:
                    gaps.append((cur, a))
                cur = max(cur, bb)
            if cur < y + h:
                gaps.append((cur, y + h))
            if gaps:
                g0, g1 = max(gaps, key=lambda g: g[1] - g[0])
                r = [x, int(g0), w, int(g1 - g0)]
        safe.append(r)
    return safe


def panel_rect(layout, panel_index, target_size=None):
    """panel_index 0 = whole page, 1..N = panel. Optionally rescale to another image size."""
    W, H = layout["width"], layout["height"]
    if panel_index <= 0 or panel_index > len(layout["panels"]):
        r = [0, 0, W, H]
    else:
        r = list(layout["panels"][panel_index - 1])
    if target_size and (target_size[0] != W or target_size[1] != H):
        sx, sy = target_size[0] / W, target_size[1] / H
        r = [r[0] * sx, r[1] * sy, r[2] * sx, r[3] * sy]
    return [int(round(v)) for v in r]


def panel_safe_rect(layout, panel_index, target_size=None):
    """Axis-aligned rect safely inside a (possibly slanted) panel - for lettering."""
    safe = layout.get("safe")
    if not safe or panel_index <= 0 or panel_index > len(safe):
        return panel_rect(layout, panel_index, target_size)
    r = list(safe[panel_index - 1])
    W, H = layout["width"], layout["height"]
    if target_size and (target_size[0] != W or target_size[1] != H):
        sx, sy = target_size[0] / W, target_size[1] / H
        r = [r[0] * sx, r[1] * sy, r[2] * sx, r[3] * sy]
    return [int(round(v)) for v in r]


def draw_layout_guide(layout, bg=(255, 255, 255), line=(0, 0, 0), numbers=False):
    img = Image.new("RGB", (layout["width"], layout["height"]), bg)
    d = ImageDraw.Draw(img)
    b = max(1, layout["border"])
    for i, (x, y, w, h) in enumerate(layout["panels"]):
        poly = panel_poly(layout, i)
        if layout.get("insets", [False] * 99)[i]:
            d.polygon(poly, fill=bg)
            d.line(_closed(poly), fill=bg, width=max(2, layout["gutter"]), joint="curve")
        d.line(_closed(poly), fill=line, width=b, joint="curve")
        if numbers:
            f = load_font("default", max(14, min(w, h) // 8))
            d.text((x + w / 2, y + h / 2), str(i + 1), fill=(200, 200, 200), font=f, anchor="mm")
    return img


def describe_position(layout, idx):
    """Human readable position of panel idx (0-based) for prompts."""
    x, y, w, h = layout["panels"][idx]
    W, H = layout["width"], layout["height"]
    cx, cy = (x + w / 2) / W, (y + h / 2) / H
    v = "top" if cy < 0.36 else ("bottom" if cy > 0.64 else "middle")
    hz = "left" if cx < 0.36 else ("right" if cx > 0.64 else "center")
    if w > 0.8 * (W - 2 * layout["margin"]):
        hz = "full width"
    shape = "wide" if w > 1.6 * h else ("tall" if h > 1.6 * w else "")
    parts = [v, hz]
    if shape:
        parts.append(shape)
    return " ".join(parts)


# --------------------------------------------------------------------------
# Image fitting / assembly
# --------------------------------------------------------------------------

def cover_fit(img, w, h):
    iw, ih = img.size
    s = max(w / iw, h / ih)
    nw, nh = max(w, int(math.ceil(iw * s))), max(h, int(math.ceil(ih * s)))
    img = img.resize((nw, nh), Image.LANCZOS)
    l, t = (nw - w) // 2, (nh - h) // 2
    return img.crop((l, t, l + w, t + h))


def assemble_panels(layout, images, bg=(255, 255, 255), line=(0, 0, 0)):
    page = Image.new("RGB", (layout["width"], layout["height"]), bg)
    d = ImageDraw.Draw(page)
    b = max(0, layout["border"])
    for i, (x, y, w, h) in enumerate(layout["panels"]):
        poly = panel_poly(layout, i)
        if layout.get("insets", [False] * 99)[i]:
            d.line(_closed(poly), fill=bg, width=max(2, layout["gutter"]), joint="curve")
        if images and w > 0 and h > 0:
            src = images[min(i, len(images) - 1)]
            mask = Image.new("L", (w, h), 0)
            ImageDraw.Draw(mask).polygon([(px - x, py - y) for px, py in poly], fill=255)
            page.paste(cover_fit(src.convert("RGB"), w, h), (x, y), mask)
        if b:
            d.line(_closed(poly), fill=line, width=b, joint="curve")
    return page


def fit_whole_page(layout, img, redraw_borders=False, line=(0, 0, 0)):
    page = cover_fit(img.convert("RGB"), layout["width"], layout["height"])
    if redraw_borders and layout["border"]:
        d = ImageDraw.Draw(page)
        for i in range(len(layout["panels"])):
            d.line(_closed(panel_poly(layout, i)), fill=line, width=layout["border"], joint="curve")
    return page


def is_blank(img, tol=12, frac=0.995, inset=0.04):
    """True if the (inset) region of img is almost uniformly white."""
    w, h = img.size
    ix, iy = int(w * inset), int(h * inset)
    g = img.convert("L").crop((ix, iy, max(ix + 1, w - ix), max(iy + 1, h - iy)))
    hist = g.histogram()
    white = sum(hist[255 - tol:])
    return white >= frac * max(1, sum(hist))


def paste_element(canvas, element, alpha, rect, cx, cy, rel_w, remove_white=False, threshold=240):
    """Paste element (RGB) with optional alpha (L) centered at (cx,cy) inside rect, width = rel_w*rect_w."""
    x, y, w, h = rect
    tw = max(4, int(rel_w * w))
    s = tw / element.width
    th = max(4, int(element.height * s))
    el = element.convert("RGB").resize((tw, th), Image.LANCZOS)
    if alpha is None:
        alpha = Image.new("L", element.size, 255)
    a = alpha.resize((tw, th), Image.LANCZOS)
    if remove_white:
        g = el.convert("L").point(lambda v: 0 if v >= threshold else 255)
        g = g.filter(ImageFilter.MinFilter(3)).filter(ImageFilter.GaussianBlur(1))
        a = Image.fromarray(np.minimum(np.array(a), np.array(g)))
    px = int(x + cx * w - tw / 2)
    py = int(y + cy * h - th / 2)
    out = canvas.copy()
    out.paste(el, (px, py), a)
    return out


# --------------------------------------------------------------------------
# Bubbles
# --------------------------------------------------------------------------

BUBBLE_STYLES = ["speech", "thought", "shout", "whisper", "caption", "sfx", "plain text"]


def _ellipse_point(cx, cy, rx, ry, ang):
    return cx + rx * math.cos(ang), cy + ry * math.sin(ang)


def _outline_from_mask(mask, stroke):
    k = stroke * 2 + 1
    return mask.filter(ImageFilter.MaxFilter(k if k % 2 else k + 1))


def draw_bubble(img, text, cx, cy, max_w, font_name="ComicNeue-Bold.ttf", font_size=28, style="speech",
                tail=None, fill=(255, 255, 255), text_color=(0, 0, 0), outline=(0, 0, 0), stroke=None,
                caption_fill=(255, 244, 196)):
    """Draw a comic balloon on a PIL RGB image (in place) and return its bbox.

    cx, cy   center in pixels
    max_w    max text width in pixels
    tail     (tx, ty) pixel point the tail points at, or None
    """
    if not str(text).strip():
        return None
    img_w, img_h = img.size
    stroke = stroke or max(2, int(font_size * 0.11))

    if style == "sfx":
        fname = "Bangers-Regular.ttf" if "Bangers-Regular.ttf" in list_fonts() else font_name
        font = load_font(fname, font_size * 3.0)
        lines, tw, th, lh = text_block(str(text).upper(), font, max_w * 1.5, spacing=0.0)
        pad = int(font_size * 0.6)
        layer = Image.new("RGBA", (tw + 2 * pad, th + 2 * pad), (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        draw_text_lines(d, lines, layer.width / 2, pad, lh, font, fill=(255, 214, 0, 255),
                        stroke_width=max(3, int(font_size * 0.22)), stroke_fill=(0, 0, 0, 255))
        layer = layer.rotate(8, expand=True, resample=Image.BICUBIC)
        img.paste(layer, (int(cx - layer.width / 2), int(cy - layer.height / 2)), layer)
        return (cx - layer.width / 2, cy - layer.height / 2, cx + layer.width / 2, cy + layer.height / 2)

    font = load_font(font_name, font_size)
    upper = style in ("shout",)
    lines, tw, th, lh = text_block(str(text).upper() if upper else str(text), font, max_w)

    if style == "plain text":
        d = ImageDraw.Draw(img)
        draw_text_lines(d, lines, cx, cy - th / 2, lh, font, fill=text_color,
                        stroke_width=max(2, font_size // 8), stroke_fill=(255, 255, 255))
        return (cx - tw / 2, cy - th / 2, cx + tw / 2, cy + th / 2)

    if style == "caption":
        pad = int(font_size * 0.5)
        x0, y0 = int(cx - tw / 2 - pad), int(cy - th / 2 - pad)
        x1, y1 = int(cx + tw / 2 + pad), int(cy + th / 2 + pad)
        d = ImageDraw.Draw(img)
        d.rectangle([x0, y0, x1, y1], fill=caption_fill, outline=outline, width=stroke)
        draw_text_lines(d, lines, cx, y0 + pad, lh, font, fill=text_color)
        return (x0, y0, x1, y1)

    # ---- balloon-shaped styles: build a mask, outline it, fill it --------
    pad_x, pad_y = font_size * 0.9, font_size * 0.7
    rx = (tw / 2 + pad_x) * 1.12
    ry = (th / 2 + pad_y) * 1.12
    if style == "shout":
        rx, ry = rx * 1.12, ry * 1.2

    margin = int(stroke * 3 + font_size)
    # local canvas covering balloon + tail
    pts_x = [cx - rx, cx + rx]
    pts_y = [cy - ry, cy + ry]
    if tail:
        pts_x.append(tail[0]); pts_y.append(tail[1])
    L = int(min(pts_x)) - margin
    T = int(min(pts_y)) - margin
    R = int(max(pts_x)) + margin
    B = int(max(pts_y)) + margin
    lw, lh_ = R - L, B - T
    lcx, lcy = cx - L, cy - T
    mask = Image.new("L", (lw, lh_), 0)
    md = ImageDraw.Draw(mask)

    if style == "shout":
        n = 22
        poly = []
        for i in range(n * 2):
            ang = math.pi * i / n
            k = 1.0 if i % 2 == 0 else 0.78
            poly.append(_ellipse_point(lcx, lcy, rx * k, ry * k, ang))
        md.polygon(poly, fill=255)
    elif style == "thought":
        md.ellipse([lcx - rx * 0.92, lcy - ry * 0.92, lcx + rx * 0.92, lcy + ry * 0.92], fill=255)
        n = 12
        for i in range(n):
            ang = 2 * math.pi * i / n
            px, py = _ellipse_point(lcx, lcy, rx * 0.86, ry * 0.86, ang)
            r = min(rx, ry) * 0.36
            md.ellipse([px - r, py - r, px + r, py + r], fill=255)
    else:  # speech / whisper
        md.ellipse([lcx - rx, lcy - ry, lcx + rx, lcy + ry], fill=255)

    if tail:
        tx, ty = tail[0] - L, tail[1] - T
        ang = math.atan2(ty - lcy, tx - lcx)
        if style == "thought":
            # trail of small circles
            ex, ey = _ellipse_point(lcx, lcy, rx, ry, ang)
            for j, f in enumerate((0.25, 0.55, 0.85)):
                px, py = ex + (tx - ex) * f, ey + (ty - ey) * f
                r = max(3, font_size * (0.42 - 0.12 * j))
                md.ellipse([px - r, py - r, px + r, py + r], fill=255)
        else:
            spread = 0.22
            b1 = _ellipse_point(lcx, lcy, rx * 0.8, ry * 0.8, ang - spread)
            b2 = _ellipse_point(lcx, lcy, rx * 0.8, ry * 0.8, ang + spread)
            md.polygon([b1, (tx, ty), b2], fill=255)

    outer = _outline_from_mask(mask, stroke)
    region = img.crop((L, T, R, B))
    region.paste(Image.new("RGB", region.size, outline), (0, 0), outer)
    region.paste(Image.new("RGB", region.size, fill), (0, 0), mask)
    if style == "whisper":
        # dashed look: punch gaps into the outline
        rd = ImageDraw.Draw(region)
        n = 40
        for i in range(0, n, 2):
            a0 = 2 * math.pi * i / n
            p = _ellipse_point(lcx, lcy, rx + stroke / 2, ry + stroke / 2, a0)
            rd.ellipse([p[0] - stroke * 1.4, p[1] - stroke * 1.4, p[0] + stroke * 1.4, p[1] + stroke * 1.4], fill=fill)
    d = ImageDraw.Draw(region)
    draw_text_lines(d, lines, lcx, lcy - th / 2, lh, font, fill=text_color)
    img.paste(region, (L, T))
    return (cx - rx, cy - ry, cx + rx, cy + ry)


# --------------------------------------------------------------------------
# Script parsing
# --------------------------------------------------------------------------

PANEL_RE = re.compile(r"^\s*(?:\[\s*)?(?:panel|paneel|p)\s*(\d+)\s*(?:\])?\s*[:.\-)]?\s*(.*)$", re.I)
LINE_RE = re.compile(r"^\s*([^:\"]{1,40}?)\s*(?:\(([^)]*)\))?\s*:\s*(.+)$")


def _classify(name, mod, text):
    n = name.strip().lower()
    m = (mod or "").strip().lower()
    if n in ("caption", "narration", "narrator", "cap", "box"):
        return "caption"
    if n in ("sfx", "sound", "fx"):
        return "sfx"
    if m in ("thought", "thinks", "thinking", "think", "gedachte"):
        return "thought"
    if m in ("shout", "shouting", "yell", "scream", "roep"):
        return "shout"
    if m in ("whisper", "whispers", "fluister"):
        return "whisper"
    letters = [c for c in text if c.isalpha()]
    if letters and text.strip().endswith("!") and all(c.isupper() for c in letters) and len(letters) > 2:
        return "shout"
    return "speech"


NOT_DIALOGUE = {"setting", "camera", "shot", "scene", "description", "action", "location", "background",
                "angle", "visual", "visuals", "art", "mood", "time", "lighting", "composition", "panel",
                "characters", "character", "setting/time", "note", "notes", "title", "page"}


def clean_line(line):
    """Strip markdown decoration an LLM might add: '**Panel 1:**', '### Panel 1', '- Mia: ...'."""
    line = line.strip()
    line = re.sub(r"^[#>\-\*•\s]+", "", line)
    line = line.replace("**", "").replace("__", "")
    return line.strip()


def _dialogue(line):
    lm = LINE_RE.match(line)
    if not lm:
        return None
    name, mod, text = lm.group(1).strip(), lm.group(2), lm.group(3).strip()
    low = name.lower()
    if low in NOT_DIALOGUE or len(name.split()) > 4:
        return None
    if any(w in ("shot", "close-up", "closeup", "angle", "view", "panel") for w in low.split()):
        return None
    text = text.strip().strip('"“”')
    if not text:
        return None
    return {"speaker": name, "style": _classify(name, mod, text), "text": text}


def parse_script(script, n_panels):
    """Returns (panels, warnings, preamble). panels: list of {description, lines:[{speaker, style, text}]}."""
    warnings = []
    panels = []
    preamble = []
    cur = None
    lines_in = [clean_line(l) for l in str(script).replace("\r", "").splitlines()]
    has_markers = any(PANEL_RE.match(l) for l in lines_in)

    if has_markers:
        for line in lines_in:
            if not line:
                continue
            m = PANEL_RE.match(line)
            if m:
                desc, shot = m.group(2).strip(), ""
                sm = re.match(r"^[\(\[]([^\)\]]{2,120})[\)\]]\s*[:.\-]?\s*(.*)$", desc)
                if sm:
                    shot, desc = sm.group(1).strip(), sm.group(2).strip()
                cur = {"description": desc, "lines": [], "shot": shot}
                panels.append(cur)
                continue
            if cur is None:  # text before the first panel = context, not a panel
                preamble.append(line)
                continue
            d = _dialogue(line)
            if d:
                cur["lines"].append(d)
            else:
                lm = LINE_RE.match(line)
                txt = line if not lm else f"{lm.group(1).strip()}: {lm.group(3).strip()}"
                if lm and lm.group(1).strip().lower() in ("camera", "shot", "angle", "composition"):
                    cur["shot"] = (cur.get("shot", "") + " " + lm.group(3).strip()).strip()
                    continue
                if lm and lm.group(1).strip().lower() in ("setting", "camera", "shot", "angle", "lighting",
                                                           "composition", "action", "background", "location",
                                                           "mood", "time", "description", "visual", "visuals", "scene"):
                    txt = lm.group(3).strip()
                cur["description"] = (cur["description"] + " " + txt).strip()
    else:
        paras = [p.strip() for p in re.split(r"\n\s*\n", "\n".join(lines_in)) if p.strip()]
        if len(paras) < n_panels and len(paras) == 1:
            sents = re.split(r"(?<=[.!?])\s+", paras[0])
            per = max(1, math.ceil(len(sents) / n_panels))
            paras = [" ".join(sents[i:i + per]) for i in range(0, len(sents), per)]
        for p in paras:
            cur = {"description": "", "lines": [], "shot": ""}
            for line in p.splitlines():
                d = _dialogue(line)
                if d and len(d["speaker"].split()) <= 3:
                    cur["lines"].append(d)
                else:
                    cur["description"] = (cur["description"] + " " + line.strip()).strip()
            panels.append(cur)

    if len(panels) > n_panels:
        warnings.append(f"Script has {len(panels)} panels but layout has {n_panels}; extra panels ignored.")
        panels = panels[:n_panels]
    while len(panels) < n_panels:
        prev = panels[-1]["description"] if panels else ""
        panels.append({"description": ("The story continues. " + prev).strip(), "lines": [], "shot": ""})
        warnings.append(f"Panel {len(panels)} had no script text; reused previous description.")
    return panels, warnings, preamble


# --------------------------------------------------------------------------
# Camera / shot direction
# --------------------------------------------------------------------------

CAMERA_MODES = ["auto: cinematic variety", "auto: dramatic action", "auto: calm & cozy", "only what I write"]

_SHOT_WORDS = ("close-up", "closeup", "close up", "wide shot", "long shot", "medium shot", "full-body", "full body",
               "angle", "bird's-eye", "birds-eye", "bird's eye", "worm's-eye", "over-the-shoulder", "over the shoulder",
               "establishing", "extreme", "pov", "point of view", "silhouette", "top-down", "aerial", "panoramic",
               "two-shot", "dutch")

SHOTS = {
    "auto: cinematic variety": {
        "wide": ["wide establishing shot, deep perspective", "high-angle wide shot looking down on the scene",
                 "wide shot, low camera, strong foreground element"],
        "tall": ["full-body shot from a low angle", "tall vertical composition looking up",
                 "full-body shot with the character framed by the setting"],
        "small": ["close-up on the face, expressive", "medium shot, eye level", "low-angle medium shot",
                  "extreme close-up on eyes or hands", "high-angle medium shot", "medium two-shot"],
    },
    "auto: dramatic action": {
        "wide": ["dynamic wide action shot, dramatic perspective, speed lines",
                 "extreme low angle worm's-eye wide shot", "panoramic shot with strong diagonal composition"],
        "tall": ["heroic full-body shot from an extreme low angle, dramatic foreshortening",
                 "vertical dutch-angle action shot, motion lines"],
        "small": ["intense extreme close-up on the eyes", "tilted dutch-angle close-up",
                  "fast action medium shot with speed lines and motion blur", "impact close-up, sharp shadows"],
    },
    "auto: calm & cozy": {
        "wide": ["gentle wide shot with lots of open sky and space", "soft eye-level establishing shot"],
        "tall": ["full-body shot at eye level, relaxed pose", "tall shot with calm negative space above"],
        "small": ["soft close-up, warm expression", "medium shot at eye level", "cozy detail shot of hands or objects"],
    },
}


def _panel_shape(layout, i):
    x, y, w, h = layout["panels"][i]
    area = w * h / float(layout["width"] * layout["height"])
    if w > 1.5 * h:
        return "wide"
    if h > 1.5 * w:
        return "tall"
    return "wide" if area > 0.28 else "small"


def assign_shots(layout, panels, mode):
    """Fill panel['shot'] where the user gave none (and the description has no shot words)."""
    if mode not in SHOTS:
        return panels
    used, prev = {}, ""
    for i, pan in enumerate(panels):
        if pan.get("shot"):
            prev = pan["shot"]
            continue
        low = pan["description"].lower()
        if any(wd in low for wd in _SHOT_WORDS):
            continue
        shape = _panel_shape(layout, i)
        opts = SHOTS[mode][shape]
        if i == 0:  # open the page with an establishing shot
            choice = SHOTS[mode]["wide"][0]
            used["wide"] = used.get("wide", 0) + 1
        else:
            k = used.get(shape, 0)
            choice = opts[k % len(opts)]
            if choice == prev and len(opts) > 1:
                k += 1
                choice = opts[k % len(opts)]
            used[shape] = k + 1
        pan["shot"] = choice
        prev = choice
    return panels


# --------------------------------------------------------------------------
# Cast (character sheet)
# --------------------------------------------------------------------------

VARIANT_RE = re.compile(r"\[([^\[\]]{1,40})\]")


def _split_without(desc, label):
    """'... without: red bulb'  ->  (desc, 'red bulb'). Default from the label: 'no bulb' -> 'bulb'."""
    parts = re.split(r"[.;,]?\s*\b(?:without|zonder|must not have|niet)\s*:\s*", desc, maxsplit=1, flags=re.I)
    desc = parts[0].strip().rstrip(".;,")
    without = parts[1].strip().rstrip(".") if len(parts) > 1 else ""
    if not without and label:
        m = re.match(r"^\s*(?:no|without|zonder|geen|lost|missing)\s+(?:his|her|its|their|zijn|haar)?\s*(.+)$", label, re.I)
        if m:
            without = m.group(1).strip()
    return desc, without


def parse_cast(cast_text):
    """'Name: description' per line  ->  list of {name, aliases, description, variants}.
    Aliases: 'Mia / Mimi: ...'.  Variants (a different look in some panels):
    'Bolt [no bulb]: small round robot ..., bare antenna. without: red bulb'."""
    cast, pending = [], []
    for raw in str(cast_text or "").splitlines():
        line = clean_line(raw)
        if not line or ":" not in line:
            continue
        # the name part ends at the first ':' that is not inside [...]
        m = re.match(r"^((?:[^:\[]|\[[^\]]*\])+?)\s*:\s*(.+)$", line)
        if not m:
            continue
        name, desc = m.group(1).strip(), m.group(2).strip()
        if not name or not desc or len(name) > 80:
            continue
        vm = VARIANT_RE.search(name)
        if vm:
            label = vm.group(1).strip()
            base = VARIANT_RE.sub("", name).strip()
            d, without = _split_without(desc, label)
            pending.append((base, label, d, without))
            continue
        aliases = [a.strip() for a in re.split(r"[/|,]", name) if a.strip()]
        cast.append({"name": aliases[0], "aliases": aliases, "description": desc, "variants": {}})
    for base, label, d, without in pending:
        aliases = [a.strip() for a in re.split(r"[/|,]", base) if a.strip()]
        c = next((c for c in cast if any(a.lower() in [x.lower() for x in c["aliases"]] for a in aliases)), None)
        if c is None:
            c = {"name": aliases[0], "aliases": aliases, "description": d, "variants": {}}
            cast.append(c)
        c["variants"][label.lower()] = {"label": label, "description": d, "without": without}
    return cast


def sheet_figures(cast):
    """Figures on the character sheet, left to right: every base character, then every variant."""
    figs = [(c["name"], None, c["description"]) for c in cast]
    for c in cast:
        for v in c.get("variants", {}).values():
            figs.append((c["name"], v["label"], v["description"]))
    return figs


def _mentions(text, alias):
    return re.search(r"(?<![\w])" + re.escape(alias) + r"(?![\w])", text, re.I) is not None


def canonical_speaker(cast, speaker):
    """'the robot [no bulb]' -> ('Bolt', 'no bulb'); unknown speakers are returned unchanged."""
    vm = VARIANT_RE.search(speaker)
    label = vm.group(1).strip() if vm else None
    sp = VARIANT_RE.sub("", speaker).strip()
    for c in cast:
        if any(sp.lower() == a.lower() for a in c["aliases"]):
            return c["name"], label
    return sp, label


def cast_in_panel(cast, panel):
    """Cast members mentioned in the description or speaking in the panel (in cast order)."""
    speakers = " ".join(l["speaker"] for l in panel["lines"])
    text = panel["description"] + " " + speakers
    return [c for c in cast if any(_mentions(text, a) for a in c["aliases"])]


def panel_variants(cast, panel):
    """{name: variant} for characters written as 'Bolt [no bulb]' in this panel's description or speakers."""
    out = {}
    texts = [panel["description"]] + [l.get("speaker_raw", l["speaker"]) for l in panel["lines"]]
    for c in cast:
        for a in c["aliases"]:
            for t in texts:
                for m in re.finditer(r"(?<![\w])" + re.escape(a) + r"\s*\[([^\[\]]{1,40})\]", t, re.I):
                    v = c.get("variants", {}).get(m.group(1).strip().lower())
                    if v:
                        out[c["name"]] = v
    return out


def strip_variant_tags(text):
    return re.sub(r"\s*\[([^\[\]]{1,40})\]", "", text)


def cast_prompt(members, variants=None):
    variants = variants or {}
    out = []
    for c in members:
        v = variants.get(c["name"])
        if v:
            out.append(f"{c['name']} = {v['description'].rstrip('.')}")
        else:
            out.append(f"{c['name']} = {c['description'].rstrip('.')}")
    return "; ".join(out)


def auto_letter_panel(img, rect, lines, font_name, font_size):
    """Place a panel's dialogue lines automatically along the top of the panel."""
    x, y, w, h = rect
    pad = max(8, int(min(w, h) * 0.04))
    cursor = y + pad
    side = 0
    sfx = [l for l in lines if l["style"] == "sfx"]
    for l in lines:
        if l["style"] == "caption":
            f = load_font(font_name, font_size)
            _, tw, th, _ = text_block(l["text"], f, w * 0.8 - font_size)
            cw = tw + font_size
            bb = draw_bubble(img, l["text"], x + pad + cw / 2 - 2, cursor + th / 2 + font_size * 0.5,
                             w * 0.8 - font_size, font_name, font_size, "caption")
            if bb:
                cursor = bb[3] + pad
    for l in lines:
        if l["style"] in ("caption", "sfx"):
            continue
        max_w = w * 0.46
        f = load_font(font_name, font_size)
        _, tw, th, _ = text_block(l["text"].upper() if l["style"] == "shout" else l["text"], f, max_w)
        rx = (tw / 2 + font_size * 0.9) * 1.12 * (1.12 if l["style"] == "shout" else 1)
        ry = (th / 2 + font_size * 0.7) * 1.12 * (1.2 if l["style"] == "shout" else 1)
        rx = min(rx, w / 2 - pad)
        cx = x + pad + rx if side == 0 else x + w - pad - rx
        cy = cursor + ry
        if cy + ry > y + h * 0.75:  # out of room -> shrink into place anyway
            cy = min(cy, y + h - ry - pad)
        tail = (cx + (w * 0.08 if side == 0 else -w * 0.08), min(y + h - pad, cy + ry + h * 0.12))
        draw_bubble(img, l["text"], cx, cy, max_w, font_name, font_size, l["style"], tail=tail)
        cursor = cy + ry * 0.9
        side = 1 - side
    for l in sfx:
        draw_bubble(img, l["text"], x + w * 0.68, y + h * 0.72, w * 0.5, font_name, font_size, "sfx")


# --------------------------------------------------------------------------
# Staging (where characters stand) + speaker-aware bubble placement
# --------------------------------------------------------------------------

def _head_y(text):
    t = text.lower()
    if "extreme close" in t:
        return 0.50
    if "close-up" in t or "close up" in t or "closeup" in t:
        return 0.45
    if any(k in t for k in ("wide", "establishing", "panoramic", "aerial", "bird's-eye", "birds-eye", "high-angle")):
        return 0.58
    if "full-body" in t or "full body" in t:
        return 0.30
    return 0.38


def stage_panel(pan, members, idx):
    """Decide where each cast member stands. Returns ({name: (x_frac, head_y)}, prompt_text)."""
    if not members:
        return {}, ""
    text = (pan.get("shot", "") + " " + pan.get("description", "")).lower()
    if any(k in text for k in ("over-the-shoulder", "over the shoulder", "from behind", "pov", "point of view")):
        return {}, ""
    order = []
    for l in pan["lines"]:
        for c in members:
            if c not in order and any(l["speaker"].strip().lower() == a.lower() for a in c["aliases"]):
                order.append(c)
    order += [c for c in members if c not in order]
    hy = _head_y(text)
    names = [c["name"] for c in order]
    if len(order) == 1:
        if "extreme close" in text:
            return {names[0]: (0.5, hy)}, f"{names[0]} is centred in the frame."
        x = 0.36 if idx % 2 == 0 else 0.64
        side = "left" if x < 0.5 else "right"
        return {names[0]: (x, hy)}, (f"{names[0]} is placed in the {side} half of the image, "
                                     f"with some open sky or background above their head.")
    xs = {2: [0.3, 0.7], 3: [0.2, 0.5, 0.8]}.get(len(order), [0.15 + 0.7 * k / (len(order) - 1) for k in range(len(order))])
    stage = {nm: (x, hy) for nm, x in zip(names, xs)}
    if len(order) == 2:
        txt = f"{names[0]} is on the left and {names[1]} is on the right."
    else:
        txt = "From left to right: " + ", ".join(names) + "."
    return stage, txt + " Keep some open background above their heads."


def _balloon_size(text, font_name, font_size, max_w, style):
    font = load_font(font_name, font_size)
    t = str(text).upper() if style == "shout" else str(text)
    _, tw, th, _ = text_block(t, font, max_w)
    rx = (tw / 2 + font_size * 0.9) * 1.12
    ry = (th / 2 + font_size * 0.7) * 1.12
    if style == "shout":
        rx, ry = rx * 1.12, ry * 1.2
    if style == "thought":
        rx, ry = rx * 1.08, ry * 1.12
    return rx, ry


def _covers(box, target):
    """Fraction of target covered by box."""
    ix = max(0, min(box[2], target[2]) - max(box[0], target[0]))
    iy = max(0, min(box[3], target[3]) - max(box[1], target[1]))
    return ix * iy / max(1, (target[2] - target[0]) * (target[3] - target[1]))


def _box_overlap(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area = max(1, (a[2] - a[0]) * (a[3] - a[1]))
    return ix * iy / area


class _Maps:
    """Low-res cost maps around one panel: heads, bodies, visual detail, where lettering may go.
    Integral images for fast box sums. The area can extend `ext` px beyond the panel so balloons may
    break out over the border into the white gutter (never onto another panel or off the page)."""

    def __init__(self, img, full_rect, heads_px, bodies_px, k=4, poly=None, inset=0, ext=0, blocked=None):
        fx, fy, fw, fh = [int(v) for v in full_rect]
        ex0, ey0 = max(0, fx - ext), max(0, fy - ext)
        ex1, ey1 = min(img.width, fx + fw + ext), min(img.height, fy + fh + ext)
        self.k, self.ox, self.oy = k, ex0, ey0
        W, H = max(1, (ex1 - ex0) // k), max(1, (ey1 - ey0) // k)
        self.W, self.H = W, H
        g = img.crop((ex0, ey0, ex1, ey1)).convert("L").resize((W, H))
        e = np.asarray(g.filter(ImageFilter.FIND_EDGES), dtype=np.float32)
        e = np.asarray(Image.fromarray(np.clip(e, 0, 255).astype(np.uint8)).filter(ImageFilter.BoxBlur(2)),
                       dtype=np.float32) / 255.0
        self.detail = self._integ(np.clip(e * 3.0, 0, 1))
        hm = np.zeros((H, W), np.float32)
        bm = np.zeros((H, W), np.float32)
        for b in heads_px:
            self._fill(hm, b)
        for b in bodies_px:   # the upper body matters more than legs and feet
            split = b[1] + (b[3] - b[1]) * 0.6
            self._fill(bm, (b[0], b[1], b[2], split), 1.0)
            self._fill(bm, (b[0], split, b[2], b[3]), 0.45, keep_max=True)
        self.heads, self.bodies = self._integ(hm), self._integ(bm)
        to_local = lambda pts: [((x - ex0) / k, (y - ey0) / k) for x, y in pts]
        own = Image.new("L", (W, H), 0)
        if poly:
            ImageDraw.Draw(own).polygon(to_local(poly), fill=255)
        else:
            ImageDraw.Draw(own).rectangle([(fx - ex0) / k, (fy - ey0) / k, (fx + fw - ex0) / k, (fy + fh - ey0) / k],
                                          fill=255)
        r = max(1, int(round(inset / k)))
        inside = np.asarray(own.filter(ImageFilter.MinFilter(2 * r + 1))) >= 128
        other = Image.new("L", (W, H), 0)
        for bp in (blocked or []):
            ImageDraw.Draw(other).polygon(to_local(bp), fill=255)
        other = np.asarray(other.filter(ImageFilter.MaxFilter(2 * max(1, int(6 / k)) + 1))) >= 128
        inside = inside & ~other            # inset panels drawn on top of this one
        gutter = np.zeros((H, W), bool)
        if ext > 0:
            near = np.asarray(own.filter(ImageFilter.MaxFilter(2 * max(1, int(ext / k)) + 1))) >= 128
            m = max(1, int(8 / k))          # keep off the very edge of the page
            onpage = np.zeros((H, W), bool)
            x_lo, y_lo = max(0, m - ex0 // k), max(0, m - ey0 // k)
            x_hi = W - max(0, m - (img.width - ex1) // k)
            y_hi = H - max(0, m - (img.height - ey1) // k)
            onpage[y_lo:y_hi, x_lo:x_hi] = True
            gutter = near & ~inside & ~other & onpage
        self.gutter = self._integ(gutter.astype(np.float32))
        self.outside = self._integ((~(inside | gutter)).astype(np.float32))

    def _fill(self, m, b, v=1.0, keep_max=False):
        x0, y0, x1, y1 = self._cells(b)
        if keep_max:
            m[y0:y1, x0:x1] = np.maximum(m[y0:y1, x0:x1], v)
        else:
            m[y0:y1, x0:x1] = v

    def _cells(self, b):
        k = self.k
        return (int(np.clip((b[0] - self.ox) / k, 0, self.W)), int(np.clip((b[1] - self.oy) / k, 0, self.H)),
                int(np.clip(np.ceil((b[2] - self.ox) / k), 0, self.W)), int(np.clip(np.ceil((b[3] - self.oy) / k), 0, self.H)))

    @staticmethod
    def _integ(m):
        return np.pad(m.cumsum(0).cumsum(1), ((1, 0), (1, 0)))

    def frac(self, integ, b):
        """Mean of the map inside box b (0..1); parts outside the mapped area count as 1 for 'outside'."""
        x0, y0, x1, y1 = self._cells(b)
        if x1 <= x0 or y1 <= y0:
            return 1.0 if integ is self.outside else 0.0
        tot = integ[y1, x1] - integ[y0, x1] - integ[y1, x0] + integ[y0, x0]
        full = (b[2] - b[0]) * (b[3] - b[1]) / float(self.k * self.k)
        cells = (x1 - x0) * (y1 - y0)
        if integ is self.outside and full > cells * 1.05:      # box sticks out of the mapped area
            return 1.0
        return float(tot) / cells


def _ellipse_core(cx, cy, rx, ry):
    """The part of an ellipse's bounding box that is really covered (ignore the empty corners)."""
    return (cx - rx * 0.86, cy - ry * 0.86, cx + rx * 0.86, cy + ry * 0.86)


def parse_balloon_adjust(text):
    """'4.1: x=0.8 y=0.15 size=0.9' -> {(4, 1): {'x': 0.8, 'y': 0.15, 'size': 0.9}}  (panel, balloon number)."""
    out = {}
    for raw in str(text or "").splitlines():
        m = re.match(r"^\s*(\d+)\s*[.,:/-]\s*(\d+)\s*[:=]?\s*(.*)$", raw)
        if not m:
            continue
        vals = {}
        for k, v in re.findall(r"(x|y|size|font|width|tail_x|tail_y)\s*[=:]\s*(-?[\d.]+)", m.group(3), re.I):
            k = k.lower()
            vals["size" if k == "font" else k] = float(v)
        if vals:
            out[(int(m.group(1)), int(m.group(2)))] = vals
    return out


def plan_balloons(img, full_rect, safe_rect, lines, font_name, font_size, stage=None, heads=None, bodies=None,
                  occupied=None, poly=None, blocked=None, adjust=None):
    """Find free space for every balloon: faces are a no-go zone, bodies nearly so (legs less than the upper body),
    then the calmest background near the speaker, above or beside them. Tries several balloon shapes and font sizes.
    When the panel is full, a balloon may break out over the panel border into the white gutter.
    adjust: {balloon_number: {x, y, size, width, tail_x, tail_y}} manual overrides (x/y = centre, 0..1 of the panel).
    Returns [{line, cx, cy, rx, ry, fs, max_w, tail, head_cov, body_cov, detail, gutter}]."""
    stage = stage or {}
    adjust = adjust or {}
    fx, fy, fw, fh = full_rect
    sx, sy, sw, sh = safe_rect
    if sw < 20 or sh < 20:
        return []
    P = lambda b: (fx + b[0] * fw, fy + b[1] * fh, fx + b[2] * fw, fy + b[3] * fh)
    head_px = {nm: P(b) for nm, b in (heads or {}).items()}
    for nm, v in stage.items():   # no detected head: estimate one around the staged position
        if nm not in head_px:
            head_px[nm] = P((v[0] - 0.12, v[1] - 0.15, v[0] + 0.12, v[1] + 0.11))
    head_list = []
    for b in head_px.values():   # a little safety margin around every face
        mx, my = (b[2] - b[0]) * 0.05, (b[3] - b[1]) * 0.05
        head_list.append((b[0] - mx, b[1] - my, b[2] + mx, b[3] + my))
    body_px = [P(b) for b in (bodies or [])]
    pad = max(8, int(min(sw, sh) * 0.03))
    ext = int(min(fw, fh) * 0.12) if blocked is not None else 0
    maps = _Maps(img, full_rect, head_list, body_px, poly=poly, inset=pad, ext=ext, blocked=blocked)
    occupied = list(occupied or [])
    if poly:   # search the whole panel shape, not only the inscribed rectangle
        sx, sy, sw, sh = fx, fy, fw, fh
    gx0, gy0, gx1, gy1 = sx - ext, sy - ext, sx + sw + ext, sy + sh + ext
    out, prev = [], None
    talk = [l for l in lines if l["style"] not in ("caption", "sfx")]
    for n_b, l in enumerate(talk, 1):
        spk = next((nm for nm in stage if nm.lower() == l["speaker"].strip().lower()), None)
        head = head_px.get(spk) if spk else None
        anchor = ((head[0] + head[2]) / 2, (head[1] + head[3]) / 2) if head else None
        adj = adjust.get(n_b, {})
        best, best_s = None, 1e18
        if "x" in adj or "y" in adj or "size" in adj or "width" in adj:
            fs = max(6, int(font_size * adj.get("size", 1.0)))
            max_w = sw * adj.get("width", 0.38)
            rx, ry = _balloon_size(l["text"], font_name, fs, max_w, l["style"])
            if "x" in adj or "y" in adj:
                cx, cy = fx + adj.get("x", 0.5) * fw, fy + adj.get("y", 0.15) * fh
                core = _ellipse_core(cx, cy, rx, ry)
                best = dict(line=l, cx=cx, cy=cy, rx=rx, ry=ry, fs=fs, max_w=max_w,
                            head_cov=maps.frac(maps.heads, core), body_cov=maps.frac(maps.bodies, core),
                            detail=maps.frac(maps.detail, core), gutter=0.0, manual=True)
            sizes = [(fs, max_w)]
        else:
            sizes = None
        if best is None:
            fsc_list = (1.0, 0.9, 0.8, 0.7)
            for fi, fsc in enumerate(fsc_list):
                fs = max(8, int(font_size * fsc))
                for wi, wr in enumerate((0.3, 0.38, 0.48, 0.6)):
                    max_w = sw * wr
                    if sizes:
                        if fi or wi:
                            break
                        fs, max_w = sizes[0]
                    rx, ry = _balloon_size(l["text"], font_name, fs, max_w, l["style"])
                    if 2 * (rx + pad) > sw + 2 * ext or 2 * (ry + pad) > sh + 2 * ext:
                        continue
                    xs = np.linspace(gx0 + pad + rx, gx1 - pad - rx, 27)
                    ys = np.linspace(gy0 + pad + ry, gy1 - pad - ry, 27)
                    for cx in xs:
                        for cy in ys:
                            core = _ellipse_core(cx, cy, rx, ry)
                            if any(_box_overlap(core, o) > 0.0 for o in occupied):
                                continue                         # never stack balloons
                            if maps.frac(maps.outside, core) > 0.0:
                                continue                         # own panel or white gutter only
                            if not (fx <= cx <= fx + fw and fy <= cy <= fy + fh):
                                continue                         # the centre stays inside the panel
                            hc = maps.frac(maps.heads, core)      # share of the balloon lying on faces
                            bc = maps.frac(maps.bodies, core)     # ... on bodies (weighted)
                            dt = maps.frac(maps.detail, core)     # ... on busy detail
                            gt = maps.frac(maps.gutter, core)     # ... sticking out over the border
                            sc = hc * 8000 + bc * 3200 + dt * 220 + gt * 1800
                            sc += fi * 320 + abs(wi - 1) * 25    # prefer the normal font size and a rounder shape
                            sc += (cy - fy) / fh * 60            # comics letter from the top
                            if anchor:
                                ax, ay = anchor
                                d = np.hypot((cx - ax) / fw, (cy - ay) / fh)
                                sc += d * 380                     # close to the speaker ...
                                if cy - ry * 0.3 > ay:
                                    sc += 120                     # ... above or beside the face ...
                                if cy - ry * 0.5 > head[3]:
                                    sc += 700                     # ... and never below the speaker
                                # never closer to someone else's face: readers would think they're talking
                                for nm2, hb in head_px.items():
                                    if nm2 != spk and not nm2.startswith("_"):
                                        d2 = np.hypot((cx - (hb[0] + hb[2]) / 2) / fw, (cy - (hb[1] + hb[3]) / 2) / fh)
                                        if d2 < d * 0.85:
                                            sc += 450
                            if prev is not None and cy + ry * 0.5 < prev[1] - ry and cx < prev[0] - rx:
                                sc += 80                          # reading order
                            if sc < best_s:
                                best_s = sc
                                best = dict(line=l, cx=cx, cy=cy, rx=rx, ry=ry, fs=fs, max_w=max_w,
                                            head_cov=hc, body_cov=bc, detail=dt, gutter=gt)
        if best is None:   # text too long for this panel: smallest font, top centre
            fs = max(8, int(font_size * 0.7))
            rx, ry = _balloon_size(l["text"], font_name, fs, sw * 0.8, l["style"])
            best = dict(line=l, cx=sx + sw / 2, cy=sy + pad + ry, rx=rx, ry=ry, fs=fs, max_w=sw * 0.8,
                        head_cov=1.0, body_cov=1.0, detail=1.0, gutter=0.0)
        cx, cy, rx, ry = best["cx"], best["cy"], best["rx"], best["ry"]
        # tail: from the balloon edge towards the speaker's face, stopping just short of it
        if "tail_x" in adj or "tail_y" in adj:
            tip = (fx + adj.get("tail_x", 0.5) * fw, fy + adj.get("tail_y", 0.5) * fh)
        elif head:
            hx = min(max(cx, head[0]), head[2])
            hy = head[1] if cy < head[1] else (head[3] if cy > head[3] else (head[1] + head[3]) / 2)
            if head[0] <= cx <= head[2] and head[1] <= cy <= head[3]:
                hx, hy = (head[0] + head[2]) / 2, head[1]
            vx, vy = hx - cx, hy - cy
            dist = max(1.0, np.hypot(vx, vy))
            edge = 1.0 / max(1e-6, np.sqrt((vx / dist / rx) ** 2 + (vy / dist / ry) ** 2))  # centre -> rim
            reach = max(edge + 0.04 * min(fw, fh), min(dist - 0.02 * min(fw, fh), edge + 0.16 * min(fw, fh)))
            tip = (cx + vx / dist * reach, cy + vy / dist * reach)
            tip = (min(max(tip[0], fx + pad), fx + fw - pad), min(max(tip[1], fy + pad), fy + fh - pad))
        else:
            tip = (cx + (0.08 * fw if cx < fx + fw / 2 else -0.08 * fw), cy + ry + 0.1 * fh)
            tip = (min(max(tip[0], fx + pad), fx + fw - pad), min(max(tip[1], fy + pad), fy + fh - pad))
        best["tail"] = tip
        best["number"] = n_b
        occupied.append((cx - rx, cy - ry, cx + rx, cy + ry))
        out.append(best)
        prev = (cx, cy)
    return out


def balloon_fit(panel_img, lines, font_name, font_size, stage, heads, bodies, poly=None, blocked=None):
    """Dry run on a raw panel image: how badly do its balloons have to cover characters? (0 = all in free space).
    poly: the panel's shape in this image's pixel coordinates (slanted panels have less room)."""
    w, h = panel_img.size
    inset = int(min(w, h) * 0.04)
    caps = [l for l in lines if l["style"] == "caption"]
    occupied = []
    if caps:   # captions take the top-left corner first
        f = load_font(font_name, font_size)
        sw = w - 2 * inset
        pad = max(8, int(min(sw, h - 2 * inset) * 0.03))
        y = inset + pad
        for l in caps:     # same geometry as smart_letter_panel's captions
            _, tw, th, _ = text_block(l["text"], f, sw * 0.7 - font_size)
            occupied.append((inset + pad, y, inset + pad + tw + font_size, y + th + font_size))
            y = occupied[-1][3] + pad
    E = int(min(w, h) * 0.12) if blocked is not None else 0
    if E:   # surround the panel with white page so balloons can test breaking out into the gutter
        big = Image.new("RGB", (w + 2 * E, h + 2 * E), (255, 255, 255))
        big.paste(panel_img.convert("RGB"), (E, E))
        sh_ = lambda pts: [(x + E, y + E) for x, y in pts]
        occupied = [(o[0] + E, o[1] + E, o[2] + E, o[3] + E) for o in occupied]
        pl = plan_balloons(big, (E, E, w, h), (E + inset, E + inset, w - 2 * inset, h - 2 * inset), lines, font_name,
                           font_size, stage, heads, bodies, occupied, sh_(poly) if poly else None,
                           [sh_(b) for b in blocked])
    else:
        pl = plan_balloons(panel_img, (0, 0, w, h), (inset, inset, w - 2 * inset, h - 2 * inset), lines, font_name,
                           font_size, stage, heads, bodies, occupied, poly, blocked)
    if not pl:
        return 0.0
    # covering faces/bodies is bad; having to shrink the text to the smallest size is a warning sign too
    return max(p["head_cov"] * 4 + p["body_cov"] + (0.4 if p["fs"] < font_size * 0.75 else 0.0) for p in pl)


def smart_letter_panel(img, full_rect, safe_rect, lines, font_name, font_size, stage=None, heads=None, bodies=None,
                       poly=None, blocked=None, adjust=None):
    """Captions top-left, balloons in free space near their speaker (tails pointing at them), SFX in a calm spot."""
    stage = stage or {}
    fx, fy, fw, fh = full_rect
    sx, sy, sw, sh = safe_rect
    if sw < 20 or sh < 20:
        return
    pad = max(8, int(min(sw, sh) * 0.03))
    occupied = []
    # 1) captions: top-left corner, like classic narration boxes
    cursor_y = sy + pad
    for l in lines:
        if l["style"] != "caption":
            continue
        f = load_font(font_name, font_size)
        _, tw, th, _ = text_block(l["text"], f, sw * 0.7 - font_size)
        bb = draw_bubble(img, l["text"], sx + pad + (tw + font_size) / 2, cursor_y + th / 2 + font_size * 0.5,
                         sw * 0.7 - font_size, font_name, font_size, "caption")
        if bb:
            occupied.append(bb)
            cursor_y = bb[3] + pad
    # 2) balloons in free space (planned on the clean art, before anything is drawn)
    plans = plan_balloons(img, full_rect, safe_rect, lines, font_name, font_size, stage, heads, bodies, occupied, poly,
                          blocked, adjust)
    for p in plans:
        draw_bubble(img, p["line"]["text"], p["cx"], p["cy"], p["max_w"], font_name, p["fs"], p["line"]["style"],
                    tail=p["tail"])
        occupied.append((p["cx"] - p["rx"], p["cy"] - p["ry"], p["cx"] + p["rx"], p["cy"] + p["ry"]))
    # 3) sound effects: lower half, on background or at most on a body - never on a face or a balloon
    P = lambda b: (fx + b[0] * fw, fy + b[1] * fh, fx + b[2] * fw, fy + b[3] * fh)
    maps = _Maps(img, full_rect, [P(b) for b in (heads or {}).values()], [P(b) for b in (bodies or [])],
                 poly=poly, inset=pad)
    for l in lines:
        if l["style"] != "sfx":
            continue
        fname = "Bangers-Regular.ttf" if "Bangers-Regular.ttf" in list_fonts() else font_name
        _, tw, th, _ = text_block(str(l["text"]).upper(), load_font(fname, font_size * 3.0), sw * 0.75, spacing=0.0)
        hw, hh = tw / 2 + font_size * 0.8, th / 2 + font_size * 0.9      # incl. outline and the slight rotation
        best, best_s = (sx + sw / 2, sy + sh * 0.7), 1e18
        for cx in np.linspace(fx + hw, fx + fw - hw, 11):
            for cy in np.linspace(fy + fh * 0.3, fy + fh - hh, 9):
                box = (cx - hw, cy - hh, cx + hw, cy + hh)
                sc = sum(_box_overlap(box, o) for o in occupied) * 4000
                sc += maps.frac(maps.heads, box) * 4000 + maps.frac(maps.bodies, box) * 900
                sc += maps.frac(maps.outside, box) * 8000
                sc += (fy + fh - cy) / fh * 60          # sound effects sit low in the panel
                if sc < best_s:
                    best, best_s = (cx, cy), sc
        bb = draw_bubble(img, l["text"], best[0], best[1], sw * 0.5, font_name, font_size, "sfx")
        if bb:
            occupied.append(bb)
    return plans


# --------------------------------------------------------------------------
# Find the characters in a finished panel (colour matching against the sheet)
# --------------------------------------------------------------------------

def _color_bins(arr):
    """arr uint8 HxWx3 -> bin index per pixel (8 levels per channel) + 'is coloured' mask."""
    a = arr.astype(np.int32)
    mx, mn = a.max(-1), a.min(-1)
    sat = (mx - mn) / np.maximum(mx, 1)
    coloured = (sat > 0.22) & (mx > 60)
    bins = (a[..., 0] // 32) * 64 + (a[..., 1] // 32) * 8 + (a[..., 2] // 32)
    return bins, coloured


def sheet_figure_boxes(sheet, n):
    """Find the n figures on a character sheet (left to right) by their non-white columns.
    Returns [(x0, y0, x1, y1)] in pixels, or None when exactly n figures can't be separated."""
    if sheet is None or n <= 0:
        return None
    g = np.asarray(sheet.convert("L").resize((max(64, sheet.width // 4), max(32, sheet.height // 4))), dtype=np.float32)
    sc = sheet.width / float(g.shape[1])
    bg = np.median(np.concatenate([g[0], g[-1], g[:, 0], g[:, -1]]))
    ink = np.abs(g - bg) > 35
    col = ink.sum(axis=0) > max(1, 0.01 * g.shape[0])
    runs, start = [], None
    for x, v in enumerate(list(col) + [False]):
        if v and start is None:
            start = x
        elif not v and start is not None:
            runs.append([start, x]); start = None
    runs = [r for r in runs if r[1] - r[0] >= 2]
    if len(runs) < n:
        return None
    while len(runs) > n:   # merge the pair with the smallest gap (an arm or tail split off a figure)
        k = min(range(len(runs) - 1), key=lambda j: runs[j + 1][0] - runs[j][1])
        runs[k] = [runs[k][0], runs[k + 1][1]]; del runs[k + 1]
    gaps = [runs[j + 1][0] - runs[j][1] for j in range(len(runs) - 1)]
    if gaps and min(gaps) < 2:
        return None
    out = []
    for x0, x1 in runs:
        rows = np.flatnonzero(ink[:, x0:x1].any(axis=1))
        y0, y1 = (rows[0], rows[-1] + 1) if len(rows) else (0, g.shape[0])
        out.append((int(x0 * sc), int(y0 * sc), int(x1 * sc), int(y1 * sc)))
    return out


def sheet_subset(sheet, boxes, picks, pad=0.06):
    """Paste only the picked figures (indices into boxes) side by side on a clean white sheet."""
    figs = []
    for k in picks:
        x0, y0, x1, y1 = boxes[k]
        px, py = int((x1 - x0) * 0.08) + 4, int((y1 - y0) * 0.03) + 4
        figs.append(sheet.convert("RGB").crop((max(0, x0 - px), max(0, y0 - py),
                                               min(sheet.width, x1 + px), min(sheet.height, y1 + py))))
    H = max(f.height for f in figs)
    gap = int(H * pad)
    W = sum(f.width for f in figs) + gap * (len(figs) + 1)
    out = Image.new("RGB", (max(W, int(H * 0.75)), H + 2 * gap), (255, 255, 255))
    x = (out.width - (W - 2 * gap)) // 2
    for f in figs:
        out.paste(f, (x, gap + (H - f.height)))
        x += f.width + gap
    return out


def character_palettes(sheet, n, owners=None):
    """Split the sheet into n vertical strips (one per character, in cast order) and return, per
    character, the colour bins that are typical for it and NOT typical for the others."""
    if sheet is None or n <= 0:
        return []
    s = sheet.convert("RGB")
    s = s.resize((max(64, s.width // 4), max(64, s.height // 4)))
    arr = np.asarray(s)
    bins, col = _color_bins(arr)
    hists = []
    w = arr.shape[1]
    fb = sheet_figure_boxes(sheet, n)
    for k in range(n):
        x0, x1 = int(w * k / n), int(w * (k + 1) / n)
        if fb:
            x0, x1 = int(fb[k][0] / sheet.width * w), max(int(fb[k][0] / sheet.width * w) + 1, int(fb[k][2] / sheet.width * w))
        b = bins[:, x0:x1][col[:, x0:x1]]
        h = np.bincount(b, minlength=512).astype(np.float64)
        hists.append(h / max(1.0, h.sum()))
    pals = []
    for k, h in enumerate(hists):
        own = owners or list(range(n))
        rest = [hh for j, hh in enumerate(hists) if j != k and own[j] != own[k]]
        others = np.max(rest, axis=0) if rest else np.zeros(512)
        keep = (h > 0.012) & (h > others * 2.5)
        pals.append(np.flatnonzero(keep))
    return pals


def locate_characters(panel, names, palettes):
    """Return {name: (x0, y0, x1, y1)} in 0..1 panel coords for characters whose colours are found."""
    out = {}
    if not palettes:
        return out
    p = panel.convert("RGB")
    sc = 160.0 / max(1, p.width)
    p = p.resize((160, max(16, int(p.height * sc))))
    arr = np.asarray(p)
    bins, col = _color_bins(arr)
    H, W = bins.shape
    # colours that dominate the panel border (sky, walls) are background, not a character
    bh, bw = max(2, H // 12), max(2, W // 12)
    regions = [bins[:bh, :], bins[:, :bw], bins[:, -bw:]]
    bg = set()
    for b in np.unique(bins[col]):
        hits = sum(1 for r in regions if (r == b).mean() > 0.2)
        if hits >= 2 or ((bins == b).mean() > 0.15 and (regions[0] == b).mean() > 0.15):
            bg.add(int(b))
    for name, pal in zip(names, palettes):
        if len(pal) == 0:
            continue
        pal = [b for b in pal if b not in bg]
        if not pal:
            continue
        m = np.isin(bins, pal) & col
        if m.mean() < 0.006:
            continue
        ys, xs = _main_blob(m)
        if len(xs) < 20:
            continue
        x0, x1 = np.percentile(xs, 4), np.percentile(xs, 96)
        y0, y1 = np.percentile(ys, 2), np.percentile(ys, 97)
        out[name] = (x0 / W, y0 / H, x1 / W, y1 / H)
    return out


def _main_blob(m):
    """Largest plausible connected blob of a boolean mask (after a small dilation so a character's
    hair / coat / boots join up). Blobs that look like sky (touching the top edge over a wide span) lose."""
    H, W = m.shape
    d = m.copy()
    for dy in (-2, -1, 1, 2):
        d |= np.roll(m, dy, 0)
    for dx in (-2, -1, 1, 2):
        d |= np.roll(m, dx, 1)
    lab = np.zeros((H, W), np.int32)
    cur = 0
    best, best_score = None, 0
    for y0 in range(H):
        for x0 in range(W):
            if not d[y0, x0] or lab[y0, x0]:
                continue
            cur += 1
            stack = [(y0, x0)]
            lab[y0, x0] = cur
            pts = []
            while stack:
                y, x = stack.pop()
                pts.append((y, x))
                for ny, nx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                    if 0 <= ny < H and 0 <= nx < W and d[ny, nx] and not lab[ny, nx]:
                        lab[ny, nx] = cur
                        stack.append((ny, nx))
            pa = np.array(pts)
            real = m[pa[:, 0], pa[:, 1]]
            area = real.sum()
            top_span = (pa[:, 0] <= 1).sum() / float(W)
            score = area * (0.25 if top_span > 0.35 else 1.0)
            if score > best_score:
                best, best_score = pa[real], score
    if best is None:
        return np.array([]), np.array([])
    return best[:, 0], best[:, 1]


def stage_from_detection(found):
    """Convert detected boxes into the (x_frac, head_y) staging format + head boxes."""
    stage, heads = {}, {}
    for nm, (x0, y0, x1, y1) in found.items():
        bh = y1 - y0
        head_h = min(bh, max(0.12, 0.35 * bh))
        hx = (x0 + x1) / 2
        stage[nm] = (hx, min(0.95, y0 + head_h * 0.5))
        heads[nm] = (x0, max(0.0, y0 - 0.03), x1, min(1.0, y0 + head_h + 0.03))
    return stage, heads


# --------------------------------------------------------------------------
# Vision check (Qwen3.5 VL): parse boxes, judge a panel
# --------------------------------------------------------------------------

def vlm_check_prompt(cast_desc, expected):
    """Pass 1: characters with labels, in Qwen-VL's native grounding format (most reliable for small models)."""
    known = "; ".join(f"{n} = {d}" for n, d in cast_desc.items()) or "none"
    return (f"Known characters: {known}.\n"
            f"Detect every character in this comic panel: every person, animal and robot, including background "
            f"figures and the same character drawn twice. Label each one with the known name only when it clearly "
            f"matches that description (hair, clothes, colours, body type); label anything else \"other\" "
            f"(for example a bird, a dog or a stranger).\n"
            f'Output JSON only: [{{"bbox_2d": [x1, y1, x2, y2], "label": "Name"}}, ...]')


def vlm_face_prompt():
    """Pass 2: every head/face, unlabelled."""
    return ("Detect the head of every person, animal and robot in this comic panel (the whole head, including hair). "
            'Output JSON only: [{"bbox_2d": [x1, y1, x2, y2], "label": "head"}, ...]')


PEOPLE_WORDS = ("people", "person", "persons", "man", "men", "woman", "women", "boy", "boys", "girl", "girls", "kid",
                "kids", "child", "children", "crowd", "stranger", "strangers", "vendor", "seller", "someone",
                "passer", "passers-by", "bystander", "bystanders", "police", "officer", "father", "mother", "family",
                "friends", "audience", "mensen", "vrouw", "jongen", "meisje", "kind", "kinderen", "iemand",
                "verkoper", "publiek", "menigte", "voorbijganger", "voorbijgangers", "agent", "vader", "moeder")


def scene_allows_people(desc):
    words = set(re.findall(r"[\w-]+", str(desc).lower()))
    return any(w in words for w in PEOPLE_WORDS)


def vlm_person_prompt():
    return ("Is there a human person (or a human hand or arm) in the middle of this image? "
            "Animals and robots do not count. Answer with one word: yes or no.")


def short_desc(desc, words=9):
    w = re.split(r"\s+", str(desc).strip().rstrip("."))
    return " ".join(w[:words]).rstrip(",;")


def where_in_panel(box):
    cx = (box[0] + box[2]) / 2
    cy = (box[1] + box[3]) / 2
    h = "on the left" if cx < 0.38 else ("on the right" if cx > 0.62 else "in the middle")
    v = " at the top" if cy < 0.3 else (" at the bottom" if cy > 0.75 else "")
    return h + v


def fix_instructions(dets, pan, cast_desc, names):
    """Edit instructions for what a panel shouldn't contain: extra copies, strangers, cast members not in it."""
    fixes = []
    expected = pan.get("cast", [])
    stage = pan.get("stage") or {}
    for nm in names:
        mine = [d for d in dets if d["label"] == nm]
        if nm in expected and len(mine) > 1:
            if nm in stage:
                keep = min(mine, key=lambda d: abs((d["box"][0] + d["box"][2]) / 2 - stage[nm][0]))
            else:
                keep = max(mine, key=lambda d: (d["box"][2] - d["box"][0]) * (d["box"][3] - d["box"][1]))
            for d in mine:
                if d is not keep:
                    fixes.append(f"remove the second {short_desc(cast_desc.get(nm, nm))} {where_in_panel(d['box'])} "
                                 f"(it is a duplicate; keep the one {where_in_panel(keep['box'])}) and paint matching "
                                 f"background where it was")
        elif nm not in expected:
            for d in mine:
                fixes.append(f"remove the {short_desc(cast_desc.get(nm, nm))} {where_in_panel(d['box'])} and paint "
                             f"matching background where it was")
    for d in dets:
        if d.get("stranger"):
            fixes.append(f"remove the person {where_in_panel(d['box'])}, including their arms and hands, and paint "
                         f"matching background where they were")
    return fixes


def vlm_identity_prompt(name, desc):
    return (f"Is the character in the middle of this image {name}: {desc.rstrip('.')}? "
            f"Answer with one word: yes or no.")


def vlm_output_unreliable(text, dets):
    """Small models sometimes loop (same box over and over, keys repeated, output cut off)."""
    t = str(text).split("</think>")[-1]
    boxes = re.findall(r"\[\s*-?[\d.]+\s*,\s*-?[\d.]+\s*,\s*-?[\d.]+\s*,\s*-?[\d.]+\s*\]", t)
    if "]" not in t.strip()[-12:] and len(boxes) > 3:
        return True
    return len(boxes) >= 6 and len(set(boxes)) <= len(boxes) / 2


def dedupe_dets(dets, iou=0.85):
    out = []
    for d in dets:
        if any(_iou(d["box"], o["box"]) > iou for o in out):
            continue
        out.append(d)
    return out


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def attach_faces(dets, faces):
    """Give each body the head box that lies (mostly) inside its upper part; leftover heads become 'other' heads."""
    used = set()
    for d in dets:
        x0, y0, x1, y1 = d["box"]
        best, bs = None, 0.0
        for k, f in enumerate(faces):
            if k in used:
                continue
            ix = max(0.0, min(x1, f[2]) - max(x0, f[0])); iy = max(0.0, min(y1, f[3]) - max(y0, f[1]))
            inside = ix * iy / max(1e-6, (f[2] - f[0]) * (f[3] - f[1]))
            high = 1.0 - max(0.0, ((f[1] + f[3]) / 2 - y0) / max(1e-6, y1 - y0))   # prefer heads near the top
            sc = inside * (0.5 + high)
            if inside > 0.6 and sc > bs:
                best, bs = k, sc
        if best is not None:
            used.add(best)
            d["face"] = faces[best]
    extra = [f for k, f in enumerate(faces) if k not in used]
    return dets, extra


def vlm_feature_prompt(name, desc, feature):
    # neutral description: drop every part that mentions the feature, so the question is not leading
    words = [w for w in re.findall(r"\w+", feature.lower()) if len(w) > 2]
    desc = ", ".join(p.strip() for p in re.split(r"[,;]", desc) if not any(w in p.lower() for w in words))
    return (f"Look at the character {name} in this image ({desc.rstrip('.')}). "
            f"Question: is there a {feature} on top of {name}'s head or antenna? "
            f"Look carefully. Answer with one word: yes or no.")


def parse_yes_no(text):
    t = str(text).split("</think>")[-1].strip().lower()
    m = re.search(r"\b(yes|no|ja|nee)\b", t)
    return bool(m) and m.group(1) in ("yes", "ja")


def _norm_box(b, w, h, scale):
    try:
        x1, y1, x2, y2 = [float(v) for v in b[:4]]
    except Exception:
        return None
    if scale == "frac":
        pass
    elif scale == "k":
        x1, y1, x2, y2 = x1 / 1000, y1 / 1000, x2 / 1000, y2 / 1000
    else:
        x1, y1, x2, y2 = x1 / w, y1 / h, x2 / w, y2 / h
    x1, x2 = sorted((min(max(x1, 0), 1), min(max(x2, 0), 1)))
    y1, y2 = sorted((min(max(y1, 0), 1), min(max(y2, 0), 1)))
    if x2 - x1 < 0.01 or y2 - y1 < 0.01:
        return None
    return (x1, y1, x2, y2)


def parse_vlm_boxes(text, img_w, img_h, names):
    """Tolerant parser for Qwen-VL style grounding output -> [{label, box, face}] in 0..1 coords.
    Reads each {...} object with regexes (small models sometimes repeat keys; the FIRST value wins)."""
    text = str(text).split("</think>")[-1]
    objs = re.findall(r"\{[^{}]*\}", text, re.S)
    num = r"\[\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\]"
    raw = []
    for o in objs:
        lm = re.search(r'"label"\s*:\s*"([^"]*)"', o)
        bm = re.search(r'"(?:bbox_2d|bbox)"\s*:\s*' + num, o)
        fm = re.search(r'"face_2d"\s*:\s*' + num, o)
        if not bm:
            continue
        raw.append((lm.group(1) if lm else "other", [float(v) for v in bm.groups()],
                    [float(v) for v in fm.groups()] if fm else None))
    vals = [v for _, b, f in raw for v in (b + (f or []))]
    mx = max(vals) if vals else 0
    scale = "frac" if mx <= 1.5 else ("k" if mx <= 1000 else "px")
    out = []
    lower = {n.lower(): n for n in names}
    for lab, b, f in raw:
        lab = lower.get(lab.strip().lower(), next((n for k, n in lower.items() if k in lab.lower()), "other"))
        box = _norm_box(b, img_w, img_h, scale)
        if box is None:
            continue
        face = _norm_box(f, img_w, img_h, scale) if f else None
        out.append({"label": lab, "box": box, "face": face})
    return out


def judge_panel(dets, expected, all_names, unreliable=False):
    """Lower is better. Duplicates of a character are the worst error."""
    if unreliable:
        return 0.3, ["check unclear"]
    counts = {n: 0 for n in all_names}
    for d in dets:
        if d["label"] in counts:
            counts[d["label"]] += 1
    err, notes = 0.0, []
    for n in all_names:
        c = counts[n]
        if n in expected:
            if c == 0:
                err += 0.5; notes.append(f"{n} missing")
            elif c > 1:
                err += 2.0 * (c - 1); notes.append(f"{n} x{c}")
        elif c > 0:
            err += 1.0 * c; notes.append(f"{n} should not be here")
    return err, notes


def stage_from_vlm(dets):
    """VLM detections -> (stage {name: (x, y)}, heads {name: box}, bodies [box...]) in 0..1 coords."""
    stage, heads, bodies = {}, {}, []
    by = {}
    for d in dets:
        bodies.append(d["box"])
        if d["label"] != "other":
            a = (d["box"][2] - d["box"][0]) * (d["box"][3] - d["box"][1])
            if d["label"] not in by or a > by[d["label"]][0]:
                by[d["label"]] = (a, d)
    for nm, (_, d) in by.items():
        x0, y0, x1, y1 = d["box"]
        top = (x0, y0, x1, y0 + min(y1 - y0, max(0.12, 0.3 * (y1 - y0))))   # head region of the body box
        # no face found (robots, animals, back views): protect the whole upper part - eyes can sit low on a body
        f = d["face"] or (x0, y0, x1, y0 + 0.65 * (y1 - y0))
        if d["face"] and (d["face"][2] - d["face"][0]) * (d["face"][3] - d["face"][1]) < \
                0.25 * (top[2] - top[0]) * (top[3] - top[1]):   # just an eye was returned: grow to the head region
            f = (min(f[0], top[0] + (top[2] - top[0]) * 0.15), min(f[1], top[1]), max(f[2], top[2] - (top[2] - top[0]) * 0.15),
                 max(f[3], top[3]))
        fx0, fy0, fx1, fy1 = f
        mx, my = (fx1 - fx0) * 0.08, (fy1 - fy0) * 0.08
        heads[nm] = (max(0, fx0 - mx), max(0, fy0 - my), min(1, fx1 + mx), min(1, fy1 + my))
        stage[nm] = ((fx0 + fx1) / 2, (fy0 + fy1) / 2)
    for k, d in enumerate(dets):   # every other face (duplicates, bystanders, animals) must stay visible too
        if d["face"] and not any(d is by[n][1] for n in by):
            heads[f"_face{k}"] = d["face"]
    return stage, heads, bodies
