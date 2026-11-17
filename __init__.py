import os
import shutil

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from . import downloader as _dl

NODE_CLASS_MAPPINGS = {**NODE_CLASS_MAPPINGS, **_dl.NODE_CLASS_MAPPINGS}
NODE_DISPLAY_NAME_MAPPINGS = {**NODE_DISPLAY_NAME_MAPPINGS, **_dl.NODE_DISPLAY_NAME_MAPPINGS}

# Copy the example input images into ComfyUI/input on first load so the
# example workflow validates out of the box.
try:
    import folder_paths

    _src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "example_inputs")
    _dst = folder_paths.get_input_directory()
    for _f in os.listdir(_src):
        if not os.path.exists(os.path.join(_dst, _f)):
            shutil.copy2(os.path.join(_src, _f), os.path.join(_dst, _f))
except Exception as e:  # never break ComfyUI start-up
    print(f"[QwenComic] could not copy example inputs: {e}")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
