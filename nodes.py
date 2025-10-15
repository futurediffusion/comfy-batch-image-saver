import os
from datetime import datetime
from typing import Any, Iterable

import numpy as np
from PIL import Image

import folder_paths


def _handle_whitespace(text: str) -> str:
    return text.strip().replace("\n", " ").replace("\r", " ").replace("\t", " ")


def _get_timestamp(time_format: str) -> str:
    now = datetime.now()
    try:
        return now.strftime(time_format)
    except Exception:
        return now.strftime("%Y-%m-%d-%H%M%S")


def _make_pathname(template: str, values: dict[str, Any], time_format: str) -> str:
    path = template
    path = path.replace("%date", _get_timestamp("%Y-%m-%d"))
    path = path.replace("%time", _get_timestamp(time_format))
    path = path.replace("%model", _handle_whitespace(str(values.get("model", "unknown"))))
    path = path.replace("%seed", _handle_whitespace(str(values.get("seed", "unknown"))))
    path = path.replace("%counter", _handle_whitespace(str(values.get("counter", 0))))
    return path


def _make_filename(template: str, values: dict[str, Any], time_format: str) -> str:
    filename = _make_pathname(template, values, time_format)
    return _get_timestamp(time_format) if filename == "" else filename


def _flatten_dict_items(data: Any) -> Iterable[tuple[str, Any]]:
    if isinstance(data, dict):
        for key, value in data.items():
            yield key, value
            yield from _flatten_dict_items(value)
    elif isinstance(data, (list, tuple)):
        for item in data:
            yield from _flatten_dict_items(item)


def _extract_first_value(data: Any, target_keys: set[str]) -> Any | None:
    for key, value in _flatten_dict_items(data):
        if key in target_keys:
            return value
    return None


class BatchImageSaver:
    def __init__(self):
        self.output_dir = folder_paths.output_directory
        self._save_counter = 0

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename": ("STRING", {"default": "%time_%seed", "multiline": False}),
                "path": ("STRING", {"default": "", "multiline": False}),
                "extension": (["png", "jpeg", "webp"],),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "ImageSaverTools"

    def save_images(self, images, filename, path, extension, prompt=None, extra_pnginfo=None):
        self._save_counter += 1

        metadata_source = extra_pnginfo if extra_pnginfo is not None else {}
        model = _extract_first_value(metadata_source, {"model", "model_name", "ckpt_name"})
        if model is None and prompt is not None:
            model = _extract_first_value(prompt, {"model", "ckpt_name"})

        seed = _extract_first_value(metadata_source, {"seed"})
        if seed is None and prompt is not None:
            seed = _extract_first_value(prompt, {"seed"})

        values = {
            "model": model if model is not None else "unknown",
            "seed": seed if seed is not None else "unknown",
            "counter": self._save_counter,
        }

        time_format = "%Y-%m-%d-%H%M%S"
        filename_base = _make_filename(filename, values, time_format)
        relative_path = _make_pathname(path, values, time_format)

        output_path = os.path.join(self.output_dir, relative_path)
        if output_path.strip() != "":
            os.makedirs(output_path, exist_ok=True)
        else:
            output_path = self.output_dir

        saved_files = self._write_images(images, output_path, filename_base, extension.lower())

        subfolder = os.path.normpath(relative_path)
        if subfolder == ".":
            subfolder = ""

        return {
            "ui": {
                "images": [
                    {"filename": name, "subfolder": subfolder, "type": "output"}
                    for name in saved_files
                ]
            }
        }

    def _write_images(self, images, output_path: str, filename_prefix: str, extension: str) -> list[str]:
        paths: list[str] = []
        batch_size = images.size()[0]

        for index, image in enumerate(images):
            array = 255.0 * image.cpu().numpy()
            img = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))

            if batch_size > 1:
                current_prefix = f"{filename_prefix}_{index + 1:02d}"
            else:
                current_prefix = filename_prefix

            filename = f"{current_prefix}.{extension}"
            file_path = os.path.join(output_path, filename)

            save_kwargs = {"optimize": True}
            if extension == "png":
                save_kwargs["compress_level"] = 4
            elif extension == "jpeg":
                save_kwargs["quality"] = 95
            elif extension == "webp":
                save_kwargs["quality"] = 95

            img.save(file_path, **save_kwargs)
            paths.append(filename)

        return paths


NODE_CLASS_MAPPINGS = {
    "Save Image Batch": BatchImageSaver,
}
