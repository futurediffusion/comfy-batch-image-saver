# ComfyUI Batch Image Saver

Custom node for ComfyUI that focuses on saving image batches quickly without embedding any metadata. The node keeps only the inputs required to define where and how files are written: output path, filename pattern and file extension.

## Features

- Saves every image in the incoming batch to the selected output directory.
- Supports `png`, `jpeg` and `webp` formats without adding EXIF or PNG chunks.
- Filename and subfolder can be customised with automatic tokens.

## Automatic tokens

You can combine any of the following tokens inside the **Path** or **Filename** fields:

| Token | Description |
| ----- | ----------- |
| `%time` | Current timestamp using `%Y-%m-%d-%H%M%S`. |
| `%date` | Current date using `%Y-%m-%d`. |
| `%seed` | First seed value found in the workflow metadata. Falls back to `unknown` if unavailable. |
| `%model` | First model or checkpoint name found in the workflow metadata. Falls back to `unknown` if unavailable. |
| `%counter` | Incremental counter that increases each time the node runs. |

Example: `outputs/%date/session_%counter` combined with `sample_%time_%seed` produces a directory structure similar to `outputs/2024-04-20/session_3/sample_2024-04-20-153010_12345.png`.

## Installation

1. Go to the `custom_nodes` directory of your ComfyUI installation.
2. Clone this repository: `git clone https://github.com/giriss/comfy-image-saver.git`
3. Restart ComfyUI.

No additional Python dependencies are required.
