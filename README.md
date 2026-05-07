# ComfyUI-VS
Visual comparison and retro transition nodes for ComfyUI

A small ComfyUI custom node pack for visual comparison tools.

## Included Nodes

### VS Image Compare Deluxe

Creates animated transitions between a source image and a generated image. Useful for before/after comparisons, upscale comparisons, prompt tests, restoration previews, and general retro nonsense.

## Effects

- Fade
- Dissolve
- Circle Wipe
- Diamond Wipe
- Star Wipe
- Clock Wipe
- Door Wipe Open
- Door Wipe Close
- Iris Open
- Iris Close
- Heart Wipe
- Spiral Wipe
- Venetian Blinds
- Checkerboard
- Falling Blocks
- Random Blocks
- Pixelate Reveal
- Luma Wipe
- Push Left
- Push Right
- Toaster Color Shift

## Basic Usage

1. Connect your original/source image to `source_image`.
2. Connect your generated/edited image to `generated_image`.
3. Choose an `effect_type`.
4. Set `frames` and `fps`.
5. Connect `frames_out` to a preview or video save node.

The node does not directly save video files. It outputs an image batch designed to work with video save/combine nodes.

## Install

Clone this repo into your ComfyUI `custom_nodes` folder:

```bash
git clone https://github.com/Jackson4242/ComfyUI-VS.git
