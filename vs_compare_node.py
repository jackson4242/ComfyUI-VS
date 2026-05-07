# ComfyUI-VS
# VS Image Compare Deluxe
# A retro transition/image comparison node for ComfyUI.
# Made for before/after comparisons, video previews

import math
import torch
import torch.nn.functional as F


def _single_frame(img: torch.Tensor) -> torch.Tensor:
    # ComfyUI IMAGE tensors are usually [B, H, W, C]
    if img.ndim == 4:
        return img[0]
    return img


def _resize_image(img: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    if img.shape[0] == target_h and img.shape[1] == target_w:
        return img

    x = img.permute(2, 0, 1).unsqueeze(0)  # HWC -> NCHW
    x = F.interpolate(x, size=(target_h, target_w), mode="bilinear", align_corners=False)
    return x.squeeze(0).permute(1, 2, 0)  # NCHW -> HWC


def _smoothstep01(x: torch.Tensor) -> torch.Tensor:
    x = torch.clamp(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _soft_mask(score: torch.Tensor, t: float, feather: float = 0.03) -> torch.Tensor:
    """
    score: H x W tensor in [0,1], lower values reveal earlier.
    returns: H x W x 1 mask in [0,1]
    """
    feather = max(float(feather), 1e-6)
    x = (t - score + feather) / (2.0 * feather)
    return _smoothstep01(x).unsqueeze(-1)


def _normalize_score(score: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """
    Normalize score map to [0, 1-eps] so feathered masks finish cleanly.
    This prevents tiny seams on the final frame.
    """
    score = score - torch.min(score)
    score = score / torch.clamp(torch.max(score), min=1e-6)
    return torch.clamp(score, 0.0, 1.0 - eps)


def _seeded_rand(shape, seed: int, device):
    # Generate on CPU for compatibility, then move to target device
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed) & 0x7FFFFFFFFFFFFFFF)
    return torch.rand(shape, generator=g, device="cpu").to(device)


def _normalized_grid(h: int, w: int, device):
    ys = torch.linspace(-1.0, 1.0, h, device=device).view(h, 1).expand(h, w)
    xs = torch.linspace(-1.0, 1.0, w, device=device).view(1, w).expand(h, w)
    return xs, ys


def _ease_value(t: float, easing: str) -> float:
    t = max(0.0, min(1.0, float(t)))

    if easing == "linear":
        return t
    if easing == "ease_in":
        return t * t
    if easing == "ease_out":
        return 1.0 - ((1.0 - t) * (1.0 - t))
    if easing == "ease_in_out":
        if t < 0.5:
            return 2.0 * t * t
        return 1.0 - ((-2.0 * t + 2.0) ** 2) / 2.0
    if easing == "smoothstep":
        return t * t * (3.0 - 2.0 * t)

    return t


class VSImageCompareDeluxe:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_image": ("IMAGE",),
                "generated_image": ("IMAGE",),
                "effect_type": (
                    [
                        "Fade",
                        "Dissolve",
                        "Circle Wipe",
                        "Diamond Wipe",
                        "Star Wipe",
                        "Clock Wipe",
                        "Door Wipe Open",
                        "Door Wipe Close",
                        "Iris Open",
                        "Iris Close",
                        "Heart Wipe",
                        "Spiral Wipe",
                        "Venetian Blinds",
                        "Checkerboard",
                        "Falling Blocks",
                        "Random Blocks",
                        "Pixelate Reveal",
                        "Luma Wipe",
                        "Push Left",
                        "Push Right",
                        "Toaster Color Shift",
                    ],
                    {"default": "Fade"},
                ),
                "frames": ("INT", {"default": 48, "min": 2, "max": 600, "step": 1}),
                "fps": ("INT", {"default": 12, "min": 1, "max": 120, "step": 1}),
                "loop_count": ("INT", {"default": 1, "min": 1, "max": 20, "step": 1}),
                "loop_style": (["restart", "ping_pong"], {"default": "restart"}),
                "hold_frames": ("INT", {"default": 0, "min": 0, "max": 120, "step": 1}),
                "easing": (
                    ["linear", "ease_in", "ease_out", "ease_in_out", "smoothstep"],
                    {"default": "smoothstep"},
                ),
                "feather": ("FLOAT", {"default": 0.025, "min": 0.001, "max": 0.25, "step": 0.001}),
                "block_size": ("INT", {"default": 48, "min": 8, "max": 256, "step": 8}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0x7FFFFFFF, "step": 1}),
                "invert_pattern": ("BOOLEAN", {"default": False}),
                "toaster_tint_strength": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01}),
                "video_flavor": (
                    ["none", "scanlines", "vhs_wobble", "crt_soft", "full_cheese"],
                    {"default": "none"},
                ),
                "video_flavor_strength": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("frames_out", "frame_count", "fps")
    FUNCTION = "build_compare"
    CATEGORY = "ComfyUI-VS"

    def _build_transition_state(self, effect_type: str, h: int, w: int, device, seed: int, block_size: int):
        state = {"effect_type": effect_type}
        xs, ys = _normalized_grid(h, w, device)

        radius = torch.sqrt(xs * xs + ys * ys)
        radius = _normalize_score(radius)

        theta = torch.atan2(ys, xs)

        if effect_type == "Dissolve":
            state["noise"] = _seeded_rand((h, w), seed, device)

        elif effect_type in ("Circle Wipe", "Iris Open"):
            state["score"] = radius

        elif effect_type == "Iris Close":
            state["score"] = 1.0 - radius

        elif effect_type == "Diamond Wipe":
            score = torch.abs(xs) + torch.abs(ys)
            state["score"] = _normalize_score(score)

        elif effect_type == "Star Wipe":
            # 5-point-ish star burst
            star_shape = 0.5 + 0.5 * torch.pow(torch.abs(torch.cos(theta * 5.0)), 0.65)
            score = radius / torch.clamp(star_shape, min=1e-6)
            state["score"] = _normalize_score(score)

        elif effect_type == "Clock Wipe":
            # Starts at top and moves clockwise
            angle = torch.atan2(xs, -ys)
            angle = torch.remainder(angle + (2.0 * math.pi), 2.0 * math.pi) / (2.0 * math.pi)
            state["score"] = _normalize_score(angle)

        elif effect_type == "Door Wipe Open":
            # Center reveals first, moving outward.
            score = torch.abs(xs)
            state["score"] = _normalize_score(score)

        elif effect_type == "Door Wipe Close":
            # Edges reveal first, finishing with both sides meeting at the center seam.
            score = 1.0 - torch.abs(xs)
            state["score"] = _normalize_score(score)

        elif effect_type == "Heart Wipe":
            # Heart-shaped implicit curve. Negative/low values reveal first from the heart center outward.
            # Coordinates adjusted so the heart fits the image.
            x = xs * 1.25
            y = -ys * 1.25 + 0.25
            heart = (x * x + y * y - 1.0) ** 3 - (x * x) * (y ** 3)

            # Inside heart is <= 0. Convert to a reveal score where heart appears first.
            score = heart
            state["score"] = _normalize_score(score)

        elif effect_type == "Spiral Wipe":
            # Spiraling score field.
            angle = torch.remainder(theta + (2.0 * math.pi), 2.0 * math.pi) / (2.0 * math.pi)
            spiral = radius + angle * 0.35
            state["score"] = _normalize_score(spiral)

        elif effect_type == "Venetian Blinds":
            stripe_h = max(4, block_size // 2)
            row_idx = torch.arange(h, device=device)
            stripe_idx = row_idx // stripe_h
            stripe_dir = (stripe_idx % 2).float().view(h, 1).expand(h, w)

            x01 = (torch.arange(w, device=device).float() / max(w - 1, 1)).view(1, w).expand(h, w)
            score = torch.where(stripe_dir < 0.5, x01, 1.0 - x01)
            state["score"] = _normalize_score(score)

        elif effect_type == "Checkerboard":
            tile = max(8, block_size)
            gy = (torch.arange(h, device=device) // tile).view(h, 1).expand(h, w)
            gx = (torch.arange(w, device=device) // tile).view(1, w).expand(h, w)

            parity = ((gx + gy) % 2).float()

            local_x = ((torch.arange(w, device=device) % tile).float() / max(tile - 1, 1)).view(1, w).expand(h, w)
            local_y = ((torch.arange(h, device=device) % tile).float() / max(tile - 1, 1)).view(h, 1).expand(h, w)
            local = (local_x + local_y) * 0.5

            # Two-phase reveal
            score = parity * 0.5 + local * 0.5
            state["score"] = _normalize_score(score)

        elif effect_type in ("Falling Blocks", "Random Blocks", "Pixelate Reveal"):
            grid_h = math.ceil(h / block_size)
            grid_w = math.ceil(w / block_size)

            delays = _seeded_rand((grid_h, grid_w), seed, device) * 0.75

            if effect_type == "Falling Blocks":
                row_bias = torch.linspace(0.0, 0.25, grid_h, device=device).view(grid_h, 1)
                delays = torch.clamp(delays * 0.65 + row_bias, 0.0, 0.95)

            by = torch.arange(h, device=device) // block_size
            bx = torch.arange(w, device=device) // block_size
            delay_map = delays[by[:, None], bx[None, :]]

            local_y = ((torch.arange(h, device=device) % block_size).float() / max(block_size - 1, 1)).view(h, 1).expand(h, w)
            local_x = ((torch.arange(w, device=device) % block_size).float() / max(block_size - 1, 1)).view(1, w).expand(h, w)

            detail_map = (local_y * 0.14) + (local_x * 0.06)

            state["delay_map"] = delay_map
            state["detail_map"] = detail_map

        elif effect_type == "Luma Wipe":
            # Luma based on generated image is built later because it needs dst.
            pass

        return state

    def _blend_from_mask(self, src: torch.Tensor, dst: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return torch.clamp(src * (1.0 - mask) + dst * mask, 0.0, 1.0)

    def _push_frame(self, src: torch.Tensor, dst: torch.Tensor, t: float, direction: str) -> torch.Tensor:
        h, w, c = src.shape
        out = torch.zeros_like(src)

        offset = int(round(w * t))
        offset = max(0, min(w, offset))

        if direction == "left":
            # src moves left out, dst comes in from right
            if offset < w:
                out[:, : w - offset, :] = src[:, offset:, :]
            if offset > 0:
                out[:, w - offset :, :] = dst[:, :offset, :]

        elif direction == "right":
            # src moves right out, dst comes in from left
            if offset < w:
                out[:, offset:, :] = src[:, : w - offset, :]
            if offset > 0:
                out[:, :offset, :] = dst[:, w - offset :, :]

        return out.clamp(0.0, 1.0)

    def _toaster_color_shift(self, src: torch.Tensor, dst: torch.Tensor, t: float, tint_strength: float) -> torch.Tensor:
        # Slight retro-blue/purple-ish broadcast cheese.
        s = max(0.0, min(1.0, float(tint_strength)))
        tint = torch.tensor(
            [1.0 + 0.15 * s, 1.0 + 0.05 * s, 1.0 + 0.45 * s],
            device=src.device,
            dtype=src.dtype,
        ).view(1, 1, 3)

        tinted_dst = torch.clamp(dst * tint, 0.0, 1.0)

        # Subtle scanline-style modulation.
        h, w, _ = src.shape
        y = torch.arange(h, device=src.device, dtype=src.dtype).view(h, 1, 1)
        scan = 1.0 - 0.03 * s * ((y % 2.0) == 0).to(src.dtype)
        tinted_dst = torch.clamp(tinted_dst * scan, 0.0, 1.0)

        return torch.clamp(src * (1.0 - t) + tinted_dst * t, 0.0, 1.0)

    def _pixelate_reveal(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        t: float,
        state: dict,
        block_size: int,
        feather: float,
        invert_pattern: bool,
    ) -> torch.Tensor:
        h, w, c = src.shape
        device = src.device

        delay_map = state["delay_map"]
        detail_map = state["detail_map"]
        score = torch.clamp(delay_map + detail_map, 0.0, 0.999)

        if invert_pattern:
            score = 1.0 - score

        mask = _soft_mask(score, t, feather=max(feather, 0.01))

        # Pixelated generated image. As t increases, pixel blocks get smaller.
        max_px = max(2, int(block_size))
        min_px = 1
        px = int(round(max_px * (1.0 - t) + min_px * t))
        px = max(1, px)

        small_h = max(1, h // px)
        small_w = max(1, w // px)

        x = dst.permute(2, 0, 1).unsqueeze(0)
        small = F.interpolate(x, size=(small_h, small_w), mode="nearest")
        pix = F.interpolate(small, size=(h, w), mode="nearest")
        pix = pix.squeeze(0).permute(1, 2, 0)

        # Blend from source into pixelated destination, which sharpens by the end.
        return self._blend_from_mask(src, pix, mask)

    def _video_flavor(
        self,
        frame: torch.Tensor,
        frame_index: int,
        total_frames: int,
        flavor: str,
        strength: float,
        seed: int,
    ) -> torch.Tensor:
        if flavor == "none" or strength <= 0.0:
            return frame

        s = max(0.0, min(1.0, float(strength)))
        h, w, c = frame.shape
        device = frame.device
        out = frame

        # Common scanline modulation.
        if flavor in ("scanlines", "crt_soft", "full_cheese"):
            y = torch.arange(h, device=device, dtype=frame.dtype).view(h, 1, 1)
            scan = 1.0 - (0.04 * s) * ((y % 2.0) == 0).to(frame.dtype)
            out = torch.clamp(out * scan, 0.0, 1.0)

        # VHS-ish horizontal wobble.
        if flavor in ("vhs_wobble", "full_cheese"):
            rows = torch.arange(h, device=device, dtype=frame.dtype)
            phase = (frame_index / max(total_frames - 1, 1)) * 2.0 * math.pi
            wobble = torch.sin(rows * 0.075 + phase * 3.0) * (2.5 * s)
            wobble = wobble.round().to(torch.int64)

            shifted = torch.empty_like(out)
            for yy in range(h):
                shifted[yy] = torch.roll(out[yy], shifts=int(wobble[yy].item()), dims=0)
            out = shifted

        # CRT softening / mild bloom-ish blur.
        if flavor in ("crt_soft", "full_cheese"):
            x = out.permute(2, 0, 1).unsqueeze(0)
            blurred = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
            blurred = blurred.squeeze(0).permute(1, 2, 0)
            out = torch.clamp(out * (1.0 - 0.25 * s) + blurred * (0.25 * s), 0.0, 1.0)

        # Tiny noise and chroma-ish shift for full cheese.
        if flavor == "full_cheese":
            noise = _seeded_rand((h, w, 1), seed + frame_index * 7919, device) - 0.5
            out = torch.clamp(out + noise * (0.035 * s), 0.0, 1.0)

            # Slight red/blue channel horizontal split.
            shift = max(1, int(round(2 * s)))
            r = torch.roll(out[:, :, 0], shifts=shift, dims=1)
            g = out[:, :, 1]
            b = torch.roll(out[:, :, 2], shifts=-shift, dims=1)
            out = torch.stack([r, g, b], dim=2).clamp(0.0, 1.0)

        return out.clamp(0.0, 1.0)

    def _mask_for_effect(
        self,
        t: float,
        effect_type: str,
        state: dict,
        src: torch.Tensor,
        dst: torch.Tensor,
        h: int,
        w: int,
        device,
        feather: float,
        invert_pattern: bool,
    ) -> torch.Tensor:
        if effect_type == "Fade":
            return torch.full((h, w, 1), float(t), device=device)

        if effect_type == "Dissolve":
            score = state["noise"]
            if invert_pattern:
                score = 1.0 - score
            return _soft_mask(score, t, feather=max(feather, 0.01))

        if effect_type == "Luma Wipe":
            # Use generated image brightness as the transition map.
            # Dark areas reveal earlier; invert_pattern flips that.
            luma = dst[:, :, 0] * 0.2126 + dst[:, :, 1] * 0.7152 + dst[:, :, 2] * 0.0722
            score = _normalize_score(luma)
            if invert_pattern:
                score = 1.0 - score
            return _soft_mask(score, t, feather=max(feather, 0.01))

        if effect_type in (
            "Circle Wipe",
            "Diamond Wipe",
            "Star Wipe",
            "Clock Wipe",
            "Door Wipe Open",
            "Door Wipe Close",
            "Iris Open",
            "Iris Close",
            "Heart Wipe",
            "Spiral Wipe",
            "Venetian Blinds",
            "Checkerboard",
        ):
            score = state["score"]
            if invert_pattern:
                score = 1.0 - score
            return _soft_mask(score, t, feather=feather)

        if effect_type in ("Falling Blocks", "Random Blocks"):
            delay_map = state["delay_map"]
            detail_map = state["detail_map"]
            score = torch.clamp(delay_map + detail_map, 0.0, 0.999)
            if invert_pattern:
                score = 1.0 - score
            return _soft_mask(score, t, feather=max(feather, 0.01))

        return torch.full((h, w, 1), float(t), device=device)

    def _render_frame(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        effect_type: str,
        t: float,
        state: dict,
        feather: float,
        block_size: int,
        invert_pattern: bool,
        toaster_tint_strength: float,
    ) -> torch.Tensor:
        h, w, _ = src.shape
        device = src.device

        if effect_type == "Push Left":
            return self._push_frame(src, dst, t, direction="left")

        if effect_type == "Push Right":
            return self._push_frame(src, dst, t, direction="right")

        if effect_type == "Toaster Color Shift":
            return self._toaster_color_shift(src, dst, t, toaster_tint_strength)

        if effect_type == "Pixelate Reveal":
            return self._pixelate_reveal(
                src=src,
                dst=dst,
                t=t,
                state=state,
                block_size=block_size,
                feather=feather,
                invert_pattern=invert_pattern,
            )

        mask = self._mask_for_effect(
            t=t,
            effect_type=effect_type,
            state=state,
            src=src,
            dst=dst,
            h=h,
            w=w,
            device=device,
            feather=feather,
            invert_pattern=invert_pattern,
        )
        return self._blend_from_mask(src, dst, mask)

    def build_compare(
        self,
        source_image,
        generated_image,
        effect_type,
        frames,
        fps,
        loop_count,
        loop_style,
        hold_frames,
        easing,
        feather,
        block_size,
        seed,
        invert_pattern,
        toaster_tint_strength,
        video_flavor,
        video_flavor_strength,
    ):
        src = _single_frame(source_image).to(torch.float32).clamp(0.0, 1.0)
        dst = _single_frame(generated_image).to(torch.float32).clamp(0.0, 1.0)

        # Match generated image to source image size.
        target_h, target_w = src.shape[0], src.shape[1]
        dst = _resize_image(dst, target_h, target_w).to(src.device)

        h, w, _ = src.shape
        device = src.device

        state = self._build_transition_state(
            effect_type=effect_type,
            h=h,
            w=w,
            device=device,
            seed=seed,
            block_size=block_size,
        )

        cycle_frames = []
        for i in range(frames):
            raw_t = i / max(frames - 1, 1)
            t = _ease_value(raw_t, easing)

            frame = self._render_frame(
                src=src,
                dst=dst,
                effect_type=effect_type,
                t=t,
                state=state,
                feather=feather,
                block_size=block_size,
                invert_pattern=bool(invert_pattern),
                toaster_tint_strength=toaster_tint_strength,
            )
            cycle_frames.append(frame)

        output_frames = []

        for _ in range(loop_count):
            output_frames.extend(cycle_frames)

            for _hf in range(hold_frames):
                output_frames.append(dst.clone())

            if loop_style == "ping_pong" and len(cycle_frames) > 2:
                reverse_frames = list(reversed(cycle_frames[1:-1]))
                output_frames.extend(reverse_frames)

                for _hf in range(hold_frames):
                    output_frames.append(src.clone())

        flavored_frames = []
        total = len(output_frames)
        for idx, frame in enumerate(output_frames):
            flavored = self._video_flavor(
                frame=frame,
                frame_index=idx,
                total_frames=total,
                flavor=video_flavor,
                strength=video_flavor_strength,
                seed=seed,
            )
            flavored_frames.append(flavored)

        batch = torch.stack(flavored_frames, dim=0)
        return (batch, int(batch.shape[0]), int(fps))


NODE_CLASS_MAPPINGS = {
    "VS Image Compare Deluxe": VSImageCompareDeluxe,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VS Image Compare Deluxe": "VS Image Compare Deluxe",
}