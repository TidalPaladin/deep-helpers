from typing import Sequence, Tuple, TypeVar, Union, cast

import matplotlib as mpl
import numpy as np
import torch
from einops import rearrange, repeat
from PIL import Image, ImageDraw
from torch import Tensor


T = TypeVar("T", bound=Union[np.ndarray, Tensor])


@torch.no_grad()
def to_colormap(heatmap: T, cmap: str = "gnuplot") -> T:
    r"""Converts a heatmap to a colormap.

    Args:
        heatmap: The heatmap to convert as a tensor or numpy array.
            Should be a single-channel image of floating point values in the range [0, 1].
        cmap: The name of the colormap to use.

    Shapes:
        * ``heatmap`` - :math:`(*, 1, H, W)`
        * Output - :math:`(*, 4, H, W)`

    Returns:
        The heatmap converted to a colormap in RGBA format.
    """

    if heatmap.shape[-3] != 1:
        raise ValueError(f"Only single channel heatmaps are supported, got shape {heatmap.shape}")

    # Get the colormap and convert
    colormap = mpl.colormaps[cmap]
    result = colormap(heatmap.float().cpu().numpy() if isinstance(heatmap, Tensor) else heatmap)
    assert isinstance(result, np.ndarray)

    # Output is channels last, so move channels to the correct position
    result = np.moveaxis(result, -1, -3)

    # Convert to be like the input
    result = torch.from_numpy(result).type_as(heatmap) if isinstance(heatmap, Tensor) else result.astype(np.float32)

    return cast(T, result)


@torch.no_grad()
def overlay_heatmap(heatmap: T, target: T, cmap: str = "gnuplot", alpha: float = 0.5) -> T:
    r"""Overlays a heatmap onto an image.

    Args:
        heatmap: The heatmap to overlay as a tensor or numpy array.
            Should be a single-channel image of floating point values in the range [0, 1].
        target: The image to overlay the heatmap on as a tensor or numpy array.
            Can be a single or three channel image. Should be in the range [0, 1].
        cmap: The name of the colormap to use.
        alpha: The alpha value to use for blending with the heatmap.

    Shapes:
        * ``heatmap`` - :math:`(B, 1, H, W)`
        * ``target`` - :math:`(B, C, H, W)`
        * Output - :math:`(B, C, H, W)`

    Returns:
        The image with the heatmap overlayed. Will be floating point in the range [0, 1].
    """
    if target.shape[1] == 1:
        target = repeat(target, "b c h w -> b (c repeat) h w", repeat=3)
    elif target.shape[1] != 3:
        raise ValueError("Only single or three channel inputs for `colormap` are supported")
    elif heatmap.shape[1] != 1:
        raise ValueError("Only single channel inputs are supported for `heatmap`")
    elif not 0 <= alpha <= 1:
        raise ValueError("`alpha` must be in [0, 1]")

    # Convert heatmap to colormap
    colormap = rearrange(to_colormap(heatmap, cmap), "b () c h w -> b c h w")

    # Extract the alpha and RGB channels
    alpha_tensor = alpha * colormap[..., -1, None, :, :]
    colormap = colormap[..., :3, :, :]

    # ignore dead space in heatmap
    alpha_tensor[rearrange((colormap == 0).all(1), "b h w -> b () h w")] = 0

    # Apply the overlay
    result = target * (1 - alpha_tensor) + colormap * alpha_tensor
    result = result.clip(0, 1)

    return cast(T, result)


@torch.no_grad()
def overlay_text(x: T, text: Union[str, Sequence[str]], pos: Tuple[float, float], **kwargs) -> T:
    r"""Overlays text onto an image.

    Args:
        x: The image to overlay the text on.
        text: The text to overlay.
        pos: The position of the text as a fraction of the image size in (H, W).
        **kwargs: Additional keyword arguments to pass to `PIL.ImageDraw.Draw.text`.

    Shapes:
        - x: :math:`(B, C, H, W)` or :math:`(C, H, W)`
        - Output: Same as :attr:`x`

    Returns:
        The image with the text overlayed.
    """
    if x.shape[-3] == 1:
        x = repeat(x, "... c h w -> ... (c repeat) h w", repeat=3)
    elif x.shape[-3] != 3:
        raise ValueError("Only single or three channel inputs for `x` are supported")

    # recurse with numpy array if x is a tensor
    if isinstance(x, Tensor):
        result = overlay_text(x.cpu().numpy(), text, pos, **kwargs)
        return cast(T, torch.from_numpy(result).type_as(x))
    assert isinstance(x, np.ndarray)

    # recurse over batch dimension
    if x.ndim == 4:
        if isinstance(text, str):
            text = [text] * x.shape[0]
        result = np.stack([overlay_text(x[i], text[i], pos, **kwargs) for i in range(x.shape[0])], axis=0)
        return cast(T, result)
    else:
        if not isinstance(text, str):
            if len(text) == 1:
                text = text[0]
            else:
                raise ValueError("Only single text strings are supported for non-batched inputs")

    assert x.ndim == 3
    assert isinstance(text, str)
    image = Image.fromarray((x * 255).astype(np.uint8).transpose(1, 2, 0))
    draw = ImageDraw.Draw(image)
    absolute_pos_xy = (int(pos[0] * image.height), int(pos[1] * image.width))
    draw.text(absolute_pos_xy, text, **kwargs)
    return cast(T, np.array(image).transpose(2, 0, 1) / 255.0)
