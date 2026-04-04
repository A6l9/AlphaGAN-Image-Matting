import uuid
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image


def build_session(onnx_path: Path) -> ort.InferenceSession:
    """Build an ONNX Runtime session.

    Args:
        onnx_path (Path): Path to the ONNX model.

    Returns:
        ort.InferenceSession: Initialized inference session.
    """
    providers = ["CPUExecutionProvider"]
    if "CUDAExecutionProvider" in ort.get_available_providers():
        providers.insert(0, "CUDAExecutionProvider")
    return ort.InferenceSession(onnx_path, providers=providers)


def run_model(session: ort.InferenceSession, x_nchw: np.ndarray) -> np.ndarray:
    """Run one model forward pass.

    Args:
        session (ort.InferenceSession): ONNX Runtime session.
        x_nchw (np.ndarray): Input tensor in NCHW format.

    Raises:
        ValueError: Raised if the model output shape is unsupported.

    Returns:
        np.ndarray: Predicted alpha map for one sample.
    """
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: x_nchw})[0]

    if output.ndim == 4:
        return output[0, 0]
    if output.ndim == 3:
        return output[0]

    raise ValueError(f'Unexpected output shape: {output.shape}')


def hann2d(h: int, w: int) -> np.ndarray:
    """Create a 2D Hann window.

    Args:
        h (int): Window height.
        w (int): Window width.

    Returns:
        np.ndarray: 2D Hann window of shape (H, W).
    """
    wy = np.hanning(h).astype(np.float32)
    wx = np.hanning(w).astype(np.float32)

    return wy[:, None] * wx[None, :]


def compute_pad(length: int, tile: int, stride: int) -> int:
    """Compute padding for tiled inference.

    Args:
        length (int): Original spatial size.
        tile (int): Tile size.
        stride (int): Tile stride.

    Returns:
        int: Required padding size.
    """
    if length <= tile:
        return tile - length
    rem = (length - tile) % stride

    return (stride - rem) % stride


def tiled_inference(
    session: ort.InferenceSession,
    mdl_inp: np.ndarray,
    trimap_u8: np.ndarray,
    tile_size: int,
    overlap: float,
) -> np.ndarray:
    """Run tiled inference on unknown trimap regions.

    Args:
        session (ort.InferenceSession): ONNX Runtime session.
        mdl_inp (np.ndarray): Input tensor in CHW format.
        trimap_u8 (np.ndarray): Trimap image in uint8 format.
        tile_size (int): Tile size for inference.
        overlap (float): Overlap ratio between tiles.

    Returns:
        np.ndarray: Predicted alpha map.
    """
    stride = max(1, int(tile_size * (1.0 - overlap)))

    _, h, w = mdl_inp.shape
    pad_h = compute_pad(h, tile_size, stride)
    pad_w = compute_pad(w, tile_size, stride)

    x4p = np.pad(
        mdl_inp,
        ((0, 0), (0, pad_h), (0, pad_w)),
        mode='reflect',
    )
    hp, wp = x4p.shape[1], x4p.shape[2]

    trimap_p = np.pad(trimap_u8, ((0, pad_h), (0, pad_w)), mode='edge')

    fg_p = trimap_p >= 254
    bg_p = trimap_p <= 1
    unknown_p = (~fg_p) & (~bg_p)

    alpha_init = np.zeros((hp, wp), dtype=np.float32)
    alpha_init[fg_p] = 1.0
    alpha_init[unknown_p] = np.nan

    acc = np.zeros((hp, wp), dtype=np.float32)
    wsum = np.zeros((hp, wp), dtype=np.float32)

    win = hann2d(tile_size, tile_size)

    for y in range(0, hp - tile_size + 1, stride):
        for x in range(0, wp - tile_size + 1, stride):
            if not unknown_p[y:y + tile_size, x:x + tile_size].any():
                continue

            patch = x4p[:, y:y + tile_size, x:x + tile_size][None].astype(np.float32, copy=False)

            alpha = run_model(session, patch)

            acc[y:y + tile_size, x:x + tile_size] += alpha * win
            wsum[y:y + tile_size, x:x + tile_size] += win

    alpha_unknown = acc / np.clip(wsum, 1e-6, None)

    alpha_out = np.nan_to_num(alpha_init, nan=0.0)
    alpha_out[unknown_p] = alpha_unknown[unknown_p]

    return alpha_out[:h, :w]


def main(
    img_path: Path,
    trim_path: Path,
    out_dir: Path,
    session: ort.InferenceSession,
    tile_size: int = 512,
    overlap: float = 0.5,
) -> None:
    """Run inference and save alpha and cutout images.

    Args:
        img_path (Path): Path to the input RGB image.
        trim_path (Path): Path to the trimap image.
        out_dir (Path): Directory for output files.
        session (ort.InferenceSession): ONNX Runtime session.
        tile_size (int, optional): Tile size for inference. Defaults to 512.
        overlap (float, optional): Tile overlap ratio. Defaults to 0.5.

    Raises:
        ValueError: Raised if image and trimap sizes do not match.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    img = Image.open(img_path).convert('RGB')
    rgb = np.asarray(img, dtype=np.float32) / 255.0

    trimap = Image.open(trim_path).convert('L')
    if trimap.size != img.size:
        raise ValueError(f'RGB and trimap sizes do not match: {img.size} != {trimap.size}')

    trimap_u8 = np.asarray(trimap, dtype=np.uint8)
    trimap_norm = (trimap_u8.astype(np.float32) / 255.0)[..., None]

    model_input = np.concatenate([rgb, trimap_norm], axis=2)
    model_input = np.transpose(model_input, (2, 0, 1)).astype(np.float32, copy=False)

    alpha = tiled_inference(
        session=session,
        mdl_inp=model_input,
        trimap_u8=trimap_u8,
        tile_size=tile_size,
        overlap=overlap,
    )

    uuid_str = str(uuid.uuid1())[:5]

    alpha_u8 = (np.clip(alpha, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(alpha_u8).save(out_dir / f'{uuid_str}-alpha.png')

    rgb_u8 = (rgb * 255.0).astype(np.uint8)
    rgba = np.dstack([rgb_u8, alpha_u8])
    Image.fromarray(rgba, mode='RGBA').save(out_dir / f'{uuid_str}-cutout.png')

    print('Saved:', out_dir / f'{uuid_str}-alpha.png', out_dir / f'{uuid_str}-cutout.png')


if __name__ == "__main__":
    ROOT_DIR = Path(__file__).parent
    IMG_PATH = ROOT_DIR / "inference_test" / "orig.png"
    TRIMAP_PATH = ROOT_DIR / "inference_test" / "trimap.png"
    ONNX_PATH = ROOT_DIR / "33_01-04-2026_13_17_41.onnx"
    OUT_DIR = ROOT_DIR / "results"

    TILE_SIZE = 512
    OVERLAP = 0.5  # 0.5 -> stride = TILE // 2

    session = build_session(ONNX_PATH)

    main(
        img_path=IMG_PATH,
        trim_path=TRIMAP_PATH,
        out_dir=OUT_DIR,
        session=session,
        tile_size=TILE_SIZE,
        overlap=OVERLAP
)