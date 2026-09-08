"""PNG sequence decode and per-frame resize."""
import cv2
import numpy as np

from pure_cacher.scan import StreamInfo, frame_timestamp_ns, list_frames


def decode_video(
    stream: StreamInfo,
    modality: str,
    num_frames: int,
    resize: tuple[int, int] | None = None,
) -> tuple[np.ndarray, float]:
    """Decode up to ``num_frames`` PNGs as ``(3, T, H, W)`` uint8 RGB.

    Frames are read one at a time and resized (``INTER_AREA``) before being
    stacked. OpenCV decodes BGR; the store holds RGB, matching the ``rgb``
    modality's channel order in the contract. The rate returned is measured
    from the filename timestamps -- PURE has no container to state one.
    """
    if modality != "rgb":
        raise ValueError(f"PURE has only an rgb modality, got {modality!r}")
    paths = list_frames(stream.source)[:num_frames]
    frames = []
    for path in paths:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"cannot decode {path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if resize is not None:
            h, w = resize
            rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)  # cv2 wants (W, H)
        frames.append(rgb)
    stacked = np.stack(frames)                                            # (T, H, W, 3)
    first, last = frame_timestamp_ns(paths[0].name), frame_timestamp_ns(paths[-1].name)
    fps = (len(paths) - 1) / ((last - first) / 1e9) if last > first else 0.0
    return np.ascontiguousarray(stacked.transpose(3, 0, 1, 2)), fps      # (3, T, H, W)
