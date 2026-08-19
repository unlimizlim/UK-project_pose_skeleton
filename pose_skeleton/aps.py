"""TSA Passenger Screening `.aps` 파일 IO.

포맷: 512바이트 헤더 + (16, 660, 512) uint16, 순서는 AYX(각도, Y, X).
파일 크기가 512 + 16*660*512*2 바이트와 정확히 일치함을 실제 파일로 검증했다.
"""
import numpy as np

HEADER_BYTES = 512
N_FRAMES = 16
HEIGHT = 660
WIDTH = 512


def read_aps(path: str) -> np.ndarray:
    """(16, 660, 512) uint16으로 읽는다."""
    with open(path, "rb") as f:
        f.seek(HEADER_BYTES)
        data = np.fromfile(f, dtype=np.uint16)
    return data.reshape(N_FRAMES, HEIGHT, WIDTH)


def read_frame(path: str, frame_idx: int) -> np.ndarray:
    """프레임 하나만 읽는다 (대량 순회 시 파일당 10MB를 다 읽지 않아도 됨)."""
    frame_bytes = HEIGHT * WIDTH * 2
    with open(path, "rb") as f:
        f.seek(HEADER_BYTES + frame_idx * frame_bytes)
        data = np.fromfile(f, dtype=np.uint16, count=HEIGHT * WIDTH)
    return data.reshape(HEIGHT, WIDTH)


def to_upright(frame: np.ndarray) -> np.ndarray:
    """Y축을 뒤집어 '위=머리, 아래=발'로 만든다. **반드시 거칠 것.**

    원본은 row 0이 발쪽이다. 이 함수를 거치면 이후 모든 코드에서 matplotlib 기본값
    (origin='upper')과 픽셀 좌표계를 그대로 쓸 수 있다. 2D 프레임과 (16,H,W) 둘 다 지원.
    """
    return np.flip(frame, axis=-2)


def to_display(frame: np.ndarray, percentile: float = 99.5) -> np.ndarray:
    """0~1 정규화. to_upright()를 먼저 적용한 프레임을 넣을 것."""
    frame = frame.astype(np.float32)
    vmax = np.percentile(frame, percentile)
    return np.clip(frame / max(vmax, 1e-6), 0, 1)
