"""몸/배경 실루엣 마스크.

**용도별로 임계값을 분리한다. 하나로 합치지 말 것.**
과거 하나로 합쳤다가 팔이 마스크에서 통째로 빠져, 팔 이탈률이 실제 0.158인데
항상 0.61로 측정되는 바람에 "어떤 방법을 써도 개선 불가"라는 오판을 했다.
"""
import numpy as np

# 신장·중심 측정용. 엄격해야 배경 노이즈에 오염되지 않는다.
# 75로 낮추면 위아래 노이즈까지 몸으로 잡혀 신장 평균이 574px -> 651px로 포화된다.
EXTENT_PERCENTILE = 90

# "이 관절이 몸 위에 있는가" 판정용. 느슨해야 팔이 포함된다.
# mmWave에서 팔은 몸통보다 어두워, 90에서는 몸통·다리만 잡히고 팔이 빠진다.
MEMBERSHIP_PERCENTILE = 75


def body_mask(frame_upright: np.ndarray) -> np.ndarray:
    """관절이 몸 위인지 판정하기 위한 마스크. 신장 측정에는 쓰지 말 것."""
    return frame_upright > np.percentile(frame_upright, MEMBERSHIP_PERCENTILE)


def vertical_extent(frame_upright: np.ndarray):
    """머리끝 row, 발끝 row."""
    mask = frame_upright > np.percentile(frame_upright, EXTENT_PERCENTILE)
    rows = np.where(mask.any(axis=1))[0]
    if len(rows) == 0:
        return 0, frame_upright.shape[0] - 1
    return int(rows.min()), int(rows.max())


def horizontal_center(frame_upright: np.ndarray) -> float:
    """몸의 좌우 중심 x."""
    mask = frame_upright > np.percentile(frame_upright, EXTENT_PERCENTILE)
    cols = np.where(mask.any(axis=0))[0]
    if len(cols) == 0:
        return frame_upright.shape[1] / 2
    return float((cols.min() + cols.max()) / 2)
