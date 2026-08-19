"""CLAHE 전처리 + YOLOv8x-pose 프레임별 검출.

전처리로 CLAHE를 쓴다. 감마·로그변환 등 5종을 비교했으나 도달거리·이탈률 모두
1~4% 차이(노이즈 수준)였고, 크롭+확대나 imgsz=1280은 오히려 나빠졌다
(conf 0.885 -> 0.512, mmWave 노이즈까지 확대되기 때문).

**검출 실패의 원인은 어두움이 아니라 각도다.** 팔꿈치 위치의 밝기(0.234)는
발목(0.229)과 같은 수준이고 밝기-신뢰도 상관은 rho=0.25에 불과하다.
전처리를 더 손봐도 얻을 것이 없다.
"""
import cv2
import numpy as np

from .aps import read_aps, to_upright, to_display
from .silhouette import body_mask

N_FRAMES = 16
N_KEYPOINTS = 17

CLAHE_CLIP = 3.0
CLAHE_TILE = (8, 8)

# 몸 밖에 찍힌 keypoint의 가중치 배율 (삼각측량에서 거의 무시하도록)
OUTSIDE_PENALTY = 0.05

COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6), (5, 7), (7, 9),
    (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]


def clahe_rgb(frame_upright: np.ndarray) -> np.ndarray:
    """모델 입력 이미지. to_upright()를 거친 프레임을 넣을 것."""
    base = (to_display(frame_upright) * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE)
    return np.stack([clahe.apply(base)] * 3, axis=-1)


def detect_all_frames(model, aps_path: str):
    """한 사람의 16프레임을 검출한다.

    반환: kp (16,17,2), frame_conf (16,), kp_conf (16,17),
          rgb_frames (list of 16), masks (list of 16)
    검출 실패 프레임은 kp=0, conf=0으로 남는다 (가중치 0이라 이후 단계에서 자동 무시).
    """
    data = read_aps(aps_path)
    kp = np.zeros((N_FRAMES, N_KEYPOINTS, 2))
    frame_conf = np.zeros(N_FRAMES)
    kp_conf = np.zeros((N_FRAMES, N_KEYPOINTS))
    rgb_frames, masks = [], []

    for fi in range(N_FRAMES):
        frame = to_upright(data[fi])
        rgb = clahe_rgb(frame)
        rgb_frames.append(rgb)
        masks.append(body_mask(frame))

        result = model(rgb, verbose=False)[0]
        if result.keypoints is not None and len(result.keypoints.xy) > 0:
            box_confs = result.boxes.conf.cpu().numpy()
            best = box_confs.argmax()
            kp[fi] = result.keypoints.xy[best].cpu().numpy()
            frame_conf[fi] = box_confs[best]
            if result.keypoints.conf is not None:
                kp_conf[fi] = result.keypoints.conf[best].cpu().numpy()

    return kp, frame_conf, kp_conf, rgb_frames, masks


def keypoint_weights(kp, frame_conf, masks, kp_conf=None):
    """(16,17) 가중치 = 관절별 신뢰도 x (실루엣 안이면 1, 밖이면 패널티).

    YOLO는 관절마다 신뢰도를 따로 낸다(예: 사선 각도에서 코 0.079 vs 골반 0.999).
    프레임 단일값을 17개 관절에 똑같이 쓰면 그 정보가 버려지므로 관절별 값을 쓴다.
    """
    weights = np.zeros((N_FRAMES, N_KEYPOINTS))
    for fi in range(N_FRAMES):
        H, W = masks[fi].shape
        for k in range(N_KEYPOINTS):
            base = frame_conf[fi] if kp_conf is None else float(kp_conf[fi, k])
            x, y = kp[fi, k]
            xi, yi = int(round(x)), int(round(y))
            inside = 0 <= yi < H and 0 <= xi < W and masks[fi][yi, xi]
            weights[fi, k] = base * (1.0 if inside else OUTSIDE_PENALTY)
    return weights
