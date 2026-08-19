"""추정 결과를 CSV로 내보낸다 (ROI 파이프라인 연동용).

**이 모듈이 내는 것은 관절 좌표이지 zone 박스가 아니다.** 두 가지는 다르다.

  이 패키지의 출력   COCO **관절** 17개의 점 좌표, 660x512 원본 픽셀
  ROI 파이프라인 요구  신체 **zone** 17개의 사각형, 224x224 캐시 뷰, [x1,y1,x2,y2)

숫자 17이 양쪽에 나오지만 **전혀 다른 것**이다 (관절 vs 구역). 관절을 zone 박스로
바꾸는 변환은 이 저장소의 범위가 아니며, 누군가 따로 맡아야 한다.
자세한 것은 README의 'ROI 파이프라인 연동' 절 참조.
"""
import csv
import os

import numpy as np

KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

COLUMNS = ["subject_id", "view_idx", "keypoint_id", "keypoint_name",
           "x", "y", "confidence", "valid"]

# 이 값 미만이면 그 관절을 신뢰하지 않는 것으로 표시한다.
# 팔은 0.7 미만에서 두 팔이 겹치기 시작한다는 측정에 근거한다 (arms.PRESERVE_CONF).
VALID_CONF = 0.70


def keypoint_rows(subject_id, skeleton, kp_conf, scale=None):
    """(16,17,2) 스켈레톤을 행 목록으로 편다.

    scale: (sx, sy)를 주면 좌표에 곱한다. 원본 660x512를 다른 해상도로 옮길 때 쓴다.
      **단순 비율 변환이 맞는지는 받는 쪽 캐시 생성 방식에 달려 있다.**
      크롭 후 리사이즈라면 비율만으로는 맞지 않으므로, 변환식을 확인하고 쓸 것.
    """
    sx, sy = scale if scale else (1.0, 1.0)
    rows = []
    for view_idx in range(skeleton.shape[0]):
        for k in range(skeleton.shape[1]):
            x, y = skeleton[view_idx, k]
            conf = float(kp_conf[view_idx, k])
            rows.append({
                "subject_id": subject_id,
                "view_idx": view_idx,
                "keypoint_id": k,
                "keypoint_name": KEYPOINT_NAMES[k],
                "x": round(float(x) * sx, 3),
                "y": round(float(y) * sy, 3),
                "confidence": round(conf, 4),
                "valid": bool(conf >= VALID_CONF),
            })
    return rows


def write_csv(path, rows, append=False):
    """행 목록을 CSV로 쓴다. append=True면 헤더 없이 이어 붙인다."""
    mode = "a" if append and os.path.exists(path) else "w"
    with open(path, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if mode == "w":
            writer.writeheader()
        writer.writerows(rows)
    return path


def export_subject(path, subject_id, skeleton, kp_conf, scale=None, append=False):
    """한 사람의 결과를 CSV에 쓴다 (16 x 17 = 272행)."""
    return write_csv(path, keypoint_rows(subject_id, skeleton, kp_conf, scale), append=append)
