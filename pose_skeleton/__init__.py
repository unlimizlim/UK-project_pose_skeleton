"""mmWave 전신 스캔(TSA Passenger Screening) 16시점 스켈레톤 추정.

사람은 정지해 있고 스캐너만 360도를 돈다는 **회전 기하**를 이용해, RGB로 학습된
YOLOv8x-pose의 검출을 mmWave 도메인에서 정제한다. 재학습 없음.

    from ultralytics import YOLO
    from pose_skeleton import estimate_pose

    model = YOLO("yolov8x-pose.pt")
    skeleton, raw, rgbs, masks = estimate_pose(model, "data/stage1_aps/<id>.aps")
    # skeleton: (16, 17, 2) — 16프레임 x COCO 17관절 x (x, y)

자세한 배경과 실패한 시도 목록은 README.md 참조.
"""
from .pipeline import estimate_pose, refine_body, refine_arms, RIGID_KEYPOINTS
from .detect import detect_all_frames, keypoint_weights, clahe_rgb, COCO_SKELETON
from .aps import read_aps, read_frame, to_upright, to_display
from .silhouette import body_mask, vertical_extent, horizontal_center

__all__ = [
    "estimate_pose", "refine_body", "refine_arms", "RIGID_KEYPOINTS",
    "detect_all_frames", "keypoint_weights", "clahe_rgb", "COCO_SKELETON",
    "read_aps", "read_frame", "to_upright", "to_display",
    "body_mask", "vertical_extent", "horizontal_center",
]
