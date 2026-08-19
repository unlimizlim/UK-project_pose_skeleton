"""사용 예제 — 한 사람의 16프레임 스켈레톤을 그려 저장한다.

    python -m pose_skeleton.example data/stage1_aps/<scan_id>.aps out.png
"""
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ultralytics import YOLO

from . import estimate_pose, COCO_SKELETON

ARM_BONES = [(5, 7), (7, 9), (6, 8), (8, 10)]
BODY_BONES = [b for b in COCO_SKELETON if b not in ARM_BONES]
ARM_KPS = [7, 8, 9, 10]


def draw(ax, rgb, skeleton):
    ax.imshow(rgb)
    xs, ys = skeleton[:, 0], skeleton[:, 1]
    for bones, color, lw in ((BODY_BONES, "#00e5ff", 1.4), (ARM_BONES, "#ff9f1c", 2.0)):
        for i, j in bones:
            ax.plot([xs[i], xs[j]], [ys[i], ys[j]], "-", c=color, lw=lw, zorder=2)
    body = [k for k in range(17) if k not in ARM_KPS]
    ax.scatter(xs[body], ys[body], c="#00e5ff", s=9, zorder=3)
    ax.scatter(xs[ARM_KPS], ys[ARM_KPS], c="#ff9f1c", s=13, zorder=3)
    ax.set_xticks([]); ax.set_yticks([])


def main():
    aps_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else "skeleton.png"

    model = YOLO("yolov8x-pose.pt")
    skeleton, raw, rgbs, _ = estimate_pose(model, aps_path)

    fig, axes = plt.subplots(2, 16, figsize=(2.2 * 16, 6.4))
    for f in range(16):
        draw(axes[0, f], rgbs[f], raw[f])
        draw(axes[1, f], rgbs[f], skeleton[f])
        axes[0, f].set_title(f"f{f}  {f * 360 // 16}°", fontsize=9)
    axes[0, 0].set_ylabel("YOLO 원본", fontsize=11)
    axes[1, 0].set_ylabel("정제 결과", fontsize=11)
    fig.suptitle(aps_path.split("/")[-1], fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=90)
    print(f"saved: {out_path}")
    print(f"skeleton shape: {skeleton.shape}  (16프레임 x COCO 17관절 x xy)")


if __name__ == "__main__":
    main()
