"""파이프라인 진입점.

    CLAHE + YOLOv8x-pose 검출
      -> 강건 삼각측량 (강체 관절만; 붕괴한 다리 쌍은 원본 유지)
      -> 하체 좌우 정렬
      -> 검출 실패 프레임 보간
      -> 팔 복원 (좌우 정렬 -> 보존/채움 -> 교차 해소 -> 상체 좌우 정렬 -> 뼈 길이 강제)

각 단계의 근거는 해당 모듈 docstring 참조.
"""
import numpy as np

from .detect import detect_all_frames, keypoint_weights
from .rotation import global_swap_triangulate, align_to_raw_sides
from .sides import UPPER_BODY_GROUP, side_reference, align_groups, align_pairs, align_with_conf
from . import arms as A

# 삼각측량을 적용할 관절: 스캔 중 몸에 고정되어 함께 회전하는 부위만.
# COCO 0=코, 11/12=골반, 13/14=무릎, 15/16=발목.
#
# **어깨(5/6)는 제외한다.** 포함시키면 팔 이탈률 0.158 -> 0.196, 또는 어깨 관절이 끊겨
# upper_arm CV 0.313 -> 0.367로 악화된다. 팔이 붙어 있어 어깨는 실제로 미세하게 움직여
# 강체 가정이 깨지는 것으로 보인다.
RIGID_KEYPOINTS = [0, 11, 12, 13, 14, 15, 16]

LEG_PAIRS = [(13, 14), (15, 16)]      # 무릎, 발목
HIP_PAIR = (11, 12)

# 좌우 간격이 원본 대비 이 비율 아래로 줄면 삼각측량이 그 쌍을 붕괴시킨 것으로 본다.
LEG_COLLAPSE_RATIO = 0.90

ARM_KEYPOINTS = [7, 8, 9, 10]


def revert_collapsed_legs(tri, kp, use_tri):
    """삼각측량이 다리를 안쪽으로 무너뜨린 쌍은 삼각측량을 적용하지 않는다.

    관측: 두 사람에서 좌/우 발목 사인파의 **위상**이 어긋나 있었다. 정상이면 두 곡선이
    옆모습 프레임(f4, f12)에서 교차하는데, 이들은 후면(f7~f8)에서 교차한다. 원본 검출의
    사인파 잔차가 29~42px로 크기 때문이다(정상 10px). 위상이 틀리면 좌우 간격도 줄어
    발목 간격이 원본의 0.66~0.74배가 됐다.

    **뼈 길이 CV로는 못 잡는다** -- 붕괴한 쪽이 오히려 CV가 좋다(0.015 vs 0.021).
    그래서 붕괴를 직접 잰다. 14명 중 2명, 4개 쌍에만 해당한다.
    """
    out = use_tri.copy()
    for a, b in LEG_PAIRS:
        raw_gap = np.abs(kp[:, a, 0] - kp[:, b, 0]).mean()
        tri_gap = np.abs(tri[:, a, 0] - tri[:, b, 0]).mean()
        if tri_gap / max(raw_gap, 1.0) < LEG_COLLAPSE_RATIO:
            out[a] = out[b] = False
    return out


def fill_failed_frames(refined, frame_conf):
    """검출이 아예 실패한 프레임(conf=0)의 비강체 관절을 이웃에서 보간한다.

    실패 시 kp가 (0,0)으로 남는데, 강체 관절은 삼각측량이 복원하지만 어깨·팔은
    (0,0)에 그대로 남아 스켈레톤이 좌상단으로 뻗는다.

    **검출이 완전히 실패한 프레임에만** 적용한다. 검출된 프레임은 신뢰도가 낮아도
    건드리지 않는다 -- 과거 팔 전체를 평활화했다가 오히려 나빠진 전례가 있다.
    """
    out = refined.copy()
    n = len(refined)
    failed = [f for f in range(n) if frame_conf[f] == 0]
    if not failed or len(failed) == n:
        return out
    good = [f for f in range(n) if frame_conf[f] > 0]
    non_rigid = [k for k in range(17) if k not in RIGID_KEYPOINTS]
    for f in failed:
        nearest = sorted(good, key=lambda g: min(abs(g - f), n - abs(g - f)))[:2]
        for k in non_rigid:
            out[f, k] = np.mean([refined[g, k] for g in nearest], axis=0)
    return out


def refine_body(kp, weights, frame_conf=None):
    """몸통·다리 정제. 상체(어깨·팔)는 원본 그대로 두고 팔은 refine_arms가 담당한다."""
    tri = align_to_raw_sides(global_swap_triangulate(kp, weights)[0], kp)

    use_tri = np.zeros(17, dtype=bool)
    use_tri[RIGID_KEYPOINTS] = True
    use_tri = revert_collapsed_legs(tri, kp, use_tri)

    refined = np.where(use_tri[None, :, None], tri, kp)
    # 하체 좌우를 기준 쌍에 맞춘다. 골반이 기준이면 골반은 자동으로 제외되고,
    # 골반이 망가져 다른 쌍이 기준이 된 경우에는 골반도 맞출 대상이 된다.
    refined = align_pairs(refined, [HIP_PAIR] + LEG_PAIRS)
    if frame_conf is not None:
        refined = fill_failed_frames(refined, frame_conf)
    return refined, use_tri


def refine_arms(kp, kp_conf, masks, refined):
    """팔 복원. refined(몸통 정제 완료)를 받아 팔만 갈아끼운 결과를 돌려준다."""
    # 피팅은 refined 위에서 한다. 원본 kp는 검출 실패 프레임이 (0,0)이라 사인파 피팅과
    # 길이 계산을 오염시키지만, refined는 그런 프레임이 이미 보간돼 좌표계가 일관된다.
    #
    # **피팅 전에** 좌우를 맞춘다. 보존 대상 안에 좌우가 뒤바뀐 것이 섞여 있으면 사인파가
    # 아예 만들어지지 않는다(cb929d44: L손목 x가 정면 352, 후면 388로 같은 쪽에 있었다.
    # 회전하면 반대쪽이어야 한다). 그 상태로 피팅하면 팔이 제자리에 붙어 있는 것처럼 보인다.
    ref = side_reference(refined)
    kp_al, conf_al = align_with_conf(refined, kp_conf, [UPPER_BODY_GROUP], ref)

    candidate = A.fit_arms_ransac(kp_al, masks=masks)
    fitted = A.clamp_filled(kp_al, A.keep_verified(kp_al, conf_al, candidate), conf_al)

    out = refined.copy()
    for k in ARM_KEYPOINTS:
        out[:, k] = fitted[:, k]

    # 순서가 중요하다: 교차 해소 -> 좌우 정렬 -> 길이 강제.
    # 길이를 먼저 강제하면, 뒤의 정렬이 맞바꾼 프레임이 상한을 다시 벗어날 수 있다.
    out = A.untangle(out)
    out = align_groups(out, [UPPER_BODY_GROUP])
    return A.enforce_bone_limits(out, kp, kp_conf)


def estimate_pose(model, aps_path: str):
    """한 사람의 16프레임 스켈레톤을 추정한다.

    model: ultralytics YOLO 인스턴스 (yolov8x-pose.pt)
    aps_path: .aps 파일 경로

    반환: (skeleton (16,17,2), raw (16,17,2), rgb_frames, masks)
      skeleton — 정제 결과, raw — YOLO 원본 검출 (비교용)
    """
    kp, frame_conf, kp_conf, rgb_frames, masks = detect_all_frames(model, aps_path)
    weights = keypoint_weights(kp, frame_conf, masks, kp_conf=kp_conf)
    refined, _ = refine_body(kp, weights, frame_conf=frame_conf)
    refined = refine_arms(kp, kp_conf, masks, refined)
    return refined, kp, rgb_frames, masks
