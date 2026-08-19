"""팔(팔꿈치·손목) 복원 — 이 파이프라인에서 가장 어려운 부분.

## 문제의 정체

팔 검출 실패의 원인은 **어두움이 아니라 각도**다. 팔꿈치 위치 밝기(0.234)는
발목(0.229)과 같고, 밝기-신뢰도 상관은 rho=0.25에 불과하다. 전처리로는 못 고친다.

옆모습에서는 먼 쪽 팔이 몸 뒤에 가려 **정보 자체가 없고**, 그때 YOLO는 두 팔을
보이는 한쪽에 겹쳐 놓는다. 팔 사슬 최소 신뢰도로 나눠 보면:

    신뢰도       프레임 비율    두 팔 간격    완전 겹침
    0.90~           18%          216px         0%
    0.70~0.90       38%          204px         0%
    0.40~0.70       28%           85px        13%
    ~0.40           16%          111px        11%

## 전략

 1. **믿을 프레임만 보존한다** — 신뢰도 >= 0.7 이고, 두 팔이 겹치지 않았고,
    회전 모델과 크게 어긋나지 않는 프레임.
 2. **나머지는 회전 사인파(RANSAC)로 채운다.**
 3. **뼈 길이로 마지막에 잡는다.**

'어디를 고칠지'와 '어떻게 고칠지'를 분리한 것이 핵심이다. 어떤 피팅 방법이든
멀쩡한 프레임을 14~37px씩 밀어내는데, 보존하면 그 손해가 0이 된다.

## 성능 (14명 x 16프레임, 정답 라벨 없는 holdout)

    복원 오차 중앙값 16.0px, 100px 초과 6.9%  (무처리: 165.1px, 64.7%)

## 한계

신뢰 프레임이 원 둘레에 고르게 없으면 복원이 불가능하다. 실측으로 cb929d44의
오른팔은 쓸 수 있는 프레임이 f0,f1,f14,f15 넷뿐이고 전부 정면 근처라, 거기서 세운
사인파를 원의 3/4로 외삽하는 셈이 된다. 이런 경우는 16시점의 한계다.
"""
import itertools

import numpy as np

N_FRAMES = 16
ANGLES = np.deg2rad(np.arange(N_FRAMES) * 360 / N_FRAMES)
DESIGN = np.stack([np.ones(N_FRAMES), np.cos(ANGLES), -np.sin(ANGLES)], axis=1)

ARM_CHAINS = [(5, 7, 9), (6, 8, 10)]   # (어깨, 팔꿈치, 손목)

# ---------------------------------------------------------------- 믿을 프레임 고르기

# 붕괴율 0%가 측정된 경계. **폴백을 두지 않는다** -- 임계값을 낮추면 신뢰도 0.4~0.5짜리
# (= 두 팔이 겹친 바로 그) 프레임을 보존해버려 대형실패가 15% -> 22%로 늘었다.
PRESERVE_CONF = 0.70

# 두 팔 사슬이 최대 어깨폭의 이 비율 안으로 붙으면 검출기가 두 팔을 겹쳐 놓은 것으로 본다.
# 신뢰도로는 이 실패를 못 거른다 -- 신뢰도 0.7 이상 217프레임 중 65개(30%)가 실제로
# 겹쳐 있었다(3775a16c f6은 신뢰도 0.89/0.75인데 손목 간격이 4px).
COLLAPSE_RATIO = 0.35

# 보존 프레임이 회전 궤도와 이만큼 넘게 어긋나면 믿지 않는다.
# 실측 잔차 중앙값 11~16px, 불량 프레임 43~155px로 확연히 갈린다.
RESIDUAL_ABS_PX = 40.0
RESIDUAL_MEDIAN_MULT = 3.0
# 중앙값 배수에 상한을 둔다. **상한이 없으면 전부 나쁜 사람에게 아무것도 못 거른다** --
# 한 사람의 오른팔은 잔차 중앙값이 73px(정상 11~16px)이라 문턱이 220px까지 벌어져,
# 팔이 몸통으로 112px 처진 프레임(잔차 151px)이 정상으로 통과했다.
RESIDUAL_CAP_PX = 80.0

# 손목이 어깨보다 이만큼 아래면 팔이 몸통으로 처진 것으로 본다.
# 이 데이터셋은 전원이 팔을 들고 서므로 처진 팔은 거의 항상 오검출이다.
# 실측: 정상 12명 원본 검출의 92.7%가 '손목이 어깨 위'이고, 보존 후보 232개 중 60px
# 넘게 처진 것은 2개(0.9%)뿐이며 그 둘도 최대 +175px로 명백한 오검출이었다.
# 반면 문제 사례는 +112px인데 신뢰도가 0.80이라 신뢰도 검사를 그냥 통과했다.
DROOP_MAX_PX = 60.0


def collapsed_frames(kp: np.ndarray) -> set:
    """두 팔이 하나로 겹친 프레임.

    기준 길이는 16프레임 중 **최대 어깨폭**이다. 프레임별 어깨폭을 분모로 쓰면
    옆모습에서 0에 가까워져 판정이 무너진다. 어깨폭은 회전과 무관하게 일정하므로
    최댓값(정면에서 관측되는 값)이 그 사람의 실제 크기다.
    """
    thresh = COLLAPSE_RATIO * np.abs(kp[:, 5, 0] - kp[:, 6, 0]).max()
    return {f for f in range(N_FRAMES)
            if min(abs(kp[f, 7, 0] - kp[f, 8, 0]), abs(kp[f, 9, 0] - kp[f, 10, 0])) < thresh}


def drooped_frames(kp, chain) -> set:
    """손목이 어깨보다 크게 내려간 프레임 (팔이 몸통으로 처짐)."""
    shoulder, _, wrist = chain
    return {f for f in range(N_FRAMES)
            if kp[f, wrist, 1] - kp[f, shoulder, 1] > DROOP_MAX_PX}


def trusted_frames(kp, kp_conf, chain) -> list:
    """세 검사를 모두 통과한 프레임만 보존한다.

    신뢰도 0.7 이상, 두 팔이 겹치지 않음, 팔이 몸통으로 처지지 않음.
    **신뢰도만으로는 뒤의 둘을 못 거른다** -- 겹침은 신뢰도 0.7 이상 프레임의 30%에서,
    처짐은 신뢰도 0.80에서도 나타났다.
    """
    conf = kp_conf[:, list(chain)].min(axis=1)
    bad = collapsed_frames(kp) | drooped_frames(kp, chain)
    return [f for f in range(N_FRAMES) if conf[f] >= PRESERVE_CONF and f not in bad]


def verify_trusted(kp, kp_conf, chain, candidate) -> list:
    """보존 대상을 회전 궤도와 대조해 재검증한다.

    신뢰도와 겹침 검사를 모두 통과하고도 틀리는 경우가 남는다:
    한쪽 팔이 몸통 아래로 처지거나(6ed62642 f0: 잔차 155px, 중앙값 11px),
    어깨 근처로 쪼그라든다(a865f755 f1: 65px). 두 경우 다 두 팔이 멀리 떨어져 있어
    겹침 검사를 통과하고 신뢰도도 0.7을 넘는다.

    기준은 절대값과 중앙값 배수 중 큰 쪽이다. 원래 잔차가 큰 사람에게 절대 기준만
    들이대면 멀쩡한 프레임까지 버리게 된다.
    """
    _, elbow, wrist = chain
    preserve = trusted_frames(kp, kp_conf, chain)
    if len(preserve) < 4:
        return preserve
    resid = {f: (np.linalg.norm(kp[f, elbow] - candidate[f, elbow])
                 + np.linalg.norm(kp[f, wrist] - candidate[f, wrist])) / 2
             for f in preserve}
    med = float(np.median(list(resid.values())))
    limit = max(RESIDUAL_ABS_PX, min(RESIDUAL_MEDIAN_MULT * med, RESIDUAL_CAP_PX))
    kept = [f for f in preserve if resid[f] <= limit]
    return kept if len(kept) >= 4 else preserve   # 너무 많이 버리면 판정 자체를 못 믿는다


# ---------------------------------------------------------------- RANSAC 궤도 피팅

INLIER_PX = 22.0        # 이 이내면 궤도에 동의하는 것으로 본다
MIN_ANGLE_SPREAD = 3    # 후보를 세울 4프레임이 최소 이만큼 떨어져 있어야 한다
MIN_INLIERS = 6
SILHOUETTE_WEIGHT = 3.0
N_REFINE_ROUNDS = 3

# inlier 사이 연속 공백 상한. 사인파는 증거 없는 구간으로 외삽하면 크게 어긋난다
# (a865f755: inlier가 [0..5,11,12,14,15]로 f6~f10이 통째로 비어 팔꿈치가 ~170px 어긋남).
# **"90도 구간 4개를 다 덮는가"로는 못 막는다** -- 위 집합도 명목상 4구간을 다 덮는다.
# 막아야 하는 것은 분포가 아니라 연속 공백이다.
MAX_INLIER_GAP = 3


def _solve(idx, x):
    A = DESIGN[idx]
    if np.linalg.matrix_rank(A) < 3:
        return None
    params, *_ = np.linalg.lstsq(A, x[idx], rcond=None)
    return params


def _candidate_subsets():
    subsets = []
    for combo in itertools.combinations(range(N_FRAMES), 4):
        gaps = [min(abs(a - b), N_FRAMES - abs(a - b))
                for a, b in itertools.combinations(combo, 2)]
        if min(gaps) >= MIN_ANGLE_SPREAD:
            subsets.append(list(combo))
    return subsets


_SUBSETS = _candidate_subsets()


def has_no_long_gap(inliers) -> bool:
    """inlier 사이 원형 최대 연속 공백이 허용치 이내인가."""
    if not inliers:
        return False
    idx = sorted(inliers)
    gaps = [b - a - 1 for a, b in zip(idx, idx[1:])]
    gaps.append(N_FRAMES - idx[-1] - 1 + idx[0])
    return max(gaps) <= MAX_INLIER_GAP


def _on_body_fraction(preds, kp, chain, masks):
    """예측 위치가 실제 몸(밝은 픽셀) 위에 있는 비율.

    검출기 출력끼리의 일치만 보면 검출기가 공통으로 틀릴 때 못 거른다.
    실루엣은 검출기와 무관한 독립 증거이므로 채점에 함께 쓴다.
    """
    if masks is None:
        return 0.0
    hits = total = 0
    for f in range(N_FRAMES):
        H, W = masks[f].shape
        for k in chain:
            xi, yi = int(round(preds[k][f])), int(round(kp[f, k, 1]))
            total += 1
            if 0 <= yi < H and 0 <= xi < W and masks[f][yi, xi]:
                hits += 1
    return hits / max(total, 1)


def ransac_fit_chain(kp, chain, masks=None):
    """한 팔 사슬의 회전 궤도를 RANSAC으로 추정.

    사슬 전체(어깨·팔꿈치·손목)의 잔차 합으로 inlier를 판정한다. 관절마다 따로 고르면
    같은 팔의 관절들이 서로 다른 기준으로 계산되어 팔이 벌어진다.
    """
    xs = {k: kp[:, k, 0] for k in chain}
    best = None
    for subset in _SUBSETS:
        params = {k: _solve(subset, xs[k]) for k in chain}
        if any(p is None for p in params.values()):
            continue
        preds = {k: DESIGN @ params[k] for k in chain}
        resid = sum(np.abs(xs[k] - preds[k]) for k in chain) / len(chain)
        inliers = [f for f in range(N_FRAMES) if resid[f] <= INLIER_PX]
        if len(inliers) < MIN_INLIERS or not has_no_long_gap(inliers):
            continue
        score = (len(inliers) + SILHOUETTE_WEIGHT * _on_body_fraction(preds, kp, chain, masks) * N_FRAMES
                 - resid[inliers].mean() / 100)
        if best is None or score > best[0]:
            best = (score, inliers, preds)
    if best is None:
        return None

    # LO-RANSAC: inlier로 재피팅하고 다시 고르기를 반복해 안정화
    inliers, final = best[1], best[2]
    for _ in range(N_REFINE_ROUNDS):
        params = {k: _solve(inliers, xs[k]) for k in chain}
        if any(p is None for p in params.values()):
            break
        preds = {k: DESIGN @ params[k] for k in chain}
        resid = sum(np.abs(xs[k] - preds[k]) for k in chain) / len(chain)
        new_inliers = [f for f in range(N_FRAMES) if resid[f] <= INLIER_PX]
        final = preds
        # 재선정은 잔차만 보므로, 그냥 두면 inlier가 한쪽 호로 뭉쳐 외삽 문제가 되살아난다
        if (len(new_inliers) < MIN_INLIERS or new_inliers == inliers
                or not has_no_long_gap(new_inliers)):
            break
        inliers = new_inliers
    return final, inliers


def fit_arms_ransac(kp, masks=None):
    """팔꿈치·손목을 RANSAC 궤도로 교체한 배열을 만든다."""
    out = kp.copy()
    for chain in ARM_CHAINS:
        result = ransac_fit_chain(kp, chain, masks)
        if result is None:
            continue
        preds, inliers = result
        _, elbow, wrist = chain
        for k in (elbow, wrist):
            out[:, k, 0] = preds[k]
            out[:, k, 1] = np.median(kp[inliers, k, 1])   # 수직축 회전이므로 y는 일정
    return out


# ---------------------------------------------------------------- 뼈 길이 / 교차

def bone_limits(kp, kp_conf, chain):
    """신뢰 프레임에서 관측된 상완/전완의 최대 길이. 없으면 None.

    최댓값을 쓰는 이유: 팔이 카메라 쪽으로 기울면 투영 길이는 짧아질 뿐 길어질 수 없다.
    따라서 최댓값이 곧 넘어서는 안 되는 상한이다. 중앙값을 쓰면 정상적인 단축까지 잘린다.
    """
    shoulder, elbow, wrist = chain
    preserve = trusted_frames(kp, kp_conf, chain)
    if len(preserve) < 3:
        return None
    return (max(np.linalg.norm(kp[f, elbow] - kp[f, shoulder]) for f in preserve),
            max(np.linalg.norm(kp[f, wrist] - kp[f, elbow]) for f in preserve))


def clamp_filled(kp, filled, kp_conf):
    """채워 넣은 프레임의 팔이 뼈 길이를 넘으면 어깨 쪽으로 끌어당긴다."""
    out = filled.copy()
    for chain in ARM_CHAINS:
        limits = bone_limits(kp, kp_conf, chain)
        if limits is None:
            continue
        shoulder, elbow, wrist = chain
        preserve = set(trusted_frames(kp, kp_conf, chain))
        for f in range(N_FRAMES):
            if f in preserve:
                continue
            for base, tip, limit in ((shoulder, elbow, limits[0]), (elbow, wrist, limits[1])):
                vec = out[f, tip] - out[f, base]
                d = float(np.linalg.norm(vec))
                if d > limit and d > 1e-6:
                    out[f, tip] = out[f, base] + vec * (limit / d)
    return out


def enforce_bone_limits(skeleton, kp, kp_conf):
    """완성된 스켈레톤 위에서, **그 스켈레톤의 어깨를 기준으로** 뼈 길이를 강제한다.

    clamp_filled와의 차이가 중요하다. 검출이 통째로 실패한 프레임은 원본 좌표가 (0,0)이라
    원본 기준으로 재면 "어깨에서 98px"이 정상으로 통과하는데, 최종 출력의 어깨는 보간된
    실제 위치다. 그 결과 상완이 292px(정상 95px)가 되어 팔이 화면 밖으로 튄다.

    상한은 **두 사슬을 합쳐 하나로** 잡는다. 뒤에 좌우를 맞바꾸는 단계가 있어 사슬 5-7-9가
    원래 6-8-10이던 값을 담을 수 있고, 사슬별 상한을 쓰면 멀쩡한 팔이 반대쪽 상한에 걸린다.
    """
    both = [b for b in (bone_limits(kp, kp_conf, c) for c in ARM_CHAINS) if b is not None]
    if not both:
        return skeleton.copy()
    max_upper = max(b[0] for b in both)
    max_fore = max(b[1] for b in both)

    out = skeleton.copy()
    for shoulder, elbow, wrist in ARM_CHAINS:
        for f in range(N_FRAMES):
            for base, tip, limit in ((shoulder, elbow, max_upper), (elbow, wrist, max_fore)):
                vec = out[f, tip] - out[f, base]
                d = float(np.linalg.norm(vec))
                if d > limit and d > 1e-6:
                    out[f, tip] = out[f, base] + vec * (limit / d)
    return out


def _ccw(a, b, c):
    return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])


def segments_cross(p1, p2, p3, p4) -> bool:
    return (_ccw(p1, p3, p4) != _ccw(p2, p3, p4)) and (_ccw(p1, p2, p3) != _ccw(p1, p2, p4))


def untangle(skeleton):
    """어깨→손목 선분이 X자로 교차하면 두 팔의 (팔꿈치,손목) 배정을 맞바꾼다.

    원본 검출 자체가 교차인 경우가 있고(3775a16c f8: 두 사슬 신뢰도 0.79/0.77),
    보존 정책이 그 오류를 그대로 통과시킨다. 길이로는 못 잡는다 -- 그 프레임의 두
    팔꿈치는 22px 차이라 어느 배정이든 길이 합이 비슷하다. 교차는 길이가 아니라
    **위상**의 문제이므로 선분 교차로 직접 판정한다.
    """
    out = skeleton.copy()
    (sl, el_l, wr_l), (sr, el_r, wr_r) = ARM_CHAINS
    for f in range(N_FRAMES):
        if not segments_cross(out[f, sl], out[f, wr_l], out[f, sr], out[f, wr_r]):
            continue
        if segments_cross(out[f, sl], out[f, wr_r], out[f, sr], out[f, wr_l]):
            continue      # 맞바꿔도 교차가 남으면 손대지 않는다
        out[f, [el_l, el_r]] = out[f, [el_r, el_l]]
        out[f, [wr_l, wr_r]] = out[f, [wr_r, wr_l]]
    return out


def keep_verified(kp, kp_conf, candidate):
    """재검증을 통과한 프레임만 원본으로 두고, 나머지는 궤도로 채운다."""
    out = kp.copy()
    for chain in ARM_CHAINS:
        _, elbow, wrist = chain
        keep = set(verify_trusted(kp, kp_conf, chain, candidate))
        for f in range(N_FRAMES):
            if f in keep:
                continue
            out[f, elbow] = candidate[f, elbow]
            out[f, wrist] = candidate[f, wrist]
    return out
