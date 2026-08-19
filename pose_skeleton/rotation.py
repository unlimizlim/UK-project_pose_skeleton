"""회전 기하 — 이 파이프라인 전체의 근거.

스캔 중 **사람은 정지해 있고 카메라(스캐너)만 360도를 돈다.** 프레임 f의 각도는
2pi*f/16 이고, f0=정면, f8=후면이다. 따라서 몸에 고정된 관절 하나의 영상 x좌표는

    x(f) = axis + b*cos(theta_f) - c*sin(theta_f)

형태의 사인파여야 하고, y좌표는 (수직축 회전이므로) 일정해야 한다.
이 성질로 노이즈를 걷어내는 것이 삼각측량 단계다.

좌우 뒤바뀜은 **프레임당 플래그 하나**로 결정한다. 몸은 강체이므로 한 프레임에서
좌우 라벨이 뒤집혔다면 몸 전체가 함께 뒤집힌 것이다. 관절 쌍마다 독립 판정하면
어깨는 이렇게 골반은 저렇게 결정되어 몸통이 X자로 꼬인다.
"""
import numpy as np

N_FRAMES = 16
ANGLES = np.deg2rad(np.arange(N_FRAMES) * 360 / N_FRAMES)
DESIGN = np.stack([np.ones(N_FRAMES), np.cos(ANGLES), -np.sin(ANGLES)], axis=1)

# COCO 좌우 대칭 쌍
SYMMETRIC_PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]

HUBER_DELTA = 15.0
N_IRLS_ITER = 6
N_GLOBAL_ROUNDS = 4


def _irls_fit(x_obs, w0):
    """가중 최소제곱 + Huber 재가중으로 사인파를 맞춘다 (이상치에 강건)."""
    w = w0.copy()
    pred = x_obs.copy()
    for _ in range(N_IRLS_ITER):
        if w.sum() < 1e-6:
            return x_obs.copy(), w
        AtW = DESIGN.T * w
        lhs = AtW @ DESIGN
        if np.linalg.matrix_rank(lhs) < 3:
            return np.full(N_FRAMES, np.average(x_obs, weights=w)), w
        pred = DESIGN @ np.linalg.solve(lhs, AtW @ x_obs)
        resid = np.abs(x_obs - pred)
        w = w0 * np.where(resid <= HUBER_DELTA, 1.0, HUBER_DELTA / np.maximum(resid, 1e-6))
    return pred, w


def _fit_all(kp, weights):
    pred = np.zeros((N_FRAMES, 17))
    for k in range(17):
        pred[:, k], _ = _irls_fit(kp[:, k, 0], weights[:, k])
    return pred


def global_swap_triangulate(kp: np.ndarray, weights: np.ndarray):
    """프레임당 '몸 전체 좌우 뒤집힘' 플래그 하나를 정하며 사인파를 맞춘다.

    반환: (정제 결과 (16,17,2), 뒤집힌 프레임 목록)
    """
    kp = kp.copy()
    weights = weights.copy()
    flipped = np.zeros(N_FRAMES, dtype=bool)

    for _ in range(N_GLOBAL_ROUNDS):
        pred = _fit_all(kp, weights)
        changed = False
        for f in range(N_FRAMES):
            cur = alt = 0.0
            for left, right in SYMMETRIC_PAIRS:
                wl, wr = weights[f, left], weights[f, right]
                cur += wl * abs(kp[f, left, 0] - pred[f, left])
                cur += wr * abs(kp[f, right, 0] - pred[f, right])
                alt += wl * abs(kp[f, right, 0] - pred[f, left])
                alt += wr * abs(kp[f, left, 0] - pred[f, right])
            if alt < cur * 0.9:            # 명확히 나을 때만 뒤집는다
                for left, right in SYMMETRIC_PAIRS:
                    kp[f, [left, right]] = kp[f, [right, left]]
                    weights[f, [left, right]] = weights[f, [right, left]]
                flipped[f] = not flipped[f]
                changed = True
        if not changed:
            break

    pred = _fit_all(kp, weights)
    refined = np.zeros_like(kp)
    refined[:, :, 0] = pred
    for k in range(17):
        w = weights[:, k]
        refined[:, k, 1] = (np.average(kp[:, k, 1], weights=w)
                            if w.sum() > 1e-6 else kp[:, k, 1])
    return refined, np.where(flipped)[0].tolist()


def align_to_raw_sides(refined: np.ndarray, raw: np.ndarray) -> np.ndarray:
    """삼각측량 결과의 좌/우 배정을 원본 검출 규약에 맞춘다 (쌍별로 전역 1회).

    하이브리드 구조에서 어깨는 원본을, 골반 아래는 삼각측량을 쓰므로 두 규약이
    전역적으로 뒤집혀 있으면 어깨-골반이 X자로 꼬인다.

    **프레임마다 개별로 뒤집으면 안 된다.** 규약 불일치는 전역적 현상이고, 한 프레임만
    교체하면 그 지점에서 궤적이 끊긴다(L발목이 f7=204, f9=164인데 f8만 346으로 튄 사례).
    옆모습에서는 좌우 간격이 0에 가까워 판정이 불안정하므로 간격을 가중치로 쓴다.
    """
    out = refined.copy()
    for left, right in SYMMETRIC_PAIRS:
        agree = disagree = 0.0
        for f in range(len(refined)):
            gap = abs(raw[f, left, 0] - raw[f, right, 0])
            if gap < 1e-6:
                continue
            if (raw[f, left, 0] < raw[f, right, 0]) == (out[f, left, 0] < out[f, right, 0]):
                agree += gap
            else:
                disagree += gap
        if disagree > agree:
            out[:, [left, right]] = out[:, [right, left]]
    return out
