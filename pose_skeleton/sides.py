"""좌우 순서 정렬 — 여러 결함의 공통 뿌리였던 부분.

YOLO는 프레임에 따라 좌우 라벨을 뒤바꿔 붙인다(14명 중 여러 명에서 1~3프레임).
섞인 채로 사인파를 맞추면 한 관절 궤적에 반대쪽 좌표가 끼어들어 피팅이 오염되고,
결과적으로 몸통이 X자로 꼬이거나 팔이 회전을 따라가지 않는다.

**원본 규약을 기준으로 삼으면 안 된다.** 원본이 그 프레임에서 틀려 있으면 틀린 것을
복원하게 된다. 측정: 원본은 몸통 X자가 0/224인데 초기 파이프라인이 24/224를 만들었다.
기준은 **좌우가 실제로 뒤바뀌는 관절 쌍**이어야 한다.
"""
import numpy as np

# 좌우 순서를 판정할 수 있는 최소 간격. 옆모습은 간격이 0에 가까워 판정이 무의미하다.
MIN_GAP = 10.0

# 함께 뒤집어야 하는 상체 묶음. 어깨-팔꿈치-손목은 한 사슬이므로 통째로 교체해야
# 사슬이 끊기지 않는다.
UPPER_BODY_GROUP = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10)]   # 눈, 귀, 어깨, 팔꿈치, 손목

REFERENCE_CANDIDATES = [(5, 6), (11, 12), (13, 14), (15, 16)]  # 어깨, 골반, 무릎, 발목
HIP_MIN_QUALITY = 0.75


def reference_quality(skeleton, pair) -> float:
    """좌우 순서가 제대로 뒤바뀌는가. 1에 가까울수록 정상, 0이면 아예 안 바뀐다.

    물리: 좌우 한 쌍의 x좌표 차이는 한 바퀴 도는 동안 부호가 두 번 바뀌어야 한다
    (평균 0 근처의 사인파). 평균이 진폭에 비해 크면 그 쌍은 기준으로 쓸 수 없다.
    """
    diff = skeleton[:, pair[0], 0] - skeleton[:, pair[1], 0]
    amp = (diff.max() - diff.min()) / 2
    return float(amp / (amp + abs(diff.mean()) + 1e-6))


def side_reference(skeleton):
    """좌우 순서의 기준 쌍을 고른다.

    기본은 골반이다(신뢰도가 항상 높고 삼각측량으로 매끄럽다). 골반이 실제로 실패했을
    때만 대체한다 -- **전원을 최고점 쌍으로 바꾸면 오히려 나빠진다.** 골반이 멀쩡한
    사람에서 발목의 잡음에 골반을 맞추게 되어 몸통 X자가 0/224 -> 9/224로 늘었다.

    실측으로 골반이 실패한 사람: a865f755 0.44, 172acab0 0.06, cb929d44 0.28.
    a865f755의 골반 좌우차는 76, 73, ..., 9, ..., 74로 한 번도 음수가 되지 않았고,
    그 골반에 다리를 맞추니 다리가 통째로 꼬였다(무릎·발목은 정상이었는데도).
    """
    if reference_quality(skeleton, (11, 12)) >= HIP_MIN_QUALITY:
        return (11, 12)
    return max(REFERENCE_CANDIDATES, key=lambda p: reference_quality(skeleton, p))


def align_groups(skeleton, groups, ref_pair=None):
    """관절 묶음의 프레임별 좌우를 기준 쌍에 맞춘다 (좌표만).

    묶음 하나는 통째로 함께 교체한다. 어깨만 뒤집고 팔꿈치를 두면 팔 사슬이 끊긴다.
    기준 쌍이 묶음 안에 있으면 그 묶음은 건너뛴다(자기 자신에 맞출 수 없다).
    """
    ra, rb = side_reference(skeleton) if ref_pair is None else ref_pair
    out = skeleton.copy()
    for pairs in groups:
        if (ra, rb) in pairs:
            continue
        for f in range(len(out)):
            ref_gap = out[f, ra, 0] - out[f, rb, 0]
            if abs(ref_gap) < MIN_GAP:
                continue
            probe = max(pairs, key=lambda ab: abs(out[f, ab[0], 0] - out[f, ab[1], 0]))
            probe_gap = out[f, probe[0], 0] - out[f, probe[1], 0]
            if abs(probe_gap) < MIN_GAP:
                continue
            if np.sign(ref_gap) != np.sign(probe_gap):
                for a, b in pairs:
                    out[f, [a, b]] = out[f, [b, a]]
    return out


def align_pairs(skeleton, pairs, ref_pair=None):
    """쌍마다 **독립적으로** 좌우를 기준에 맞춘다 (하체용).

    상체와 달리 무릎·발목은 각자 판정한다. 원본으로 되돌린 다리 쌍이 원본의 단발
    라벨 오류를 안고 오기 때문이다(172acab0 f6: L무릎 249 < R무릎 321인데
    골반은 L 289 > R 243 -- 한 프레임만 뒤집혀 허벅지가 X자였다).
    """
    ra, rb = side_reference(skeleton) if ref_pair is None else ref_pair
    out = skeleton.copy()
    for a, b in pairs:
        if (a, b) == (ra, rb):
            continue
        for f in range(len(out)):
            ref_gap = out[f, ra, 0] - out[f, rb, 0]
            pair_gap = out[f, a, 0] - out[f, b, 0]
            if abs(ref_gap) < MIN_GAP or abs(pair_gap) < MIN_GAP:
                continue
            if np.sign(ref_gap) != np.sign(pair_gap):
                out[f, [a, b]] = out[f, [b, a]]
    return out


def align_with_conf(kp, kp_conf, groups, ref_pair):
    """좌표와 신뢰도를 **함께** 정렬한다 (팔 피팅 입력을 만들 때 필요).

    신뢰도를 같이 옮기지 않으면 A쪽 좌표에 B쪽 신뢰도가 붙어 엉뚱한 프레임을 신뢰하게 된다.
    """
    ra, rb = ref_pair
    out_kp, out_conf = kp.copy(), kp_conf.copy()
    for pairs in groups:
        if (ra, rb) in pairs:
            continue
        for f in range(len(out_kp)):
            ref_gap = out_kp[f, ra, 0] - out_kp[f, rb, 0]
            if abs(ref_gap) < MIN_GAP:
                continue
            probe = max(pairs, key=lambda ab: abs(out_kp[f, ab[0], 0] - out_kp[f, ab[1], 0]))
            probe_gap = out_kp[f, probe[0], 0] - out_kp[f, probe[1], 0]
            if abs(probe_gap) < MIN_GAP:
                continue
            if np.sign(ref_gap) != np.sign(probe_gap):
                for a, b in pairs:
                    out_kp[f, [a, b]] = out_kp[f, [b, a]]
                    out_conf[f, [a, b]] = out_conf[f, [b, a]]
    return out_kp, out_conf
