# pose_skeleton

mmWave 전신 스캔(TSA Passenger Screening `.aps`)에서 **COCO 17관절 스켈레톤**을 추정한다.
사람당 16각도(660×512, 16bit)를 입력받아 `(16, 17, 2)` 배열을 낸다.

**재학습 없음.** RGB로 학습된 YOLOv8x-pose의 검출을 회전 기하로 정제한다.

---

## 1. 핵심 아이디어

스캔 중 **사람은 정지해 있고 스캐너만 360°를 돈다.** 프레임 `f`의 각도는 `2π·f/16`
(f0 = 정면, f8 = 후면)이므로, 몸에 고정된 관절의 영상 x좌표는

```
x(f) = axis + b·cos(θ_f) − c·sin(θ_f)
```

형태의 사인파여야 하고, y좌표는 (수직축 회전이므로) 일정해야 한다.
이 물리적 제약으로 검출 노이즈를 걷어내는 것이 이 패키지가 하는 일 전부다.

---

## 2. 설치

### 2-1. 파이썬 패키지

```bash
pip install ultralytics numpy opencv-python
pip install matplotlib            # example.py를 쓸 때만 필요
```

`torch`는 `ultralytics`가 알아서 함께 설치한다.

검증된 버전 조합 (다른 버전도 대체로 동작하나, 문제가 생기면 이 조합으로 맞춰볼 것):

| 패키지 | 버전 |
|---|---|
| ultralytics | 8.4.118 |
| numpy | 2.5.2 |
| opencv-python | 5.0.0.93 |
| torch | 2.13.0 |
| matplotlib | 3.11.1 |

### 2-2. 모델 가중치

`yolov8x-pose.pt` (133MB)가 필요하지만 **따로 받을 필요 없다.**
`YOLO("yolov8x-pose.pt")`를 처음 호출하면 ultralytics가 자동으로 내려받는다.

주의: 상대경로로 주면 **실행한 디렉터리에 133MB가 떨어진다.** 위치를 고정하고 싶으면
절대경로를 쓸 것.

```python
model = YOLO("/원하는/경로/yolov8x-pose.pt")
```

### 2-3. 설치 확인

```bash
python -m pose_skeleton.example <스캔파일>.aps check.png
```

`check.png`가 생기고 스켈레톤이 몸을 따라가면 정상이다.

---

## 3. 파일 구조

```
pose_skeleton/
├── __init__.py       공개 API 모음 (여기서 전부 import 가능)
├── pipeline.py       estimate_pose — 진입점, 단계 조립
├── detect.py         CLAHE 전처리 + YOLOv8x-pose 검출 + 관절별 가중치
├── rotation.py       사인파 모델, IRLS 강건 삼각측량, 전역 좌우 뒤집힘 판정
├── sides.py          좌우 순서 기준 선택 및 정렬
├── arms.py           팔 복원 (신뢰 프레임 선별 → RANSAC 궤도 → 뼈 길이 → 교차 해소)
├── aps.py            .aps 파일 IO
├── silhouette.py     몸/배경 실루엣 마스크
├── export.py         결과를 CSV로 내보내기 (ROI 연동용)
├── example.py        시각 확인용 데모 (파이프라인과 무관, 지워도 됨)
└── README.md         이 문서
```

의존 관계는 한 방향이다: `pipeline → {detect, rotation, sides, arms} → {aps, silhouette}`.
바깥 프로젝트 파일을 전혀 참조하지 않으므로 **폴더째 복사하면 그대로 동작한다.**

고칠 곳을 찾을 때:

| 증상 | 볼 파일 |
|---|---|
| 검출 자체가 안 됨 | `detect.py` |
| 몸통·다리가 흔들림 | `rotation.py`, `pipeline.refine_body` |
| 좌우가 뒤바뀜 / 몸통이 X자 | `sides.py` |
| 팔이 이상함 | `arms.py`, `pipeline.refine_arms` |

---

## 4. 사용법

### 4-1. 기본

```python
from ultralytics import YOLO
from pose_skeleton import estimate_pose

model = YOLO("yolov8x-pose.pt")          # 한 번만 로드해서 재사용할 것
skeleton, raw, rgbs, masks = estimate_pose(model, "data/stage1_aps/<scan_id>.aps")
```

반환값 4개:

| 이름 | 형태 | 설명 |
|---|---|---|
| `skeleton` | `(16, 17, 2)` float | **정제 결과.** 보통 이것만 쓰면 된다 |
| `raw` | `(16, 17, 2)` float | YOLO 원본 검출 (비교·디버깅용) |
| `rgbs` | list of 16 | 모델에 넣은 CLAHE 이미지 `(660, 512, 3)` uint8 |
| `masks` | list of 16 | 몸/배경 마스크 `(660, 512)` bool |

좌표는 **`to_upright` 적용 후 픽셀 좌표**다 (원점 좌상단, y는 아래로 증가).
`rgbs[f]`와 그대로 겹쳐 그리면 맞는다.

### 4-2. 관절 인덱스 (COCO 17)

```
 0 코        1 왼눈      2 오른눈    3 왼귀      4 오른귀
 5 왼어깨    6 오른어깨   7 왼팔꿈치   8 오른팔꿈치  9 왼손목   10 오른손목
11 왼골반   12 오른골반  13 왼무릎   14 오른무릎  15 왼발목  16 오른발목
```

```python
left_wrist_track = skeleton[:, 9]        # (16, 2) — 16프레임에 걸친 왼손목 궤적
frame8 = skeleton[8]                     # (17, 2) — 후면 프레임의 전체 관절
```

**좌/우 라벨은 "영상 기준"이 아니라 검출기 라벨을 정렬한 결과다.** 후면 프레임에서는
해부학적 좌우가 영상에서 반대로 보이는 것이 정상이다.

### 4-3. 여러 명 처리

```python
import glob
from ultralytics import YOLO
from pose_skeleton import estimate_pose

model = YOLO("yolov8x-pose.pt")          # 루프 밖에서 한 번만
results = {}
for path in sorted(glob.glob("data/stage1_aps/*.aps")):
    scan_id = path.split("/")[-1][:-4]
    results[scan_id], *_ = estimate_pose(model, path)
```

한 사람당 16회 추론이라 CPU에서 5~10초, GPU에서 1~2초 걸린다.
결과만 필요하면 `rgbs`·`masks`는 버려서 메모리를 아낄 것 (한 사람당 약 30MB).

### 4-4. 저장·불러오기

```python
import numpy as np
np.savez_compressed("skeletons.npz", **results)          # scan_id -> (16,17,2)

data = np.load("skeletons.npz")
skeleton = data["<scan_id>"]
```

### 4-5. 중간 단계 직접 쓰기

단계를 갈아끼우거나 진단할 때:

```python
from pose_skeleton import detect_all_frames, keypoint_weights
from pose_skeleton.pipeline import refine_body, refine_arms

kp, frame_conf, kp_conf, rgbs, masks = detect_all_frames(model, path)
weights = keypoint_weights(kp, frame_conf, masks, kp_conf=kp_conf)

body, _ = refine_body(kp, weights, frame_conf=frame_conf)   # 몸통·다리까지만
full = refine_arms(kp, kp_conf, masks, body)                # 팔까지
```

`estimate_pose`는 정확히 이 세 줄을 부르는 것이 전부다.

### 4-6. 그림 그리기

```python
import matplotlib.pyplot as plt
from pose_skeleton import COCO_SKELETON

f = 0
fig, ax = plt.subplots(figsize=(5, 6))
ax.imshow(rgbs[f])
xs, ys = skeleton[f, :, 0], skeleton[f, :, 1]
for i, j in COCO_SKELETON:
    ax.plot([xs[i], xs[j]], [ys[i], ys[j]], "-", c="cyan", lw=1.5)
ax.scatter(xs, ys, c="cyan", s=10)
ax.axis("off")
```

원본과 나란히 16프레임을 보려면 `example.py`를 쓰는 편이 빠르다.

### 4-7. 파일 하나만 읽기

```python
from pose_skeleton import read_aps, read_frame, to_upright, to_display

data = read_aps(path)                     # (16, 660, 512) uint16
frame = to_upright(read_frame(path, 8))   # 프레임 하나만 (파일 전체를 안 읽음)
plt.imshow(to_display(frame), cmap="gray")
```

`to_upright`는 **반드시** 거칠 것. 원본은 row 0이 발쪽이라 그대로 쓰면 상하가 뒤집힌다.

### 4-8. CSV로 내보내기

```python
from pose_skeleton.export import export_subject

export_subject("pose_keypoints.csv", scan_id, skeleton, kp_conf)   # 16 x 17 = 272행
```

컬럼: `subject_id, view_idx, keypoint_id, keypoint_name, x, y, confidence, valid`
(`valid`는 관절 신뢰도 ≥ 0.70 기준). 여러 명은 `append=True`로 이어 붙인다.

```python
export_subject(out, sid, skeleton, kp_conf, append=True)
```

---

## 5. ROI 파이프라인 연동 (`spwhay/UK-project_ROI`)

팀의 ROI 파이프라인은 pose 결과를 받아 zone ROI와 병합한다
(`roi_stage1_pose_handoff/POSE_ROI_HANDOFF.md`). **현재 이 패키지의 출력을 그대로
넣을 수는 없다.** 세 가지가 다르다.

| | 이 패키지가 내는 것 | ROI 파이프라인이 요구하는 것 |
|---|---|---|
| 대상 | COCO **관절** 17개 | 신체 **zone** 17개 |
| 형태 | 점 `(x, y)` | 사각형 `[x1, y1, x2, y2)` |
| 좌표계 | 660×512 원본 픽셀 | 224×224 캐시 뷰 |

### ⚠ 숫자 17의 함정

양쪽 다 17이지만 **전혀 다른 것**이다.

- 이 패키지의 17 = COCO 관절 (코, 눈, 귀, 어깨, 팔꿈치, 손목, 골반, 무릎, 발목)
- ROI의 17 = 신체 구역 (bicep, forearm, chest, abdomen, thigh, calf, ankle, upper back …)

`keypoint_id`를 `zone_id`로 그냥 매핑하면 안 된다. 예를 들어 관절 5는 왼어깨지만
zone 5는 chest다.

### 남아 있는 두 가지 작업 (이 저장소 범위 밖)

1. **관절 → zone 박스 변환.** 관절 좌표에서 각 zone의 사각형을 만들어야 한다.
   본 저장소는 **스켈레톤까지가 범위**이며 zone 판정은 하지 않기로 정했다.
   ROI 쪽 `zone_roi_pipeline_local.py`의 `ANCHORS` 표(zone별 신체 높이 비율)가
   출발점이 될 수 있다.
2. **좌표 변환 (660×512 → 224×224).** ROI 쪽 224 뷰는 `aps_cache_all/` 캐시에서 오는데
   그 캐시를 만드는 코드가 공개 저장소에 없다. **단순 리사이즈인지 크롭 후 리사이즈인지
   확인해야 한다.** 단순 비율이면 `export_subject(..., scale=(224/512, 224/660))`으로
   충분하지만, 크롭이 섞이면 비율만으로는 틀린다.

### 연동 시 반드시 지킬 것

- 조인 키는 `subject_id`, `view_idx`(0–15), `zone_id`(1–17) 세 개다.
- **뷰 순서와 해부학적 좌우를 바꾸지 말 것.** 이 패키지의 `view_idx`는 `.aps` 프레임
  순서 그대로이고, 좌우는 검출기 라벨을 정렬한 결과다 (§4-2 참조).
- 좌표는 half-open `[x1, y1, x2, y2)`, `0 <= x1 < x2 <= 224`.
- 신뢰할 수 없으면 `pose_valid=false`로 두고 좌표를 비우는 편이 낫다. ROI 쪽이 기존
  ROI로 폴백하도록 설계돼 있다.
- **§7의 시점별 품질표를 폴백 정책에 반영할 것.** f4(90°)는 14명 전원 팔을 믿을 수
  없으므로, 그 각도의 팔 관련 zone(1–4)은 애초에 `pose_valid=false`로 두는 편이 안전하다.

---

## 6. 파이프라인 단계

```
CLAHE + YOLOv8x-pose 검출                       detect.py
  → 강건 삼각측량 (강체 관절만)                  rotation.py
  → 붕괴한 다리 쌍은 원본 유지                   pipeline.revert_collapsed_legs
  → 하체 좌우 정렬                               sides.align_pairs
  → 검출 실패 프레임 보간                        pipeline.fill_failed_frames
  → 팔 복원                                      arms.py
      좌우 정렬 → 보존/채움 → 교차 해소
      → 상체 좌우 정렬 → 뼈 길이 강제
```

두 가지 설계 결정이 성능의 대부분을 만든다.

- 삼각측량은 **강체 관절에만** 적용한다 (코·골반·무릎·발목). 어깨를 포함시키면
  팔 이탈률이 0.158 → 0.196으로 악화된다. 팔이 붙어 있어 어깨는 실제로 미세하게 움직인다.
- 팔은 **믿을 프레임만 보존하고 나머지만 채운다.** 어떤 피팅 방법이든 멀쩡한 프레임을
  14~37px씩 밀어내므로, 보존하면 그 손해가 0이 된다.

---

## 7. 성능 (14명 × 16프레임)

정답 라벨이 없으므로 **holdout 방식**으로 잰다 — 믿을 수 있는 프레임 하나를 실제
관측된 실패 모습대로 망가뜨린 뒤 복원시켜 픽셀 오차를 잰다 (사람 손 개입 없음).

| 항목 | 원본 검출 | 이 파이프라인 |
|---|---|---|
| 팔 복원 오차 중앙값 | 165.1px | **16.0px** |
| 팔 100px 초과 실패 | 64.7% | **6.9%** |
| 팔 X자 교차 | 11/224 | **0/224** |
| 몸통으로 처진 팔 | 있음 | **0/448** |
| 두 팔 겹침 | 68/224 | 30/224 |
| 몸통 X자 | 0/224 | **0/224** |
| 뼈 길이 상한 위반 | — | **0/448** |
| 몸통·다리 실루엣 이탈률 | 0.067 | **0.039** |

---

## 8. 시점별 품질 — 각도가 성능을 지배한다

| 프레임 | 각도 | 팔 신뢰 가능 | 두 팔 겹침 |
|---|---|---|---|
| f14, f15, f0, f1, f2 | 315~45° | 0.93~1.00 | 0.00 |
| f8, f9 | 180~202° | 0.86 | 0.07 |
| f6, f7, f10 | 135~225° | 0.71~0.79 | 0.07~0.14 |
| f3, f13 | 67°, 292° | 0.57~0.64 | 0.36~0.43 |
| **f4, f5, f11, f12** | **90~270°** | **0.00~0.14** | **0.79~0.93** |

정확히 옆모습(f4 = 90°)에서는 **14명 전원 팔을 하나도 믿을 수 없다.** 먼 쪽 팔이 몸 뒤에
완전히 숨기 때문이며, 이미지 개선으로는 고칠 수 없다. 후면(f8)은 의외로 좋다.

시점 수를 줄이는 실험을 한다면 정면부(f14, f15, f0, f1, f2)와 후면(f8, f9)을 먼저 남길 것.

---

## 9. 알려진 한계

- **신뢰 프레임이 원 둘레에 고르게 없으면 복원이 불가능하다.** 14명 중 2명이 해당한다.
  한 사람은 오른팔에 쓸 수 있는 프레임이 f0, f1, f14, f15 넷뿐이고 전부 정면 근처라,
  거기서 세운 사인파를 원의 3/4로 외삽하는 셈이 된다. 이 두 명은 holdout 오차가
  178px / 41px로, 아무 처리도 안 한 것과 사실상 같다.
- **좌우 순서의 기준이 될 관절이 없는 사람이 있다.** 좌우 쌍의 x차이는 회전하며 부호가
  두 번 바뀌어야 하는데(점수 1에 가까움), 정상인 사람은 0.97~0.99인 반면 이 두 명은
  네 쌍(어깨·골반·무릎·발목) 모두 0.66 이하다.
- 손목이 실제 손보다 안쪽에 찍히는 경향이 남아 있다. 검출기가 원래 짧게 잡는 것이라
  실루엣 끝까지 늘리는 보정은 위험하다 (이 프로젝트에서 두 번 실패했다).
- 옆모습에서 두 다리가 투영상 겹치는 것은 **물리적으로 정상**이다. 다리 교차를 0으로
  만들려 하지 말 것 (원본도 33/224에서 교차한다).

---

## 10. 다시 시도하지 말 것 (이미 실패)

- **전처리 개선** — CLAHE 강화·감마·로그변환 5종 모두 1~4% 차이(노이즈 수준).
  팔이 어두워서가 아니라 각도상 가려져서다. 팔꿈치 밝기(0.234)는 발목(0.229)과 같고
  밝기-신뢰도 상관은 ρ = 0.25다.
- **크롭 + 확대, imgsz=1280** — 오히려 악화(conf 0.885 → 0.512). mmWave 노이즈도 확대된다.
- **yolo11x-pose, yolov8x-pose-p6** — 낱장에선 좋아 보이나 전체로는 v8x보다 나쁘다.
- **pseudo-label 파인튜닝** — 삼각측량 결과로 YOLO 미세조정. 뼈 길이 CV는 좋아졌으나
  옆모습 팔을 놓친다.
- **회전축 공유 제약** — 모든 관절의 사인파 offset을 하나로 고정. 30.7 → 40.0px 악화.
- **3D 강체 사슬** — 팔을 어깨에 붙은 3D 강체로 놓고 뼈 길이를 제약. 16.0 → 24.6px 악화.
- **전 프레임을 궤도로 대체** — 문제가 있는 사람에서도 개선되지 않고 나머지가 나빠진다.
- **좌우 대칭 사전정보** — 두 팔이 거울상이라는 관계는 실제로 성립한다(방위각 합
  −179.2° ± 6.2°). 그런데 그것으로 고장난 팔을 복구해도 나아지지 않았다. 팔꿈치 각도가
  v9와 거울 복구본 모두 정상 범위(74~172°) 안이라 구분이 안 되고, holdout도 16.0px로
  동일했다. 전제가 맞아도 기준으로 삼을 반대쪽 팔 자체가 불안정하면 소용이 없다.

---

## 11. 지표를 믿기 전에 읽을 것

이 프로젝트에서 **지표가 여덟 번 잘못된 결론을 유도했다.** 공통 패턴은 하나다 —
**팔이나 다리가 몸통 쪽으로 붕괴할수록 좋아지는 지표**가 많다.

- 뼈 길이 CV는 붕괴한 쪽이 오히려 좋다 (다리 붕괴 사례: 0.015 vs 정상 0.021).
- 실루엣 이탈률·밝기도 팔이 몸통에 붙을수록 좋아진다.
- 오차 계산에서 좌우 순열의 최소값을 취하면, 두 팔을 하나로 겹친 방법이 유리해진다.

그래서 이 파이프라인의 판정은 **(1) 서로 게임하기 어려운 두 지표를 함께 보고,
(2) 반드시 눈으로 확인**하는 절차를 거쳤다. 실제로 마지막 다섯 개 결함은 전부
수치가 통과시킨 것을 육안으로 잡았다.

**새 방법을 시도한다면 `example.py`로 16프레임을 눈으로 보는 것을 반드시 포함할 것.**

---

## 12. 데이터 취급

대회 규칙상 스캔 원본과 원본이 그대로 보이는 결과 그림은 **재배포 금지**다.
`.aps` 파일과 파생 이미지를 저장소에 커밋하지 말 것.
