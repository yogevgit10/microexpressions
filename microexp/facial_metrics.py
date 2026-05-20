from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


METRIC_COLUMNS = [
    "au06_cheek_raise_proxy",
    "au12_lip_corner_pull_proxy",
    "au09_nose_wrinkle_proxy",
    "au10_upper_lip_raise_proxy",
    "au24_lip_press_proxy",
    "au45_blink_proxy",
    "cheek_composite_proxy",
    "mouth_open_proxy",
    "mouth_asymmetry_proxy",
    "head_yaw_proxy",
    "head_roll_deg",
]


KEY_EVENT_METRICS = [
    "au06_cheek_raise_proxy",
    "au12_lip_corner_pull_proxy",
    "au09_nose_wrinkle_proxy",
    "au10_upper_lip_raise_proxy",
    "au24_lip_press_proxy",
    "au45_blink_proxy",
    "cheek_composite_proxy",
]


@dataclass
class FaceMetricExtractor:
    refine_landmarks: bool = True
    model_path: str | Path | None = None

    def __post_init__(self):
        try:
            import mediapipe as mp  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "mediapipe is required. Install dependencies with: "
                "python -m pip install -r requirements.txt"
            ) from exc
        self._mp = mp
        self._api = None
        self._face_mesh = None
        self._landmarker = None

        if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
            self._api = "solutions"
            self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=self.refine_landmarks,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            return

        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        self._api = "tasks"
        model_path = Path(self.model_path) if self.model_path else default_model_path()
        if not model_path.exists():
            raise RuntimeError(
                "MediaPipe Tasks requires a local face_landmarker.task model. "
                f"Expected it at: {model_path}"
            )

        options = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)

    def close(self):
        if self._face_mesh is not None:
            self._face_mesh.close()
        if self._landmarker is not None:
            self._landmarker.close()

    def process_bgr(self, cv2, frame, frame_idx: int, timestamp_s: float) -> dict:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        base = {
            "frame_idx": frame_idx,
            "timestamp_s": timestamp_s,
            "face_detected": 0,
        }
        for col in METRIC_COLUMNS:
            base[col] = np.nan

        if self._api == "solutions":
            result = self._face_mesh.process(rgb)
            if not result.multi_face_landmarks:
                return base
            landmarks = result.multi_face_landmarks[0].landmark
        else:
            mp_image = self._mp.Image(
                image_format=self._mp.ImageFormat.SRGB,
                data=np.ascontiguousarray(rgb),
            )
            timestamp_ms = int(round(timestamp_s * 1000.0))
            result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
            if not result.face_landmarks:
                return base
            landmarks = result.face_landmarks[0]

        metrics = compute_metrics(landmarks)
        base.update(metrics)
        base["face_detected"] = 1
        return base


def default_model_path() -> Path:
    return Path(__file__).resolve().parents[1] / "models" / "face_landmarker.task"


def _pt(landmarks, idx: int) -> np.ndarray:
    lm = landmarks[idx]
    return np.array([float(lm.x), float(lm.y), float(lm.z)], dtype=float)


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a[:2] - b[:2]))


def _eye_aspect_ratio(landmarks, left: bool) -> float:
    if left:
        outer, inner = _pt(landmarks, 33), _pt(landmarks, 133)
        upper_a, lower_a = _pt(landmarks, 159), _pt(landmarks, 145)
        upper_b, lower_b = _pt(landmarks, 160), _pt(landmarks, 144)
    else:
        outer, inner = _pt(landmarks, 263), _pt(landmarks, 362)
        upper_a, lower_a = _pt(landmarks, 386), _pt(landmarks, 374)
        upper_b, lower_b = _pt(landmarks, 385), _pt(landmarks, 380)
    width = max(_dist(outer, inner), 1e-9)
    return (_dist(upper_a, lower_a) + _dist(upper_b, lower_b)) / (2.0 * width)


def compute_metrics(landmarks) -> dict:
    left_eye_outer = _pt(landmarks, 33)
    right_eye_outer = _pt(landmarks, 263)
    scale = max(_dist(left_eye_outer, right_eye_outer), 1e-9)

    left_eye_center = (_pt(landmarks, 33) + _pt(landmarks, 133)) / 2.0
    right_eye_center = (_pt(landmarks, 263) + _pt(landmarks, 362)) / 2.0
    nose_tip = _pt(landmarks, 1)
    nose_base = _pt(landmarks, 2)
    left_nostril = _pt(landmarks, 98)
    right_nostril = _pt(landmarks, 327)
    upper_lip = _pt(landmarks, 13)
    lower_lip = _pt(landmarks, 14)
    left_mouth = _pt(landmarks, 61)
    right_mouth = _pt(landmarks, 291)
    left_cheek = _pt(landmarks, 205)
    right_cheek = _pt(landmarks, 425)
    left_lower_eyelid = _pt(landmarks, 145)
    right_lower_eyelid = _pt(landmarks, 374)

    cheek_eye_left = _dist(left_cheek, left_lower_eyelid) / scale
    cheek_eye_right = _dist(right_cheek, right_lower_eyelid) / scale
    au06 = -float(np.mean([cheek_eye_left, cheek_eye_right]))

    mouth_corner_y = float(np.mean([left_mouth[1], right_mouth[1]]))
    upper_lip_y = float(upper_lip[1])
    au12 = -(mouth_corner_y - upper_lip_y) / scale

    nose_width = _dist(left_nostril, right_nostril) / scale
    au09 = nose_width

    au10 = -(_dist(nose_base, upper_lip) / scale)
    mouth_open = _dist(upper_lip, lower_lip) / scale
    au24 = -mouth_open

    left_ear = _eye_aspect_ratio(landmarks, left=True)
    right_ear = _eye_aspect_ratio(landmarks, left=False)
    au45 = -float(np.mean([left_ear, right_ear]))

    mouth_asymmetry = abs(float(left_mouth[1] - right_mouth[1])) / scale
    cheek_composite = float(np.mean([au06, au12, au10]))

    left_nose_dist = _dist(nose_tip, left_eye_center)
    right_nose_dist = _dist(nose_tip, right_eye_center)
    head_yaw = (left_nose_dist - right_nose_dist) / scale
    dy = float(right_eye_center[1] - left_eye_center[1])
    dx = float(right_eye_center[0] - left_eye_center[0])
    head_roll = math.degrees(math.atan2(dy, dx))

    return {
        "au06_cheek_raise_proxy": au06,
        "au12_lip_corner_pull_proxy": au12,
        "au09_nose_wrinkle_proxy": au09,
        "au10_upper_lip_raise_proxy": au10,
        "au24_lip_press_proxy": au24,
        "au45_blink_proxy": au45,
        "cheek_composite_proxy": cheek_composite,
        "mouth_open_proxy": mouth_open,
        "mouth_asymmetry_proxy": mouth_asymmetry,
        "head_yaw_proxy": head_yaw,
        "head_roll_deg": head_roll,
    }
