"""Real-time BigSmall demo for a webcam, video, or UBFC-rPPG clip directory."""

from __future__ import annotations

import argparse
import csv
import math
import sys
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
from scipy import signal

from neural_methods.model.BigSmall import BigSmall


ROOT = Path(__file__).resolve().parent
DEFAULT_CLIPS = ROOT / "data" / "demo" / "UBFC-rPPG" / "faces" / "subject1"
DEFAULT_GT = ROOT / "data" / "demo" / "UBFC-rPPG" / "subject1" / "ground_truth.txt"
DEFAULT_MODEL = ROOT / "final_model_release" / "BP4D_BigSmall_Multitask_Fold2.pth"
DEFAULT_FACE_LANDMARKER = ROOT / "assets" / "models" / "face_landmarker.task"
AU_NAMES = [
    "AU01", "AU02", "AU04", "AU06", "AU07", "AU10",
    "AU12", "AU14", "AU15", "AU17", "AU23", "AU24",
]


class FrameSource:
    fps: float = 25.0
    pre_cropped: bool = False

    def read(self):
        raise NotImplementedError

    def close(self):
        pass


class CaptureSource(FrameSource):
    def __init__(self, value, camera=False, camera_fps=30.0, camera_backend="auto"):
        backend_names = {"any": cv2.CAP_ANY}
        if hasattr(cv2, "CAP_MSMF"):
            backend_names["msmf"] = cv2.CAP_MSMF
        if hasattr(cv2, "CAP_DSHOW"):
            backend_names["dshow"] = cv2.CAP_DSHOW
        if camera and camera_backend == "auto" and sys.platform == "win32":
            candidates = [name for name in ("msmf", "dshow", "any") if name in backend_names]
        else:
            candidates = [camera_backend if camera else "any"]
        self.capture = None
        self.backend_name = candidates[0]
        for name in candidates:
            candidate = cv2.VideoCapture(value, backend_names[name])
            if candidate.isOpened():
                self.capture = candidate
                self.backend_name = name
                break
            candidate.release()
        if self.capture is None:
            raise RuntimeError(f"Cannot open input: {value} (tried {', '.join(candidates)})")
        if camera:
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.capture.set(cv2.CAP_PROP_FPS, camera_fps)
        measured_fps = self.capture.get(cv2.CAP_PROP_FPS)
        self.fps = measured_fps if measured_fps and measured_fps > 1 else camera_fps
        self.camera = camera
        self.index = 0
        self.start_time = time.perf_counter()
        self.pre_cropped = False
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._stop_event = threading.Event()
        self._latest_lock = threading.Lock()
        self._latest = None
        self._latest_sequence = -1
        self._returned_sequence = -1
        self._capture_thread = None
        if self.camera:
            self._capture_thread = threading.Thread(
                target=self._capture_latest_loop, name="camera-latest-frame", daemon=True
            )
            self._capture_thread.start()

    def _capture_latest_loop(self):
        sequence = 0
        while not self._stop_event.is_set():
            ok, frame = self.capture.read()
            if not ok:
                time.sleep(0.005)
                continue
            timestamp = time.perf_counter() - self.start_time
            with self._latest_lock:
                self._latest = (frame, timestamp, sequence)
                self._latest_sequence = sequence
            sequence += 1

    def read(self):
        if self.camera:
            deadline = time.perf_counter() + 2.0
            while time.perf_counter() < deadline and not self._stop_event.is_set():
                with self._latest_lock:
                    if self._latest is not None and self._latest_sequence > self._returned_sequence:
                        frame, timestamp, sequence = self._latest
                        self._returned_sequence = sequence
                        return frame, timestamp, sequence
                time.sleep(0.001)
            return None
        ok, frame = self.capture.read()
        if not ok:
            return None
        timestamp = self.index / self.fps
        self.index += 1
        return frame, timestamp, self.index - 1

    def close(self):
        self._stop_event.set()
        if self._capture_thread is not None:
            self._capture_thread.join(timeout=0.5)
        self.capture.release()
        if self._capture_thread is not None and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=0.5)


class ClipDirectorySource(FrameSource):
    """Streams the final frame from consecutive five-frame UBFC face clips."""

    def __init__(self, directory: Path, fps=29.326, start_timestamp=0.109):
        self.files = sorted(directory.glob("clip_*.avi"))
        if not self.files:
            raise FileNotFoundError(f"No clip_*.avi files found in {directory}")
        self.fps = fps
        self.start_timestamp = start_timestamp
        self.index = 0
        self.pre_cropped = True

    def read(self):
        if self.index >= len(self.files):
            return None
        capture = cv2.VideoCapture(str(self.files[self.index]))
        frame = None
        while True:
            ok, candidate = capture.read()
            if not ok:
                break
            frame = candidate
        capture.release()
        if frame is None:
            raise RuntimeError(f"Cannot decode {self.files[self.index]}")
        result = frame, self.start_timestamp + self.index / self.fps, self.index
        self.index += 1
        return result


class GroundTruth:
    def __init__(self, path: Path | None):
        self.ppg = self.hr = self.timestamps = None
        if path and path.exists():
            rows = np.loadtxt(path)
            if rows.ndim == 2 and rows.shape[0] >= 3:
                self.ppg, self.hr, self.timestamps = rows[:3]

    def at(self, timestamp):
        if self.timestamps is None:
            return math.nan, math.nan
        index = int(np.searchsorted(self.timestamps, timestamp, side="left"))
        index = min(max(index, 0), len(self.timestamps) - 1)
        return float(self.ppg[index]), float(self.hr[index])


@dataclass
class FaceObservation:
    crop: np.ndarray | None = None
    box: tuple[int, int, int, int] | None = None
    status: str = "NO FACE"
    motion: float = math.nan
    blendshapes: dict[str, float] = field(default_factory=dict)


class FaceTracker:
    """Landmark-based video face tracking with a short loss grace period."""

    def __init__(self, pre_cropped=False, model_path=DEFAULT_FACE_LANDMARKER,
                 grace_seconds=1.0):
        self.pre_cropped = pre_cropped
        self.grace_seconds = float(grace_seconds)
        self.box = None
        self.last_success_timestamp = -math.inf
        self.last_timestamp_ms = -1
        self.previous_landmarks = None
        self.landmarker = None
        if not pre_cropped:
            if not Path(model_path).exists():
                raise FileNotFoundError(f"MediaPipe face model not found: {model_path}")
            vision = mp.tasks.vision
            self._model_buffer = Path(model_path).read_bytes()
            options = vision.FaceLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_buffer=self._model_buffer),
                running_mode=vision.RunningMode.VIDEO,
                num_faces=1,
                min_face_detection_confidence=0.35,
                min_face_presence_confidence=0.35,
                min_tracking_confidence=0.35,
                output_face_blendshapes=False,
            )
            self.landmarker = vision.FaceLandmarker.create_from_options(options)

    @staticmethod
    def _square_crop(frame, box, scale=1.12):
        height, width = frame.shape[:2]
        x, y, w, h = [float(value) for value in box]
        side = max(w, h) * scale
        center_x = x + w / 2.0
        center_y = y + h / 2.0 + 0.02 * h
        x1 = int(max(0, round(center_x - side / 2.0)))
        y1 = int(max(0, round(center_y - side / 2.0)))
        x2 = int(min(width, round(center_x + side / 2.0)))
        y2 = int(min(height, round(center_y + side / 2.0)))
        if x2 <= x1 or y2 <= y1:
            return None
        return frame[y1:y2, x1:x2]

    @staticmethod
    def _landmark_box(points, frame_shape):
        height, width = frame_shape[:2]
        x1, y1 = np.quantile(points, 0.01, axis=0)
        x2, y2 = np.quantile(points, 0.99, axis=0)
        face_w, face_h = x2 - x1, y2 - y1
        x1 -= 0.06 * face_w
        x2 += 0.06 * face_w
        y1 -= 0.10 * face_h
        y2 += 0.05 * face_h
        x1, y1 = max(0.0, x1), max(0.0, y1)
        x2, y2 = min(float(width - 1), x2), min(float(height - 1), y2)
        return np.asarray([x1, y1, x2 - x1, y2 - y1], dtype=np.float32)

    def _observation_from_box(self, frame, box, status, motion=math.nan,
                              blendshapes=None):
        crop = self._square_crop(frame, box)
        if crop is None:
            return FaceObservation()
        display_box = tuple(int(round(value)) for value in box)
        return FaceObservation(
            crop=crop,
            box=display_box,
            status=status,
            motion=float(motion),
            blendshapes=blendshapes or {},
        )

    def process(self, frame, timestamp):
        if self.pre_cropped:
            height, width = frame.shape[:2]
            return FaceObservation(
                crop=frame,
                box=(0, 0, width, height),
                status="TRACKING",
                motion=0.0,
            )

        timestamp_ms = max(self.last_timestamp_ms + 1, int(round(timestamp * 1000.0)))
        self.last_timestamp_ms = timestamp_ms
        rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result = self.landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp_ms
        )
        if result.face_landmarks:
            height, width = frame.shape[:2]
            points = np.asarray([
                (landmark.x * width, landmark.y * height)
                for landmark in result.face_landmarks[0]
            ], dtype=np.float32)
            detected_box = self._landmark_box(points, frame.shape)
            if self.box is None:
                self.box = detected_box
            else:
                old_center = self.box[:2] + self.box[2:] / 2.0
                new_center = detected_box[:2] + detected_box[2:] / 2.0
                displacement = float(np.linalg.norm(new_center - old_center))
                reference = max(float(self.box[2]), float(self.box[3]), 1.0)
                new_weight = 0.70 if displacement > 0.10 * reference else 0.38
                self.box = (1.0 - new_weight) * self.box + new_weight * detected_box
            motion = 0.0
            if self.previous_landmarks is not None:
                motion = float(np.median(np.linalg.norm(
                    points - self.previous_landmarks, axis=1
                )) / max(self.box[2], self.box[3], 1.0))
            self.previous_landmarks = points
            self.last_success_timestamp = float(timestamp)
            blendshapes = {}
            if result.face_blendshapes:
                blendshapes = {
                    category.category_name: float(category.score)
                    for category in result.face_blendshapes[0]
                }
            return self._observation_from_box(
                frame, self.box, "TRACKING", motion, blendshapes
            )

        if self.box is not None and timestamp - self.last_success_timestamp <= self.grace_seconds:
            return self._observation_from_box(frame, self.box, "TRACKING WEAK")
        self.previous_landmarks = None
        return FaceObservation()

    def close(self):
        if self.landmarker is not None:
            self.landmarker.close()


class OnlineBigSmallPreprocessor:
    def __init__(self, stats_frames=125):
        self.history = deque(maxlen=stats_frames)
        self.pending = deque()

    def reset_pending(self):
        self.pending.clear()

    def push(self, crop, timestamp):
        resized = cv2.resize(crop, (144, 144), interpolation=cv2.INTER_AREA).astype(np.float32)
        self.history.append(resized)
        self.pending.append((resized, float(timestamp)))
        if len(self.pending) < 4:
            return None

        pending = list(self.pending)
        pending_frames = [item[0] for item in pending]
        sample_timestamps = np.asarray([item[1] for item in pending[:3]], dtype=np.float64)
        big_raw = np.stack(pending_frames[:3])
        history = np.stack(self.history)
        mean, std = float(history.mean()), float(history.std())
        big = (big_raw - mean) / max(std, 1e-6)

        small = []
        for first, second in zip(pending_frames[:3], pending_frames[1:4]):
            diff = (second - first) / (second + first + 1e-7)
            small.append(diff)
        small = np.stack(small)

        recent = history[-min(len(history), 30):]
        if len(recent) > 1:
            recent_diff = (recent[1:] - recent[:-1]) / (recent[1:] + recent[:-1] + 1e-7)
            diff_std = float(recent_diff.std())
        else:
            diff_std = float(small.std())
        small = small / max(diff_std, 1e-6)
        small = np.stack([
            cv2.resize(image, (9, 9), interpolation=cv2.INTER_AREA) for image in small
        ])

        for _ in range(3):
            self.pending.popleft()

        big_tensor = torch.from_numpy(big.transpose(0, 3, 1, 2)).float()
        small_tensor = torch.from_numpy(small.transpose(0, 3, 1, 2)).float()
        return big_tensor, small_tensor, sample_timestamps


class BigSmallPredictor:
    def __init__(self, checkpoint: Path, device="cuda:0"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = BigSmall(n_segment=3).to(self.device).eval()
        state_dict = torch.load(checkpoint, map_location=self.device, weights_only=True)
        if state_dict and next(iter(state_dict)).startswith("module."):
            state_dict = OrderedDict((name[7:], value) for name, value in state_dict.items())
        self.model.load_state_dict(state_dict)

    def predict(self, tensors):
        big, small = (tensor.to(self.device) for tensor in tensors)
        with torch.inference_mode():
            au, bvp, respiration = self.model((big, small))
        return (
            torch.sigmoid(au).cpu().numpy(),
            bvp[:, 0].cpu().numpy(),
            respiration[:, 0].cpu().numpy(),
        )


@dataclass
class Metrics:
    hr: float = math.nan
    hr_bigsmall: float = math.nan
    hr_source: str = "--"
    rr: float = math.nan
    rmssd: float = math.nan
    sdnn: float = math.nan
    snr: float = math.nan
    snr_bigsmall: float = math.nan
    beats: int = 0
    quality: str = "WARMING UP"
    sample_rate: float = math.nan


@dataclass
class AffectiveState:
    state: str = "CALIBRATING"
    valence: float = 0.0
    arousal: float = 0.0
    confidence: float = 0.0
    calibration_progress: float = 0.0
    face_calibrated: bool = False
    evidence: dict[str, float] = field(default_factory=lambda: {
        "smile": 0.0,
        "tension": 0.0,
        "brow_raise": 0.0,
        "sadness_like": 0.0,
        "disgust_like": 0.0,
        "lip_tightening": 0.0,
    })
    physiology_arousal: float = 0.5


class PhysiologyAnalyzer:
    def __init__(self, fps=25.0, max_seconds=300):
        self.fps = fps
        self.bvp_diff = deque(maxlen=int(fps * max_seconds))
        self.resp_diff = deque(maxlen=int(fps * max_seconds))
        self.bvp_time = deque(maxlen=int(fps * max_seconds))
        self.resp_time = deque(maxlen=int(fps * max_seconds))
        self.au = deque(maxlen=int(fps * 3))
        self.last_bvp = np.empty(0)
        self.last_resp = np.empty(0)
        self.last_hr = math.nan
        self.last_hr_bigsmall = math.nan

    def add(self, au, bvp, respiration, timestamps, physiology_ok=True):
        if physiology_ok:
            self.bvp_diff.extend(float(value) for value in bvp)
            self.resp_diff.extend(float(value) for value in respiration)
            self.bvp_time.extend(float(value) for value in timestamps)
            self.resp_time.extend(float(value) for value in timestamps)
        self.au.extend(row for row in au)

    def smooth_au(self, seconds=1.0):
        if not self.au:
            return np.zeros(len(AU_NAMES), dtype=np.float32)
        count = max(1, int(self.fps * seconds))
        return np.mean(np.stack(list(self.au)[-count:]), axis=0)

    @staticmethod
    def _recover(values, low, high, fs):
        values = np.asarray(values, dtype=np.float64)
        if len(values) < max(30, int(fs * 3)):
            return np.empty(0)
        recovered = signal.detrend(np.cumsum(values))
        sos = signal.butter(3, [low, high], btype="bandpass", fs=fs, output="sos")
        return signal.sosfiltfilt(sos, recovered)

    @staticmethod
    def _rate_fft(waveform, low, high, fs, previous=math.nan):
        frequencies, power = signal.periodogram(
            waveform, fs=fs, window="hann", nfft=max(2048, len(waveform))
        )
        mask = (frequencies >= low) & (frequencies <= high)
        if not mask.any():
            return math.nan, math.nan
        band_f, band_p = frequencies[mask], power[mask]
        peak_index = int(np.argmax(band_p))
        if math.isfinite(previous):
            continuity = np.abs(band_f - previous / 60.0) <= 0.35
            if continuity.any():
                local_indices = np.flatnonzero(continuity)
                local_index = int(local_indices[np.argmax(band_p[continuity])])
                if band_p[local_index] >= 0.20 * band_p[peak_index]:
                    peak_index = local_index
        peak_frequency = float(band_f[peak_index])
        signal_mask = np.abs(band_f - peak_frequency) <= 0.10
        signal_power = float(band_p[signal_mask].sum())
        noise_power = float(band_p[~signal_mask].sum()) + 1e-12
        return peak_frequency * 60.0, 10.0 * math.log10(signal_power / noise_power + 1e-12)

    def compute(self, face_ok=True, brightness=128.0, motion=0.0):
        result = Metrics()
        bvp_fps = self.fps
        resp_fps = self.fps
        bvp_window = np.asarray(list(self.bvp_diff)[-int(self.fps * 60):], dtype=np.float64)
        resp_window = np.asarray(list(self.resp_diff)[-int(self.fps * 45):], dtype=np.float64)
        if len(self.bvp_time) > 1:
            time_window = np.asarray(list(self.bvp_time)[-int(self.fps * 60):], dtype=np.float64)
            duration = float(time_window[-1] - time_window[0])
            if duration > 0:
                result.sample_rate = (len(time_window) - 1) / duration
        self.last_bvp = self._recover(bvp_window, 0.7, 3.0, bvp_fps)
        self.last_resp = self._recover(resp_window, 0.13, 0.5, resp_fps)

        big_snr = math.nan
        if len(self.last_bvp) >= int(bvp_fps * 8):
            raw_bigsmall, big_snr = self._rate_fft(
                self.last_bvp, 0.7, 3.0, bvp_fps, self.last_hr_bigsmall
            )
            if math.isfinite(self.last_hr_bigsmall):
                raw_bigsmall = float(np.clip(
                    raw_bigsmall, self.last_hr_bigsmall - 3.0, self.last_hr_bigsmall + 3.0
                ))
                self.last_hr_bigsmall = 0.80 * self.last_hr_bigsmall + 0.20 * raw_bigsmall
            else:
                self.last_hr_bigsmall = raw_bigsmall
            result.hr_bigsmall = self.last_hr_bigsmall
        result.snr_bigsmall = big_snr
        result.hr = result.hr_bigsmall
        result.hr_source = "BigSmall"
        result.snr = big_snr
        if math.isfinite(result.hr):
            self.last_hr = result.hr
        if len(self.last_resp) >= int(resp_fps * 15):
            result.rr, _ = self._rate_fft(self.last_resp, 0.13, 0.5, resp_fps)

        if len(self.last_bvp) >= int(bvp_fps * 30):
            upsampled = signal.resample_poly(self.last_bvp, 4, 1)
            upsampled_fps = bvp_fps * 4
            prominence = max(float(np.std(upsampled)) * 0.20, 1e-6)
            peaks, _ = signal.find_peaks(
                upsampled, distance=int(upsampled_fps * 0.30), prominence=prominence
            )
            intervals = np.diff(peaks) / upsampled_fps * 1000.0
            intervals = intervals[(intervals >= 300.0) & (intervals <= 2000.0)]
            if len(intervals) >= 15:
                median = float(np.median(intervals))
                intervals = intervals[(intervals >= 0.70 * median) & (intervals <= 1.30 * median)]
            result.beats = len(intervals) + 1 if len(intervals) else 0
            if len(intervals) >= 15:
                result.rmssd = float(np.sqrt(np.mean(np.diff(intervals) ** 2)))
                result.sdnn = float(np.std(intervals, ddof=1))

        if not face_ok:
            result.quality = "NO FACE"
        elif math.isfinite(motion) and motion > 0.035:
            result.quality = "MOVING"
        elif brightness < 45:
            result.quality = "TOO DARK"
        elif brightness > 220:
            result.quality = "TOO BRIGHT"
        elif len(self.bvp_diff) < int(self.fps * 8):
            result.quality = "WARMING UP"
        elif not math.isfinite(result.snr) or result.snr < -10:
            result.quality = "LOW SIGNAL"
        else:
            result.quality = "GOOD" if result.snr >= -3 else "FAIR"
        return result


class AffectiveStateEstimator:
    """Experimental subject-relative affect estimator, not an emotion classifier.

    BigSmall supplies AU, pulse, and respiration estimates.  A short neutral-ish
    baseline converts their absolute, dataset-dependent outputs into changes for
    the current subject.  Facial evidence estimates valence; physiology mainly
    contributes to arousal because it cannot reliably identify emotion valence.
    """

    def __init__(self, baseline_seconds=20.0):
        self.baseline_seconds = max(float(baseline_seconds), 5.0)
        self.face_baseline_seconds = min(5.0, self.baseline_seconds)
        self.first_timestamp = None
        self.baseline_au = []
        self.baseline_values = {"hr": [], "rr": [], "rmssd": []}
        self.au_center = np.zeros(len(AU_NAMES), dtype=np.float64)
        self.au_scale = np.full(len(AU_NAMES), 0.08, dtype=np.float64)
        self.phys_center = {}
        self.phys_scale = {}
        self.finalized = False
        self.rmssd_bootstrap = []
        self.hr_history = deque(maxlen=600)
        self.smoothed_evidence = None
        self.smoothed_valence = 0.0
        self.smoothed_arousal = 0.25

    @staticmethod
    def _robust_center_scale(values, floor):
        values = np.asarray(values, dtype=np.float64)
        values = values[np.isfinite(values)]
        if not len(values):
            return math.nan, float(floor)
        center = float(np.median(values))
        mad = float(np.median(np.abs(values - center)))
        return center, max(1.4826 * mad, float(floor))

    def _collect_baseline(self, au, metrics, collect_au=True):
        if collect_au:
            self.baseline_au.append(np.asarray(au, dtype=np.float64))
            if len(self.baseline_au) >= 5:
                values = np.stack(self.baseline_au)
                self.au_center = np.median(values, axis=0)
                mad = np.median(np.abs(values - self.au_center), axis=0)
                self.au_scale = np.maximum(1.4826 * mad, 0.06)
        for name in self.baseline_values:
            value = getattr(metrics, name)
            if math.isfinite(value):
                self.baseline_values[name].append(float(value))

    def _finalize_baseline(self):
        if self.baseline_au:
            values = np.stack(self.baseline_au)
            self.au_center = np.median(values, axis=0)
            mad = np.median(np.abs(values - self.au_center), axis=0)
            self.au_scale = np.maximum(1.4826 * mad, 0.06)
        floors = {"hr": 5.0, "rr": 2.0, "rmssd": 15.0}
        for name, floor in floors.items():
            center, scale = self._robust_center_scale(self.baseline_values[name], floor)
            self.phys_center[name] = center
            self.phys_scale[name] = scale
        self.finalized = True

    def _au_evidence(self, au):
        au = np.asarray(au, dtype=np.float64)
        z = np.clip((au - self.au_center) / self.au_scale, -3.0, 3.0)
        relative = np.clip(z / 2.5, 0.0, 1.0)
        absolute = np.clip((au - 0.15) / 0.55, 0.0, 1.0)
        activation = np.clip(0.72 * relative + 0.28 * absolute, 0.0, 1.0)
        score = dict(zip(AU_NAMES, activation))

        def soft_and(*names):
            values = np.clip([score[name] for name in names], 1e-5, 1.0)
            return float(np.prod(values) ** (1.0 / len(values)))

        evidence = {
            "smile": soft_and("AU06", "AU12"),
            "brow_raise": soft_and("AU01", "AU02"),
            "sadness_like": soft_and("AU01", "AU04", "AU15"),
            "disgust_like": soft_and("AU10", "AU17"),
            "lip_tightening": soft_and("AU23", "AU24"),
        }
        evidence["tension"] = float(np.clip(
            0.34 * score["AU04"]
            + 0.26 * score["AU07"]
            + 0.40 * evidence["lip_tightening"],
            0.0, 1.0,
        ))
        if self.smoothed_evidence is None:
            self.smoothed_evidence = evidence
        else:
            self.smoothed_evidence = {
                name: 0.78 * self.smoothed_evidence[name] + 0.22 * value
                for name, value in evidence.items()
            }
        return self.smoothed_evidence

    def _standardized_change(self, name, value):
        center = self.phys_center.get(name, math.nan)
        scale = self.phys_scale.get(name, 1.0)
        if not math.isfinite(value) or not math.isfinite(center):
            return 0.0, False
        return float(np.clip((value - center) / scale, -3.0, 3.0)), True

    def _physiology_arousal(self, timestamp, metrics):
        if math.isfinite(metrics.hr):
            self.hr_history.append((timestamp, float(metrics.hr)))

        if self.finalized and not math.isfinite(self.phys_center.get("rmssd", math.nan)):
            if math.isfinite(metrics.rmssd):
                self.rmssd_bootstrap.append(float(metrics.rmssd))
                if len(self.rmssd_bootstrap) >= 5:
                    center, scale = self._robust_center_scale(self.rmssd_bootstrap, 15.0)
                    self.phys_center["rmssd"], self.phys_scale["rmssd"] = center, scale

        hr_z, has_hr = self._standardized_change("hr", metrics.hr)
        rr_z, has_rr = self._standardized_change("rr", metrics.rr)
        rmssd_z, has_rmssd = self._standardized_change("rmssd", metrics.rmssd)

        hr_trend = 0.0
        if has_hr and len(self.hr_history) > 1:
            target = timestamp - 10.0
            older = min(self.hr_history, key=lambda item: abs(item[0] - target))
            if timestamp - older[0] >= 4.0:
                hr_trend = float(np.clip((metrics.hr - older[1]) / 8.0, -2.0, 2.0))

        terms = []
        if has_hr:
            terms.append((0.48, hr_z))
            terms.append((0.16, hr_trend))
        if has_rr:
            terms.append((0.20, rr_z))
        if has_rmssd:
            terms.append((0.16, -rmssd_z))
        raw = sum(weight * value for weight, value in terms) / max(
            sum(weight for weight, _ in terms), 1e-6
        )
        arousal = 1.0 / (1.0 + math.exp(-1.15 * raw)) if terms else 0.5
        quality_weight = {
            "GOOD": 1.0,
            "FAIR": 0.70,
            "LOW SIGNAL": 0.25,
            "MOVING": 0.0,
            "TRACKING WEAK": 0.0,
            "TOO DARK": 0.10,
            "TOO BRIGHT": 0.10,
        }.get(metrics.quality, 0.0)
        return float(arousal), quality_weight

    @staticmethod
    def _state_label(valence, arousal, evidence):
        if evidence["smile"] >= 0.43 and valence >= 0.12:
            return "POSITIVE / EXCITED" if arousal >= 0.58 else "POSITIVE / PLEASANT"
        if evidence["tension"] >= 0.43 and arousal >= 0.56:
            return "TENSE / HIGH AROUSAL"
        if evidence["sadness_like"] >= 0.42 and valence <= -0.15 and arousal < 0.58:
            return "NEGATIVE / LOW AROUSAL"
        if evidence["brow_raise"] >= 0.48:
            return "SURPRISE / ATTENTION CUE"
        if arousal >= 0.66:
            return "HIGH AROUSAL"
        if abs(valence) <= 0.16 and arousal <= 0.52:
            return "NEUTRAL / CALM"
        return "MIXED / UNCERTAIN"

    def update(self, timestamp, au, metrics):
        timestamp = float(timestamp)
        if self.first_timestamp is None:
            self.first_timestamp = timestamp
        elapsed = max(timestamp - self.first_timestamp, 0.0)
        progress = min(elapsed / self.baseline_seconds, 1.0)

        face_calibrated = elapsed >= self.face_baseline_seconds and len(self.baseline_au) >= 20
        if not self.finalized:
            self._collect_baseline(
                au, metrics,
                collect_au=elapsed <= self.face_baseline_seconds or len(self.baseline_au) < 20,
            )
            if progress >= 1.0 and len(self.baseline_au) >= 20:
                self._finalize_baseline()

        evidence = self._au_evidence(au)
        face_valence = float(np.clip(
            1.18 * evidence["smile"]
            - 0.58 * evidence["tension"]
            - 0.46 * evidence["sadness_like"]
            - 0.28 * evidence["disgust_like"],
            -1.0, 1.0,
        ))
        face_arousal = float(np.clip(
            0.20 + 0.68 * max(
                evidence["smile"], evidence["tension"],
                evidence["brow_raise"], evidence["disgust_like"],
            ),
            0.0, 1.0,
        ))
        physiology_arousal, physiology_quality = self._physiology_arousal(timestamp, metrics)
        combined_arousal = (
            0.72 * face_arousal + physiology_quality * physiology_arousal
        ) / (0.72 + physiology_quality)

        self.smoothed_valence = 0.82 * self.smoothed_valence + 0.18 * face_valence
        self.smoothed_arousal = 0.82 * self.smoothed_arousal + 0.18 * combined_arousal
        state = self._state_label(self.smoothed_valence, self.smoothed_arousal, evidence)
        evidence_strength = max(
            max(evidence.values()), abs(self.smoothed_valence),
            abs(self.smoothed_arousal - 0.5) * 2.0,
        )
        confidence = progress * (0.55 + 0.45 * physiology_quality) * (
            0.42 + 0.58 * evidence_strength
        )
        if not self.finalized:
            if face_calibrated and evidence["smile"] >= 0.43:
                state = "SMILE CUE / CALIBRATING"
            else:
                state = "CALIBRATING"
            confidence *= 0.35

        return AffectiveState(
            state=state,
            valence=float(self.smoothed_valence),
            arousal=float(np.clip(self.smoothed_arousal, 0.0, 1.0)),
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            calibration_progress=progress,
            face_calibrated=face_calibrated or self.finalized,
            evidence={name: float(value) for name, value in evidence.items()},
            physiology_arousal=physiology_arousal,
        )


def format_value(value, suffix="", digits=1):
    return "--" if not math.isfinite(value) else f"{value:.{digits}f}{suffix}"


def draw_waveform(canvas, waveform, rect, color, label):
    x, y, w, h = rect
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (55, 55, 55), 1)
    cv2.putText(canvas, label, (x + 8, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    if waveform is None or len(waveform) < 2:
        return
    values = np.asarray(waveform[-min(len(waveform), 400):], dtype=np.float64)
    values = (values - values.mean()) / max(values.std(), 1e-6)
    xs = np.linspace(x + 4, x + w - 4, len(values))
    ys = y + h / 2 - values * (h * 0.20)
    points = np.column_stack((xs, np.clip(ys, y + 4, y + h - 4))).astype(np.int32)
    cv2.polylines(canvas, [points], False, color, 1, cv2.LINE_AA)


def draw_bar(canvas, label, value, y, color, width=180):
    value = float(np.clip(value, 0.0, 1.0))
    cv2.putText(canvas, label, (18, y + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (205, 205, 205), 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (155, y), (155 + width, y + 14), (58, 58, 58), -1)
    cv2.rectangle(canvas, (155, y), (155 + int(width * value), y + 14), color, -1)
    cv2.putText(canvas, f"{value:.2f}", (345, y + 13), cv2.FONT_HERSHEY_SIMPLEX,
                0.38, (205, 205, 205), 1, cv2.LINE_AA)


def render(frame, box, face_status, metrics, analyzer, au, affect, gt_hr, elapsed,
           source_name):
    display_h, display_w = 600, 800
    video = cv2.resize(frame, (display_w, display_h))
    if box is not None:
        sx, sy = display_w / frame.shape[1], display_h / frame.shape[0]
        x, y, w, h = box
        box_color = (0, 220, 0) if face_status == "TRACKING" else (0, 190, 255)
        cv2.rectangle(
            video, (int(x * sx), int(y * sy)), (int((x + w) * sx), int((y + h) * sy)),
            box_color, 2,
        )
        cv2.putText(video, face_status, (int(x * sx), max(18, int(y * sy) - 7)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_color, 1, cv2.LINE_AA)
    if source_name == "camera" and affect.calibration_progress < 1.0:
        hint = (
            "Keep a neutral face for 5 s"
            if not affect.face_calibrated
            else "Face baseline ready - you may smile"
        )
        cv2.rectangle(video, (12, 12), (425, 48), (10, 10, 10), -1)
        cv2.putText(video, hint, (24, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                    (0, 220, 255), 1, cv2.LINE_AA)

    panel = np.full((display_h, 460, 3), 24, dtype=np.uint8)
    quality_color = (0, 210, 0) if metrics.quality == "GOOD" else (0, 190, 255)
    lines = [
        ("BigSmall Live Demo", (255, 255, 255), 0.75),
        (f"Source: {source_name}", (180, 180, 180), 0.48),
        (f"Time: {elapsed:5.1f} s", (180, 180, 180), 0.48),
        (f"Quality: {metrics.quality}", quality_color, 0.62),
        (f"HR: {format_value(metrics.hr, ' bpm')} [{metrics.hr_source}]", (80, 210, 255), 0.70),
        (("Face tracker: MediaPipe" if source_name == "camera"
          else f"GT HR: {format_value(gt_hr, ' bpm')}"),
         (150, 220, 150), 0.50),
        (f"Resp: {format_value(metrics.rr, ' rpm')}", (255, 180, 80), 0.62),
        (f"RMSSD: {format_value(metrics.rmssd, ' ms')}", (220, 170, 255), 0.58),
        (f"SDNN: {format_value(metrics.sdnn, ' ms')}", (220, 170, 255), 0.58),
        (f"SNR: {format_value(metrics.snr, ' dB')}  Signal FPS: "
         f"{format_value(metrics.sample_rate, '', 1)}", (180, 180, 180), 0.43),
    ]
    y = 32
    for text, color, scale in lines:
        cv2.putText(panel, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
        y += 31 if scale >= 0.6 else 25

    y += 2
    cv2.line(panel, (16, y), (438, y), (65, 65, 65), 1)
    y += 27
    cv2.putText(panel, "Experimental affect estimate", (16, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1, cv2.LINE_AA)
    y += 28
    state_color = (90, 220, 255) if affect.state != "CALIBRATING" else (0, 190, 255)
    cv2.putText(panel, f"State: {affect.state}", (16, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.53, state_color, 1, cv2.LINE_AA)
    y += 25
    cv2.putText(panel, f"Valence: {affect.valence:+.2f}   Arousal: {affect.arousal:.2f}",
                (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (205, 205, 205), 1, cv2.LINE_AA)
    y += 23
    cv2.putText(panel, f"Confidence: {affect.confidence:.2f}", (16, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)
    if affect.calibration_progress < 1.0:
        cv2.putText(panel, f"Phys baseline: {affect.calibration_progress * 100:3.0f}%", (215, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.43, (180, 180, 180), 1, cv2.LINE_AA)
    y += 28
    cv2.putText(panel, "Independent AU evidence", (16, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.49, (205, 205, 205), 1, cv2.LINE_AA)
    au_scores = dict(zip(AU_NAMES, au))
    cv2.putText(panel, f"AU06/12 {au_scores['AU06']:.2f}/{au_scores['AU12']:.2f}",
                (278, y), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (150, 150, 150), 1, cv2.LINE_AA)
    y += 10
    evidence_rows = [
        ("Smile (AU06+12)", "smile", (90, 220, 120)),
        ("Tension", "tension", (90, 150, 255)),
        ("Brow raise", "brow_raise", (255, 190, 90)),
        ("Sadness-like", "sadness_like", (190, 130, 230)),
        ("Disgust-like", "disgust_like", (130, 190, 130)),
        ("Lip tighten 23+24", "lip_tightening", (180, 120, 220)),
    ]
    for label, name, color in evidence_rows:
        draw_bar(panel, label, affect.evidence.get(name, 0.0), y, color)
        y += 22

    canvas = np.hstack((video, panel))
    draw_waveform(canvas, analyzer.last_bvp, (20, 455, 360, 120), (80, 210, 255), "Recovered BVP")
    draw_waveform(canvas, analyzer.last_resp, (410, 455, 360, 120), (255, 180, 80), "Respiration")
    cv2.putText(canvas, "Affect tendency only; experimental PRV; not a medical device", (810, 582),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 100, 255), 1, cv2.LINE_AA)
    return canvas


def build_source(args):
    if args.source == "camera":
        return CaptureSource(
            args.camera_id, camera=True, camera_fps=args.camera_fps,
            camera_backend=args.camera_backend,
        )
    input_path = Path(args.input).resolve()
    if args.source == "clips" or input_path.is_dir():
        return ClipDirectorySource(input_path)
    return CaptureSource(str(input_path), camera=False)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "time_s", "hr_bpm", "hr_bigsmall_bpm", "hr_source",
        "gt_hr_bpm", "resp_rpm", "rmssd_ms", "sdnn_ms",
        "snr_db", "snr_bigsmall_db", "signal_fps", "beats",
        "quality", "face_status", "face_motion", "affective_state", "valence", "arousal",
        "affect_confidence", "calibration_progress", "face_calibrated",
        "physiology_arousal",
        "evidence_smile", "evidence_tension", "evidence_brow_raise",
        "evidence_sadness_like", "evidence_disgust_like", "evidence_lip_tightening",
        *AU_NAMES,
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["clips", "video", "camera"], default="clips")
    parser.add_argument("--input", default=str(DEFAULT_CLIPS))
    parser.add_argument("--ground-truth", default=str(DEFAULT_GT))
    parser.add_argument("--checkpoint", default=str(DEFAULT_MODEL))
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--camera-fps", type=float, default=30.0)
    parser.add_argument("--camera-backend", choices=["auto", "msmf", "dshow", "any"],
                        default="auto")
    parser.add_argument("--face-model", default=str(DEFAULT_FACE_LANDMARKER))
    parser.add_argument("--face-loss-grace", type=float, default=1.0)
    parser.add_argument("--baseline-seconds", type=float, default=20.0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--no-realtime", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--output-dir", default=str(ROOT / "runs" / "bigsmall_demo"))
    args = parser.parse_args()

    source = build_source(args)
    gt_path = None if args.source == "camera" else (Path(args.ground_truth) if args.ground_truth else None)
    ground_truth = GroundTruth(gt_path)
    tracker = FaceTracker(
        source.pre_cropped, model_path=Path(args.face_model),
        grace_seconds=max(args.face_loss_grace, 0.0),
    )
    preprocessor = OnlineBigSmallPreprocessor()
    predictor = BigSmallPredictor(Path(args.checkpoint))
    analyzer = PhysiologyAnalyzer(fps=25.0)
    affect_estimator = AffectiveStateEstimator(args.baseline_seconds)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    writer = None
    start_wall = time.perf_counter()
    processed_frames = 0
    last_metrics = Metrics()
    last_au = np.zeros(len(AU_NAMES), dtype=np.float32)
    last_affect = AffectiveState()
    last_canvas = None
    next_model_timestamp = 0.0
    last_face_status = "NO FACE"
    last_face_motion = math.nan

    print(f"Source FPS: {source.fps:.3f}")
    if isinstance(source, CaptureSource) and source.camera:
        print(f"Camera backend: {source.backend_name}")
    print(f"Inference device: {predictor.device}")
    print("Press Q or ESC to exit.")

    try:
        while True:
            item = source.read()
            if item is None:
                break
            frame, timestamp, frame_index = item
            frame_brightness = float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean())
            observation = tracker.process(frame, timestamp)
            crop, box = observation.crop, observation.box
            face_ok = observation.status == "TRACKING"
            last_face_status = observation.status
            last_face_motion = observation.motion
            brightness = float(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).mean()) if crop is not None else 0.0

            if observation.status == "TRACKING WEAK":
                last_metrics.quality = "TRACKING WEAK"
                preprocessor.reset_pending()
            elif crop is None:
                preprocessor.reset_pending()
                last_metrics.quality = (
                    "CAMERA BLACK"
                    if isinstance(source, CaptureSource) and source.camera and frame_brightness < 3.0
                    else "NO FACE"
                )
                if last_metrics.quality == "CAMERA BLACK" and frame_index == 30:
                    print("Camera is returning black frames. Check the privacy shutter and "
                          "Windows camera permission, or try --camera-backend dshow.")

            if face_ok and crop is not None and timestamp + 1e-6 >= next_model_timestamp:
                while next_model_timestamp <= timestamp:
                    next_model_timestamp += 1.0 / 25.0
                prepared = preprocessor.push(crop, timestamp)
                if prepared is not None:
                    big, small, sample_timestamps = prepared
                    au, bvp, respiration = predictor.predict((big, small))
                    analyzer.add(
                        au, bvp, respiration, sample_timestamps,
                        physiology_ok=True,
                    )
                    last_metrics = analyzer.compute(
                        face_ok=True, brightness=brightness, motion=observation.motion,
                    )
                    last_au = analyzer.smooth_au(seconds=1.0)
                    last_affect = affect_estimator.update(timestamp, last_au, last_metrics)
                    _, gt_hr = ground_truth.at(timestamp)
                    row = {
                        "time_s": round(timestamp, 3),
                        "hr_bpm": last_metrics.hr,
                        "hr_bigsmall_bpm": last_metrics.hr_bigsmall,
                        "hr_source": last_metrics.hr_source,
                        "gt_hr_bpm": gt_hr,
                        "resp_rpm": last_metrics.rr,
                        "rmssd_ms": last_metrics.rmssd,
                        "sdnn_ms": last_metrics.sdnn,
                        "snr_db": last_metrics.snr,
                        "snr_bigsmall_db": last_metrics.snr_bigsmall,
                        "signal_fps": last_metrics.sample_rate,
                        "beats": last_metrics.beats,
                        "quality": last_metrics.quality,
                        "face_status": observation.status,
                        "face_motion": observation.motion,
                        "affective_state": last_affect.state,
                        "valence": last_affect.valence,
                        "arousal": last_affect.arousal,
                        "affect_confidence": last_affect.confidence,
                        "calibration_progress": last_affect.calibration_progress,
                        "face_calibrated": last_affect.face_calibrated,
                        "physiology_arousal": last_affect.physiology_arousal,
                    }
                    row.update({
                        f"evidence_{name}": value
                        for name, value in last_affect.evidence.items()
                    })
                    row.update({name: float(value) for name, value in zip(AU_NAMES, last_au)})
                    rows.append(row)

            _, gt_hr = ground_truth.at(timestamp)
            elapsed = timestamp if not isinstance(source, CaptureSource) or not source.camera else time.perf_counter() - start_wall
            last_canvas = render(
                frame, box, observation.status, last_metrics, analyzer, last_au, last_affect,
                gt_hr, elapsed, args.source
            )

            if args.record:
                if writer is None:
                    output_path = output_dir / "preview.mp4"
                    writer = cv2.VideoWriter(
                        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"),
                        min(max(source.fps, 10.0), 30.0),
                        (last_canvas.shape[1], last_canvas.shape[0]),
                    )
                writer.write(last_canvas)

            if not args.headless:
                cv2.imshow("BigSmall Live Demo", last_canvas)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    break

            processed_frames += 1
            if args.max_frames and processed_frames >= args.max_frames:
                break
            if not args.no_realtime and not (isinstance(source, CaptureSource) and source.camera):
                target = start_wall + processed_frames / source.fps
                delay = target - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
    finally:
        tracker.close()
        source.close()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        write_csv(output_dir / "session.csv", rows)
        if last_canvas is not None:
            ok, encoded = cv2.imencode(".jpg", last_canvas)
            if ok:
                (output_dir / "last_frame.jpg").write_bytes(encoded.tobytes())

    print(f"Processed frames: {processed_frames}")
    print(f"Prediction rows: {len(rows)}")
    print(f"Final quality: {last_metrics.quality}")
    print(f"Final HR: {format_value(last_metrics.hr, ' bpm')}")
    print(f"Final affect state: {last_affect.state}")
    print(f"Final valence/arousal: {last_affect.valence:+.2f} / {last_affect.arousal:.2f}")
    print(f"Results: {output_dir}")


if __name__ == "__main__":
    main()
