"""Convert Isaac Lab HDF5 datasets to the GR00T LeRobot schema.

This script creates the directory layout and metadata required by GR00T's
flavor of the LeRobot v2.0 dataset specification. It reads Isaac Lab HDF5
recordings, converts state/action arrays and camera streams, and writes
Parquet episodes alongside the supporting ``meta`` files and MP4 videos.

The expected output directory tree matches the schema described in the
``Robot Data Conversion Guide``:

```
└── <output>
    ├── data
    │   └── chunk-000
    │       └── episode_000000.parquet
    ├── videos
    │   └── chunk-000
    │       └── observation.images.front
    │           └── episode_000000.mp4
    └── meta
        ├── episodes.jsonl
        ├── info.json
        ├── modality.json
        └── tasks.jsonl
```

Notes
-----
* Run this script inside the LeRobot environment so that ``pyarrow`` and
  video encoders (``imageio`` or ``opencv``) are available.
* Only the fields required by GR00T are emitted. Extend the metadata section
  if your dataset exposes additional modalities.
* The script is intentionally conservative with destructive actions. If the
  output directory already exists you must pass ``--overwrite``.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    import h5py  # type: ignore
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise ImportError(
        "h5py is required to decode Isaac Lab datasets. Please install it in the "
        "active environment."
    ) from exc

try:
    import pyarrow as pa  # type: ignore
    import pyarrow.parquet as pq  # type: ignore
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise ImportError(
        "pyarrow is required to write Parquet files compatible with LeRobot."
    ) from exc

try:  # pragma: no cover - optional dependency
    import imageio.v3 as imageio_v3  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    imageio_v3 = None

try:  # pragma: no cover - optional dependency
    import cv2  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    cv2 = None

from tqdm import tqdm


# IsaacLab → LeRobot joint limit mapping (degrees).
ISAACLAB_JOINT_POS_LIMIT_RANGE = [
    (-110.0, 110.0),
    (-100.0, 100.0),
    (-100.0, 90.0),
    (-95.0, 95.0),
    (-160.0, 160.0),
    (-10.0, 100.0),
]
LEROBOT_JOINT_POS_LIMIT_RANGE = [
    (-100.0, 100.0),
    (-100.0, 100.0),
    (-100.0, 100.0),
    (-100.0, 100.0),
    (-100.0, 100.0),
    (0.0, 100.0),
]


@dataclass(frozen=True)
class RobotModalityConfig:
    """Holds per-robot modality description."""

    state_segments: List[Tuple[str, int, int]]
    action_segments: List[Tuple[str, int, int]]
    video_keys: Dict[str, str]


SINGLE_ARM_MODALITY = RobotModalityConfig(
    state_segments=[
        ("shoulder_pan", 0, 1),
        ("shoulder_lift", 1, 2),
        ("elbow_flex", 2, 3),
        ("wrist_flex", 3, 4),
        ("wrist_roll", 4, 5),
        ("gripper", 5, 6),
    ],
    action_segments=[
        ("shoulder_pan", 0, 1),
        ("shoulder_lift", 1, 2),
        ("elbow_flex", 2, 3),
        ("wrist_flex", 3, 4),
        ("wrist_roll", 4, 5),
        ("gripper", 5, 6),
    ],
    video_keys={
        "front": "observation.images.front",
        "wrist": "observation.images.wrist",
    },
)


BI_ARM_MODALITY = RobotModalityConfig(
    state_segments=[
        ("left_shoulder_pan", 0, 1),
        ("left_shoulder_lift", 1, 2),
        ("left_elbow_flex", 2, 3),
        ("left_wrist_flex", 3, 4),
        ("left_wrist_roll", 4, 5),
        ("left_gripper", 5, 6),
        ("right_shoulder_pan", 6, 7),
        ("right_shoulder_lift", 7, 8),
        ("right_elbow_flex", 8, 9),
        ("right_wrist_flex", 9, 10),
        ("right_wrist_roll", 10, 11),
        ("right_gripper", 11, 12),
    ],
    action_segments=[
        ("left_shoulder_pan", 0, 1),
        ("left_shoulder_lift", 1, 2),
        ("left_elbow_flex", 2, 3),
        ("left_wrist_flex", 3, 4),
        ("left_wrist_roll", 4, 5),
        ("left_gripper", 5, 6),
        ("right_shoulder_pan", 6, 7),
        ("right_shoulder_lift", 7, 8),
        ("right_elbow_flex", 8, 9),
        ("right_wrist_flex", 9, 10),
        ("right_wrist_roll", 10, 11),
        ("right_gripper", 11, 12),
    ],
    video_keys={
        "left": "observation.images.left",
        "top": "observation.images.top",
        "right": "observation.images.right",
    },
)


def preprocess_joint_positions(joint_values: np.ndarray) -> np.ndarray:
    """Convert Isaac Lab joint angles to LeRobot's normalized degree ranges."""

    if joint_values.ndim != 2:
        raise ValueError("Joint arrays must be two-dimensional (T, D).")

    processed = joint_values.astype(np.float32, copy=True)
    processed = processed / math.pi * 180.0

    isaac_limits = np.asarray(ISAACLAB_JOINT_POS_LIMIT_RANGE, dtype=np.float32)
    lerobot_limits = np.asarray(LEROBOT_JOINT_POS_LIMIT_RANGE, dtype=np.float32)
    base_dim = isaac_limits.shape[0]
    if processed.shape[1] % base_dim != 0:
        raise ValueError(
            f"Expected joint dimension to be a multiple of {base_dim}, "
            f"got {processed.shape[1]}."
        )

    for joint_index in range(processed.shape[1]):
        src_min, src_max = isaac_limits[joint_index % base_dim]
        dst_min, dst_max = lerobot_limits[joint_index % base_dim]
        processed[:, joint_index] = (processed[:, joint_index] - src_min) / (src_max - src_min)
        processed[:, joint_index] = processed[:, joint_index] * (dst_max - dst_min) + dst_min

    return processed


def _decode_task_attribute(demo_group: h5py.Group) -> str | None:
    """Attempt to read task metadata stored as HDF5 attributes."""

    for key in ("language_instruction", "instruction", "task", "task_description"):
        if key in demo_group.attrs:
            value = demo_group.attrs[key]
            if isinstance(value, bytes):
                return value.decode("utf-8")
            return str(value)
    return None


def _ensure_uint8_video(frames: np.ndarray) -> np.ndarray:
    """Cast image array to uint8 RGB for MP4 writers."""

    frames = np.asarray(frames)
    if frames.ndim != 4:
        raise ValueError("Video arrays must have shape (T, H, W, C).")
    if frames.dtype == np.uint8:
        return frames
    return np.clip(frames, 0, 255).astype(np.uint8)


def _write_video(path: Path, frames: np.ndarray, fps: float, crf: int) -> None:
    """Persist frames to an MP4 file."""

    frames = _ensure_uint8_video(frames)
    path.parent.mkdir(parents=True, exist_ok=True)

    if imageio_v3 is not None:  # pragma: no branch - prefer imageio when available
        ffmpeg_params = [
            "-crf",
            str(crf),
        ]
        imageio_v3.imwrite(
            path,
            frames,
            fps=fps,
            codec="libx264",
            ffmpeg_params=ffmpeg_params,
        )
        return

    if cv2 is not None:
        height, width = frames.shape[1:3]
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():  # pragma: no cover - runtime guard
            raise RuntimeError(f"Failed to open video writer for {path}.")
        for frame in frames:
            if frame.shape[-1] == 3:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            else:
                writer.write(frame)
        writer.release()
        return

    raise ImportError(
        "No suitable video writer found. Install imageio[ffmpeg] or opencv-python."
    )


def _write_parquet(path: Path, columns: Dict[str, pa.Array]) -> None:
    """Write a Parquet file from pyarrow arrays."""

    table = pa.Table.from_pydict(columns)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd")


def _modality_to_dict(modality: RobotModalityConfig) -> Dict[str, Dict[str, Dict[str, object]]]:
    """Translate robot modality configuration to modality.json structure."""

    base_range = LEROBOT_JOINT_POS_LIMIT_RANGE
    state_config: Dict[str, Dict[str, object]] = {}
    action_config: Dict[str, Dict[str, object]] = {}

    for name, start, end in modality.state_segments:
        entry: Dict[str, object] = {"start": start, "end": end, "dtype": "float32"}
        if end - start == 1:
            entry["range"] = base_range[start % len(base_range)]
        state_config[name] = entry

    for name, start, end in modality.action_segments:
        entry = {"start": start, "end": end, "dtype": "float32"}
        if end - start == 1:
            entry["range"] = base_range[start % len(base_range)]
        action_config[name] = entry

    video_config = {
        alias: {"original_key": original_key}
        for alias, original_key in modality.video_keys.items()
    }

    annotation_config = {
        "human.action.task_description": {},
        "human.validity": {},
    }

    return {
        "state": state_config,
        "action": action_config,
        "video": video_config,
        "annotation": annotation_config,
    }


def _select_modality(robot_type: str) -> RobotModalityConfig:
    if robot_type == "so101_follower":
        return SINGLE_ARM_MODALITY
    if robot_type == "bi_so101_follower":
        return BI_ARM_MODALITY
    raise ValueError(f"Unsupported robot type: {robot_type}")


def _extract_episode(
    robot_type: str,
    demo_group: h5py.Group,
    skip_frames: int,
    preprocess: bool,
) -> tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Return action/state arrays and camera frames for one episode."""

    try:
        actions = np.asarray(demo_group["actions"], dtype=np.float32)
    except KeyError as exc:
        raise KeyError(f"Demo {demo_group.name} is missing 'actions'.") from exc

    video_tensors: Dict[str, np.ndarray] = {}

    if robot_type == "so101_follower":
        joint_pos = np.asarray(demo_group["obs/joint_pos"], dtype=np.float32)
        video_tensors["front"] = np.asarray(demo_group["obs/front"])
        video_tensors["wrist"] = np.asarray(demo_group["obs/wrist"])
        state = joint_pos
    elif robot_type == "bi_so101_follower":
        left = np.asarray(demo_group["obs/left_joint_pos"], dtype=np.float32)
        right = np.asarray(demo_group["obs/right_joint_pos"], dtype=np.float32)
        state = np.concatenate([left, right], axis=1)
        video_tensors["left"] = np.asarray(demo_group["obs/left"])
        video_tensors["right"] = np.asarray(demo_group["obs/right"])
        video_tensors["top"] = np.asarray(demo_group["obs/top"])
    else:  # pragma: no cover - guarded upstream
        raise ValueError(robot_type)

    if preprocess:
        actions = preprocess_joint_positions(actions)
        state = preprocess_joint_positions(state)

    if actions.shape[0] <= skip_frames:
        return np.empty((0, actions.shape[1]), dtype=np.float32), np.empty((0, state.shape[1]), dtype=np.float32), {}

    frames_slice = slice(skip_frames, None)
    actions = actions[frames_slice]
    state = state[frames_slice]
    for alias, tensor in list(video_tensors.items()):
        video_tensors[alias] = tensor[frames_slice]

    if state.shape[0] != actions.shape[0]:
        raise ValueError(
            f"State/action length mismatch for {demo_group.name}: "
            f"{state.shape[0]} vs {actions.shape[0]}"
        )

    return actions, state, video_tensors


def convert_dataset(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output directory {output_dir} already exists. Use --overwrite to replace it."
            )
        shutil.rmtree(output_dir)

    meta_dir = output_dir / "meta"
    data_dir = output_dir / "data"
    videos_dir = output_dir / "videos"
    meta_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)

    modality = _select_modality(args.robot_type)
    modality_dict = _modality_to_dict(modality)

    state_feature_names: List[str] = []
    action_feature_names: List[str] = []
    for field_name, start, end in modality.state_segments:
        length = end - start
        if length == 1:
            state_feature_names.append(field_name)
        else:
            state_feature_names.extend(f"{field_name}_{i}" for i in range(length))
    for field_name, start, end in modality.action_segments:
        length = end - start
        if length == 1:
            action_feature_names.append(field_name)
        else:
            action_feature_names.extend(f"{field_name}_{i}" for i in range(length))

    episode_index = 0
    global_observation_index = 0
    total_observations = 0
    episodes_meta: List[Dict[str, object]] = []
    task_to_index: Dict[str, int] = {}
    chunks_used: set[int] = set()
    info_features: Dict[str, Dict[str, object]] = {}
    state_samples: List[np.ndarray] = []
    action_samples: List[np.ndarray] = []

    for hdf5_path in args.input_files:
        file_path = Path(hdf5_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Input file {file_path} does not exist.")

        with h5py.File(file_path, "r") as handle:
            if "data" not in handle:
                raise KeyError(f"File {file_path} is missing the 'data' group.")

            demo_names = list(handle["data"].keys())
            iter_bar = tqdm(demo_names, desc=f"Processing {file_path.name}")
            for demo_name in iter_bar:
                demo_group = handle["data"][demo_name]
                if "success" in demo_group.attrs and not bool(demo_group.attrs["success"]):
                    continue

                actions, state, videos = _extract_episode(
                    robot_type=args.robot_type,
                    demo_group=demo_group,
                    skip_frames=args.skip_frames,
                    preprocess=not args.disable_preprocessing,
                )

                if actions.size == 0:
                    continue

                episode_length = actions.shape[0]

                state_f64 = state.astype(np.float64, copy=True)
                action_f64 = actions.astype(np.float64, copy=True)

                state_samples.append(state_f64)
                action_samples.append(action_f64)

                task_text = _decode_task_attribute(demo_group) or args.task
                if task_text is None:
                    raise ValueError(
                        "Unable to determine a task description. Provide --task or store it as an HDF5 attribute."
                    )

                task_index = task_to_index.setdefault(task_text, len(task_to_index))
                success_flag = bool(demo_group.attrs.get("success", True))

                timestamps = np.arange(episode_length, dtype=np.float64) / args.fps
                annotation_task = np.full(episode_length, task_index, dtype=np.int64)

                valid_index = task_to_index.setdefault("valid", len(task_to_index))
                if success_flag:
                    validity_index = valid_index
                else:
                    validity_index = task_to_index.setdefault("invalid", len(task_to_index))
                annotation_validity = np.full(episode_length, validity_index, dtype=np.int64)

                episode_index_column = np.full(episode_length, episode_index, dtype=np.int64)
                global_index_column = np.arange(
                    global_observation_index,
                    global_observation_index + episode_length,
                    dtype=np.int64,
                )
                next_done = np.zeros(episode_length, dtype=bool)
                next_done[-1] = True
                next_reward = np.zeros(episode_length, dtype=np.float64)

                chunk_id = episode_index // args.chunk_size
                chunk_name = f"chunk-{chunk_id:03d}"
                parquet_path = data_dir / chunk_name / f"episode_{episode_index:06d}.parquet"
                chunks_used.add(chunk_id)

                columns = {
                    "observation.state": pa.array(state_f64.tolist(), type=pa.list_(pa.float64())),
                    "action": pa.array(action_f64.tolist(), type=pa.list_(pa.float64())),
                    "timestamp": pa.array(timestamps),
                    "annotation.human.action.task_description": pa.array(annotation_task.astype(np.int64)),
                    "annotation.human.validity": pa.array(annotation_validity.astype(np.int64)),
                    "task_index": pa.array(annotation_task.astype(np.int64)),
                    "episode_index": pa.array(episode_index_column.astype(np.int64)),
                    "index": pa.array(global_index_column.astype(np.int64)),
                    "next.reward": pa.array(next_reward.astype(np.float64)),
                    "next.done": pa.array(next_done),
                }
                _write_parquet(parquet_path, columns)

                if not info_features:
                    info_features = {
                        "observation.state": {
                            "dtype": "float64",
                            "shape": [state_f64.shape[1]],
                            "names": state_feature_names,
                        },
                        "action": {
                            "dtype": "float64",
                            "shape": [action_f64.shape[1]],
                            "names": action_feature_names,
                        },
                        "timestamp": {
                            "dtype": "float64",
                            "shape": [1],
                        },
                        "annotation.human.action.task_description": {
                            "dtype": "int64",
                            "shape": [1],
                        },
                        "annotation.human.validity": {
                            "dtype": "int64",
                            "shape": [1],
                        },
                        "task_index": {
                            "dtype": "int64",
                            "shape": [1],
                        },
                        "episode_index": {
                            "dtype": "int64",
                            "shape": [1],
                        },
                        "index": {
                            "dtype": "int64",
                            "shape": [1],
                        },
                        "next.reward": {
                            "dtype": "float64",
                            "shape": [1],
                        },
                        "next.done": {
                            "dtype": "bool",
                            "shape": [1],
                        },
                    }

                for alias, frames in videos.items():
                    original_key = modality.video_keys[alias]
                    video_chunk_dir = videos_dir / chunk_name / original_key
                    video_path = video_chunk_dir / f"episode_{episode_index:06d}.mp4"
                    _write_video(video_path, frames, args.fps, args.video_crf)

                    if original_key not in info_features:
                        height, width, channels = frames.shape[1:4]
                        info_features[original_key] = {
                            "dtype": "video",
                            "shape": [int(height), int(width), int(channels)],
                            "names": ["height", "width", "channel"],
                            "video_info": {
                                "video.fps": args.fps,
                                "video.codec": "h264",
                                "video.pix_fmt": "yuv420p",
                                "video.is_depth_map": False,
                                "has_audio": False,
                            },
                        }

                episodes_meta.append(
                    {
                        "episode_index": episode_index,
                        "tasks": [task_index],
                        "length": episode_length,
                        "success": success_flag,
                    }
                )

                episode_index += 1
                total_observations += episode_length
                global_observation_index += episode_length

    if episode_index == 0:
        raise RuntimeError("No valid episodes were converted.")

    total_chunks = (max(chunks_used) + 1) if chunks_used else 0

    def _stats_from_samples(samples: List[np.ndarray]) -> Dict[str, List[float]]:
        stacked = np.vstack(samples).astype(np.float64)
        return {
            "mean": stacked.mean(axis=0).tolist(),
            "std": stacked.std(axis=0).tolist(),
            "min": stacked.min(axis=0).tolist(),
            "max": stacked.max(axis=0).tolist(),
            "q01": np.quantile(stacked, 0.01, axis=0).tolist(),
            "q99": np.quantile(stacked, 0.99, axis=0).tolist(),
        }

    stats_payload = {
        "observation.state": _stats_from_samples(state_samples),
        "action": _stats_from_samples(action_samples),
    }
    info = {
        "dataset_name": args.dataset_name or output_dir.name,
        "format": "GR00T LeRobot",
        "codebase_version": "v2.0",
        "robot_type": args.robot_type,
        "total_episodes": episode_index,
        "total_frames": int(total_observations),
        "total_tasks": len(task_to_index),
        "total_videos": len(modality.video_keys),
        "total_chunks": total_chunks,
        "chunks_size": args.chunk_size,
        "fps": args.fps,
        "splits": {"train": f"0:{episode_index}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": info_features,
        "num_observations": int(total_observations),
        "source": "Isaac Lab conversion",
    }

    (meta_dir / "info.json").write_text(json.dumps(info, indent=2))
    (meta_dir / "modality.json").write_text(json.dumps(modality_dict, indent=2))
    (meta_dir / "stats.json").write_text(json.dumps(stats_payload, indent=2))

    with (meta_dir / "episodes.jsonl").open("w", encoding="utf-8") as episode_file:
        for entry in episodes_meta:
            episode_file.write(json.dumps(entry) + "\n")

    with (meta_dir / "tasks.jsonl").open("w", encoding="utf-8") as tasks_file:
        for task_text, index in sorted(task_to_index.items(), key=lambda item: item[1]):
            tasks_file.write(json.dumps({"task_index": index, "task": task_text}) + "\n")

    print(f"Converted {episode_index} episodes containing {total_observations} observations.")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert Isaac Lab HDF5 demos to GR00T LeRobot format.")
    parser.add_argument(
        "--input_files",
        nargs="+",
        required=True,
        help="One or more Isaac Lab HDF5 files to convert.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Destination directory for the GR00T dataset.",
    )
    parser.add_argument(
        "--robot_type",
        default="so101_follower",
        choices=["so101_follower", "bi_so101_follower"],
        help="Robot embodiment recorded in the dataset.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Frame rate used when recording the data (controls timestamps and video fps).",
    )
    parser.add_argument(
        "--skip_frames",
        type=int,
        default=5,
        help="Number of warm-up frames to discard from the beginning of each episode.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=100,
        help="Number of episodes per chunk directory.",
    )
    parser.add_argument(
        "--task",
        default=None,
        help="Fallback task description when the HDF5 demo lacks task metadata.",
    )
    parser.add_argument(
        "--dataset_name",
        default=None,
        help="Custom dataset name recorded in meta/info.json.",
    )
    parser.add_argument(
        "--video_crf",
        type=int,
        default=20,
        help="x264 constant rate factor for MP4 encoding when imageio/ffmpeg is used (lower is higher quality).",
    )
    parser.add_argument(
        "--disable_preprocessing",
        action="store_true",
        help="Skip joint normalization (keeps original Isaac Lab ranges).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove the output directory before writing conversions.",
    )
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    convert_dataset(args)


if __name__ == "__main__":
    main()
