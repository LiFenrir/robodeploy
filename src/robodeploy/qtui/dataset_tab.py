"""数据检查区：数据集浏览 / 逐帧回放 / 完整性检查 / 维护脚本 / v3.0 导出。"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PyQt6.QtCore import QProcess, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSlider,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from robodeploy.scripts.filter_valid_episodes import validate_episode

from .widgets.video_grid import CameraView

DEFAULT_CONVERT_SCRIPT = (
    "/home/kemove/INNOV/projects/lerobot/src/lerobot/scripts/convert_dataset_v21_to_v30.py"
)
DEFAULT_CHUNK_SIZE = 1000


class _FuncWorker(QThread):
    """通用后台执行线程：fn() 结果经 done 信号返回。"""

    done = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, fn, parent=None):
        super().__init__(parent)
        self._fn = fn

    def run(self) -> None:
        try:
            self.done.emit(self._fn())
        except Exception as exc:
            self.failed.emit(str(exc))


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class FfmpegVideoReader:
    """ffmpeg 子进程逐帧解码：rawvideo RGB24 管道输出，-ss 输入侧精确 seek。

    read(idx) 返回 RGB uint8 (H, W, 3)；顺序读零开销，跳读自动重定位。
    """

    MAX_FORWARD = 30  # 超过该跨度的前跳直接重新 seek

    def __init__(self, path: str, fps: float, width: int, height: int):
        self._path = path
        self._fps = fps
        self._width = width
        self._height = height
        self._proc: subprocess.Popen | None = None
        self._next_idx = 0  # 管道中下一帧对应的帧序号

    def _spawn(self, start_idx: int) -> None:
        self.close()
        cmd = ["ffmpeg", "-v", "error"]
        if start_idx > 0:
            cmd += ["-ss", f"{start_idx / self._fps:.4f}"]
        cmd += ["-i", self._path, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
        self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self._next_idx = start_idx

    def read(self, idx: int) -> np.ndarray | None:
        if self._proc is None or idx < self._next_idx or idx > self._next_idx + self.MAX_FORWARD:
            self._spawn(idx)
        frame_size = self._width * self._height * 3
        frame = None
        while self._next_idx <= idx:
            data = self._proc.stdout.read(frame_size)
            if len(data) < frame_size:
                return None
            frame = np.frombuffer(data, np.uint8).reshape(self._height, self._width, 3)
            self._next_idx += 1
        return frame

    def close(self) -> None:
        if self._proc is not None:
            self._proc.kill()
            self._proc.wait()
            self._proc = None


class DatasetTab(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._root: Path | None = None
        self._info: dict = {}
        self._episodes: list[dict] = []
        self._ep_by_index: dict[int, dict] = {}
        self._video_keys: list[str] = []
        self._worker: _FuncWorker | None = None
        self._process = None

        # 回放状态
        self._replay_columns: dict[str, np.ndarray] = {}
        self._replay_row_map: list[tuple[str, int]] = []  # values 表行 → (列名, 维度下标)
        self._replay_videos: dict[str, str] = {}
        self._replay_length = 0
        self._replay_frame_idx = 0
        self._video_reader: FfmpegVideoReader | None = None
        self._replay_timer = QTimer(self)
        self._replay_timer.timeout.connect(self._on_replay_tick)

        self._build_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        top_row = QHBoxLayout()
        self.edit_root = QLineEdit()
        self.edit_root.setPlaceholderText("数据集根目录（含 meta/info.json）")
        btn_browse = QPushButton("浏览…")
        btn_browse.clicked.connect(self._on_browse)
        btn_load = QPushButton("加载")
        btn_load.clicked.connect(self._on_load)
        top_row.addWidget(self.edit_root, 1)
        top_row.addWidget(btn_browse)
        top_row.addWidget(btn_load)
        layout.addLayout(top_row)

        self.lbl_info = QLabel("未加载")
        layout.addWidget(self.lbl_info)

        # 回放区
        replay_box = QGroupBox("逐帧回放（仅可视化，不下发机器人）")
        replay_layout = QVBoxLayout(replay_box)
        replay_controls = QHBoxLayout()
        replay_controls.addWidget(QLabel("剧集"))
        self.spin_episode = QSpinBox()
        self.spin_episode.setEnabled(False)
        replay_controls.addWidget(self.spin_episode)
        self.btn_replay_load = QPushButton("加载剧集")
        self.btn_replay_load.setEnabled(False)
        self.btn_replay_load.clicked.connect(self._on_replay_load)
        self.combo_camera = QComboBox()
        self.combo_camera.currentTextChanged.connect(self._on_camera_changed)
        self.btn_play = QPushButton("▶ 播放")
        self.btn_play.setEnabled(False)
        self.btn_play.clicked.connect(self._on_play_toggle)
        self.btn_prev = QPushButton("|◀")
        self.btn_prev.setEnabled(False)
        self.btn_prev.clicked.connect(lambda: self._seek(self._replay_frame_idx - 1))
        self.btn_next = QPushButton("▶|")
        self.btn_next.setEnabled(False)
        self.btn_next.clicked.connect(lambda: self._seek(self._replay_frame_idx + 1))
        self.combo_speed = QComboBox()
        self.combo_speed.addItems(["0.25", "0.5", "1", "2", "4"])
        self.combo_speed.setCurrentText("1")
        self.combo_speed.currentTextChanged.connect(self._on_speed_changed)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setEnabled(False)
        self.slider.sliderMoved.connect(self._seek)
        self.lbl_frame = QLabel("0 / 0")
        self.lbl_ep_info = QLabel("")
        replay_controls.addWidget(self.btn_replay_load)
        replay_controls.addWidget(self.lbl_ep_info)
        replay_controls.addWidget(QLabel("相机"))
        replay_controls.addWidget(self.combo_camera)
        replay_controls.addWidget(self.btn_play)
        replay_controls.addWidget(self.btn_prev)
        replay_controls.addWidget(self.btn_next)
        replay_controls.addWidget(QLabel("倍速"))
        replay_controls.addWidget(self.combo_speed)
        replay_controls.addWidget(self.slider, 1)
        replay_controls.addWidget(self.lbl_frame)
        replay_layout.addLayout(replay_controls)

        replay_body = QHBoxLayout()
        self.replay_view = CameraView("replay")
        self.values_table = QTableWidget(0, 2)
        self.values_table.setHorizontalHeaderLabels(["字段", "值"])
        self.values_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        replay_body.addWidget(self.replay_view, 2)
        replay_body.addWidget(self.values_table, 1)
        replay_layout.addLayout(replay_body)
        layout.addWidget(replay_box, 1)

        # 维护脚本 + v3.0 导出
        ops_row = QHBoxLayout()
        ops_box = QGroupBox("完整性检查 / 维护脚本")
        ops_layout = QVBoxLayout(ops_box)
        self.btn_validate = QPushButton("批量完整性检查")
        self.btn_validate.clicked.connect(self._on_validate)
        ops_layout.addWidget(self.btn_validate)
        for text, handler in (
            ("删除剧集", self._on_delete_episodes),
            ("替换 task", self._on_replace_task),
            ("重新统计 stats", lambda: self._run_script("regenerate_stats", ["--dataset", self._root_str()])),
            ("gripper 二值化", lambda: self._run_script("binarize_gripper", ["--dataset", self._root_str()])),
            ("过滤导出", self._on_filter_dataset),
            ("合并数据集", self._on_merge_datasets),
        ):
            btn = QPushButton(text)
            btn.clicked.connect(handler)
            ops_layout.addWidget(btn)
        ops_row.addWidget(ops_box)

        export_box = QGroupBox("v3.0 导出")
        export_form = QFormLayout(export_box)
        self.edit_convert_script = QLineEdit(DEFAULT_CONVERT_SCRIPT)
        self.edit_convert_repo = QLineEdit()
        self.edit_convert_repo.setPlaceholderText("repo-id（默认取数据集目录名）")
        self.btn_convert = QPushButton("导出 v3.0")
        self.btn_convert.clicked.connect(self._on_convert_v30)
        export_form.addRow("转换脚本", self.edit_convert_script)
        export_form.addRow("repo-id", self.edit_convert_repo)
        export_form.addRow(self.btn_convert)
        ops_row.addWidget(export_box)
        layout.addLayout(ops_row)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumHeight(160)
        layout.addWidget(self.log_view)

    # ------------------------------------------------------------------
    # 数据集加载
    # ------------------------------------------------------------------

    def _root_str(self) -> str:
        return str(self._root) if self._root else ""

    def _require_root(self) -> bool:
        if self._root is None:
            QMessageBox.warning(self, "提示", "请先加载数据集")
            return False
        return True

    def _on_browse(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择数据集根目录")
        if path:
            self.edit_root.setText(path)

    def _on_load(self) -> None:
        root = Path(self.edit_root.text()).expanduser()
        try:
            self._info = json.loads((root / "meta" / "info.json").read_text())
            self._episodes = _read_jsonl(root / "meta" / "episodes.jsonl")
        except Exception as exc:
            QMessageBox.critical(self, "加载失败", str(exc))
            return

        self._root = root
        features = self._info.get("features", {})
        self._video_keys = [k for k, f in features.items() if f.get("dtype") == "video"]
        self.lbl_info.setText(
            f"codebase={self._info.get('codebase_version', '?')} | "
            f"episodes={self._info.get('total_episodes', len(self._episodes))} | "
            f"fps={self._info.get('fps', '?')} | 相机: {', '.join(self._video_keys)}"
        )

        self._ep_by_index = {ep["episode_index"]: ep for ep in self._episodes}
        indices = sorted(self._ep_by_index)
        self.spin_episode.setEnabled(bool(indices))
        self.btn_replay_load.setEnabled(bool(indices))
        if indices:
            self.spin_episode.setRange(indices[0], indices[-1])
        self.combo_camera.clear()
        self.combo_camera.addItems(self._video_keys)
        self._log(f"已加载 {root}，{len(self._episodes)} 个剧集")

    # ------------------------------------------------------------------
    # 完整性检查
    # ------------------------------------------------------------------

    def _on_validate(self) -> None:
        if not self._require_root() or (self._worker is not None and self._worker.isRunning()):
            return
        root, info, episodes = self._root, self._info, self._episodes
        video_keys = self._video_keys
        features = info.get("features", {})
        expected_columns = [k for k in features if k not in video_keys]
        for k in ("episode_index", "index", "task_index", "timestamp", "frame_index"):
            if k not in expected_columns:
                expected_columns.append(k)
        template = info.get(
            "video_path", "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
        )

        def _run() -> list[tuple[int, bool, str]]:
            results = []
            for ep in episodes:
                ok, reason = validate_episode(root, ep, video_keys, expected_columns, template)
                results.append((ep["episode_index"], ok, reason))
            return results

        self.btn_validate.setEnabled(False)
        self._log("完整性检查中…")
        self._worker = _FuncWorker(_run, self)
        self._worker.done.connect(self._on_validate_done)
        self._worker.failed.connect(lambda msg: self._log(f"检查失败: {msg}"))
        self._worker.start()

    def _on_validate_done(self, results: list) -> None:
        self.btn_validate.setEnabled(True)
        bad = [(ep_idx, reason) for ep_idx, ok, reason in results if not ok]
        for ep_idx, reason in bad:
            self._log(f"无效剧集 #{ep_idx}: {reason}")
        self._log(f"检查完成: {len(results) - len(bad)} 有效 / {len(bad)} 无效")

    # ------------------------------------------------------------------
    # 逐帧回放（parquet + ffmpeg 直读，不走 LeRobotDataset/torch）
    # ------------------------------------------------------------------

    def _on_replay_load(self) -> None:
        ep = self._ep_by_index.get(self.spin_episode.value())
        if ep is None:
            QMessageBox.warning(self, "提示", f"剧集 {self.spin_episode.value()} 不存在")
            return
        self._stop_replay()
        tasks = ep.get("tasks", [""])
        self.lbl_ep_info.setText(
            f"#{ep['episode_index']} | {ep.get('length', 0)}帧 | {tasks[0] if tasks else ''}"
        )

        root, video_keys = self._root, self._video_keys
        template = self._info.get(
            "video_path", "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
        )

        def _run():
            ep_idx = ep["episode_index"]
            chunk = ep_idx // DEFAULT_CHUNK_SIZE
            table = pq.read_table(root / f"data/chunk-{chunk:03d}/episode_{ep_idx:06d}.parquet")
            columns = {}
            for col in ("action", "observation.state"):
                if col in table.column_names:
                    columns[col] = np.vstack(table.column(col).to_numpy())
            videos = {}
            for vk in video_keys:
                vpath = root / template.format(episode_chunk=chunk, video_key=vk, episode_index=ep_idx)
                if vpath.exists():
                    videos[vk] = str(vpath)
            return {"columns": columns, "videos": videos, "length": ep.get("length", 0)}

        self._worker = _FuncWorker(_run, self)
        self._worker.done.connect(self._on_replay_loaded)
        self._worker.failed.connect(lambda msg: QMessageBox.critical(self, "回放加载失败", msg))
        self._worker.start()

    def _on_replay_loaded(self, data: dict) -> None:
        self._replay_columns = data["columns"]
        self._replay_videos = data["videos"]
        self._replay_length = data["length"]
        self._replay_frame_idx = 0
        self.slider.setEnabled(True)
        self.slider.setRange(0, max(0, self._replay_length - 1))
        self.btn_play.setEnabled(True)
        self.btn_prev.setEnabled(True)
        self.btn_next.setEnabled(True)

        features = self._info.get("features", {})
        # values 表：state 在前 action 在后，字段名加前缀区分来源
        prefix = {"observation.state": "state", "action": "action"}
        self._replay_row_map = []
        rows = []
        for col in ("observation.state", "action"):
            arr = self._replay_columns.get(col)
            if arr is None:
                continue
            names = features.get(col, {}).get("names") or [f"{col}[{i}]" for i in range(arr.shape[1])]
            for i, name in enumerate(names):
                self._replay_row_map.append((col, i))
                rows.append(f"{prefix[col]}.{name}")
        self.values_table.setRowCount(len(rows))
        for r, label in enumerate(rows):
            self.values_table.setItem(r, 0, QTableWidgetItem(label))
            self.values_table.setItem(r, 1, QTableWidgetItem("-"))
        self._open_video()
        self._seek(0)
        self._log(f"回放已加载: {self._replay_length} 帧")

    def _open_video(self) -> None:
        if self._video_reader is not None:
            self._video_reader.close()
            self._video_reader = None
        cam = self.combo_camera.currentText()
        path = self._replay_videos.get(cam)
        if not path:
            return
        feature = self._info.get("features", {}).get(cam, {})
        height, width = (feature.get("shape") or [480, 640, 3])[:2]
        fps = self._info.get("fps", 30) or 30
        self._video_reader = FfmpegVideoReader(path, fps=fps, width=width, height=height)

    def _on_camera_changed(self) -> None:
        if self._replay_length > 0:
            self._open_video()
            self._seek(self._replay_frame_idx)

    def _on_speed_changed(self) -> None:
        if self._replay_timer.isActive():
            self._replay_timer.setInterval(self._tick_interval_ms())

    def _tick_interval_ms(self) -> int:
        fps = self._info.get("fps", 30) or 30
        return max(1, int(1000 / (fps * float(self.combo_speed.currentText()))))

    def _on_play_toggle(self) -> None:
        if self._replay_timer.isActive():
            self._stop_replay()
        else:
            self._replay_timer.start(self._tick_interval_ms())
            self.btn_play.setText("⏸ 暂停")

    def _stop_replay(self) -> None:
        self._replay_timer.stop()
        self.btn_play.setText("▶ 播放")

    def _on_replay_tick(self) -> None:
        if self._replay_frame_idx >= self._replay_length - 1:
            self._stop_replay()
            return
        self._seek(self._replay_frame_idx + 1)

    def _seek(self, idx: int) -> None:
        if self._replay_length <= 0:
            return
        idx = max(0, min(idx, self._replay_length - 1))
        self._replay_frame_idx = idx
        self.slider.blockSignals(True)
        self.slider.setValue(idx)
        self.slider.blockSignals(False)
        self.lbl_frame.setText(f"{idx} / {self._replay_length - 1}")

        if self._video_reader is not None:
            frame = self._video_reader.read(idx)
            if frame is not None:
                self.replay_view.set_frame(frame)

        for r, (col, i) in enumerate(self._replay_row_map):
            arr = self._replay_columns[col]
            item = self.values_table.item(r, 1)
            if item is not None and idx < arr.shape[0]:
                item.setText(f"{arr[idx, i]:.4f}")

    # ------------------------------------------------------------------
    # 维护脚本（QProcess 子进程）
    # ------------------------------------------------------------------

    def _run_script(self, module: str, args: list[str]) -> None:
        if self._process is not None:
            QMessageBox.warning(self, "提示", "已有脚本在运行")
            return
        cmd = [sys.executable, "-m", f"robodeploy.scripts.{module}", *args]
        self._log(f"$ {' '.join(cmd)}")
        self._process = QProcess(self)
        self._process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self._process.readyReadStandardOutput.connect(self._on_proc_output)
        self._process.finished.connect(self._on_proc_finished)
        self._process.start(cmd[0], cmd[1:])

    def _on_proc_output(self) -> None:
        data = bytes(self._process.readAllStandardOutput()).decode("utf-8", errors="replace")
        self._log(data.rstrip())

    def _on_proc_finished(self) -> None:
        self._log("[脚本已退出]")
        self._process = None

    def _on_delete_episodes(self) -> None:
        if not self._require_root():
            return
        text, ok = _ask_text(self, "删除剧集", "要删除的 episode 序号（空格分隔）：")
        if not ok or not text.strip():
            return
        try:
            indices = [int(s) for s in text.split()]
        except ValueError:
            QMessageBox.warning(self, "输入错误", "序号必须为整数")
            return
        reply = QMessageBox.question(
            self, "确认", f"将从 {self._root.name} 删除剧集 {indices}，不可恢复。继续？"
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._run_script("delete_episodes", [self._root_str(), *[str(i) for i in indices]])

    def _on_replace_task(self) -> None:
        if not self._require_root():
            return
        task, ok = _ask_text(self, "替换 task", "新的 task 描述：")
        if not ok or not task.strip():
            return
        tgt, ok = _ask_text(self, "替换 task", "输出父目录：")
        if not ok or not tgt.strip():
            return
        self._run_script("replace_task", ["--src", self._root_str(), "--tgt", tgt, "--task", task])

    def _on_filter_dataset(self) -> None:
        if not self._require_root():
            return
        text, ok = _ask_text(
            self,
            "过滤导出",
            "输出父目录 | repo_id | is_failure(true/false/留空) | is_infer(true/false/mixed/留空)，空格分隔：",
        )
        if not ok or not text.strip():
            return
        parts = text.split()
        if len(parts) < 2:
            QMessageBox.warning(self, "输入错误", "至少需要 输出父目录 和 repo_id")
            return
        args = ["--dataset", self._root_str(), "--output_dir", parts[0], "--repo_id", parts[1]]
        if len(parts) > 2 and parts[2]:
            args += ["--is-failure", parts[2]]
        if len(parts) > 3 and parts[3]:
            args += ["--is-infer", parts[3]]
        self._run_script("filter_lerobot_dataset", args)

    def _on_merge_datasets(self) -> None:
        text, ok = _ask_text(self, "合并数据集", "数据集路径列表（空格分隔）| 输出父目录 | repo_id：")
        if not ok or not text.strip():
            return
        parts = text.split()
        if len(parts) < 3:
            QMessageBox.warning(self, "输入错误", "至少需要 2 个数据集路径 + 输出目录 + repo_id")
            return
        self._run_script(
            "merge_lerobot_datasets",
            ["--datasets", *parts[:-2], "--output_dir", parts[-2], "--repo_id", parts[-1]],
        )

    # ------------------------------------------------------------------
    # v3.0 导出
    # ------------------------------------------------------------------

    def _on_convert_v30(self) -> None:
        if not self._require_root():
            return
        script = self.edit_convert_script.text().strip()
        if not Path(script).exists():
            QMessageBox.warning(self, "提示", f"转换脚本不存在: {script}\n请在配置中修改路径")
            return
        repo_id = self.edit_convert_repo.text().strip() or self._root.name
        args = [script, f"--repo-id={repo_id}", f"--root={self._root_str()}", "--push-to-hub=false"]

        if self._process is not None:
            QMessageBox.warning(self, "提示", "已有脚本在运行")
            return
        self._log(f"$ {sys.executable} {' '.join(args)}")
        self._process = QProcess(self)
        self._process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self._process.readyReadStandardOutput.connect(self._on_proc_output)
        self._process.finished.connect(self._on_convert_finished)
        self._process.start(sys.executable, args)

    def _on_convert_finished(self) -> None:
        self._process = None
        try:
            info = json.loads((self._root / "meta" / "info.json").read_text())
            version = info.get("codebase_version", "?")
            self._log(
                f"转换完成，codebase_version={version}"
                + (" ✓" if version == "v3.0" else "（非 v3.0，请检查日志）")
            )
        except Exception as exc:
            self._log(f"校验 info.json 失败: {exc}")

    # ------------------------------------------------------------------
    # 杂项
    # ------------------------------------------------------------------

    def _log(self, text: str) -> None:
        for line in text.splitlines() or [""]:
            self.log_view.appendPlainText(line)

    def shutdown(self) -> None:
        self._stop_replay()
        if self._video_reader is not None:
            self._video_reader.close()
            self._video_reader = None
        if self._process is not None:
            self._process.kill()
            self._process = None


def _ask_text(parent: QWidget, title: str, label: str) -> tuple[str, bool]:
    """单行文本输入对话框。"""
    dialog = QDialog(parent)
    dialog.setWindowTitle(title)
    form = QFormLayout(dialog)
    edit = QLineEdit()
    form.addRow(label, edit)
    buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    form.addRow(buttons)
    return edit.text(), dialog.exec() == QDialog.DialogCode.Accepted
