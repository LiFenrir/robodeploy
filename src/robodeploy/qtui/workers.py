"""后台线程：相机枚举 / 取流 / URDF 渲染。硬件 IO 一律在 QThread 内，帧经 Signal 跨线程传递。

调试预览直接用 cv2 / pyrealsense2 底层 API（不走驱动抽象层的严格校验），
参数尽力设置，界面显示实际生效的流参数。
"""

import time

import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from robodeploy.cameras.opencv.camera_opencv import OpenCVCamera

try:
    from robodeploy.cameras.realsense.camera_realsense import RealSenseCamera
except ImportError:
    RealSenseCamera = None

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None

try:
    import mujoco
except ImportError:
    mujoco = None

# rotation 配置值 → cv2 旋转常量
_CV2_ROTATION = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, -90: cv2.ROTATE_90_COUNTERCLOCKWISE}


def res_pixels(res: str) -> int:
    """ "640x480" → 像素数，用于档位排序。"""
    w, h = res.split("x")
    return int(w) * int(h)


def _rs_color_modes() -> dict[str, dict[str, list[int]]]:
    """枚举所有 RealSense 彩色流档位：serial → {"宽x高": [fps...]}（全格式并集）。"""
    if rs is None:
        return {}
    modes = {}
    for dev in rs.context().query_devices():
        serial = dev.get_info(rs.camera_info.serial_number)
        res: dict[str, set] = {}
        for sensor in dev.query_sensors():
            if not sensor.is_color_sensor():
                continue
            for p in sensor.get_stream_profiles():
                if p.is_video_stream_profile() and p.stream_type() == rs.stream.color:
                    v = p.as_video_stream_profile()
                    res.setdefault(f"{v.width()}x{v.height()}", set()).add(v.fps())
        modes[serial] = {k: sorted(v) for k, v in sorted(res.items(), key=lambda kv: -res_pixels(kv[0]))}
    return modes


class EnumWorker(QThread):
    """枚举 opencv / realsense 相机，结果 [{type, name, id, default_stream_profile, supported_modes}]。"""

    cameras_found = pyqtSignal(list)

    def run(self) -> None:
        cameras = []
        try:
            cameras.extend(OpenCVCamera.find_cameras())
        except Exception as exc:
            cameras.append({"type": "error", "name": f"OpenCV 枚举失败: {exc}", "id": None})
        if RealSenseCamera is not None:
            try:
                cameras.extend(RealSenseCamera.find_cameras())
            except Exception as exc:
                cameras.append({"type": "error", "name": f"RealSense 枚举失败: {exc}", "id": None})
            modes = _rs_color_modes()
            for info in cameras:
                if info.get("type") == "RealSense":
                    info["supported_modes"] = modes.get(str(info["id"]), {})
        self.cameras_found.emit(cameras)


class CameraWorker(QThread):
    """调试取流线程：底层 API 直连，frame_ready 输出 RGB uint8 (H,W,3)。

    source 字典：
      opencv:    {"kind", "id", "width", "height", "fps", "rotation"}
      realsense: {"kind", "serial", "width", "height", "fps", "rotation", "use_depth"}
    连接后 actual 为实际生效的 {"width", "height", "fps"}；
    RealSense 额外填充 intrinsics（color 内参 + depth_scale）。
    """

    frame_ready = pyqtSignal(np.ndarray)
    opened = pyqtSignal(str)  # 实际流参数，如 "848x480@30"
    failed = pyqtSignal(str)

    def __init__(self, source: dict, parent=None):
        super().__init__(parent)
        self.source = source
        self.actual: dict = {}
        self.intrinsics: dict | None = None
        self._running = False
        self._cancelled = False
        self._usb2 = False
        self._cap: cv2.VideoCapture | None = None
        self._pipe = None
        self._rs_format = None
        self._fallback_note = ""

    # ------------------------------------------------------------------
    # 连接
    # ------------------------------------------------------------------

    def _open_cv2(self, use_mjpg: bool) -> tuple[cv2.VideoCapture, dict]:
        cap = cv2.VideoCapture(self.source["id"])
        if not cap.isOpened():
            cap.release()
            raise ConnectionError(f"无法打开 {self.source['id']}")
        if use_mjpg:
            # 高分辨率/高帧率需要 MJPG 码流；YUYV 带宽不足时 set 会被驱动静默忽略
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        for prop, key in (
            (cv2.CAP_PROP_FRAME_WIDTH, "width"),
            (cv2.CAP_PROP_FRAME_HEIGHT, "height"),
            (cv2.CAP_PROP_FPS, "fps"),
        ):
            if self.source.get(key):
                cap.set(prop, float(self.source[key]))
        actual = {
            "width": int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))),
            "height": int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))),
            "fps": cap.get(cv2.CAP_PROP_FPS),
        }
        return cap, actual

    def _matches_request(self, actual: dict) -> bool:
        src = self.source
        return (
            actual["width"] == src.get("width")
            and actual["height"] == src.get("height")
            and abs(actual["fps"] - (src.get("fps") or actual["fps"])) < 0.5
        )

    def _connect_opencv(self) -> None:
        cap, actual = self._open_cv2(use_mjpg=True)
        if not self._matches_request(actual):
            # MJPG 下未命中 → 退回默认格式重试，取更贴近请求的一档
            cap2, actual2 = self._open_cv2(use_mjpg=False)
            if self._matches_request(actual2):
                cap.release()
                cap, actual = cap2, actual2
            else:
                cap2.release()
        self.actual = actual
        self._cap = cap

    def _pick_rs_profile(self, profiles: list, notes: list[str]):
        """按请求在设备真实档位中选最优 profile：分辨率精确→最近，fps 精确→最近。

        格式优先级按链路带宽：USB3 rgb8>bgr8>yuyv（零转换）；USB2 yuyv 优先
        （2 字节/像素，rgb8 高分辨率档带宽超 USB2 上限会在 S_FMT 报 EIO）。
        """
        w, h, fps = self.source.get("width"), self.source.get("height"), self.source.get("fps")
        if not (w and h and fps):
            default = next((p for p in profiles if p.is_default()), None)
            return default or profiles[0]

        pool = [p for p in profiles if p.width() == w and p.height() == h]
        if not pool:
            # 无该分辨率 → 取像素数最接近的一档
            best_res = min(
                {(p.width(), p.height()) for p in profiles}, key=lambda r: abs(r[0] * r[1] - w * h)
            )
            pool = [p for p in profiles if (p.width(), p.height()) == best_res]
            notes.append(f"无 {w}x{h} 档")
        fps_pool = [p for p in pool if p.fps() == fps]
        if fps_pool:
            pool = fps_pool
        else:
            best_fps = min({p.fps() for p in pool}, key=lambda f: abs(f - fps))
            pool = [p for p in pool if p.fps() == best_fps]
            notes.append(f"无 {fps}fps 档")
        formats = (
            (rs.format.yuyv, rs.format.rgb8, rs.format.bgr8)
            if self._usb2
            else (rs.format.rgb8, rs.format.bgr8, rs.format.yuyv)
        )
        for fmt in formats:
            match = next((p for p in pool if p.format() == fmt), None)
            if match is not None:
                return match
        return pool[0]

    def _connect_realsense(self) -> None:
        if rs is None:
            raise RuntimeError("pyrealsense2 未安装")
        serial = str(self.source["serial"])
        device = next(
            (d for d in rs.context().query_devices() if d.get_info(rs.camera_info.serial_number) == serial),
            None,
        )
        if device is None:
            raise ConnectionError(f"未找到序列号 {serial} 的 RealSense 设备（重新枚举确认是否掉线）")

        # USB2 链路带宽不足以支撑 rgb8 高分辨率档 → 格式优先级切换为 yuyv 优先
        try:
            usb_type = device.get_info(rs.camera_info.usb_type_descriptor) or ""
        except RuntimeError:
            usb_type = ""
        self._usb2 = usb_type.startswith("2")

        profiles = []
        for sensor in device.query_sensors():
            if sensor.is_color_sensor():
                profiles = [
                    p.as_video_stream_profile()
                    for p in sensor.get_stream_profiles()
                    if p.is_video_stream_profile() and p.stream_type() == rs.stream.color
                ]
                break

        notes: list[str] = []
        chosen = self._pick_rs_profile(profiles, notes)
        self._rs_format = chosen.format()
        if self._usb2 and self._rs_format == rs.format.yuyv:
            notes.append("USB2 链路 yuyv 格式")

        # can_resolve 预检：配置不可解析时先尝试去掉深度流，再报占用错误
        use_depth = bool(self.source.get("use_depth"))
        self._pipe = rs.pipeline()
        for attempt in range(2):
            cfg = rs.config()
            cfg.enable_device(serial)
            cfg.enable_stream(
                rs.stream.color,
                chosen.stream_index(),
                chosen.width(),
                chosen.height(),
                chosen.format(),
                chosen.fps(),
            )
            if use_depth:
                cfg.enable_stream(
                    rs.stream.depth, chosen.width(), chosen.height(), rs.format.z16, chosen.fps()
                )
            if cfg.can_resolve(self._pipe):
                break
            if attempt == 0 and use_depth:
                use_depth = False
                notes.append("深度流不可用")
                continue
            raise ConnectionError("配置无法解析：设备可能被其他进程占用")

        # pipe.start 失败恢复，按错误特征分流：
        # - "Cannot open /dev/videoN"（ENOENT）：进程退出后 uvc 驱动重绑定的节点重建窗口，只能等 → 纯重试
        # - "xioctl VIDIOC_S_FMT"（EIO）：传感器拒绝其宣称支持的档位，卡在异常状态 → 硬件复位一次后重试
        # （复位本身会触发节点重建窗口，后续 ENOENT 由重试自然吸收；总预算 90s，可被取消）
        profile = None
        last_exc = None
        reset_done = False
        deadline = time.monotonic() + 90.0
        while time.monotonic() < deadline:
            try:
                profile = self._pipe.start(cfg)
                break
            except RuntimeError as exc:
                last_exc = exc
                if self._cancelled:
                    # 重试期间用户点了断开 → 立即退出，不再等待预算耗尽
                    raise ConnectionError("连接已取消") from None
                if not reset_done and "xioctl" in str(exc):
                    reset_done = True
                    notes.append("设备档位异常已硬件复位")
                    try:
                        device.hardware_reset()
                    except RuntimeError:
                        pass
                    self._wait_reenumerate(serial)
                    # 复位自身触发数十秒节点重建窗口 → 重设计时，给足重试预算
                    deadline = time.monotonic() + 90.0
                time.sleep(2.0)
                self._pipe = rs.pipeline()
        if profile is None:
            raise ConnectionError(
                f"重试超时仍无法取流: {last_exc}。"
                "若刚退出过其他 RealSense 程序，USB 节点可能在重建，请稍后重试；否则请重新插拔设备"
            ) from last_exc
        color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        self.actual = {
            "width": color_profile.width(),
            "height": color_profile.height(),
            "fps": color_profile.fps(),
        }
        intr = color_profile.get_intrinsics()
        self.intrinsics = {
            "color": {
                "width": intr.width,
                "height": intr.height,
                "fx": intr.fx,
                "fy": intr.fy,
                "ppx": intr.ppx,
                "ppy": intr.ppy,
                "model": str(intr.model),
                "coeffs": list(intr.coeffs),
            }
        }
        if use_depth:
            try:
                self.intrinsics["depth_scale"] = profile.get_device().first_depth_sensor().get_depth_scale()
            except RuntimeError:
                pass
        self._fallback_note = f"（{'，'.join(notes)}）" if notes else ""

    @staticmethod
    def _wait_reenumerate(serial: str, timeout_s: float = 10.0) -> None:
        """hardware_reset 后轮询等设备重新出现，再稍等节点就绪。"""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            devs = rs.context().query_devices()
            if any(d.get_info(rs.camera_info.serial_number) == serial for d in devs):
                time.sleep(2.0)
                return
            time.sleep(0.5)
        raise ConnectionError(f"硬件复位后设备 {serial} 未重新出现")

    # ------------------------------------------------------------------
    # 取流
    # ------------------------------------------------------------------

    def _rotate(self, frame: np.ndarray) -> np.ndarray:
        code = _CV2_ROTATION.get(self.source.get("rotation", 0))
        return cv2.rotate(frame, code) if code is not None else frame

    def run(self) -> None:
        try:
            if self.source["kind"] == "opencv":
                self._connect_opencv()
            else:
                self._connect_realsense()
        except Exception as exc:
            if not self._cancelled:
                self.failed.emit(f"连接失败: {exc}")
            return

        if self._cancelled:
            # 连接耗时重试期间用户点了断开 → 直接收掉，不进取流循环
            if self._pipe is not None:
                try:
                    self._pipe.stop()
                except Exception:
                    pass
                self._pipe = None
            return

        note = self._fallback_note
        fps = self.actual["fps"]
        fps_text = f"{fps:.0f}" if float(fps).is_integer() else f"{fps:.1f}"
        self.opened.emit(f"{self.actual['width']}x{self.actual['height']}@{fps_text}{note}")

        self._running = True
        while self._running:
            try:
                if self._cap is not None:
                    ret, raw = self._cap.read()
                    if not ret or raw is None:
                        raise RuntimeError("read 失败")
                    frame = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
                else:
                    frames = self._pipe.wait_for_frames(5000)
                    color = frames.get_color_frame()
                    if not color:
                        continue
                    raw = np.asanyarray(color.get_data())
                    if self._rs_format == rs.format.yuyv:
                        # yuyv 原始帧 numpy 解释不固定（可能 (H,W) uint16 或 (H,W*2) uint8）
                        # → 按帧元数据统一还原为 (H, W, 2) uint8 再转 RGB
                        raw = (
                            np.ascontiguousarray(raw)
                            .view(np.uint8)
                            .reshape(color.get_height(), color.get_width(), 2)
                        )
                        frame = cv2.cvtColor(raw, cv2.COLOR_YUV2RGB_YUY2)
                    elif self._rs_format == rs.format.bgr8:
                        frame = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
                    else:
                        frame = raw
                self.frame_ready.emit(self._rotate(np.ascontiguousarray(frame)))
            except Exception as exc:
                if self._running:
                    self.failed.emit(f"取流失败: {exc}")
                break

    def stop(self) -> None:
        self._cancelled = True
        self._running = False
        self.wait(6000)
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        if self._pipe is not None:
            try:
                self._pipe.stop()
            except Exception:
                pass
            self._pipe = None


class UrdfWorker(QThread):
    """URDF 离屏渲染线程：qpos 更新 → mujoco MjRenderer → frame_ready(RGB uint8)。

    set_qpos(joint_values) 更新关节角；渲染上下文在线程内惰性初始化，
    初始化失败时发 failed 信号，由上层降级为数值表。
    """

    frame_ready = pyqtSignal(np.ndarray)
    ready = pyqtSignal(int)  # 关节数 nq
    failed = pyqtSignal(str)

    def __init__(self, urdf_path: str, width: int = 640, height: int = 480, fps: int = 30, parent=None):
        super().__init__(parent)
        self._urdf_path = urdf_path
        self._width = width
        self._height = height
        self._interval_ms = 1000 // fps
        self._running = False
        self._qpos: np.ndarray | None = None

    def set_qpos(self, joint_values: np.ndarray) -> None:
        self._qpos = np.asarray(joint_values, dtype=np.float64)

    def run(self) -> None:
        try:
            if mujoco is None:
                raise ImportError("mujoco 未安装")
            model = mujoco.MjModel.from_xml_path(self._urdf_path)
            data = mujoco.MjData(model)
            renderer = mujoco.Renderer(model, height=self._height, width=self._width)
        except Exception as exc:
            self.failed.emit(f"URDF 渲染初始化失败: {exc}")
            return

        self._running = True
        self.ready.emit(model.nq)
        while self._running:
            if self._qpos is not None:
                n = min(model.nq, len(self._qpos))
                data.qpos[:n] = self._qpos[:n]
                mujoco.mj_forward(model, data)
            renderer.update_scene(data)
            self.frame_ready.emit(renderer.render().copy())
            self.msleep(self._interval_ms)
        renderer.close()

    def stop(self) -> None:
        self._running = False
        self.wait(3000)
