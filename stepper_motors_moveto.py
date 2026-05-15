from __future__ import annotations
import math, time
from PyQt6.QtCore import QObject, QTimer, pyqtSignal

class StepperMotorsMoveto(QObject):
    """
    GOTO planner with safe Z sequencing for precision stages.

    When a move involves BOTH XY and Z changes AND Z delta > Z_HOME_THRESHOLD:
      1. Z homes to true zero (via photoelectric switch)
      2. XY moves to target
      3. Z moves to target position

    Small Z deltas (e.g. spiral scan tilt corrections) do XY first,
    then Z, without Z homing.

    Pure XY or pure Z moves skip the irrelevant phases.

    Speed is dynamic: read from the hardware controller's current profile.
    """
    finished = pyqtSignal(bool, str)
    statusChanged = pyqtSignal(str, str)

    # Only trigger Z homing for Z deltas larger than this (full-steps)
    Z_HOME_THRESHOLD = 5.0

    def __init__(self, motion_controller, motors_model, mc=None, homing=None,
                 vmax: float = 400.0, parent=None):
        super().__init__(parent)
        self.motion = motion_controller
        self.MOTORS = motors_model
        self.mc = mc
        self.homing = homing  # StepperMotorsHoming reference for Z-home
        self._default_vmax = float(vmax)
        self._active = False

        self._phase = "idle"
        self._target = {"x": 0.0, "y": 0.0, "z": 0.0}
        self._final_target = {"x": 0.0, "y": 0.0, "z": 0.0}
        self._saved_profile = None
        self._z_home_connected = False  # track signal connection

        self._tol = 0.5
        self._slow_radius = 5.0
        self._vmin = 30.0

        self._stall_timeout_s = 3.0
        self._progress_eps = 0.02

        self._last_dist = None
        self._last_progress_t = 0.0

        self._timer = QTimer(self)
        self._timer.setInterval(10)
        self._timer.timeout.connect(self._plan_tick)

    def set_homing(self, homing):
        """Set homing controller reference (can be called after construction)."""
        self.homing = homing

    @property
    def vmax(self) -> float:
        """Dynamic vmax from the hardware controller's active profile."""
        if self.mc is not None:
            try:
                return float(self.mc.get_profile_vmax())
            except Exception:
                pass
        return self._default_vmax

    def is_active(self) -> bool:
        return self._active

    def start(self, tx_fs: float, ty_fs: float, tz_fs: float, profile: str = None):
        """Start a move with safe Z sequencing.

        Combined XY+Z moves with large Z delta: home Z first, then XY, then Z.
        Combined XY+Z moves with small Z delta: XY first, then Z (no homing).
        Pure XY: move directly.
        Pure Z to ~0: home Z.
        Pure Z to other: move directly.
        """
        if self._active:
            self.finished.emit(False, "Busy: move-to already active.")
            return

        self._final_target = {"x": float(tx_fs), "y": float(ty_fs), "z": float(tz_fs)}
        cur = self._cur()

        # Only switch profile if explicitly requested by caller
        self._saved_profile = None
        if profile is not None and self.mc is not None:
            self._saved_profile = self.mc.get_current_profile()
            self.mc.apply_speed_profile(profile)

        moving_xy = (abs(self._final_target["x"] - cur["x"]) > self._tol or
                     abs(self._final_target["y"] - cur["y"]) > self._tol)
        moving_z = abs(self._final_target["z"] - cur["z"]) > self._tol
        z_delta = abs(self._final_target["z"] - cur["z"])
        large_z = z_delta > self.Z_HOME_THRESHOLD

        if moving_xy and moving_z and large_z:
            # Large combined XY+Z move: home Z to true zero first
            self._active = True
            self.statusChanged.emit("Homing Z...", "red")
            if self.homing is not None:
                self._phase = "z_homing"
                self._connect_z_home_signal()
                self.homing.home_z_only()
            else:
                # No homing controller — fall back to driving Z toward 0
                print("[MoveTo] WARNING: no homing controller, driving Z toward 0")
                self._phase = "z_up"
                self._target = {"x": cur["x"], "y": cur["y"], "z": 0.0}
                self._last_dist = self._calc_dist(cur, self._target)
                self._last_progress_t = time.monotonic()
                self.motion.set_mode("goto")
                self._timer.start()
        elif moving_xy:
            # Pure XY move, or small combined XY+Z (XY first, Z handled in _advance_phase)
            self._active = True
            self._phase = "xy"
            self._target = {"x": self._final_target["x"],
                            "y": self._final_target["y"],
                            "z": cur["z"]}
            self._last_dist = self._calc_dist(cur, self._target)
            self._last_progress_t = time.monotonic()
            self.statusChanged.emit("Moving...", "red")
            self.motion.set_mode("goto")
            self._timer.start()
        elif moving_z:
            # Pure Z move
            if self._final_target["z"] < 1.0 and self.homing is not None:
                # Z targeting zero — use homing for true zero
                self._active = True
                self._phase = "z_homing"
                self.statusChanged.emit("Homing Z...", "red")
                self._connect_z_home_signal()
                self.homing.home_z_only()
            else:
                self._active = True
                self._phase = "z_only"
                self._target = self._final_target
                self._last_dist = self._calc_dist(cur, self._target)
                self._last_progress_t = time.monotonic()
                self.statusChanged.emit("Moving...", "red")
                self.motion.set_mode("goto")
                self._timer.start()
        else:
            # Already at target
            self.finished.emit(True, "")
            return

    def stop(self, reason="Move stopped by user."):
        if not self._active:
            return
        # Stop Z homing if in progress
        if self._phase == "z_homing" and self.homing is not None:
            self.homing.stop(reason)
        self._timer.stop()
        self._active = False
        self._phase = "idle"
        self.motion.stop_immediate()
        self._disconnect_z_home_signal()
        self._restore_profile()
        self.statusChanged.emit("Idle", "gray")
        self.finished.emit(False, reason)

    def _connect_z_home_signal(self):
        """Connect to homing z_home_finished signal."""
        if not self._z_home_connected and self.homing is not None:
            self.homing.z_home_finished.connect(self._on_z_home_done)
            self._z_home_connected = True

    def _disconnect_z_home_signal(self):
        """Disconnect from homing z_home_finished signal."""
        if self._z_home_connected and self.homing is not None:
            try:
                self.homing.z_home_finished.disconnect(self._on_z_home_done)
            except Exception:
                pass
            self._z_home_connected = False

    def _on_z_home_done(self, success: bool, message: str):
        self._disconnect_z_home_signal()

        if not self._active or self._phase != "z_homing":
            return

        if not success:
            print(f"[MoveTo] Z homing failed: {message}")
            self._timer.stop()
            self._active = False
            self._phase = "idle"
            self._restore_profile()
            self.statusChanged.emit("Idle", "gray")
            self.finished.emit(False, f"Z homing failed: {message}")
            return

        cur = self._cur()
        moving_xy = (abs(self._final_target["x"] - cur["x"]) > self._tol or
                     abs(self._final_target["y"] - cur["y"]) > self._tol)

        if moving_xy:
            print(f"[MoveTo] Z homed to zero, starting XY move")
            self._phase = "xy"
            self._target = {"x": self._final_target["x"],
                            "y": self._final_target["y"],
                            "z": cur["z"]}
            self._last_dist = self._calc_dist(cur, self._target)
            self._last_progress_t = time.monotonic()
            self.statusChanged.emit("Moving XY...", "red")
            self.motion.set_mode("goto")
            self._timer.start()
        else:
            # Pure Z home — we're done
            print(f"[MoveTo] Z homed to zero, no XY needed")
            self._finish()

    def _restore_profile(self):
        """Restore speed profile only if we explicitly overrode it."""
        if self._saved_profile is not None and self.mc is not None:
            try:
                self.mc.apply_speed_profile(self._saved_profile)
            except Exception:
                pass
            self._saved_profile = None

    def _cur(self):
        return {
            "x": float(getattr(self.MOTORS, "x_current_position_full_step", 0.0)),
            "y": float(getattr(self.MOTORS, "y_current_position_full_step", 0.0)),
            "z": float(getattr(self.MOTORS, "z_current_position_full_step", 0.0)),
        }

    def _calc_dist(self, p1, p2):
        return math.sqrt((p1["x"]-p2["x"])**2 + (p1["y"]-p2["y"])**2 + (p1["z"]-p2["z"])**2)

    def _finish(self):
        """Complete the move."""
        self._timer.stop()
        self._active = False
        self._phase = "idle"
        self.motion.stop_immediate()
        self._restore_profile()
        self.statusChanged.emit("Idle", "gray")
        self.finished.emit(True, "")

    def _advance_phase(self):
        """Transition to the next phase after arriving at current target."""
        cur = self._cur()
        now = time.monotonic()

        if self._phase == "z_up":
            # Z retract done (fallback path) → now do XY
            moving_xy = (abs(self._final_target["x"] - cur["x"]) > self._tol or
                         abs(self._final_target["y"] - cur["y"]) > self._tol)
            if moving_xy:
                self._phase = "xy"
                self._target = {"x": self._final_target["x"],
                                "y": self._final_target["y"],
                                "z": cur["z"]}
                self._last_dist = self._calc_dist(cur, self._target)
                self._last_progress_t = now
                self.statusChanged.emit("Moving XY...", "red")
                return
            # No XY needed, check Z target
            dz = abs(self._final_target["z"] - cur["z"])
            if dz > self._tol:
                self._phase = "z_down"
                self._target = self._final_target
                self._last_dist = self._calc_dist(cur, self._target)
                self._last_progress_t = now
                self.statusChanged.emit("Moving Z...", "red")
                return
            self._finish()

        elif self._phase == "xy":
            # XY done → check if Z needs to move to target
            dz = abs(self._final_target["z"] - cur["z"])
            if dz > self._tol:
                self._phase = "z_down"
                self._target = self._final_target
                self._last_dist = self._calc_dist(cur, self._target)
                self._last_progress_t = now
                self.statusChanged.emit("Moving Z...", "red")
                return
            self._finish()

        else:
            # z_only, z_down, or any other terminal phase
            self._finish()

    def _plan_tick(self):
        if not self._active:
            return

        # Don't tick during Z homing — homing controller handles that
        if self._phase == "z_homing":
            return

        cur = self._cur()
        dist = self._calc_dist(cur, self._target)
        now = time.monotonic()

        # Stall Detection
        if self._last_dist is not None:
            if dist < (self._last_dist - self._progress_eps):
                self._last_dist = dist
                self._last_progress_t = now
            elif (now - self._last_progress_t) > self._stall_timeout_s:
                print(f"[MoveTo] Stall detected: dist={dist:.3f}, "
                      f"last_dist={self._last_dist:.3f}, phase={self._phase}")
                self.stop("Move blocked (Stalled).")
                return

        # Arrival at current phase target
        if dist <= self._tol:
            self._advance_phase()
            return

        # Velocity math — use dynamic vmax
        ex = self._target["x"] - cur["x"]
        ey = self._target["y"] - cur["y"]
        ez = self._target["z"] - cur["z"]

        ux, uy, uz = ex/dist, ey/dist, ez/dist

        current_vmax = self.vmax
        speed = current_vmax
        if dist < self._slow_radius:
            vmin = min(self._vmin, current_vmax)
            alpha = max(0.0, min(1.0, dist / self._slow_radius))
            speed = vmin + (current_vmax - vmin) * alpha

        self.motion.set_desired_velocity(ux*speed, uy*speed, uz*speed, mode="goto")