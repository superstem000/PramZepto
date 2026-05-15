# motor.py
from dataclasses import dataclass
from typing import Callable, Dict, Optional
from PyQt6.QtCore import QObject, QThread, pyqtSignal
from tmc_driver.tmc_2209 import (
    Tmc2209, TmcEnableControlPin, TmcMotionControlStepDir, TmcComUart, MovementAbsRel
)
import time

# -------------------- Patch: per-step callback with signed accumulator --------------------

_orig_make_a_step = TmcMotionControlStepDir.make_a_step

def _patched_make_a_step(self: TmcMotionControlStepDir):
    _orig_make_a_step(self)

    if getattr(self, "_suppress_step_cb", False):
        #print(f"[STEP] SUPPRESSED id={id(self)}")
        return

    # Check if original library has a call back
    cb = getattr(self, "_on_step_cb", None)
    if not cb:
        return

    sign = 1 if getattr(self, "_dir_sign", 1) >= 0 else -1
    #print(f"[STEP] REAL id={id(self)}")

    # Maintain signed logical position (microstep units)
    lp = getattr(self, "_logical_pos", 0)
    lp += sign
    self._logical_pos = lp

    try:
        cb(lp)
    except Exception:
        pass

TmcMotionControlStepDir.make_a_step = _patched_make_a_step


# -------------------- Signals --------------------

class MotorSignals(QObject):
    print("stepper_motor_module.py: step updated")
    step = pyqtSignal(str, int)  # axis_name, logical_micro_pos


# -------------------- Data --------------------

@dataclass
class Targets:
    x: float # full-step
    y: float # full-step
    z: float

# -------------------- Axis wrapper --------------------

class Axis:
    def __init__(self, axis_cfg, uart: TmcComUart):
        # axis_cfg is an AxisCfg dataclass
        self.tmc = Tmc2209(
            TmcEnableControlPin(axis_cfg.enable),
            TmcMotionControlStepDir(axis_cfg.step, axis_cfg.dir),   # (STEP, DIR)
            uart,
            driver_address=axis_cfg.driver_address,
        )
        self.tmc.set_direction_reg(False)
        self.invert = bool(axis_cfg.invert_dir)

        self.tmc.set_current(axis_cfg.current_ma)
        self.tmc.set_interpolation(True)
        self.tmc.set_spreadcycle(axis_cfg.spreadcycle)
        self.tmc.set_microstepping_resolution(axis_cfg.microstepping)
        self.tmc.set_internal_rsense(False)
        self.tmc.set_motor_enabled(True)

        # initialize signed accumulator
        self.tmc.tmc_mc._logical_pos = 0
        self.tmc.tmc_mc._dir_sign = 1

    def on_step(self, fn):
        """Register a callable that receives (relative_current_pos) on every step."""
        # NOTE: this value is whatever the driver's tmc_mc.current_pos is.
        self.tmc.tmc_mc._on_step_cb = fn

    def clear_step_cb(self):
        """Remove step callback."""
        if hasattr(self.tmc.tmc_mc, "_on_step_cb"):
            self.tmc.tmc_mc._on_step_cb = None
    
    def set_profile(self, accel_fullstep: int, max_speed_fullstep: int):
        self.tmc.tmc_mc._suppress_step_cb = True
        self.tmc.acceleration_fullstep = int(accel_fullstep)
        self.tmc.max_speed_fullstep = int(max_speed_fullstep)
        self.tmc.tmc_mc.acceleration_fullstep = int(accel_fullstep)
        self.tmc.tmc_mc.max_speed_fullstep = int(max_speed_fullstep)

    def set_logical_pos(self, logical_micro: int):
        self.tmc.tmc_mc._logical_pos = int(logical_micro)

    def get_logical_pos(self) -> int:
        return int(getattr(self.tmc.tmc_mc, "_logical_pos", 0))

    def move_rel(self, microsteps: int):
        """
            Actual Microstepping (signed)
        """
        logical = int(microsteps)
        if logical == 0:
            return
        
        self.tmc.tmc_mc._suppress_step_cb = False

        # physical direction to driver
        physical = -logical if self.invert else logical

        # set direction sign for accumulator callback
        self.tmc.tmc_mc._dir_sign = 1 if logical > 0 else -1

        # run threaded relative
        self.tmc.tmc_mc.run_to_position_steps_threaded(physical, MovementAbsRel.RELATIVE)
        
    def wait(self):
        self.tmc.tmc_mc.wait_for_movement_finished_threaded()

    def stop(self):
        self.tmc.tmc_mc.stop()

    def enable(self, on: bool):
        self.tmc.set_motor_enabled(bool(on))


# -------------------- Jog loop worker --------------------

class _JogWorker(QObject):
    step = pyqtSignal(str, int)   # axis_name, relative_pos
    """Background loop that ticks run_speed() for all axes while speeds are non-zero.

    The TMC library's run_speed() uses internal microsecond timing to decide
    when to fire step pulses. We must call it frequently enough that it can
    distinguish between different speed settings. A 1ms (1000us) sleep is too
    slow - both fast and slow speeds have _step_interval under 1000us, so both
    step on every tick, making them identical. We use ~20us busy-poll when
    stepping so run_speed's internal _step_interval timing controls the rate.
    """
    def __init__(self, mc, tick_s=0.000020):
        super().__init__()
        self.mc = mc
        self._tick_s = float(tick_s)    # ~20us when busy
        self._idle_tick_s = 0.002       # 2ms when idle (save CPU)
        self._running = True

    def stop(self):
        self._running = False

    def run(self):
        while self._running:
            busy = False
            for ax in ("x", "y", "z"):
                a = self.mc.axes[ax].tmc.tmc_mc
                if getattr(a, "speed_fullstep", 0) != 0:
                    a.run_speed()
                    busy = True
            # Short sleep when stepping so run_speed's interval timing works;
            # longer sleep when idle to avoid burning CPU
            time.sleep(self._tick_s if busy else self._idle_tick_s)


# -------------------- MotorController (always-on step signal) --------------------

class StepperMotorHardware:
    """
    UI-agnostic controller: create once, keep around.
    Microstepping is locked at 32 for all axes.
    Speed is controlled via named profiles (travel / normal / fine).
    """
    LOCKED_MRES = 32  # always 32 microsteps

    def __init__(self, cfg):
        self.signals = MotorSignals()
        self.cfg = cfg

        self.uart = TmcComUart(cfg.uart.port)
        self.axes = {
            "x": Axis(cfg.axes["x"], self.uart),
            "y": Axis(cfg.axes["y"], self.uart),
            "z": Axis(cfg.axes["z"], self.uart),
        }

        # Force microstepping to 32 on all axes regardless of config
        for name, ax in self.axes.items():
            ax.tmc.set_microstepping_resolution(self.LOCKED_MRES)
            cfg.axes[name].microstepping = self.LOCKED_MRES

        # global callbacks always installed
        for name, ax in self.axes.items():
            ax.on_step(lambda pos, n=name: self.signals.step.emit(n, pos))

        # Store speed control config
        sc = getattr(cfg, "speed_control", None)
        if sc is not None:
            self._base_speed = int(sc.base_speed_fullstep)
            self._base_accel = int(sc.base_accel_fullstep)
            self._max_speed  = int(sc.max_speed_fullstep)
            self._max_accel  = int(sc.max_accel_fullstep)
            self._min_speed  = int(sc.min_speed_fullstep)
        else:
            self._base_speed = int(cfg.motion.max_speed_fullstep)
            self._base_accel = int(cfg.motion.accel_fullstep)
            self._max_speed  = 800
            self._max_accel  = 400
            self._min_speed  = 3

        self._speed_multiplier = 1.0
        self._current_vmax = self._base_speed
        self._current_accel = self._base_accel

        # Apply default speed (x1.0)
        self.set_speed_multiplier(1.0)
        
        self._jog_thread: Optional[QThread] = None
        self._jog_worker: Optional[_JogWorker] = None

    def set_speed_multiplier(self, multiplier: float):
        """Set speed as a multiplier of base speed. x1.0 = normal, x0.01 = very slow, x2.67 = max.
        Accel scales proportionally. Clamped to [min_speed, max_speed]."""
        multiplier = max(0.001, float(multiplier))
        self._speed_multiplier = multiplier

        raw_speed = self._base_speed * multiplier
        raw_accel = self._base_accel * multiplier

        self._current_vmax = int(max(self._min_speed, min(self._max_speed, raw_speed)))
        self._current_accel = int(max(1, min(self._max_accel, raw_accel)))

        for a in self.axes.values():
            a.set_profile(self._current_accel, self._current_vmax)
        print(f"[MC] Speed x{multiplier:.2f} -> vmax={self._current_vmax}, accel={self._current_accel}")

    def set_speed_for_homing(self):
        """Temporarily set max speed for homing. Call restore_speed_multiplier() after."""
        self._saved_multiplier = self._speed_multiplier
        for a in self.axes.values():
            a.set_profile(self._max_accel, self._max_speed)
        self._current_vmax = self._max_speed
        self._current_accel = self._max_accel
        print(f"[MC] Speed -> HOMING (vmax={self._max_speed}, accel={self._max_accel})")

    def restore_speed_multiplier(self):
        """Restore the user's speed multiplier after homing."""
        self.set_speed_multiplier(self._saved_multiplier)

    def get_speed_multiplier(self) -> float:
        return self._speed_multiplier

    def get_current_vmax(self) -> float:
        return float(self._current_vmax)

    def get_max_multiplier(self) -> float:
        """Max multiplier before hitting max_speed."""
        return float(self._max_speed) / float(self._base_speed)

    def get_min_multiplier(self) -> float:
        """Min multiplier before hitting min_speed."""
        return float(self._min_speed) / float(self._base_speed)

    # Keep old API for compatibility (motion controller uses this)
    def get_profile_vmax(self, profile_name: str = None) -> float:
        return float(self._current_vmax)

    def get_current_profile(self) -> str:
        return f"x{self._speed_multiplier:.2f}"

    def apply_speed_profile(self, profile_name: str):
        """Compatibility shim: 'travel' maps to max speed, others are no-ops."""
        if profile_name == "travel":
            self.set_speed_for_homing()
        # Other profile names are ignored; use set_speed_multiplier instead

    def _ensure_jog_loop(self):
        if getattr(self, "_jog_thread", None):
            return
        self._jog_thread = QThread()
        self._jog_worker = _JogWorker(self)
        self._jog_worker.moveToThread(self._jog_thread)
        self._jog_thread.started.connect(self._jog_worker.run)
        self._jog_thread.start()

    def stop_jog_loop(self):
        """(Optional) Call on app exit to clean up the jog thread."""
        if getattr(self, "_jog_thread", None):
            self._jog_worker.stop()
            self._jog_thread.quit()
            self._jog_thread.wait()
            self._jog_thread = None
            self._jog_worker = None

    def halt_all_speed(self):
        """Set all jog speeds to 0 (does not kill the jog loop)."""
        for ax in ("x", "y", "z"):
            self.axes[ax].tmc.tmc_mc.speed_fullstep = 0

    def stop_all(self):
        self.halt_all_speed()
        self.stop_jog_loop()
        for a in self.axes.values():
            a.stop()

    def enable_all(self, on: bool):
        for a in self.axes.values():
            a.enable(on)

    def set_speed_fullstep(self, axis: str, speed_fullstep: float):
        """
        Set continuous speed for an axis in full-steps per second.
        The TMC library's speed_fullstep property setter automatically
        scales by mres internally, so we pass full-step values directly.
        """
        axis = axis.lower()
        if axis not in self.axes:
            return
        self._ensure_jog_loop()

        a = self.axes[axis].tmc.tmc_mc
        

        # Clamp to profile vmax (in full-step units)
        vmax = self.axes[axis].tmc.tmc_mc.max_speed_fullstep

        # LOGICAL command (what the UI means by + / -)
        logical_speed = max(-vmax, min(vmax, float(speed_fullstep)))
        # PHYSICAL command to the driver (apply invert_dir here)
        physical_speed = -logical_speed if self.axes[axis].invert else logical_speed

        # The property setter handles mres scaling and _step_interval computation
        a.speed_fullstep = int(physical_speed)
        # For your patched callback: report LOGICAL direction
        if logical_speed != 0:
            a._suppress_step_cb = False
            a._dir_sign = 1 if logical_speed > 0 else -1
        else:
            a._suppress_step_cb = True

    def set_axis_microstepping(self, axis: str, mres: int):
        axis = axis.lower()
        if axis in self.axes:
            self.axes[axis].tmc.set_microstepping_resolution(int(mres))

    def get_axis_microstepping(self, axis: str) -> int:
        axis = axis.lower()
        if axis in self.axes:
            return int(self.axes[axis].tmc.tmc_mc.mres)  # microsteps/fullstep as tracked by MC
        return 0
    
    def set_logical_position_micro(self, axis: str, pos_micro: int):
        axis = axis.lower()
        if axis in self.axes:
            self.axes[axis].set_logical_pos(int(pos_micro))

    def get_logical_position_micro(self, axis: str) -> int:
        axis = axis.lower()
        if axis in self.axes:
            return self.axes[axis].get_logical_pos()
        return 0
    
    # Make sure close() shuts the jog loop down as well
    def close(self):
        self.halt_all_speed()
        self.stop_jog_loop()
        self.stop_all()
        self.enable_all(False)