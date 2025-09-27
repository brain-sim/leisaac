import numpy as np

import carb
import omni

from .so101_leader import SO101Leader
from ..device_base import Device
from termcolor import colored

class XLeRobotTeleop(Device):
    """Teleoperation interface combining keyboard base control with dual SO101 leaders."""

    def __init__(
        self,
        env,
        left_port: str = '/dev/ttyACM0',
        right_port: str = '/dev/ttyACM1',
        recalibrate: bool = False,
    ):
        super().__init__(env)

        print(colored(f"Connecting to left_so101_leader using left_port : {left_port}...", "cyan", attrs=["bold"]))
        self.left_leader = SO101Leader(env, port=left_port, recalibrate=recalibrate, calibration_file_name="xlerobot_left_so101_leader.json")
        print(colored(f"Connecting to right_so101_leader using right_port : {right_port}...", "cyan", attrs=["bold"]))
        self.right_leader = SO101Leader(env, port=right_port, recalibrate=recalibrate, calibration_file_name="xlerobot_right_so101_leader.json")
        # avoid duplicate keyboard listeners for start/reset keys
        self.right_leader.stop_keyboard_listener()
        self.left_leader.stop_keyboard_listener()

        # base command keyboard handling
        self._base_command = np.zeros(3, dtype=np.float32)
        self._pressed_keys: set[str] = set()
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            self._on_keyboard_event,
        )

        self._base_speed = 10.0

        self._BASE_KEY_MAPPING = {
            "W": 1.25 * self._base_speed * np.asarray([0.0, 1.0, -1.0], dtype=np.float32),
            "UP": 1.25 * self._base_speed * np.asarray([0.0, 1.0, -1.0], dtype=np.float32),
            "S": self._base_speed * np.asarray([0.0, -1.0, 1.0], dtype=np.float32),
            "DOWN": self._base_speed * np.asarray([0.0, -1.0, 1.0], dtype=np.float32),
            "A": self._base_speed * np.asarray([1.0, -0.45, -0.45], dtype=np.float32),
            "LEFT": self._base_speed * np.asarray([1.0, -0.45, -0.45], dtype=np.float32),
            "D": self._base_speed * np.asarray([-1.0, 0.45, 0.45], dtype=np.float32),
            "RIGHT": self._base_speed * np.asarray([-1.0, 0.45, 0.45], dtype=np.float32),
            "Q": self._base_speed * np.asarray([-1.0, -1.0, -1.0], dtype=np.float32),
            "E": self._base_speed * np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
        }

    def __del__(self):
        self.stop_keyboard_listener()
        if hasattr(self, 'right_leader'):
            self.right_leader.stop_keyboard_listener()

    def __str__(self) -> str:
        msg = "XLEROBOT teleop (keyboard base + dual SO101 leaders).\n"
        msg += "\t----------------------------------------------\n"
        msg += "\tBase move: W/S or Up/Down (forward/backward)\n"
        msg += "\t           A/D or Left/Right (strafe left/right)\n"
        msg += "\t           Q/E (rotate left/right)\n"
        msg += "\tSpace resets base command to zero\n"
        msg += "\tArms: controlled via attached SO101 leaders\n"
        msg += "\tUse B / R / N on left leader to start/reset\n"
        return msg

    def stop_keyboard_listener(self):
        if hasattr(self, '_input') and hasattr(self, '_keyboard') and hasattr(self, '_keyboard_sub') and self._keyboard_sub is not None:
            self._input.unsubscribe_to_keyboard_events(self._keyboard, self._keyboard_sub)
            self._keyboard_sub = None

    def add_callback(self, key, func):
        self.left_leader.add_callback(key, func)
        self.right_leader.add_callback(key, func)

    def reset(self):
        self.left_leader.reset()
        self.right_leader.reset()
        self._base_command[:] = 0.0
        self._pressed_keys.clear()

    @property
    def started(self) -> bool:
        return self.left_leader.started

    @property
    def reset_state(self) -> bool:
        return self.left_leader.reset_state or self.right_leader.reset_state

    @reset_state.setter
    def reset_state(self, value: bool):
        self.left_leader.reset_state = value
        self.right_leader.reset_state = value

    def get_device_state(self):
        return {
            "base": self._base_command,
            "left_arm": self.left_leader.get_device_state(),
            "right_arm": self.right_leader.get_device_state(),
        }

    def input2action(self):
        state = {}
        reset = state["reset"] = self.reset_state
        state['started'] = self.started
        if reset:
            self.reset_state = False
            return state

        state['joint_state'] = self.get_device_state()

        ac_dict = {
            "reset": reset,
            "started": self.started,
            "xlerobot": True,
        }
        if reset:
            return ac_dict

        ac_dict['joint_state'] = {
            "base": state['joint_state']['base'],
            "left_arm": state['joint_state']['left_arm'],
            "right_arm": state['joint_state']['right_arm'],
            "head_pan": np.asarray([0.0], dtype=np.float32),
        }
        ac_dict['motor_limits'] = {
            'left_arm': self.left_leader.motor_limits,
            'right_arm': self.right_leader.motor_limits,
        }
        return ac_dict

    def _on_keyboard_event(self, event, *args, **kwargs):
        key_name = event.input.name if hasattr(event.input, "name") else event.input
        print(colored(self._base_command, "magenta", attrs=["bold"]))
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if key_name in self._BASE_KEY_MAPPING:
                if key_name not in self._pressed_keys:
                    self._base_command += self._BASE_KEY_MAPPING[key_name]
                    self._pressed_keys.add(key_name)
                    np.clip(self._base_command, -10.0, 10.0, out=self._base_command)
            elif key_name == "SPACE":
                self._base_command[:] = 0.0
                self._pressed_keys.clear()
            elif key_name == "B":
                self.left_leader._started = True
                self.left_leader._reset_state = False
                self.right_leader._reset_state = False
                self.right_leader._started = True
            elif key_name == "R":
                self.left_leader._started = False
                self.left_leader._reset_state = True
                self.right_leader._reset_state = True
                self.right_leader._started = False
                callback = self.left_leader._additional_callbacks.get("R")
                if callback is not None:
                    callback()
            elif key_name == "N":
                self.left_leader._started = False
                self.left_leader._reset_state = True
                self.right_leader._reset_state = True
                self.right_leader._started = False
                callback = self.left_leader._additional_callbacks.get("N")
                if callback is not None:
                    callback()
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if key_name in self._BASE_KEY_MAPPING and key_name in self._pressed_keys:
                self._base_command[:] = 0.0
                self._pressed_keys.remove(key_name)
                np.clip(self._base_command, -10.0, 10.0, out=self._base_command)

        if not self._pressed_keys:
            # clear residual drift when no key is active
            self._base_command[:] = 0.0
        return True
