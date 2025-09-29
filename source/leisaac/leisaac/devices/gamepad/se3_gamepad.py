import weakref
import numpy as np

from collections.abc import Callable
from typing import Optional

import carb
import omni

from ..device_base import Device


class Se3Gamepad(Device):
    """A gamepad teleoperation interface that commands Cartesian twists.

    For bi-arm tasks, use L1 and R1 to select which arm to control.
    The mappings are:
        ============================== =================
        Description                    Button/Stick
        ============================== =================
        Select left arm                L1 (press)
        Select right arm               R1 (press)
        Gripper open                   L2 (trigger)
        Gripper close                  R2 (trigger)
        Move forward/backward (x)      Left stick up/down
        Move left/right (y)            Left stick left/right
        Move up/down (z)               Right stick up/down
        Rotate about x (roll)          D-pad up/down
        Rotate about y (pitch)         D-pad left/right
        Rotate about z (yaw)           Right stick left/right
        ============================== =================

    """

    def __init__(
        self,
        env,
        sensitivity: float = 0.02,
        rotation_sensitivity: Optional[float] = None,
        gripper_sensitivity: Optional[float] = None,
    ):
        super().__init__(env)
        """Initialize the gamepad layer.
        """
        # disable simulator gamepad camera control
        carb_settings_iface = carb.settings.get_settings()
        carb_settings_iface.set_bool("/persistent/app/omniverse/gamepadCameraControl", False)
        # store inputs
        self.translation_sensitivity = sensitivity
        self.rotation_sensitivity = rotation_sensitivity if rotation_sensitivity is not None else sensitivity
        self.gripper_sensitivity = gripper_sensitivity if gripper_sensitivity is not None else sensitivity

        # detect arm configuration (single vs bi-arm)
        bi_arm_actions = getattr(env.cfg, "actions", None)
        has_left_arm = hasattr(bi_arm_actions, "left_arm_action")
        has_right_arm = hasattr(bi_arm_actions, "right_arm_action")
        self._num_arms = 2 if has_left_arm and has_right_arm else 1
        self._dofs_per_arm = 6

        # command buffers
        self._delta_twist = np.zeros(self._num_arms * self._dofs_per_arm)
        self._delta_gripper = np.zeros(self._num_arms)

        # acquire omniverse interfaces
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._gamepad = self._appwindow.get_gamepad(0)
        # note: Use weakref on callbacks to ensure that this object can be deleted when its destructor is called.
        self._gamepad_sub = self._input.subscribe_to_gamepad_events(
            self._gamepad,
            lambda event, *args, obj=weakref.proxy(self): obj._on_gamepad_event(event, *args),
        )

        # input values buffer
        self._input_values = {}

        # current arm
        self._current_arm = 0

        # some flags and callbacks
        self.started = False
        self._reset_state = 0
        self._additional_callbacks = {}

    def __del__(self):
        """Release the gamepad interface."""
        self._input.unsubscribe_to_gamepad_events(self._gamepad, self._gamepad_sub)
        self._gamepad_sub = None

    def __str__(self) -> str:
        """Returns: A string containing the information of gamepad."""
        msg = "Gamepad Controller for SE(3).\n"
        msg += f"\tGamepad name: {self._input.get_gamepad_name(self._gamepad)}\n"
        msg += "\t----------------------------------------------\n"
        msg += "\tSelect left arm:                     L1\n"
        msg += "\tSelect right arm:                    R1\n"
        msg += "\tGripper open:                        L2\n"
        msg += "\tGripper close:                       R2\n"
        msg += "\tMove forward/backward (x):           Left Stick Up / Down\n"
        msg += "\tMove left/right (y):                 Left Stick Left / Right\n"
        msg += "\tMove up/down (z):                    Right Stick Up / Down\n"
        msg += "\tRotate about x (roll):               D-pad Up / Down\n"
        msg += "\tRotate about y (pitch):              D-pad Left / Right\n"
        msg += "\tRotate about z (yaw):                Right Stick Left / Right\n"
        msg += "\t----------------------------------------------\n"
        msg += "\tStart Control: Y\n"
        msg += "\tTask Failed and Reset: B\n"
        msg += "\tTask Success and Reset: A\n"
        msg += "\tControl+C: quit"
        return msg

    def get_device_state(self):
        # compute twist based on current input values
        self._delta_twist.fill(0)
        offset = self._current_arm * self._dofs_per_arm
        ls_up = self._input_values.get("LEFT_STICK_UP", 0.0)
        ls_down = self._input_values.get("LEFT_STICK_DOWN", 0.0)
        ls_left = self._input_values.get("LEFT_STICK_LEFT", 0.0)
        ls_right = self._input_values.get("LEFT_STICK_RIGHT", 0.0)
        rs_up = self._input_values.get("RIGHT_STICK_UP", 0.0)
        rs_down = self._input_values.get("RIGHT_STICK_DOWN", 0.0)
        rs_left = self._input_values.get("RIGHT_STICK_LEFT", 0.0)
        rs_right = self._input_values.get("RIGHT_STICK_RIGHT", 0.0)
        pad_up = self._input_values.get("PAD_UP", 0.0)
        pad_down = self._input_values.get("PAD_DOWN", 0.0)
        pad_left = self._input_values.get("PAD_LEFT", 0.0)
        pad_right = self._input_values.get("PAD_RIGHT", 0.0)
        self._delta_twist[offset + 0] = (ls_up - ls_down) * self.translation_sensitivity
        self._delta_twist[offset + 1] = (ls_left - ls_right) * self.translation_sensitivity
        self._delta_twist[offset + 2] = (rs_up - rs_down) * self.translation_sensitivity
        self._delta_twist[offset + 3] = (pad_up - pad_down) * self.rotation_sensitivity
        self._delta_twist[offset + 4] = (pad_left - pad_right) * self.rotation_sensitivity
        self._delta_twist[offset + 5] = (rs_left - rs_right) * self.rotation_sensitivity
        return self._delta_twist

    def get_gripper_state(self):
        # compute gripper based on current input values
        self._delta_gripper.fill(0)
        lt = self._input_values.get("LEFT_TRIGGER", 0.0)
        rt = self._input_values.get("RIGHT_TRIGGER", 0.0)
        self._delta_gripper[self._current_arm] = (lt - rt) * self.gripper_sensitivity
        return self._delta_gripper

    def input2action(self):
        state = {}
        reset = state["reset"] = self._reset_state
        state['started'] = self.started
        if reset:
            self._reset_state = False
            return state
        state['ee_twist'] = self.get_device_state()
        state['gripper'] = self.get_gripper_state()

        ac_dict = {}
        ac_dict["reset"] = reset
        ac_dict['started'] = self.started
        ac_dict['keyboard'] = True  # Note: This is leftover from keyboard; can change to False or remove if needed
        if reset:
            return ac_dict
        ac_dict['ee_twist'] = state['ee_twist']
        ac_dict['joint_state'] = state['ee_twist']  # backward compatibility
        ac_dict['gripper'] = state['gripper']
        return ac_dict

    def reset(self):
        self._delta_twist = np.zeros(self._num_arms * self._dofs_per_arm)
        self._delta_gripper = np.zeros(self._num_arms)
        # Note: Not clearing _input_values to preserve hardware state

    def add_callback(self, key: str, func: Callable):
        self._additional_callbacks[key] = func

    def _on_gamepad_event(self, event, *args, **kwargs):
        # update the current value for the input
        self._input_values[event.input.name] = event.value

        # handle arm selection on button press
        if event.input == carb.input.GamepadInput.LEFT_SHOULDER and event.value == 1.0:
            self._current_arm = 0
        elif event.input == carb.input.GamepadInput.RIGHT_SHOULDER and event.value == 1.0:
            self._current_arm = 1

        # handle start / reset
        if event.input == carb.input.GamepadInput.Y and event.value == 1.0:
            self.started = True
            self._reset_state = False
        elif event.input == carb.input.GamepadInput.B and event.value == 1.0:
            self.started = False
            self._reset_state = True
            if "R" in self._additional_callbacks:
                self._additional_callbacks["R"]()
        elif event.input == carb.input.GamepadInput.A and event.value == 1.0:
            self.started = False
            self._reset_state = True
            if "N" in self._additional_callbacks:
                self._additional_callbacks["N"]()

        return True