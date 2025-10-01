import weakref
import numpy as np

from collections.abc import Callable
from typing import Optional

import carb
import omni

from ..device_base import Device


class Se3Keyboard(Device):
    """A keyboard teleoperation interface that commands Cartesian twists.

    For single-arm tasks, the default bindings apply velocity commands to the
    end-effector frame:
        ============================== ================= =================
        Description                    Key (+ve axis)    Key (-ve axis)
        ============================== ================= =================
        Forward / Backward (x)         W                 S
        Left / Right (y)               A                 D
        Up / Down (z)                  Q                 E
        Rotate about x (roll)          F                 V
        Rotate about y (pitch)         T                 G
        Rotate about z (yaw)           Y                 H
        Open / Close gripper           Z                 X
        ============================== ================= =================

    For bi-arm tasks, the left-arm bindings remain the same and an additional
    block of keys operates the right arm:
        ============================== ================= =================
        Description                    Key (+ve axis)    Key (-ve axis)
        ============================== ================= =================
        Forward / Backward (x)         I                 K
        Left / Right (y)               J                 L
        Up / Down (z)                  U                 O
        Rotate about x (roll)          P                 ;
        Rotate about y (pitch)         ' (apostrophe)    / (slash)
        Rotate about z (yaw)           . (period)        , (comma)
        Open / Close gripper           [ (left bracket)  ] (right bracket)
        ============================== ================= =================

    """

    def __init__(
        self,
        env,
        sensitivity: float = 0.05,
        rotation_sensitivity: Optional[float] = None,
        gripper_sensitivity: Optional[float] = None,
    ):
        super().__init__(env)
        """Initialize the keyboard layer.
        """
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
        self._keyboard = self._appwindow.get_keyboard()
        # note: Use weakref on callbacks to ensure that this object can be deleted when its destructor is called.
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            lambda event, *args, obj=weakref.proxy(self): obj._on_keyboard_event(event, *args),
        )
        # bindings for keyboard to command
        self._create_key_bindings()

        # some flags and callbacks
        self.started = False
        self._reset_state = 0
        self._additional_callbacks = {}

    def __del__(self):
        """Release the keyboard interface."""
        self._input.unsubscribe_to_keyboard_events(self._keyboard, self._keyboard_sub)
        self._keyboard_sub = None

    def __str__(self) -> str:
        """Returns: A string containing the information of joystick."""
        msg = "Keyboard Controller for SE(3).\n"
        msg += f"\tKeyboard name: {self._input.get_keyboard_name(self._keyboard)}\n"
        msg += "\t----------------------------------------------\n"
        msg += "\tLeft arm forward/backward:           W / S\n"
        msg += "\tLeft arm left/right:                 A / D\n"
        msg += "\tLeft arm up/down:                    Q / E\n"
        msg += "\tLeft arm rotate about x:             F / V\n"
        msg += "\tLeft arm rotate about y:             T / G\n"
        msg += "\tLeft arm rotate about z:             Y / H\n"
        msg += "\tLeft gripper open/close:             Z / X\n"
        if self._num_arms == 2:
            msg += "\tRight arm forward/backward:          I / K\n"
            msg += "\tRight arm left/right:                J / L\n"
            msg += "\tRight arm up/down:                   U / O\n"
            msg += "\tRight arm rotate about x:            P / ;\n"
            msg += "\tRight arm rotate about y:            ' (apostrophe) / (slash)\n"
            msg += "\tRight arm rotate about z:            . (period) / , (comma)\n"
            msg += "\tRight gripper open/close:            [ / ]\n"
        msg += "\t----------------------------------------------\n"
        msg += "\tStart Control: B\n"
        msg += "\tTask Failed and Reset: R\n"
        msg += "\tTask Success and Reset: N\n"
        msg += "\tControl+C: quit"
        return msg

    def get_device_state(self):
        return self._delta_twist

    def get_gripper_state(self):
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
        ac_dict['keyboard'] = True
        if reset:
            return ac_dict
        ac_dict['ee_twist'] = state['ee_twist']
        ac_dict['joint_state'] = state['ee_twist']  # backward compatibility
        ac_dict['gripper'] = state['gripper']
        return ac_dict

    def reset(self):
        self._delta_twist = np.zeros(self._num_arms * self._dofs_per_arm)
        self._delta_gripper = np.zeros(self._num_arms)

    def add_callback(self, key: str, func: Callable):
        self._additional_callbacks[key] = func

    def _on_keyboard_event(self, event, *args, **kwargs):
        # apply the command when pressed
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name in self._INPUT_KEY_MAPPING.keys():
                self._delta_twist += self._INPUT_KEY_MAPPING[event.input.name]
            elif event.input.name in self._GRIPPER_KEY_MAPPING.keys():
                self._delta_gripper += self._GRIPPER_KEY_MAPPING[event.input.name]
            elif event.input.name == "B":
                self.started = True
                self._reset_state = False
            elif event.input.name == "R":
                self.started = False
                self._reset_state = True
                if "R" in self._additional_callbacks:
                    self._additional_callbacks["R"]()
            elif event.input.name == "N":
                self.started = False
                self._reset_state = True
                if "N" in self._additional_callbacks:
                    self._additional_callbacks["N"]()
        # remove the command when un-pressed
        if event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if event.input.name in self._INPUT_KEY_MAPPING.keys():
                self._delta_twist -= self._INPUT_KEY_MAPPING[event.input.name]
            elif event.input.name in self._GRIPPER_KEY_MAPPING.keys():
                self._delta_gripper -= self._GRIPPER_KEY_MAPPING[event.input.name]
        return True

    def _create_key_bindings(self):
        """Creates default key binding."""
        self._INPUT_KEY_MAPPING = {}
        self._GRIPPER_KEY_MAPPING = {}


        def _register_keys(aliases_per_dof, offset, sign):
            for idx, aliases in enumerate(aliases_per_dof):
                if not aliases:
                    continue
                scale = self.translation_sensitivity if idx < 3 else self.rotation_sensitivity
                for key in aliases:
                    vec = np.zeros(self._num_arms * self._dofs_per_arm)
                    vec[offset + idx] = sign * scale
                    self._INPUT_KEY_MAPPING[key] = vec

        left_positive_aliases = [["W"], ["A"], ["Q"], ["F"], ["T"], ["Y"]]
        left_negative_aliases = [["S"], ["D"], ["E"], ["V"], ["G"], ["H"]]
        _register_keys(left_positive_aliases, 0, 1.0)
        _register_keys(left_negative_aliases, 0, -1.0)

        if self._num_arms == 2:
            right_positive_aliases = [
                ["I"],
                ["J"],
                ["U"],
                ["P"],
                ["APOSTROPHE", "'", "QUOTE", "OEM_APOSTROPHE", "OEM_QUOTE", "OEM_7"],
                ["PERIOD", ".", "DOT", "OEM_PERIOD"],
            ]
            right_negative_aliases = [
                ["K"],
                ["L"],
                ["O"],
                ["SEMICOLON", ";", "OEM_SEMICOLON", "OEM_1"],
                ["SLASH", "/", "FORWARD_SLASH", "OEM_SLASH", "OEM_2"],
                ["COMMA", ",", "OEM_COMMA"],
            ]
            _register_keys(right_positive_aliases, self._dofs_per_arm, 1.0)
            _register_keys(right_negative_aliases, self._dofs_per_arm, -1.0)

        def _register_gripper_keys(aliases, arm_index, sign):
            for key in aliases:
                vec = np.zeros(self._num_arms)
                vec[arm_index] = sign * self.gripper_sensitivity
                self._GRIPPER_KEY_MAPPING[key] = vec

        _register_gripper_keys(["Z"], 0, 1.0)
        _register_gripper_keys(["X"], 0, -1.0)

        if self._num_arms == 2:
            _register_gripper_keys(["LEFT_BRACKET", "BRACKETLEFT", "[", "OEM_4"], 1, 1.0)
            _register_gripper_keys(["RIGHT_BRACKET", "BRACKETRIGHT", "]", "OEM_6"], 1, -1.0)
