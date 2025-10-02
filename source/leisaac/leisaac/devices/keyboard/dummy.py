"""Placeholder teleoperation device that intentionally refrains from commanding actions."""

import weakref

from collections.abc import Callable
from typing import Any

import carb
import omni

from ..device_base import Device


class DummyTeleop(Device):
    """A no-op teleoperation interface used as a placeholder."""

    def __init__(self, env) -> None:
        super().__init__(env)
        self.started = False
        self._callbacks: dict[Any, Callable] = {}

        # Acquire the keyboard interface so registered callbacks (e.g. R/N) still fire.
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            lambda event, *args, obj=weakref.proxy(self): obj._on_keyboard_event(event, *args),
        )

    def __del__(self) -> None:  # pragma: no cover - defensive cleanup
        try:
            self._input.unsubscribe_to_keyboard_events(self._keyboard, self._keyboard_sub)
        except AttributeError:
            pass

    def __str__(self) -> str:  # pragma: no cover - simple metadata method
        return "Dummy teleop device (no inputs)."

    def reset(self) -> None:
        self.started = False

    def add_callback(self, key: Any, func: Callable) -> None:
        self._callbacks[key] = func

    def advance(self):  # noqa: D401 - signature mandated by base class
        """Return no command so the environment keeps idling."""
        return None

    def _on_keyboard_event(self, event, *args) -> bool:
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            callback = self._callbacks.get(event.input.name)
            if callback is not None:
                callback()
        return True
