import json
import random
import dataclasses
from typing import Optional, Union

_no_zmq: Union[bool, str] = False
try:
    from ball_launcher_beepy import BallLauncherClient
except ModuleNotFoundError as me:
    if "zmq" in str(me):
        _no_zmq = str(me)
        pass
    else:
        raise me


@dataclasses.dataclass
class BallLauncherConfig:
    """Configuration of a ball launcher"""

    IP: str
    port: int
    phi: tuple[float, float]
    theta: tuple[float, float]
    top_left_motor: tuple[float, float]
    top_right_motor: tuple[float, float]
    bottom_motor: tuple[float, float]


class BallLauncher:
    _ranges: tuple[str, ...] = (
        "phi",
        "theta",
        "top_left_motor",
        "top_right_motor",
        "bottom_motor",
    )

    def __init__(self, config: BallLauncherConfig) -> None:
        if _no_zmq:
            raise ModuleNotFoundError(_no_zmq)
        self._config = config
        self._client = BallLauncherClient(self._config.IP, self._config.port)

    def launch(
        self,
        phi: Optional[float] = None,
        theta: Optional[float] = None,
        top_left_motor: Optional[float] = None,
        top_right_motor: Optional[float] = None,
        bottom_motor: Optional[float] = None,
    ) -> None:
        """
        Launch a ball using either random parameters (if corresponding argument
        is None) or the specified value.
        """

        def _get_value(key: str, locals_: dict, config: BallLauncherConfig) -> float:
            l = locals_[key]
            if l is not None:
                return l
            return random.uniform(*getattr(self._config, key))

        locals_ = locals()

        values: dict[str, float] = {
            key: _get_value(key, locals_, self._config) for key in self._ranges
        }
        self._client.set_state(**values)
        self._client.launch_ball()
