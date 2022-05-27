from enum import Enum


class HindsightStateSelectionStrategy(Enum):
    """
    The strategies for selecting new hindsight when
    creating artificial transitions.
    """

    RANDOM = 0
    REWARD = 1
    ADVANTAGE = 2

class HindsightStateSelectionStrategyHorizon(Enum):
    """
    The strategies for selecting new hindsight when
    creating artificial transitions.
    """

    STEP = 0
    FUTURE = 1
    EPISODE = 2


# For convenience
# that way, we can use string to select a strategy
KEY_TO_HINDSIGHT_STATE_STRATEGY = {
    "random": HindsightStateSelectionStrategy.RANDOM,
    "reward": HindsightStateSelectionStrategy.REWARD,
    "advantage": HindsightStateSelectionStrategy.ADVANTAGE,
}

KEY_TO_HINDSIGHT_STATE_STRATEGY_HORIZON = {
    "step": HindsightStateSelectionStrategyHorizon.STEP,
    "future": HindsightStateSelectionStrategyHorizon.FUTURE,
    "episode": HindsightStateSelectionStrategyHorizon.EPISODE,
}