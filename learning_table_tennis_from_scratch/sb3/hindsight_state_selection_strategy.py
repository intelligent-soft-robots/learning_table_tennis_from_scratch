from enum import Enum


class HindsightStateSelectionStrategy(Enum):
    """
    The strategies for selecting new hindsight when
    creating artificial transitions.
    """

    RANDOM = 0
    REWARD = 1
    ADVANTAGE = 2
    ACHIEVED_GOAL = 3
    REWARD_ACHIEVED_GOAL_002 = 4

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
    "achieved_goal": HindsightStateSelectionStrategy.ACHIEVED_GOAL,
    "reward_achieved_goal_002": HindsightStateSelectionStrategy.REWARD_ACHIEVED_GOAL_002,
}

KEY_TO_HINDSIGHT_STATE_STRATEGY_HORIZON = {
    "step": HindsightStateSelectionStrategyHorizon.STEP,
    "future": HindsightStateSelectionStrategyHorizon.FUTURE,
    "episode": HindsightStateSelectionStrategyHorizon.EPISODE,
}