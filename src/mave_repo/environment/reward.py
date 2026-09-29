from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from mave_repro.chemistry.reaction import Reaction
from mave_repro.environment.state import RetroState


class RewardFunction(ABC):
    """
    Interface for retrosynthetic task reward R(s, a).

    IMPORTANT
    ---------
    The paper defines R(s_t, a_t) but does not uniquely specify
    its numerical implementation. Concrete reward functions here
    are therefore reproduction choices unless later evidence
    establishes the original definition.
    """

    @abstractmethod
    def step_reward(
        self,
        state: RetroState,
        reaction: Reaction,
        next_state: RetroState,
        *,
        terminated: bool,
    ) -> float:
        """
        Reward for one reaction transition.
        """

    def failure_reward(
        self,
        state: RetroState,
    ) -> float:
        """
        Reward assigned when planning terminates externally
        without finding a complete route, e.g. search budget
        exhaustion.
        """
        return 0.0


@dataclass(frozen=True, slots=True)
class SparseTerminalReward(
    RewardFunction
):
    """
    Simple sparse success reward.

    This is a REPRODUCTION CHOICE, not a paper-specified reward.

        non-terminal transition -> step_reward_value
        solved terminal state   -> success_reward
        search failure          -> failure_reward_value
    """

    success_reward: float = 1.0
    step_reward_value: float = 0.0
    failure_reward_value: float = 0.0

    def step_reward(
        self,
        state: RetroState,
        reaction: Reaction,
        next_state: RetroState,
        *,
        terminated: bool,
    ) -> float:

        if terminated:
            return float(
                self.success_reward
            )

        return float(
            self.step_reward_value
        )

    def failure_reward(
        self,
        state: RetroState,
    ) -> float:
        return float(
            self.failure_reward_value
        )


@dataclass(frozen=True, slots=True)
class ZeroReward(
    RewardFunction
):
    """
    Debug reward function.
    """

    def step_reward(
        self,
        state: RetroState,
        reaction: Reaction,
        next_state: RetroState,
        *,
        terminated: bool,
    ) -> float:
        return 0.0


def discounted_return(
    rewards: list[float]
    | tuple[float, ...],
    *,
    gamma: float,
) -> float:
    """
    Compute

        sum_t gamma^t * reward_t

    Used to construct downstream returns for escalation and
    reaction-policy training.
    """

    if not 0.0 <= gamma <= 1.0:
        raise ValueError(
            "gamma must lie in [0, 1]."
        )

    total = 0.0
    discount = 1.0

    for reward in rewards:
        total += (
            discount
            * float(reward)
        )

        discount *= gamma

    return total


def discounted_returns_to_go(
    rewards: list[float]
    | tuple[float, ...],
    *,
    gamma: float,
) -> tuple[float, ...]:
    """
    Return discounted return-to-go for every trajectory step.

    For rewards [R_t, ..., R_T], output:

        G_t, G_{t+1}, ..., G_T
    """

    if not 0.0 <= gamma <= 1.0:
        raise ValueError(
            "gamma must lie in [0, 1]."
        )

    output = [
        0.0
    ] * len(rewards)

    running = 0.0

    for index in range(
        len(rewards) - 1,
        -1,
        -1,
    ):
        running = (
            float(rewards[index])
            + gamma * running
        )

        output[index] = running

    return tuple(output)