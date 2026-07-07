import abc
from typing import Dict


class BasePolicy(abc.ABC):
    @abc.abstractmethod
    def infer(self, obs: Dict, **rtc_kwargs) -> Dict:
        """Infer actions from observations.

        Args:
            obs: Observation dictionary.
            **rtc_kwargs: Optional RTC parameters (prev_chunk_left_over,
                inference_delay, execution_horizon). Ignored by non-RTC policies.
        """

    def reset(self) -> None:
        """Reset the policy to its initial state."""
        pass
