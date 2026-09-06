"""Joint-policy adapter, separate from the two-actor comparator-only path."""

from comlrl.trainers.preference.centralized import CentralizedComparatorParseError

from single_turn.centralized_comparator import TravelCentralizedComparatorAdapter
from single_turn.formatting import wrap_formatter_with_chat_template


CENTRAL_SYSTEM_PROMPT = (
    "You are one centralized travel planner. Plan logistics and experiences "
    "jointly and return only the requested fixed-schema JSON."
)


class TravelCentralizedCollaborationAdapter(TravelCentralizedComparatorAdapter):
    """One chat prompt and joint action; split only at the task boundary."""

    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer

    def build_prompt(self, batch_item, agent_prompts):
        raw = super().build_prompt(batch_item, agent_prompts)
        if self.tokenizer is None:
            return raw  # Data/contract dry runs do not need model downloads.
        formatter = wrap_formatter_with_chat_template(
            lambda item: raw, self.tokenizer, system_prompt=CENTRAL_SYSTEM_PROMPT
        )
        return formatter(batch_item)

    def parse_completion(self, completion, batch_item, num_agents):
        if num_agents != 2:
            raise ValueError("Centralized Travel requires two logical roles.")
        try:
            return super().parse_completion(completion, batch_item, num_agents)
        except (ValueError, TypeError, KeyError) as exc:
            # A truncated/malformed sample is a failed task action, not a reason
            # to crash training. CoMLRL maps this exception to two empty roles.
            raise CentralizedComparatorParseError(str(exc)) from exc
