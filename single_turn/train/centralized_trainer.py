"""Opt-in centralized actors; existing Travel trainers are not modified."""

from comlrl.trainers.reinforce import CentralizedMAGRPOTrainer
from comlrl.utils.distributed import unwrap_model

from single_turn.centralized_comparator import joint_slots
from single_turn.centralized_native import (
    SectionedFixedSlotJSONLogitsProcessor,
    build_native_prefill,
    native_slots,
)
from single_turn.train.preference_trainer import (
    TravelMADPOTrainer,
    TravelMADPOIterTrainer,
    TravelMARLHFTrainer,
    TravelMARLHFIterTrainer,
)
from single_turn.train.structured_trainer import StructuredOutputMAGRPOTrainer


class TravelCentralizedActorMixin:
    """Retain joint trajectories for learning and Travel's value-only loss."""

    def __init__(self, *args, **kwargs):
        config = kwargs.get("args")
        if getattr(config, "collaboration_mode", None) != "centralized":
            raise ValueError("Use centralized configuration with this trainer.")
        super().__init__(*args, **kwargs)
        if self.num_agents != 1 or len(self.agents) != 1 or len(self.optimizers) != 1:
            raise RuntimeError(
                "Centralized Travel must construct exactly one actor/optimizer."
            )
        if not self.constrain_json_skeleton:
            raise ValueError("Centralized Travel requires the joint JSON grammar.")
        # Preserve Travel's bounded diagnostic queue through the core reward
        # adapter. This does not change the reward or its numeric range.
        runtime = self.centralized_collaboration
        drain = getattr(runtime.reward_func, "drain_details", None)
        if callable(drain):
            runtime.drain_details = drain

    def _generate_completions(
        self,
        agent,
        batch_items,
        agent_idx=0,
        num_return_sequences=1,
        max_new_tokens=2048,
        prompts_override=None,
        external_prompts=None,
        **kwargs,
    ):
        if agent_idx != 0 or len(batch_items) != 1:
            raise ValueError(
                "Centralized Travel generates one joint prompt with actor 0."
            )
        if "slot_order_override" in kwargs:
            raise ValueError(
                "The centralized actor owns all joint slots; do not override them."
            )
        prompts = self._resolve_prompts(
            batch_items,
            agent_idx=0,
            prompts_override=prompts_override,
            external_prompts=external_prompts,
        )
        tokenizer = self.tokenizers[0]
        prompt_length = len(
            tokenizer.encode(
                prompts[0] + self._travel_assistant_prefill(0, batch_items[0]),
                add_special_tokens=True,
            )
        )
        model_limit = getattr(
            unwrap_model(agent).config, "max_position_embeddings", None
        )
        if model_limit and prompt_length + max_new_tokens > int(model_limit):
            raise ValueError(
                "Centralized Travel prompt plus output budget exceeds the model context window; refusing truncation."
            )
        tokenizer_limit = getattr(tokenizer, "model_max_length", None)
        if tokenizer_limit and prompt_length > int(tokenizer_limit):
            raise ValueError(
                "Centralized Travel prompt exceeds tokenizer context window; refusing truncation."
            )
        return super()._generate_completions(
            agent,
            batch_items,
            agent_idx=0,
            num_return_sequences=num_return_sequences,
            max_new_tokens=max_new_tokens,
            prompts_override=prompts,
            external_prompts=None,
            slot_order_override=self._centralized_slot_order(batch_items[0]["days"]),
            **kwargs,
        )

    def _centralized_slot_order(self, days):
        return joint_slots(days)

    def _log_eval_metrics(self, all_agent_completions_turns, *args, **kwargs):
        # Travel's compact logger precedes the core logger in the MRO. Split
        # here exactly once, using the items recorded by the core eval loop.
        roles = self.centralized_collaboration.split_eval(
            all_agent_completions_turns, self._centralized_eval_items
        )
        return super()._log_eval_metrics(roles, *args, **kwargs)


class TravelCentralizedPreferenceMixin(TravelCentralizedActorMixin):
    def _build_centralized_comparator_prompt(self, batch_item):
        # Actor and teacher must use IDENTICAL chat context and assistant
        # prefill; no second wrapping and no role-projected replay tokens.
        return self.formatters[0](batch_item)

    def _generate_centralized_policy_output_for_agent(
        self,
        policy_agent,
        batch_item,
        *,
        prompt,
        num_candidates,
        **kwargs,
    ):
        unused = [
            i
            for i, actor in enumerate(self.agents)
            if unwrap_model(actor) is not unwrap_model(policy_agent)
        ]
        with self._offload_actors(unused), self._comparator_rng(policy_agent):
            result = self._generate_completions(
                policy_agent,
                [batch_item],
                agent_idx=0,
                num_return_sequences=num_candidates,
                max_new_tokens=self.args.max_new_tokens,
                prompts_override=[prompt],
                **kwargs,
            )
        # Includes the exact joint tokens/masks cached by TravelPreferenceMixin
        # and the same prefilled prompt as target candidates. Do not split here.
        return [result]

    @staticmethod
    def _preference_pair_to_record(pair):
        record = TravelMADPOIterTrainer._preference_pair_to_record(pair)
        record["travel_collaboration_mode"] = "centralized"
        return record

    def _preference_pair_from_record(self, record):
        if record.get("travel_collaboration_mode") != "centralized":
            raise ValueError(
                "Centralized actors require their own fresh joint-policy replay directory."
            )
        return super()._preference_pair_from_record(record)


class TravelCentralizedMAGRPOTrainer(
    TravelCentralizedActorMixin, StructuredOutputMAGRPOTrainer, CentralizedMAGRPOTrainer
):
    pass


class TravelNativeCentralizedActorMixin(TravelCentralizedActorMixin):
    """Generate the logistics answer, then the experience answer, in one call."""

    def _centralized_slot_order(self, days):
        return native_slots(days)

    def _travel_assistant_prefill(self, agent_idx, batch_item):
        return build_native_prefill(int(batch_item.get("days", 0)))

    def _travel_skeleton_processor(
        self, tokenizer, *, prompt_length, slots, max_new_tokens
    ):
        return SectionedFixedSlotJSONLogitsProcessor(
            tokenizer,
            prompt_length=prompt_length,
            slots=slots,
            max_value_tokens=int(getattr(self, "max_value_tokens", 32)),
            max_new_tokens=max_new_tokens,
        )


class TravelNativeCentralizedMAGRPOTrainer(
    TravelNativeCentralizedActorMixin,
    StructuredOutputMAGRPOTrainer,
    CentralizedMAGRPOTrainer,
):
    pass


class TravelCentralizedMADPOTrainer(
    TravelCentralizedPreferenceMixin, TravelMADPOTrainer
):
    pass


class TravelCentralizedMADPOIterTrainer(
    TravelCentralizedPreferenceMixin, TravelMADPOIterTrainer
):
    pass


class TravelCentralizedMARLHFTrainer(
    TravelCentralizedPreferenceMixin, TravelMARLHFTrainer
):
    pass


class TravelCentralizedMARLHFIterTrainer(
    TravelCentralizedPreferenceMixin, TravelMARLHFIterTrainer
):
    pass
