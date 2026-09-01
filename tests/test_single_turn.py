from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from single_turn.aggregation import (
    PLAN_FIELDS,
    merge_agent_assignments,
    owned_slots,
    slot_owner,
)
from single_turn.data import normalize_travelplanner_row, partition_rows
from single_turn.formatting import get_single_turn_formatters
from single_turn.parsing import parse_assignments
from single_turn.rewards.single_turn_reward import (
    TravelJointReward,
    score_single_turn_response,
)


def fixture_item():
    plan = [
        {
            "day": 1,
            "current_city": "from Alpha to Beta",
            "transportation": "Flight Number: F100, from Alpha to Beta",
            "breakfast": "-",
            "attraction": "Museum, Beta",
            "lunch": "Cafe, Beta",
            "dinner": "Bistro, Beta",
            "accommodation": "Hotel, Beta",
        }
    ]
    return {
        "id": "fixture",
        "prompt": "Plan a one-day trip from Alpha to Beta.",
        "query": "Plan a one-day trip from Alpha to Beta.",
        "org": "Alpha",
        "dest": "Beta",
        "days": 1,
        "gold_plan": plan,
        "reference_information": repr(
            [
                {"Description": "Attractions in Beta", "Content": "Museum Beta"},
                {"Description": "Restaurants in Beta", "Content": "Cafe Bistro Beta"},
                {"Description": "Accommodations in Beta", "Content": "Hotel Beta"},
                {"Description": "Flight Alpha Beta", "Content": "F100 Alpha Beta"},
            ]
        ),
        "reference_records": [
            {"Description": "All candidates", "Content": "Museum Cafe Bistro Hotel F100"}
        ],
        "local_constraint": {},
        "test": "",
        "entry_point": "",
    }


def assignment(day, field, value):
    return {"day": day, "field": field, "value": value}


def completion(agent_id, assignments):
    return json.dumps(
        {"agent_id": agent_id, "assignments": assignments},
        ensure_ascii=False,
    )


def exact_completions():
    logistics = [
        assignment(1, "current_city", "from Alpha to Beta"),
        assignment(
            1,
            "transportation",
            "Flight Number: F100, from Alpha to Beta",
        ),
        assignment(1, "accommodation", "Hotel, Beta"),
    ]
    experience = [
        assignment(1, "breakfast", "-"),
        assignment(1, "attraction", "Museum, Beta"),
        assignment(1, "lunch", "Cafe, Beta"),
        assignment(1, "dinner", "Bistro, Beta"),
    ]
    return [completion(0, logistics), completion(1, experience)]


def all_dash_completions():
    return [
        completion(
            0,
            [assignment(1, field, "-") for field in (
                "current_city",
                "transportation",
                "accommodation",
            )],
        ),
        completion(
            1,
            [assignment(1, field, "-") for field in (
                "breakfast",
                "attraction",
                "lunch",
                "dinner",
            )],
        ),
    ]


class ParsingTests(unittest.TestCase):
    def test_exact_strict_schema_is_accepted(self):
        parsed = parse_assignments(
            completion(0, [assignment(1, "breakfast", "-")]),
            expected_agent_id=0,
            capacity=1,
        )
        self.assertTrue(parsed.parse_success)
        self.assertTrue(parsed.strict_json)
        self.assertTrue(parsed.schema_valid)
        self.assertTrue(parsed.agent_id_match)
        self.assertTrue(parsed.capacity_valid)
        self.assertEqual(parsed.error_codes, ())
        self.assertEqual(parsed.assignments[0].field, "breakfast")

    def test_fenced_json_is_salvaged_but_not_strict(self):
        parsed = parse_assignments(
            '```json\n'
            '{"agent_id": 0, "assignments": '
            '[{"day": 1, "field": "breakfast", "value": "-"}]}\n'
            '```',
            expected_agent_id=0,
            capacity=1,
        )
        self.assertFalse(parsed.parse_success)
        self.assertTrue(parsed.decode_success)
        self.assertFalse(parsed.strict_json)
        self.assertIn("not_strict_json", parsed.error_codes)
        self.assertEqual(parsed.assignments[0].field, "breakfast")

    def test_trailing_text_and_multiple_objects_are_only_salvaged(self):
        payload = completion(0, [assignment(1, "breakfast", "-")])
        cases = {
            "trailing": payload + "\nHere is the answer.",
            "multiple": payload + "\n" + completion(0, []),
        }
        for label, text in cases.items():
            with self.subTest(label=label):
                parsed = parse_assignments(
                    text,
                    expected_agent_id=0,
                    capacity=1,
                )
                self.assertFalse(parsed.parse_success)
                self.assertTrue(parsed.decode_success)
                self.assertFalse(parsed.strict_json)
                self.assertEqual(len(parsed.assignments), 1)
                self.assertIn("not_strict_json", parsed.error_codes)

    def test_wrong_agent_id_is_recovered_but_invalid(self):
        parsed = parse_assignments(
            completion(1, [assignment(1, "breakfast", "-")]),
            expected_agent_id=0,
            capacity=1,
        )
        self.assertFalse(parsed.parse_success)
        self.assertTrue(parsed.decode_success)
        self.assertFalse(parsed.agent_id_match)
        self.assertEqual(len(parsed.assignments), 1)
        self.assertIn("agent_id_mismatch", parsed.error_codes)

    def test_over_capacity_is_recovered_but_invalid(self):
        parsed = parse_assignments(
            completion(
                0,
                [
                    assignment(1, "breakfast", "-"),
                    assignment(1, "lunch", "Cafe, Beta"),
                ],
            ),
            expected_agent_id=0,
            capacity=1,
        )
        self.assertFalse(parsed.parse_success)
        self.assertTrue(parsed.decode_success)
        self.assertFalse(parsed.capacity_valid)
        self.assertEqual(len(parsed.assignments), 2)
        self.assertIn("capacity_exceeded", parsed.error_codes)

    def test_duplicate_and_out_of_domain_slots_are_not_strict(self):
        duplicate = assignment(1, "breakfast", "-")
        parsed_duplicate = parse_assignments(
            completion(0, [duplicate, duplicate]),
            expected_agent_id=0,
            capacity=2,
            days=1,
            valid_fields=PLAN_FIELDS,
        )
        self.assertFalse(parsed_duplicate.parse_success)
        self.assertIn("self_duplicate", parsed_duplicate.error_codes)

        for invalid_assignment in (
            assignment(2, "breakfast", "-"),
            assignment(1, "unknown_field", "-"),
            assignment(1, "Breakfast", "-"),
            assignment(1, "breakfast", ""),
        ):
            with self.subTest(assignment=invalid_assignment):
                parsed_invalid = parse_assignments(
                    completion(0, [invalid_assignment]),
                    expected_agent_id=0,
                    capacity=1,
                    days=1,
                    valid_fields=PLAN_FIELDS,
                )
                self.assertFalse(parsed_invalid.parse_success)
                self.assertIn("assignment_value", parsed_invalid.error_codes)

    def test_non_json_fails_without_crashing(self):
        parsed = parse_assignments("I would visit the museum.")
        self.assertFalse(parsed.parse_success)
        self.assertFalse(parsed.decode_success)
        self.assertEqual(parsed.assignments, [])


class AggregationTests(unittest.TestCase):
    def test_different_values_for_same_slot_conflict(self):
        result = merge_agent_assignments(
            [
                completion(0, [assignment(1, "lunch", "Cafe A")]),
                completion(1, [assignment(1, "lunch", "Cafe B")]),
            ],
            days=1,
        )
        self.assertIn((1, "lunch"), result.conflict_slots)
        self.assertNotIn((1, "lunch"), result.merged_assignments)
        self.assertEqual(result.plan[0]["lunch"], "-")

    def test_same_value_overlap_is_merged_and_counted(self):
        result = merge_agent_assignments(
            [
                completion(0, [assignment(1, "lunch", "Cafe A")]),
                completion(1, [assignment(1, "lunch", "Cafe A")]),
            ],
            days=1,
        )
        self.assertEqual(result.merged_assignments[(1, "lunch")], "Cafe A")
        self.assertEqual(result.overlap_count, 1)

    def test_partition_has_one_owner_per_slot(self):
        for days in (1, 2, 3, 5, 7):
            with self.subTest(days=days):
                agent_0 = owned_slots(0, days)
                agent_1 = owned_slots(1, days)
                all_slots = {
                    (day, field)
                    for day in range(1, days + 1)
                    for field in PLAN_FIELDS
                }
                self.assertFalse(agent_0 & agent_1)
                self.assertEqual(agent_0 | agent_1, all_slots)
                self.assertLessEqual(abs(len(agent_0) - len(agent_1)), 1)
                self.assertTrue(
                    all(slot_owner(day, field, days) == 0 for day, field in agent_0)
                )
                self.assertTrue(
                    all(slot_owner(day, field, days) == 1 for day, field in agent_1)
                )

    def test_overcapacity_action_is_discarded_atomically(self):
        overcapacity = [
            assignment(1, "current_city", "from Alpha to Beta"),
            assignment(1, "transportation", "Flight Number: F100"),
            assignment(1, "breakfast", "-"),
            assignment(1, "attraction", "Museum, Beta"),
            assignment(1, "lunch", "Cafe, Beta"),
        ]
        result = merge_agent_assignments(
            [completion(0, overcapacity), completion(1, [])],
            days=1,
        )
        self.assertEqual(result.per_agent_assignments[0], {})
        self.assertFalse(result.parsed[0].capacity_valid)


class RewardTests(unittest.TestCase):
    def test_exact_complementary_plan_gets_max_phase_one_reward(self):
        reward, detail = score_single_turn_response(
            exact_completions(), batch_item=fixture_item()
        )
        self.assertAlmostEqual(reward, 1.2, places=6)
        self.assertEqual(detail["exact_match"], 1.0)
        self.assertEqual(detail["rewarded_exact_match"], 1.0)
        self.assertEqual(detail["team_action_valid"], 1.0)
        self.assertEqual(detail["cooperative_contribution"], 1.0)
        self.assertEqual(detail["coverage"], 1.0)
        self.assertEqual(detail["conflict_count"], 0.0)
        self.assertEqual(detail["role_score"], 1.0)

    def test_complementary_agents_beat_identical_agents(self):
        exact_reward, _ = score_single_turn_response(
            exact_completions(), batch_item=fixture_item()
        )
        logistics = json.loads(exact_completions()[0])["assignments"]
        duplicate_reward, detail = score_single_turn_response(
            [completion(0, logistics), completion(1, logistics)],
            batch_item=fixture_item(),
        )
        self.assertLess(duplicate_reward, exact_reward)
        self.assertGreater(detail["overlap_count"], 0)
        self.assertLess(detail["coverage"], 1.0)
        self.assertEqual(detail["cooperative_contribution"], 0.0)

    def test_all_dash_assignments_receive_low_reward(self):
        reward, detail = score_single_turn_response(
            all_dash_completions(), batch_item=fixture_item()
        )
        self.assertLessEqual(reward, -0.17)
        self.assertEqual(detail["team_action_valid"], 1.0)
        self.assertEqual(detail["cooperative_contribution"], 0.0)
        self.assertEqual(detail["coverage"], 0.0)
        self.assertEqual(detail["raw_coverage"], 1.0)
        self.assertEqual(detail["empty_recall"], 1.0)
        self.assertEqual(detail["empty_match_score"], 0.0)
        self.assertEqual(detail["empty_mismatch_rate"], 1.0)

    def test_one_sided_semantic_contribution_is_negative(self):
        experience = json.loads(all_dash_completions()[1])["assignments"]
        reward, detail = score_single_turn_response(
            [exact_completions()[0], completion(1, experience)],
            batch_item=fixture_item(),
        )
        self.assertLess(reward, 0.0)
        self.assertEqual(detail["contribution_ratios"], [1.0, 0.0])
        self.assertEqual(detail["cooperative_contribution"], 0.0)
        self.assertEqual(detail["contribution_deficit"], 1.0)

    def test_swapped_ownership_cannot_claim_exact_bonus(self):
        logistics = json.loads(exact_completions()[0])["assignments"]
        experience = json.loads(exact_completions()[1])["assignments"]
        reward, detail = score_single_turn_response(
            [completion(0, experience), completion(1, logistics)],
            batch_item=fixture_item(),
        )
        self.assertEqual(detail["exact_match"], 1.0)
        self.assertEqual(detail["rewarded_exact_match"], 0.0)
        self.assertEqual(detail["ownership_validity"], [0.0, 0.0])
        self.assertEqual(detail["team_action_valid"], 0.0)
        self.assertEqual(detail["slot_quality"], 1.0)
        self.assertEqual(detail["role_score"], 0.0)
        self.assertLess(reward, 1.2)

    def test_overcapacity_action_cannot_improve_exact_team(self):
        exact_reward, _ = score_single_turn_response(
            exact_completions(), batch_item=fixture_item()
        )
        logistics = json.loads(exact_completions()[0])["assignments"]
        overcapacity = logistics + [
            assignment(1, "attraction", "Museum, Beta"),
            assignment(1, "lunch", "Cafe, Beta"),
        ]
        reward, detail = score_single_turn_response(
            [completion(0, overcapacity), exact_completions()[1]],
            batch_item=fixture_item(),
        )
        self.assertLess(reward, exact_reward)
        self.assertEqual(detail["overcapacity_agent_rate"], 0.5)
        self.assertGreater(detail["extra_assignment_count"], 0.0)
        self.assertEqual(detail["agent_action_validity"][0], 0.0)

    def test_spurious_fill_of_gold_dash_is_worse_than_explicit_dash(self):
        exact_reward, _ = score_single_turn_response(
            exact_completions(), batch_item=fixture_item()
        )
        experience = json.loads(exact_completions()[1])["assignments"]
        experience[0] = assignment(1, "breakfast", "Cafe, Beta")
        reward, detail = score_single_turn_response(
            [exact_completions()[0], completion(1, experience)],
            batch_item=fixture_item(),
        )
        self.assertLess(reward, exact_reward)
        self.assertEqual(detail["exact_match"], 0.0)
        self.assertEqual(detail["correct_empty_slots"], 0.0)
        self.assertEqual(detail["empty_recall"], 0.0)
        self.assertEqual(detail["spurious_fill_rate"], 1.0)

    def test_one_grounded_slot_has_global_nonempty_coverage(self):
        one_slot = completion(
            0,
            [assignment(1, "current_city", "from Alpha to Beta")],
        )
        _, detail = score_single_turn_response(
            [one_slot, completion(1, [])],
            batch_item=fixture_item(),
        )
        self.assertAlmostEqual(detail["coverage"], 1.0 / 6.0)

    def test_trailing_explanation_loses_strict_team_bonus(self):
        exact_reward, _ = score_single_turn_response(
            exact_completions(), batch_item=fixture_item()
        )
        malformed = exact_completions()[0] + "\nThis is the completed assignment."
        reward, detail = score_single_turn_response(
            [malformed, exact_completions()[1]],
            batch_item=fixture_item(),
        )
        self.assertLess(reward, exact_reward)
        self.assertEqual(detail["parse_success"], 0.5)
        self.assertEqual(detail["rewarded_exact_match"], 0.0)

    def test_short_substring_cannot_exploit_grounding(self):
        logistics = [
            assignment(1, "current_city", "a"),
            assignment(1, "transportation", "a"),
            assignment(1, "accommodation", "a"),
        ]
        experience = [
            assignment(1, "breakfast", "a"),
            assignment(1, "attraction", "a"),
            assignment(1, "lunch", "a"),
            assignment(1, "dinner", "a"),
        ]
        reward, detail = score_single_turn_response(
            [
                completion(0, logistics),
                completion(1, experience),
            ],
            batch_item=fixture_item(),
        )
        self.assertEqual(detail["grounding"], 0.0)
        self.assertEqual(detail["coverage"], 0.0)
        self.assertLess(reward, 0.1)

    def test_joint_reward_returns_one_shared_scalar_per_sample(self):
        expected, _ = score_single_turn_response(
            exact_completions(), batch_item=fixture_item()
        )
        reward_model = TravelJointReward()
        rewards = reward_model(
            [exact_completions()[0]],
            [exact_completions()[1]],
            batch_items=[fixture_item()],
            prompts=[["agent 0 prompt"], ["agent 1 prompt"]],
        )
        self.assertEqual(len(rewards), 1)
        self.assertIsInstance(rewards[0], float)
        self.assertAlmostEqual(rewards[0], expected, places=6)
        self.assertAlmostEqual(reward_model.last_details[0]["reward"], expected)


class DataAndPromptTests(unittest.TestCase):
    def test_official_annotated_plan_shape_is_normalized(self):
        gold = fixture_item()["gold_plan"]
        raw = {
            "query": "Plan a one-day trip from Alpha to Beta.",
            "days": 1,
            "annotated_plan": repr([{"org": "Alpha"}, gold]),
            "reference_information": "[]",
            "local_constraint": "{}",
        }
        normalized = normalize_travelplanner_row(raw, 0)
        self.assertEqual(normalized["gold_plan"][0]["day"], 1)
        self.assertEqual(set(normalized["gold_plan"][0]) - {"day"}, set(PLAN_FIELDS))

    def test_missing_annotated_plan_is_rejected(self):
        raw = {
            "query": "Plan a one-day trip from Alpha to Beta.",
            "days": 1,
            "annotated_plan": "not a plan",
            "reference_information": "[]",
        }
        with self.assertRaisesRegex(ValueError, "requires annotated"):
            normalize_travelplanner_row(raw, 0)

    def test_partition_is_disjoint_and_reproducible(self):
        rows = [{"id": str(index)} for index in range(10)]
        train_a, eval_a = partition_rows(rows, train_samples=8, eval_samples=2, seed=7)
        train_b, eval_b = partition_rows(rows, train_samples=8, eval_samples=2, seed=7)
        self.assertEqual(train_a, train_b)
        self.assertEqual(eval_a, eval_b)
        self.assertFalse({x["id"] for x in train_a} & {x["id"] for x in eval_a})

    def test_role_prompts_share_task_but_have_distinct_guidance(self):
        item = fixture_item()
        prompts = [formatter(item) for formatter in get_single_turn_formatters(num_agents=2)]
        self.assertIn("LOGISTICS AND FEASIBILITY", prompts[0])
        self.assertIn("DAILY EXPERIENCE", prompts[1])
        self.assertIn(item["query"], prompts[0])
        self.assertIn(item["query"], prompts[1])
        self.assertIn("at most 4 assignments", prompts[0])
        self.assertIn("exactly 3 assignments", prompts[0])
        self.assertIn("exactly 4 assignments", prompts[1])
        self.assertIn("(day 1, current_city)", prompts[0])
        self.assertNotIn("(day 1, current_city)", prompts[1])
        self.assertIn('"agent_id" must be the integer 0', prompts[0])
        self.assertIn('"agent_id" must be the integer 1', prompts[1])
        self.assertIn("no Markdown fence", prompts[0])
        self.assertNotIn("exact candidate text", prompts[0])


if __name__ == "__main__":
    unittest.main()
