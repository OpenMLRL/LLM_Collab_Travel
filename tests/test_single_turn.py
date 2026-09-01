from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from single_turn.aggregation import PLAN_FIELDS, merge_agent_assignments
from single_turn.data import normalize_travelplanner_row, partition_rows
from single_turn.formatting import get_single_turn_formatters
from single_turn.parsing import parse_assignments
from single_turn.rewards.single_turn_reward import score_single_turn_response


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


def exact_completions():
    logistics = [
        [1, "current_city", "from Alpha to Beta"],
        [1, "transportation", "Flight Number: F100, from Alpha to Beta"],
        [1, "accommodation", "Hotel, Beta"],
    ]
    experience = [
        [1, "breakfast", "-"],
        [1, "attraction", "Museum, Beta"],
        [1, "lunch", "Cafe, Beta"],
        [1, "dinner", "Bistro, Beta"],
    ]
    return [
        json.dumps({"assignments": logistics}),
        json.dumps({"assignments": experience}),
    ]


class ParsingTests(unittest.TestCase):
    def test_json_fence_and_triplet_format(self):
        parsed = parse_assignments(
            '```json\n{"assignments": [[1, "breakfast", "-"]]}\n```'
        )
        self.assertTrue(parsed.parse_success)
        self.assertEqual(parsed.assignments[0].field, "breakfast")

    def test_non_json_fails_without_crashing(self):
        parsed = parse_assignments("I would visit the museum.")
        self.assertFalse(parsed.parse_success)
        self.assertEqual(parsed.assignments, [])


class AggregationTests(unittest.TestCase):
    def test_different_values_for_same_slot_conflict(self):
        result = merge_agent_assignments(
            [
                json.dumps({"assignments": [[1, "lunch", "Cafe A"]]}),
                json.dumps({"assignments": [[1, "lunch", "Cafe B"]]}),
            ],
            days=1,
        )
        self.assertIn((1, "lunch"), result.conflict_slots)
        self.assertNotIn((1, "lunch"), result.merged_assignments)
        self.assertEqual(result.plan[0]["lunch"], "-")

    def test_same_value_overlap_is_merged_and_counted(self):
        completion = json.dumps({"assignments": [[1, "lunch", "Cafe A"]]})
        result = merge_agent_assignments([completion, completion], days=1)
        self.assertEqual(result.merged_assignments[(1, "lunch")], "Cafe A")
        self.assertEqual(result.overlap_count, 1)


class RewardTests(unittest.TestCase):
    def test_exact_complementary_plan_gets_max_phase_one_reward(self):
        reward, detail = score_single_turn_response(
            exact_completions(), batch_item=fixture_item()
        )
        self.assertAlmostEqual(reward, 1.1, places=6)
        self.assertEqual(detail["exact_match"], 1.0)
        self.assertEqual(detail["coverage"], 1.0)
        self.assertEqual(detail["conflict_count"], 0.0)
        self.assertEqual(detail["role_score"], 1.0)

    def test_redundant_agents_score_below_complementary_agents(self):
        exact_reward, _ = score_single_turn_response(
            exact_completions(), batch_item=fixture_item()
        )
        duplicate = exact_completions()[0]
        duplicate_reward, detail = score_single_turn_response(
            [duplicate, duplicate], batch_item=fixture_item()
        )
        self.assertLess(duplicate_reward, exact_reward)
        self.assertGreater(detail["overlap_count"], 0)
        self.assertLess(detail["coverage"], 1.0)

    def test_all_empty_assignments_cannot_exploit_grounding(self):
        logistics = [
            [1, "current_city", "-"],
            [1, "transportation", "-"],
            [1, "accommodation", "-"],
        ]
        experience = [
            [1, "breakfast", "-"],
            [1, "attraction", "-"],
            [1, "lunch", "-"],
            [1, "dinner", "-"],
        ]
        completions = [
            json.dumps({"assignments": logistics}),
            json.dumps({"assignments": experience}),
        ]
        reward, detail = score_single_turn_response(
            completions, batch_item=fixture_item()
        )
        self.assertLess(reward, 0.2)
        self.assertLess(detail["grounding"], 0.5)
        self.assertLess(detail["coverage"], 0.5)
        self.assertEqual(detail["raw_coverage"], 1.0)
        self.assertGreater(detail["empty_mismatch_rate"], 0.5)

    def test_short_substring_cannot_exploit_grounding(self):
        logistics = [
            [1, "current_city", "a"],
            [1, "transportation", "a"],
            [1, "accommodation", "a"],
        ]
        experience = [
            [1, "breakfast", "a"],
            [1, "attraction", "a"],
            [1, "lunch", "a"],
            [1, "dinner", "a"],
        ]
        reward, detail = score_single_turn_response(
            [
                json.dumps({"assignments": logistics}),
                json.dumps({"assignments": experience}),
            ],
            batch_item=fixture_item(),
        )
        self.assertEqual(detail["grounding"], 0.0)
        self.assertEqual(detail["coverage"], 0.0)
        self.assertLess(reward, 0.1)


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


if __name__ == "__main__":
    unittest.main()
