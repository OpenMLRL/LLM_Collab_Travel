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
from single_turn.data import filter_rows, normalize_travelplanner_row, partition_rows
from single_turn.evaluation import evaluate_single_turn_response
from single_turn.formatting import (
    build_agent_json_prefill,
    build_compact_reference_context,
    build_role_budget_contract,
    get_single_turn_formatters,
)
from single_turn.logger import (
    aggregate_single_turn_metrics,
    build_single_turn_eval_logger,
)
from single_turn.parsing import parse_assignments
from single_turn.rewards.reference_evaluator import (
    CatalogEntry,
    accommodation_satisfies_house_rule,
    accommodation_satisfies_room_type,
    parse_reference_catalog,
    scaled_party_cost,
    transportation_satisfies_rule,
)
from single_turn.rewards.single_turn_reward import (
    TravelJointReward,
    TravelRewardConfig,
    score_single_turn_response,
)


def reference_records():
    return [
        {
            "Description": "Attractions in Beta",
            "Content": (
                "Name Latitude Longitude Address Phone Website City\n"
                "Museum 1.0 2.0 Main-Street 555 site Beta\n"
                "Park 1.1 2.1 Park-Street 556 site Beta"
            ),
        },
        {
            "Description": "Restaurants in Beta",
            "Content": (
                "Name Average Cost Cuisines Aggregate Rating City\n"
                "Cafe 10 Indian 4.0 Beta\n"
                "Diner 15 American 4.1 Beta\n"
                "Bistro 20 Italian 4.2 Beta\n"
                "Brunch 12 French 4.3 Beta"
            ),
        },
        {
            "Description": "Accommodations in Beta",
            "Content": (
                "NAME price room type house_rules minimum nights maximum occupancy review rate number city\n"
                "Hotel 100 Private room No parties 1 2 4.0 Beta\n"
                "Motel 80 Entire home/apt No smoking 1 4 3.0 Beta"
            ),
        },
        {
            "Description": "Flight from Alpha to Beta on 2022-01-01",
            "Content": (
                "Flight Number Price DepTime ArrTime ActualElapsedTime FlightDate "
                "OriginCityName DestCityName Distance\n"
                "F100 20 09:00 10:00 1 hours 0 minutes 2022-01-01 Alpha Beta 100"
            ),
        },
        {
            "Description": "Flight from Beta to Alpha on 2022-01-03",
            "Content": (
                "Flight Number Price DepTime ArrTime ActualElapsedTime FlightDate "
                "OriginCityName DestCityName Distance\n"
                "F200 25 18:00 19:00 1 hours 0 minutes 2022-01-03 Beta Alpha 100"
            ),
        },
    ]


def fixture_item():
    records = reference_records()
    return {
        "id": "travelplanner-validation-fixture",
        "prompt": "Plan a three-day round trip from Alpha to Beta.",
        "query": "Plan a three-day round trip from Alpha to Beta.",
        "org": "Alpha",
        "dest": "Beta",
        "days": 3,
        "visiting_city_number": 1,
        "date": repr(["2022-01-01", "2022-01-02", "2022-01-03"]),
        "dates": ["2022-01-01", "2022-01-02", "2022-01-03"],
        "people_number": 1,
        "budget": 1000,
        "level": "easy",
        "reference_information": repr(records),
        "reference_records": records,
        "local_constraint": {
            "house rule": None,
            "cuisine": None,
            "room type": None,
            "transportation": None,
        },
        "test": "",
        "entry_point": "",
    }


def five_day_fixture_item():
    records = []
    for city in ("Beta", "Gamma"):
        records.extend(
            [
                {
                    "Description": f"Attractions in {city}",
                    "Content": (
                        "Name Latitude Longitude Address Phone Website City\n"
                        f"Museum-{city} 1.0 2.0 Main-Street 555 site {city}"
                    ),
                },
                {
                    "Description": f"Restaurants in {city}",
                    "Content": (
                        "Name Average Cost Cuisines Aggregate Rating City\n"
                        f"Cafe-{city} 10 American 4.0 {city}\n"
                        f"Diner-{city} 15 Italian 4.1 {city}\n"
                        f"Bistro-{city} 20 French 4.2 {city}"
                    ),
                },
                {
                    "Description": f"Accommodations in {city}",
                    "Content": (
                        "NAME price room type house_rules minimum nights maximum "
                        "occupancy review rate number city\n"
                        f"Hotel-{city} 100 Private room No parties 1 2 4.0 {city}"
                    ),
                },
            ]
        )
    for number, origin, destination, date in (
        (100, "Alpha", "Beta", "2022-01-01"),
        (200, "Beta", "Gamma", "2022-01-03"),
        (300, "Gamma", "Alpha", "2022-01-05"),
    ):
        records.append(
            {
                "Description": (
                    f"Flight from {origin} to {destination} on {date}"
                ),
                "Content": (
                    "Flight Number Price DepTime ArrTime ActualElapsedTime "
                    "FlightDate OriginCityName DestCityName Distance\n"
                    f"F{number} 20 09:00 10:00 1 hours 0 minutes {date} "
                    f"{origin} {destination} 100"
                ),
            }
        )
    for origin, destination in (("Gamma", "Beta"), ("Beta", "Alpha")):
        records.append(
            {
                "Description": f"Taxi from {origin} to {destination}",
                "Content": (
                    f"taxi, from {origin} to {destination}, duration: "
                    "2 hours, distance: 100 km, cost: 5"
                ),
            }
        )
    return {
        "id": "travelplanner-five-day-fixture",
        "prompt": "Plan a five-day round trip from Alpha through Beta and Gamma.",
        "query": "Plan a five-day round trip from Alpha through Beta and Gamma.",
        "org": "Alpha",
        "dest": "Beta and Gamma",
        "days": 5,
        "visiting_city_number": 2,
        "date": repr([f"2022-01-0{day}" for day in range(1, 6)]),
        "dates": [f"2022-01-0{day}" for day in range(1, 6)],
        "people_number": 1,
        "budget": 2000,
        "level": "easy",
        "reference_information": repr(records),
        "reference_records": records,
        "local_constraint": {
            "house rule": None,
            "cuisine": None,
            "room type": None,
            "transportation": None,
        },
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


def valid_plan_values(*, attraction="Museum, Beta", accommodation="Hotel, Beta"):
    return {
        (1, "current_city"): "from Alpha to Beta",
        (1, "transportation"): "Flight Number: F100, from Alpha to Beta",
        (1, "breakfast"): "-",
        (1, "attraction"): "-",
        (1, "lunch"): "-",
        (1, "dinner"): "-",
        (1, "accommodation"): accommodation,
        (2, "current_city"): "Beta",
        (2, "transportation"): "-",
        (2, "breakfast"): "Cafe, Beta",
        (2, "attraction"): attraction,
        (2, "lunch"): "Diner, Beta",
        (2, "dinner"): "Bistro, Beta",
        (2, "accommodation"): accommodation,
        (3, "current_city"): "from Beta to Alpha",
        (3, "transportation"): "Flight Number: F200, from Beta to Alpha",
        (3, "breakfast"): "-",
        (3, "attraction"): "-",
        (3, "lunch"): "-",
        (3, "dinner"): "-",
        (3, "accommodation"): "-",
    }


def completions_from_values(values, *, days=3):
    outputs = []
    for agent_idx in range(2):
        slots = sorted(
            owned_slots(agent_idx, days),
            key=lambda slot: (slot[0], PLAN_FIELDS.index(slot[1])),
        )
        outputs.append(
            completion(
                agent_idx,
                [assignment(day, field, values[(day, field)]) for day, field in slots],
            )
        )
    return outputs


def valid_completions(**kwargs):
    return completions_from_values(valid_plan_values(**kwargs))


def all_dash_completions():
    return completions_from_values(
        {(day, field): "-" for day in range(1, 4) for field in PLAN_FIELDS}
    )


class ParsingTests(unittest.TestCase):
    def test_exact_strict_schema_is_accepted(self):
        parsed = parse_assignments(
            completion(0, [assignment(1, "breakfast", "-")]),
            expected_agent_id=0,
            capacity=1,
        )
        self.assertTrue(parsed.parse_success)
        self.assertTrue(parsed.strict_json)
        self.assertEqual(parsed.error_codes, ())

    def test_fenced_json_is_recovered_but_not_strict(self):
        parsed = parse_assignments(
            "```json\n" + completion(0, [assignment(1, "breakfast", "-")]) + "\n```",
            expected_agent_id=0,
            capacity=1,
        )
        self.assertFalse(parsed.parse_success)
        self.assertTrue(parsed.decode_success)
        self.assertIn("not_strict_json", parsed.error_codes)

    def test_wrong_id_and_overcapacity_are_invalid(self):
        wrong_id = parse_assignments(
            completion(1, [assignment(1, "breakfast", "-")]),
            expected_agent_id=0,
            capacity=1,
        )
        self.assertIn("agent_id_mismatch", wrong_id.error_codes)
        over = parse_assignments(
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
        self.assertIn("capacity_exceeded", over.error_codes)

    def test_invalid_slots_and_non_json_fail(self):
        parsed = parse_assignments(
            completion(0, [assignment(4, "unknown", "x")]),
            expected_agent_id=0,
            capacity=1,
            days=3,
            valid_fields=PLAN_FIELDS,
        )
        self.assertIn("assignment_value", parsed.error_codes)
        self.assertFalse(parse_assignments("not json").decode_success)


class AggregationTests(unittest.TestCase):
    def test_conflicts_are_removed(self):
        result = merge_agent_assignments(
            [
                completion(0, [assignment(1, "lunch", "Cafe, Beta")]),
                completion(1, [assignment(1, "lunch", "Diner, Beta")]),
            ],
            days=1,
        )
        self.assertIn((1, "lunch"), result.conflict_slots)
        self.assertNotIn((1, "lunch"), result.merged_assignments)

    def test_partition_has_one_role_consistent_owner_per_slot(self):
        for days in (1, 3, 5, 7):
            first = owned_slots(0, days)
            second = owned_slots(1, days)
            all_slots = {
                (day, field) for day in range(1, days + 1) for field in PLAN_FIELDS
            }
            self.assertFalse(first & second)
            self.assertEqual(first | second, all_slots)
            self.assertEqual(len(first), 3 * days)
            self.assertEqual(len(second), 4 * days)
            self.assertTrue(
                all(slot_owner(day, field, days) == 0 for day, field in first)
            )
            self.assertTrue(
                all(slot_owner(day, "dinner", days) == 1 for day in range(1, days + 1))
            )


class ReferenceRewardTests(unittest.TestCase):
    def test_reference_catalog_is_structured(self):
        catalog = parse_reference_catalog(fixture_item()["reference_information"])
        self.assertTrue(catalog.parse_success)
        self.assertEqual(len(catalog.restaurants), 4)
        self.assertEqual(len(catalog.attractions), 2)
        self.assertEqual(len(catalog.accommodations), 2)
        self.assertEqual(len(catalog.transportation), 2)

    def test_shared_hard_constraint_and_party_cost_helpers(self):
        restaurant = CatalogEntry(
            category="restaurant", name="Cafe", cost=10
        )
        flight = CatalogEntry(
            category="transportation", name="F1", cost=10, mode="flight"
        )
        taxi = CatalogEntry(
            category="transportation", name="Taxi", cost=10, mode="taxi"
        )
        drive = CatalogEntry(
            category="transportation",
            name="Self-driving",
            cost=10,
            mode="self-driving",
        )
        private_room = CatalogEntry(
            category="accommodation",
            name="Hotel",
            cost=10,
            room_type="Private room",
            house_rules="No parties",
            maximum_occupancy=3,
        )

        self.assertEqual(scaled_party_cost(restaurant, 7), 70.0)
        self.assertEqual(scaled_party_cost(flight, 7), 70.0)
        self.assertEqual(scaled_party_cost(taxi, 7), 20.0)
        self.assertEqual(scaled_party_cost(drive, 7), 20.0)
        self.assertEqual(scaled_party_cost(private_room, 7), 30.0)
        self.assertTrue(
            accommodation_satisfies_room_type(private_room, "Private room")
        )
        self.assertTrue(
            accommodation_satisfies_room_type(private_room, "not shared room")
        )
        self.assertFalse(
            accommodation_satisfies_room_type(private_room, "entire room")
        )
        self.assertFalse(
            accommodation_satisfies_house_rule(private_room, "parties")
        )
        self.assertTrue(
            accommodation_satisfies_house_rule(private_room, "smoking")
        )
        self.assertFalse(transportation_satisfies_rule(flight, "no flight"))
        self.assertTrue(transportation_satisfies_rule(taxi, "no flight"))
        self.assertFalse(
            transportation_satisfies_rule(drive, "no self-driving")
        )

    def test_terminal_hard_constraint_checks_remain_strict(self):
        room_item = fixture_item()
        room_item["local_constraint"] = {
            **room_item["local_constraint"],
            "room type": "Private room",
        }
        _, matching_room = score_single_turn_response(
            valid_completions(accommodation="Hotel, Beta"),
            batch_item=room_item,
        )
        _, wrong_room = score_single_turn_response(
            valid_completions(accommodation="Motel, Beta"),
            batch_item=room_item,
        )
        self.assertEqual(matching_room["hard_pass/room_type"], 1.0)
        self.assertEqual(matching_room["ultimate/reference_hard_macro"], 1.0)
        self.assertEqual(wrong_room["hard_pass/room_type"], 0.0)
        self.assertEqual(wrong_room["ultimate/reference_hard_macro"], 0.0)

        house_item = fixture_item()
        house_item["local_constraint"] = {
            **house_item["local_constraint"],
            "house rule": "parties",
        }
        _, forbidden_house_rule = score_single_turn_response(
            valid_completions(accommodation="Hotel, Beta"),
            batch_item=house_item,
        )
        self.assertEqual(forbidden_house_rule["hard_pass/room_rule"], 0.0)

        transport_item = fixture_item()
        transport_item["local_constraint"] = {
            **transport_item["local_constraint"],
            "transportation": "no flight",
        }
        _, forbidden_transport = score_single_turn_response(
            valid_completions(), batch_item=transport_item
        )
        self.assertEqual(
            forbidden_transport["hard_pass/transportation"], 0.0
        )

    def test_valid_plan_reaches_maximum_without_gold(self):
        reward, detail = score_single_turn_response(
            valid_completions(), batch_item=fixture_item()
        )
        self.assertAlmostEqual(reward, 1.0, places=6)
        self.assertEqual(
            detail["reward_backend"],
            "reference_constraint_terminal_weighted_v10",
        )
        self.assertAlmostEqual(
            detail["reward_component/budget_pass"], 0.12
        )
        self.assertAlmostEqual(
            detail["reward_component/commonsense_macro"], 0.10
        )
        self.assertAlmostEqual(
            detail["reward_component/hard_macro"], 0.18
        )
        self.assertAlmostEqual(
            detail["reward_component/final_success"], 0.25
        )
        self.assertEqual(detail["ultimate/reference_plan_success"], 1.0)
        self.assertEqual(detail["ultimate/reference_plan_delivery"], 1.0)
        self.assertEqual(detail["ultimate/required_plan_completion"], 1.0)
        self.assertEqual(detail["reference_plan_nonempty"], 1.0)
        self.assertEqual(detail["ultimate/collaboration_success"], 1.0)
        self.assertEqual(detail["ultimate/team_action_success"], 1.0)
        self.assertEqual(detail["required_grounded_recall"], 1.0)
        self.assertEqual(detail["required_cooperative_contribution"], 1.0)
        self.assertNotIn("exact_match", detail)
        self.assertNotIn("slot_quality", detail)

    def test_budget_shaping_is_dense_without_weakening_terminal_pass(self):
        exact_budget = fixture_item()
        exact_budget["budget"] = 290
        reward, valid = score_single_turn_response(
            valid_completions(), batch_item=exact_budget
        )
        self.assertAlmostEqual(reward, 1.0, places=6)
        self.assertEqual(valid["required_cost_completeness"], 1.0)
        self.assertEqual(valid["budget_margin_score"], 1.0)
        self.assertEqual(valid["budget_constraint_soft"], 1.0)
        self.assertEqual(valid["ultimate/reference_budget_pass"], 1.0)

        values = valid_plan_values()
        values[(2, "lunch")] = "Imaginary Cafe, Beta"
        _, unknown_meal = score_single_turn_response(
            completions_from_values(values), batch_item=exact_budget
        )
        self.assertAlmostEqual(
            unknown_meal["required_cost_completeness"], 6.0 / 7.0
        )
        self.assertAlmostEqual(
            unknown_meal["required_grounded_recall"], 10.0 / 11.0
        )
        self.assertAlmostEqual(
            unknown_meal["budget_constraint_soft"], 60.0 / 77.0
        )
        self.assertGreater(unknown_meal["budget_constraint_soft"], 0.0)
        self.assertEqual(unknown_meal["cost_complete"], 0.0)
        self.assertEqual(unknown_meal["ultimate/reference_budget_pass"], 0.0)

        values = valid_plan_values()
        values[(2, "attraction")] = "Imaginary Museum, Beta"
        _, unknown_attraction = score_single_turn_response(
            completions_from_values(values), batch_item=exact_budget
        )
        self.assertEqual(unknown_attraction["required_cost_completeness"], 1.0)
        self.assertAlmostEqual(
            unknown_attraction["budget_constraint_soft"], 10.0 / 11.0
        )
        self.assertEqual(unknown_attraction["cost_complete"], 0.0)
        self.assertEqual(
            unknown_attraction["ultimate/reference_budget_pass"], 0.0
        )

        # Optional travel-day cost slots are not part of the required-plan
        # denominator. If the model nevertheless emits one, it must be costed
        # before dense budget shaping can reach one.
        values = valid_plan_values()
        values[(1, "breakfast")] = "Imaginary Cafe, Beta"
        _, unknown_optional_cost = score_single_turn_response(
            completions_from_values(values), batch_item=exact_budget
        )
        self.assertEqual(
            unknown_optional_cost["required_cost_completeness"], 1.0
        )
        self.assertAlmostEqual(
            unknown_optional_cost["budget_cost_completeness"], 7.0 / 8.0
        )
        self.assertAlmostEqual(
            unknown_optional_cost["budget_constraint_soft"], 7.0 / 8.0
        )
        self.assertEqual(unknown_optional_cost["cost_complete"], 0.0)
        self.assertEqual(
            unknown_optional_cost["ultimate/reference_budget_pass"], 0.0
        )

    def test_budget_margin_penalizes_overage_continuously(self):
        expected = ((290, 1.0, 1.0), (250, 0.84, 0.0), (145, 0.0, 0.0))
        for budget, margin, terminal_pass in expected:
            with self.subTest(budget=budget):
                item = fixture_item()
                item["budget"] = budget
                _, detail = score_single_turn_response(
                    valid_completions(), batch_item=item
                )
                self.assertAlmostEqual(detail["estimated_cost"], 290.0)
                self.assertAlmostEqual(detail["budget_margin_score"], margin)
                self.assertAlmostEqual(detail["budget_constraint_soft"], margin)
                self.assertEqual(
                    detail["ultimate/reference_budget_pass"], terminal_pass
                )

        _, dash = score_single_turn_response(
            all_dash_completions(), batch_item=fixture_item()
        )
        self.assertEqual(dash["required_cost_completeness"], 0.0)
        self.assertEqual(dash["budget_constraint_soft"], 0.0)

    def test_two_different_valid_plans_can_both_score_maximum(self):
        first, first_detail = score_single_turn_response(
            valid_completions(attraction="Museum, Beta", accommodation="Hotel, Beta"),
            batch_item=fixture_item(),
        )
        second, second_detail = score_single_turn_response(
            valid_completions(attraction="Park, Beta", accommodation="Motel, Beta"),
            batch_item=fixture_item(),
        )
        self.assertAlmostEqual(first, 1.0, places=6)
        self.assertAlmostEqual(second, 1.0, places=6)
        self.assertEqual(first_detail["ultimate/reference_plan_success"], 1.0)
        self.assertEqual(second_detail["ultimate/reference_plan_success"], 1.0)

    def test_annotated_plan_is_ignored_even_when_present(self):
        plain = fixture_item()
        poisoned = dict(plain)
        poisoned["annotated_plan"] = "malicious target"
        poisoned["gold_plan"] = [{"do": "not read me"}]
        first = score_single_turn_response(valid_completions(), batch_item=plain)
        second = score_single_turn_response(valid_completions(), batch_item=poisoned)
        self.assertEqual(first, second)

    def test_all_dash_and_hallucinations_fail_end_metrics(self):
        dash_reward, dash = score_single_turn_response(
            all_dash_completions(), batch_item=fixture_item()
        )
        values = valid_plan_values()
        values[(2, "lunch")] = "Imaginary Cafe, Beta"
        fake_reward, fake = score_single_turn_response(
            completions_from_values(values), batch_item=fixture_item()
        )
        good_reward, _ = score_single_turn_response(
            valid_completions(), batch_item=fixture_item()
        )
        self.assertLess(dash_reward, fake_reward)
        self.assertLess(fake_reward, good_reward)
        self.assertEqual(dash["required_grounded_recall"], 0.0)
        self.assertEqual(dash["reference_plan_nonempty"], 0.0)
        self.assertEqual(dash["ultimate/reference_plan_delivery"], 0.0)
        self.assertEqual(dash["ultimate/required_plan_completion"], 0.0)
        self.assertEqual(dash["ultimate/reference_plan_success"], 0.0)
        self.assertLess(fake["entity_grounding_precision"], 1.0)
        self.assertEqual(fake["reference_plan_nonempty"], 1.0)
        self.assertEqual(fake["ultimate/reference_plan_delivery"], 1.0)
        self.assertEqual(fake["ultimate/required_plan_completion"], 1.0)
        self.assertEqual(fake["ultimate/reference_within_reference"], 0.0)

    def test_required_plan_completion_is_stricter_than_delivery(self):
        outputs = valid_completions()

        _, recovered = score_single_turn_response(
            ["prefix\n" + outputs[0], outputs[1]],
            batch_item=fixture_item(),
        )
        self.assertEqual(recovered["reference_plan_nonempty"], 1.0)
        self.assertEqual(recovered["assignment_coverage"], 1.0)
        self.assertEqual(recovered["required_fill_rate"], 1.0)
        self.assertEqual(recovered["ultimate/team_action_success"], 0.0)
        self.assertEqual(recovered["ultimate/reference_plan_delivery"], 1.0)
        self.assertEqual(recovered["ultimate/required_plan_completion"], 0.0)

        truncated = '{"agent_id":0,"assignments":['
        _, one_agent = score_single_turn_response(
            [truncated, outputs[1]],
            batch_item=fixture_item(),
        )
        self.assertEqual(one_agent["reference_plan_nonempty"], 1.0)
        self.assertEqual(one_agent["ultimate/reference_plan_delivery"], 1.0)
        self.assertEqual(one_agent["ultimate/required_plan_completion"], 0.0)

        values = valid_plan_values()
        values[(2, "breakfast")] = "-"
        _, missing_required = score_single_turn_response(
            completions_from_values(values),
            batch_item=fixture_item(),
        )
        self.assertEqual(missing_required["ultimate/team_action_success"], 1.0)
        self.assertEqual(missing_required["assignment_coverage"], 1.0)
        self.assertLess(missing_required["required_fill_rate"], 1.0)
        self.assertEqual(missing_required["ultimate/reference_plan_delivery"], 1.0)
        self.assertEqual(
            missing_required["ultimate/required_plan_completion"], 0.0
        )

    def test_wrong_route_and_lazy_agent_are_penalized(self):
        values = valid_plan_values()
        values[(1, "current_city")] = "from Gamma to Beta"
        wrong_reward, wrong = score_single_turn_response(
            completions_from_values(values), batch_item=fixture_item()
        )
        good_reward, _ = score_single_turn_response(
            valid_completions(), batch_item=fixture_item()
        )
        self.assertLess(wrong_reward, good_reward)
        self.assertEqual(wrong["ultimate/reference_reasonable_route"], 0.0)

        lazy_reward, lazy = score_single_turn_response(
            [valid_completions()[0], completion(1, [])],
            batch_item=fixture_item(),
        )
        self.assertLess(lazy_reward, good_reward)
        self.assertEqual(lazy["agent_1/verified_contribution"], 0.0)
        self.assertEqual(lazy["ultimate/both_agent_verified_contribution"], 0.0)

    def test_non_strict_output_loses_team_success(self):
        outputs = valid_completions()
        malformed = "prefix\n" + outputs[0]
        reward, detail = score_single_turn_response(
            [malformed, outputs[1]], batch_item=fixture_item()
        )
        self.assertLess(reward, 1.0)
        self.assertEqual(detail["ultimate/team_action_success"], 0.0)
        self.assertEqual(detail["ultimate/collaboration_success"], 0.0)
        self.assertEqual(detail["agent_0/decode_success"], 1.0)

    def test_reward_recovers_complete_triples_but_ultimate_stays_strict(self):
        outputs = valid_completions()
        first = json.loads(outputs[0])
        malformed = '{"agent_id":0,"assignments":[' + ",".join(
            json.dumps(entry, ensure_ascii=False)[1:-1]
            for entry in first["assignments"]
        )
        reward, detail = score_single_turn_response(
            [malformed, outputs[1]], batch_item=fixture_item()
        )
        self.assertGreater(reward, 0.0)
        self.assertLess(reward, 0.5)
        self.assertEqual(detail["ultimate/team_action_success"], 0.0)
        self.assertEqual(detail["agent_0/decode_success"], 0.0)
        self.assertGreater(detail["shaping/agent_0/regex_triple_count"], 0.0)
        self.assertGreater(detail["shaping/agent_0/recovered_owned_coverage"], 0.8)
        self.assertGreater(
            detail["shaping/recovered_plan/required_grounded_recall"], 0.8
        )

    def test_format_progress_distinguishes_list_from_copied_string(self):
        reference = fixture_item()["reference_information"]
        as_string = '{"agent_id":0,"assignments":"' + reference
        as_list = (
            '{"agent_id":0,"assignments":['
            '"day":1,"field":"current_city",'
            '"value":"from Alpha to Beta"}'
        )
        _, string_detail = score_single_turn_response(
            [as_string, ""], batch_item=fixture_item()
        )
        _, list_detail = score_single_turn_response(
            [as_list, ""], batch_item=fixture_item()
        )
        self.assertEqual(string_detail["shaping/agent_0/assignments_string"], 1.0)
        self.assertEqual(string_detail["shaping/agent_0/assignments_list"], 0.0)
        self.assertEqual(list_detail["shaping/agent_0/assignments_list"], 1.0)
        self.assertGreater(
            list_detail["shaping/agent_0/format_progress"],
            string_detail["shaping/agent_0/format_progress"],
        )
        self.assertGreater(string_detail["reward_penalty/reference_copy"], 0.0)

        overlong = as_list + (" unrelated-padding" * 300)
        _, overlong_detail = score_single_turn_response(
            [overlong, ""], batch_item=fixture_item()
        )
        self.assertGreater(overlong_detail["reward_penalty/overlong"], 0.0)

    def test_recovered_owned_triple_gets_bounded_repair_only(self):
        malformed = (
            '{"agent_id":1,"assignments":['
            '"day":2,"field":"lunch","value":"Diner, Beta"},'
        )
        reward, detail = score_single_turn_response(
            ["", malformed], batch_item=fixture_item()
        )
        empty_reward, _ = score_single_turn_response(
            ["", ""], batch_item=fixture_item()
        )
        self.assertGreater(reward, empty_reward)
        self.assertEqual(detail["entity_grounding_precision"], 0.0)
        self.assertEqual(detail["ultimate/reference_grounding"], 0.0)
        self.assertEqual(detail["shaping/quoted_grounded_count"], 1.0)
        self.assertGreater(detail["reward_component/recovered_semantic"], 0.0)
        self.assertLessEqual(
            detail["reward_component/recovered_semantic"], 0.01
        )
        self.assertEqual(detail["reward_component/strict_quality"], 0.0)

        repeated = malformed.replace(
            '"value":"Diner, Beta"',
            '"value":"Diner, Beta"},'
            '"day":2,"field":"breakfast","value":"Diner, Beta"},'
            '"day":2,"field":"lunch","value":"Diner, Beta"',
        )
        _, repeated_detail = score_single_turn_response(
            ["", repeated], batch_item=fixture_item()
        )
        self.assertEqual(repeated_detail["shaping/quoted_unique_grounded_count"], 1.0)
        self.assertLess(repeated_detail["shaping/quoted_grounding_progress"], 1.0)

        bare_name = malformed.replace("Diner, Beta", "Diner")
        _, bare_detail = score_single_turn_response(
            ["", bare_name], batch_item=fixture_item()
        )
        self.assertEqual(bare_detail["shaping/quoted_grounded_count"], 0.0)
        self.assertEqual(bare_detail["shaping/quoted_grounding_score"], 0.0)

    def test_detached_value_spam_cannot_earn_grounding(self):
        spam = (
            '{"value":"Cafe, Beta","value":"Diner, Beta",'
            '"value":"Museum, Beta"}'
        )
        reward, detail = score_single_turn_response(
            [spam, ""], batch_item=fixture_item()
        )
        self.assertEqual(detail["shaping/quoted_grounded_count"], 0.0)
        self.assertEqual(detail["reward_component/recovered_semantic"], 0.0)
        self.assertLess(reward, 0.005)

    def test_malformed_grounded_progress_is_rankable_but_strictly_bounded(self):
        grounded = (
            '{"agent_id":1,"assignments":['
            '"day":2,"field":"lunch","value":"Diner, Beta"},'
        )
        ungrounded = grounded.replace("Diner, Beta", "Imaginary Cafe, Beta")
        grounded_reward, grounded_detail = score_single_turn_response(
            ["", grounded], batch_item=fixture_item()
        )
        ungrounded_reward, ungrounded_detail = score_single_turn_response(
            ["", ungrounded], batch_item=fixture_item()
        )

        self.assertGreater(grounded_reward, ungrounded_reward)
        self.assertLessEqual(grounded_reward, 0.02)
        self.assertEqual(grounded_detail["ultimate/team_action_success"], 0.0)
        self.assertEqual(ungrounded_detail["ultimate/team_action_success"], 0.0)
        self.assertGreater(
            grounded_detail["recovered_semantic_balance"],
            ungrounded_detail["recovered_semantic_balance"],
        )

    def test_all_dash_and_route_only_do_not_form_a_high_reward_plateau(self):
        dash_reward, _ = score_single_turn_response(
            all_dash_completions(), batch_item=fixture_item()
        )
        route_only = valid_plan_values()
        for field in ("breakfast", "attraction", "lunch", "dinner"):
            route_only[(2, field)] = "-"
        route_reward, route_detail = score_single_turn_response(
            completions_from_values(route_only), batch_item=fixture_item()
        )
        valid_reward, _ = score_single_turn_response(
            valid_completions(), batch_item=fixture_item()
        )
        self.assertAlmostEqual(dash_reward, 0.04)
        self.assertLess(route_reward, 0.75)
        self.assertGreater(valid_reward - route_reward, 0.30)
        self.assertEqual(route_detail["ultimate/collaboration_success"], 0.0)

    def test_one_extra_assignment_does_not_erase_reward_contribution(self):
        outputs = valid_completions()
        over = json.loads(outputs[1])
        over["assignments"].append(assignment(2, "dinner", "Bistro, Beta"))
        reward, detail = score_single_turn_response(
            [outputs[0], json.dumps(over)], batch_item=fixture_item()
        )
        self.assertGreater(reward, 0.0)
        self.assertLess(reward, 0.5)
        self.assertEqual(detail["agent_1/verified_contribution"], 0.0)
        self.assertGreater(detail["shaping/agent_1/recovered_owned_coverage"], 0.9)
        self.assertGreater(
            detail["shaping/agent_1/recovered_required_contribution"], 0.9
        )
        self.assertGreater(detail["reward_penalty/invalid_action"], 0.0)

    def test_dense_reward_order_and_component_accounting(self):
        dash_reward, _ = score_single_turn_response(
            all_dash_completions(), batch_item=fixture_item()
        )
        values = valid_plan_values()
        values[(2, "lunch")] = "Imaginary Cafe, Beta"
        partial_reward, partial_detail = score_single_turn_response(
            completions_from_values(values), batch_item=fixture_item()
        )
        valid_reward, valid_detail = score_single_turn_response(
            valid_completions(), batch_item=fixture_item()
        )
        self.assertLess(dash_reward, partial_reward)
        self.assertLess(partial_reward, valid_reward)
        positive = sum(
            value
            for key, value in partial_detail.items()
            if key.startswith("reward_component/")
        )
        penalties = sum(
            value
            for key, value in partial_detail.items()
            if key.startswith("reward_penalty/")
        )
        self.assertAlmostEqual(partial_detail["unclamped_reward"], positive - penalties)
        self.assertAlmostEqual(valid_detail["reward"], 1.0)

    def test_malformed_agent_cannot_create_a_high_reward_shortcut(self):
        outputs = valid_completions()
        one_malformed = ["prefix\n" + outputs[0], outputs[1]]
        both_malformed = ["prefix\n" + output for output in outputs]
        one_reward, one_detail = score_single_turn_response(
            one_malformed, batch_item=fixture_item()
        )
        both_reward, both_detail = score_single_turn_response(
            both_malformed, batch_item=fixture_item()
        )
        self.assertLessEqual(one_reward, 0.03)
        self.assertLessEqual(both_reward, 0.02)
        self.assertEqual(one_detail["ultimate/team_action_success"], 0.0)
        self.assertEqual(both_detail["action_validity"], 0.0)
        # The plan evaluator can still recover perfect plan-level outcomes
        # from prefixed JSON. Those outcomes must remain behind the strict J
        # gate instead of creating a malformed-output shortcut.
        self.assertEqual(one_detail["ultimate/reference_budget_pass"], 1.0)
        self.assertEqual(
            one_detail["ultimate/reference_commonsense_macro"], 1.0
        )
        self.assertEqual(one_detail["ultimate/reference_hard_macro"], 1.0)
        self.assertLessEqual(
            one_detail["reward_component/recovered_semantic"], 0.01
        )
        self.assertLessEqual(
            both_detail["reward_component/recovered_semantic"], 0.01
        )
        for detail in (one_detail, both_detail):
            self.assertEqual(detail["reward_component/budget_pass"], 0.0)
            self.assertEqual(
                detail["reward_component/commonsense_macro"], 0.0
            )
            self.assertEqual(detail["reward_component/hard_macro"], 0.0)

    def test_v10_reserves_most_reward_for_strict_terminal_outcomes(self):
        config = TravelRewardConfig()
        non_terminal = sum(
            (
                config.protocol_progress_weight,
                config.recovered_semantic_weight,
                config.action_validity_weight,
                config.team_action_weight,
                config.strict_balance_weight,
                config.strict_quality_weight,
                config.strict_grounding_weight,
            )
        )
        terminal = sum(
            (
                config.budget_pass_bonus,
                config.commonsense_macro_bonus,
                config.hard_macro_bonus,
                config.final_success_bonus,
            )
        )
        self.assertAlmostEqual(non_terminal, 0.35)
        self.assertAlmostEqual(terminal, 0.65)

        with self.assertRaisesRegex(ValueError, "must sum to 1.0"):
            TravelRewardConfig(budget_pass_bonus=0.13).validate()
        with self.assertRaisesRegex(ValueError, "must be non-negative"):
            TravelRewardConfig(
                budget_pass_bonus=-0.01,
                final_success_bonus=0.38,
            ).validate()

    def test_strict_all_dash_agent_cannot_free_ride_on_either_role(self):
        outputs = valid_completions()
        for lazy_agent in (0, 1):
            payloads = [json.loads(output) for output in outputs]
            for assignment_row in payloads[lazy_agent]["assignments"]:
                assignment_row["value"] = "-"
            lazy_outputs = [
                json.dumps(payload, ensure_ascii=False) for payload in payloads
            ]
            reward, detail = score_single_turn_response(
                lazy_outputs, batch_item=fixture_item()
            )
            self.assertEqual(detail["ultimate/team_action_success"], 1.0)
            self.assertEqual(
                detail[f"agent_{lazy_agent}/required_grounded_contribution"],
                0.0,
            )
            self.assertLessEqual(
                detail["required_cooperative_contribution"], 0.10
            )
            self.assertLess(reward, 0.30)

    def test_self_loop_cannot_erase_experience_role_requirements(self):
        values = valid_plan_values()
        values[(2, "current_city")] = "from Beta to Beta"
        values[(2, "transportation")] = "-"
        outputs = completions_from_values(values)
        lazy_experience = json.loads(outputs[1])
        for row in lazy_experience["assignments"]:
            row["value"] = "-"
        outputs[1] = json.dumps(lazy_experience)

        reward, detail = score_single_turn_response(
            outputs, batch_item=fixture_item()
        )

        self.assertLess(detail["route_parse_rate"], 1.0)
        self.assertEqual(
            detail["agent_1/required_grounded_contribution"], 0.0
        )
        self.assertLess(reward, 0.25)

    def test_empty_required_role_is_zero_contribution_not_vacuous_success(self):
        values = {
            (day, field): "-"
            for day in range(1, 4)
            for field in PLAN_FIELDS
        }
        values.update(
            {
                (1, "current_city"): "from Alpha to Beta",
                (1, "transportation"): (
                    "Flight Number: F100, from Alpha to Beta"
                ),
                (1, "accommodation"): "Hotel, Beta",
                (2, "current_city"): "from Beta to Alpha",
                (2, "transportation"): (
                    "Flight Number: F200, from Beta to Alpha"
                ),
                (3, "current_city"): "from Alpha to Beta",
                (3, "transportation"): (
                    "Flight Number: F100, from Alpha to Beta"
                ),
            }
        )

        reward, detail = score_single_turn_response(
            completions_from_values(values), batch_item=fixture_item()
        )

        self.assertEqual(
            detail["agent_1/required_grounded_contribution"], 0.0
        )
        self.assertLess(reward, 0.25)

    def test_logistics_agent_cannot_shrink_teammate_work_with_extra_moves(self):
        values = {
            (day, field): "-"
            for day in range(1, 6)
            for field in PLAN_FIELDS
        }
        values.update(
            {
                (1, "current_city"): "from Alpha to Beta",
                (1, "transportation"): (
                    "Flight Number: F100, from Alpha to Beta"
                ),
                (1, "accommodation"): "Hotel-Beta, Beta",
                (2, "current_city"): "Beta",
                (2, "breakfast"): "Cafe-Beta, Beta",
                (2, "attraction"): "Museum-Beta, Beta",
                (2, "lunch"): "Diner-Beta, Beta",
                (2, "dinner"): "Bistro-Beta, Beta",
                (2, "accommodation"): "Hotel-Beta, Beta",
                (3, "current_city"): "from Beta to Gamma",
                (3, "transportation"): (
                    "Flight Number: F200, from Beta to Gamma"
                ),
                (3, "accommodation"): "Hotel-Gamma, Gamma",
                (4, "current_city"): "Gamma",
                # Agent 1 uses grounded but wrong-city entities on the second
                # stay day. Agent 0 must not be able to make them valid (or
                # delete their requirements) by relabeling the day as a move
                # to Beta.
                (4, "breakfast"): "Cafe-Beta, Beta",
                (4, "attraction"): "Museum-Beta, Beta",
                (4, "lunch"): "Diner-Beta, Beta",
                (4, "dinner"): "Bistro-Gamma, Gamma",
                (4, "accommodation"): "Hotel-Gamma, Gamma",
                (5, "current_city"): "from Gamma to Alpha",
                (5, "transportation"): (
                    "Flight Number: F300, from Gamma to Alpha"
                ),
            }
        )
        honest_outputs = completions_from_values(values, days=5)
        honest_reward, honest = score_single_turn_response(
            honest_outputs, batch_item=five_day_fixture_item()
        )

        attacked = dict(values)
        attacked.update(
            {
                (4, "current_city"): "from Gamma to Beta",
                (4, "transportation"): "Taxi, from Gamma to Beta",
                (4, "accommodation"): "Hotel-Beta, Beta",
                (5, "current_city"): "from Beta to Alpha",
                (5, "transportation"): "Taxi, from Beta to Alpha",
            }
        )
        attacked_outputs = completions_from_values(attacked, days=5)
        attacked_reward, attack = score_single_turn_response(
            attacked_outputs, batch_item=five_day_fixture_item()
        )

        self.assertEqual(attacked_outputs[1], honest_outputs[1])
        self.assertEqual(honest["route_scaffold_match_rate"], 1.0)
        self.assertLess(attack["route_scaffold_match_rate"], 1.0)
        self.assertLessEqual(
            attack["required_grounded_recall"],
            honest["required_grounded_recall"],
        )
        self.assertLessEqual(
            attack["required_cooperative_contribution"],
            honest["required_cooperative_contribution"],
        )
        self.assertLess(attacked_reward, honest_reward)

    def test_conflicted_slot_is_not_a_verified_contribution(self):
        outputs = valid_completions()
        first = json.loads(outputs[0])
        first["assignments"][-1] = assignment(1, "breakfast", "Diner, Beta")
        _, detail = score_single_turn_response(
            [json.dumps(first), outputs[1]],
            batch_item=fixture_item(),
        )
        self.assertEqual(detail["conflict_count"], 1.0)
        self.assertLess(detail["agent_0/verified_contribution"], 1.0)
        self.assertLess(detail["agent_1/verified_contribution"], 1.0)
        self.assertEqual(detail["ultimate/both_agent_verified_contribution"], 0.0)

    def test_bare_entity_name_does_not_receive_grounding_credit(self):
        values = valid_plan_values()
        values[(2, "lunch")] = "Diner"
        _, detail = score_single_turn_response(
            completions_from_values(values), batch_item=fixture_item()
        )
        self.assertLess(detail["entity_grounding_precision"], 1.0)
        self.assertEqual(detail["ultimate/reference_grounding"], 0.0)
        self.assertEqual(detail["ultimate/reference_plan_success"], 0.0)

    def test_abbreviated_transport_does_not_receive_route_credit(self):
        values = valid_plan_values()
        values[(1, "transportation")] = "F100"
        values[(3, "transportation")] = "F200"
        reward, detail = score_single_turn_response(
            completions_from_values(values), batch_item=fixture_item()
        )
        self.assertLess(reward, 1.0)
        self.assertEqual(detail["ultimate/reference_transport_consistency"], 0.0)
        self.assertEqual(detail["ultimate/reference_plan_success"], 0.0)

    def test_appended_factual_claims_do_not_receive_grounding_credit(self):
        values = valid_plan_values()
        values[(2, "lunch")] = "Diner, Beta; cost: $0 invented"
        values[(1, "transportation")] = (
            "Flight Number: F100, from Alpha to Beta, on 2099-12-31, teleportation"
        )
        reward, detail = score_single_turn_response(
            completions_from_values(values), batch_item=fixture_item()
        )
        self.assertLess(reward, 1.0)
        self.assertLess(detail["entity_grounding_precision"], 1.0)
        self.assertEqual(detail["ultimate/reference_plan_success"], 0.0)

    def test_accommodation_must_be_in_end_of_day_city(self):
        item = fixture_item()
        records = [
            *reference_records(),
            {
                "Description": "Accommodations in Alpha",
                "Content": (
                    "NAME price room type house_rules minimum nights maximum occupancy "
                    "review rate number city\n"
                    "Origin Inn 50 Private room No parties 1 2 4.0 Alpha"
                ),
            },
        ]
        item["reference_information"] = repr(records)
        item["reference_records"] = records
        values = valid_plan_values()
        values[(1, "accommodation")] = "Origin Inn, Alpha"
        reward, detail = score_single_turn_response(
            completions_from_values(values), batch_item=item
        )
        self.assertLess(reward, 1.0)
        self.assertEqual(detail["ultimate/reference_within_current_city"], 0.0)
        self.assertEqual(detail["ultimate/reference_plan_success"], 0.0)

    def test_joint_reward_returns_one_shared_scalar(self):
        outputs = valid_completions()
        reward_model = TravelJointReward()
        rewards = reward_model(
            [outputs[0]],
            [outputs[1]],
            batch_items=[fixture_item()],
            prompts=[["agent 0"], ["agent 1"]],
        )
        self.assertEqual(rewards, [1.0])
        self.assertEqual(reward_model.last_details[0]["reward"], 1.0)
        self.assertEqual(
            reward_model.last_details[0]["unclamped_reward"], 1.0
        )
        pending = reward_model.drain_details()
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0]["reward"], 1.0)
        self.assertEqual(reward_model.drain_details(), [])


class LoggerTests(unittest.TestCase):
    def test_eval_logger_emits_only_headline_metrics(self):
        item = fixture_item()
        outputs = valid_completions()
        logger = build_single_turn_eval_logger([item])
        metrics = logger(
            [[[outputs[0]]], [[outputs[1]]]],
            test_cases=[""],
            entry_points=[""],
            prompts=[item["prompt"]],
        )
        self.assertEqual(metrics[0]["turn_1/ultimate/collaboration_success"], 1.0)
        self.assertEqual(
            metrics[0]["turn_1/ultimate/reference_commonsense_micro"], 1.0
        )
        public_keys = {
            key
            for key in metrics[0]
            if key.startswith("turn_1/") and "/_aggregate/" not in key
        }
        self.assertEqual(
            public_keys,
            {
                "turn_1/action_validity",
                "turn_1/ultimate/team_action_success",
                "turn_1/required_cooperative_contribution",
                "turn_1/required_grounded_recall",
                "turn_1/entity_grounding_precision",
                "turn_1/required_cost_completeness",
                "turn_1/budget_constraint_soft",
                "turn_1/route_scaffold_match_rate",
                "turn_1/ultimate/reference_plan_delivery",
                "turn_1/ultimate/required_plan_completion",
                "turn_1/ultimate/reference_commonsense_micro",
                "turn_1/ultimate/reference_commonsense_macro",
                "turn_1/ultimate/reference_hard_micro",
                "turn_1/ultimate/reference_hard_macro",
                "turn_1/ultimate/reference_budget_pass",
                "turn_1/ultimate/reference_plan_success",
                "turn_1/ultimate/collaboration_success",
            },
        )
        self.assertNotIn("turn_1/agent_0/parser_error/any", metrics[0])

    def test_eval_logger_builds_a_wandb_sample_table(self):
        item = fixture_item()
        outputs = valid_completions()
        logger = build_single_turn_eval_logger(
            [item], reward_config=TravelRewardConfig(), panel_size=4
        )
        metrics = logger(
            [[[outputs[0]]], [[outputs[1]]]],
            test_cases=[""],
            entry_points=[""],
            prompts=[item["prompt"]],
        )
        aggregate = aggregate_single_turn_metrics(metrics)
        self.assertEqual(
            set(aggregate),
            {
                "turn_1/action_validity",
                "turn_1/ultimate/team_action_success",
                "turn_1/required_cooperative_contribution",
                "turn_1/required_grounded_recall",
                "turn_1/entity_grounding_precision",
                "turn_1/required_cost_completeness",
                "turn_1/budget_constraint_soft",
                "turn_1/grounding_f1",
                "turn_1/route_scaffold_match_rate",
                "turn_1/ultimate/reference_plan_delivery",
                "turn_1/ultimate/required_plan_completion",
                "turn_1/ultimate/reference_commonsense_micro",
                "turn_1/ultimate/reference_commonsense_macro",
                "turn_1/ultimate/reference_hard_micro",
                "turn_1/ultimate/reference_hard_macro",
                "turn_1/ultimate/reference_budget_pass",
                "turn_1/ultimate/reference_plan_success",
                "turn_1/ultimate/collaboration_success",
                "turn_1/eval_samples",
            },
        )
        table = aggregate["turn_1/eval_samples"]
        self.assertIn("agent_0_output", table.columns)
        self.assertIn("agent_1_output", table.columns)
        self.assertIn("merged_plan", table.columns)
        self.assertIn("reward", table.columns)
        self.assertIn("protocol_progress", table.columns)
        self.assertIn("required_cooperative_contribution", table.columns)
        self.assertIn("plan_delivery", table.columns)
        self.assertIn("required_plan_completion", table.columns)
        self.assertIn("required_cost_completeness", table.columns)
        self.assertIn("budget_constraint_soft", table.columns)
        self.assertIn("budget_pass", table.columns)
        self.assertIn("commonsense_macro", table.columns)
        self.assertIn("hard_macro", table.columns)
        self.assertEqual(len(table.data), 1)
        self.assertEqual(table.data[0][table.columns.index("reward")], 1.0)
        self.assertEqual(
            table.data[0][table.columns.index("plan_delivery")], 1.0
        )
        self.assertEqual(
            table.data[0][table.columns.index("required_plan_completion")],
            1.0,
        )

    def test_eval_logger_fails_loudly_on_unknown_prompt(self):
        logger = build_single_turn_eval_logger([fixture_item()])
        outputs = valid_completions()
        with self.assertRaises(KeyError):
            logger(
                [[[outputs[0]]], [[outputs[1]]]],
                test_cases=[""],
                entry_points=[""],
                prompts=["not in the fixed split"],
            )

    def test_eval_logger_rejects_duplicate_prompts(self):
        item = fixture_item()
        duplicate = dict(item, id="duplicate")
        with self.assertRaises(ValueError):
            build_single_turn_eval_logger([item, duplicate])

    def test_evaluator_and_logger_ignore_target_fields(self):
        plain = fixture_item()
        poisoned = dict(plain)
        poisoned["annotated_plan"] = "do not read"
        poisoned["gold_plan"] = [{"secret": True}]
        outputs = valid_completions()
        self.assertEqual(
            evaluate_single_turn_response(outputs, batch_item=plain),
            evaluate_single_turn_response(outputs, batch_item=poisoned),
        )
        plain_metrics = build_single_turn_eval_logger([plain])(
            [[[outputs[0]]], [[outputs[1]]]], [""], [""], [plain["prompt"]]
        )
        poisoned_metrics = build_single_turn_eval_logger([poisoned])(
            [[[outputs[0]]], [[outputs[1]]]], [""], [""], [poisoned["prompt"]]
        )
        self.assertEqual(plain_metrics, poisoned_metrics)

    def test_micro_aggregation_uses_global_denominators(self):
        metrics = [
            {
                "turn_1/ultimate/reference_hard_micro": 1.0,
                "turn_1/_aggregate/hard_pass_count": 1.0,
                "turn_1/_aggregate/hard_applicable_count": 1.0,
                "turn_1/grounding_f1": 1.0,
                "turn_1/_aggregate/required_valid_count": 1.0,
                "turn_1/_aggregate/required_slot_count": 1.0,
                "turn_1/_aggregate/grounded_entity_count": 1.0,
                "turn_1/_aggregate/predicted_entity_count": 1.0,
                "turn_1/_aggregate/required_cost_known_count": 6.0,
                "turn_1/_aggregate/required_cost_slot_count": 7.0,
            },
            {
                "turn_1/ultimate/reference_hard_micro": 0.25,
                "turn_1/_aggregate/hard_pass_count": 1.0,
                "turn_1/_aggregate/hard_applicable_count": 4.0,
                "turn_1/grounding_f1": 0.5,
                "turn_1/_aggregate/required_valid_count": 1.0,
                "turn_1/_aggregate/required_slot_count": 3.0,
                "turn_1/_aggregate/grounded_entity_count": 3.0,
                "turn_1/_aggregate/predicted_entity_count": 4.0,
                "turn_1/_aggregate/required_cost_known_count": 13.0,
                "turn_1/_aggregate/required_cost_slot_count": 13.0,
            },
        ]
        aggregate = aggregate_single_turn_metrics(metrics)
        self.assertAlmostEqual(aggregate["turn_1/ultimate/reference_hard_micro"], 0.4)
        self.assertAlmostEqual(aggregate["turn_1/required_grounded_recall"], 0.5)
        self.assertAlmostEqual(aggregate["turn_1/entity_grounding_precision"], 0.8)
        self.assertAlmostEqual(
            aggregate["turn_1/required_cost_completeness"], 19.0 / 20.0
        )
        self.assertAlmostEqual(aggregate["turn_1/grounding_f1"], 8.0 / 13.0)
        self.assertNotIn("turn_1/_aggregate/hard_pass_count", aggregate)


class DataAndPromptTests(unittest.TestCase):
    def test_validation_row_without_annotation_is_normalized(self):
        raw = {
            "query": "Plan a trip.",
            "days": 3,
            "date": repr(["2022-01-01", "2022-01-02", "2022-01-03"]),
            "reference_information": repr(reference_records()),
            "local_constraint": "{}",
        }
        normalized = normalize_travelplanner_row(raw, 7, source_split="validation")
        self.assertEqual(normalized["id"], "travelplanner-validation-7")
        self.assertNotIn("gold_plan", normalized)
        self.assertNotIn("annotated_plan", normalized)
        self.assertEqual(len(normalized["dates"]), 3)

    def test_annotation_is_removed_from_normalized_row(self):
        raw = {
            "query": "Plan a trip.",
            "days": 3,
            "annotated_plan": "secret",
            "reference_information": repr(reference_records()),
        }
        normalized = normalize_travelplanner_row(raw, 0, source_split="train")
        self.assertNotIn("annotated_plan", normalized)

    def test_filters_and_partition_are_disjoint_and_reproducible(self):
        rows = [
            {
                "id": str(index),
                "days": 3 if index < 6 else 5,
                "level": "easy",
                "visiting_city_number": 1 if index < 6 else 2,
                "reference_chars": 100 + index,
                "source_index": index,
            }
            for index in range(10)
        ]
        selected = filter_rows(
            rows,
            days=[3],
            levels=["easy"],
            visiting_city_numbers=[1],
            select_shortest=5,
        )
        self.assertEqual([row["id"] for row in selected], ["0", "1", "2", "3", "4"])
        first = partition_rows(selected, train_samples=4, eval_samples=1, seed=7)
        second = partition_rows(selected, train_samples=4, eval_samples=1, seed=7)
        self.assertEqual(first, second)
        self.assertFalse(
            {row["id"] for row in first[0]} & {row["id"] for row in first[1]}
        )

    def test_stratified_partition_balances_and_interleaves_eval_panels(self):
        rows = [
            {
                "id": f"{days}-{level}-{index}",
                "days": days,
                "level": level,
            }
            for days in (3, 5)
            for level in ("easy", "medium")
            for index in range(6)
        ]
        train, evaluation = partition_rows(
            rows,
            train_samples=16,
            eval_samples=8,
            seed=42,
            stratify_by=["days", "level"],
            interleave_eval=True,
        )
        for days in (3, 5):
            for level in ("easy", "medium"):
                self.assertEqual(
                    sum(row["days"] == days and row["level"] == level for row in train),
                    4,
                )
        for panel_start in (0, 4):
            panel = evaluation[panel_start : panel_start + 4]
            self.assertEqual(
                {(row["days"], row["level"]) for row in panel},
                {(3, "easy"), (3, "medium"), (5, "easy"), (5, "medium")},
            )
        self.assertFalse(
            {row["id"] for row in train} & {row["id"] for row in evaluation}
        )

    def test_role_budget_contract_is_feasible_exact_and_target_free(self):
        item = fixture_item()
        contract = build_role_budget_contract(item)
        self.assertTrue(contract.feasible)
        self.assertEqual(contract.reason, "feasible")
        self.assertEqual(contract.logistics_floor_cents, 20_500)
        self.assertEqual(contract.experience_floor_cents, 3_700)
        self.assertEqual(contract.logistics_cap_cents, 58_400)
        self.assertEqual(contract.experience_cap_cents, 41_600)
        self.assertEqual(
            contract.logistics_cap_cents + contract.experience_cap_cents,
            contract.team_budget_cents,
        )

        prompts = [
            formatter(item) for formatter in get_single_turn_formatters(num_agents=2)
        ]
        for prompt in prompts:
            self.assertIn("team_budget=1000.00", prompt)
            self.assertIn("logistics_cap=584.00", prompt)
            self.assertIn("experience_cap=416.00", prompt)
            self.assertIn("feasible=yes", prompt)
        self.assertIn("YOUR_ROLE=LOGISTICS; YOUR_CAP=584.00", prompts[0])
        self.assertIn("YOUR_ROLE=EXPERIENCE; YOUR_CAP=416.00", prompts[1])

        poisoned = dict(
            item,
            annotated_plan="forbidden target",
            gold_plan=[{"forbidden": True}],
        )
        poisoned_prompts = [
            formatter(poisoned)
            for formatter in get_single_turn_formatters(num_agents=2)
        ]
        self.assertEqual(prompts, poisoned_prompts)

    def test_role_prompts_make_hard_eligibility_and_required_slots_explicit(self):
        item = fixture_item()
        item["local_constraint"] = {
            "house rule": None,
            "cuisine": ["Indian"],
            "room type": "Private room",
            "transportation": None,
        }
        logistics, experience = [
            formatter(item) for formatter in get_single_turn_formatters(num_agents=2)
        ]
        hotel_line = next(
            line for line in logistics.splitlines() if line.startswith('- "Hotel, Beta"')
        )
        motel_line = next(
            line for line in logistics.splitlines() if line.startswith('- "Motel, Beta"')
        )
        cafe_line = next(
            line for line in experience.splitlines() if line.startswith('- "Cafe, Beta"')
        )
        diner_line = next(
            line for line in experience.splitlines() if line.startswith('- "Diner, Beta"')
        )
        self.assertIn("room=Private room", hotel_line)
        self.assertIn("eligible=yes", hotel_line)
        self.assertIn("room=Entire home/apt", motel_line)
        self.assertIn("eligible=no", motel_line)
        self.assertIn("cuisines=Indian", cafe_line)
        self.assertIn("eligible=yes", cafe_line)
        self.assertIn("cuisines=American", diner_line)
        self.assertIn("eligible=yes", diner_line)
        self.assertIn(
            'day=1: breakfast, attraction, lunch, dinner are EMPTY; use "-" for all four',
            experience,
        )
        self.assertIn(
            "day=2: breakfast, lunch, and dinner are REQUIRED distinct",
            experience,
        )
        self.assertIn("never output multiple attractions in one slot", experience)

    def test_accommodation_prompt_excludes_infeasible_minimum_stay(self):
        item = fixture_item()
        item["reference_information"] = item["reference_information"].replace(
            "Motel 80 Entire home/apt No smoking 1 4 3.0 Beta",
            "Motel 80 Entire home/apt No smoking 3 4 3.0 Beta",
        )
        item["reference_records"] = []
        logistics = get_single_turn_formatters(num_agents=2)[0](item)
        hotel_line = next(
            line for line in logistics.splitlines() if line.startswith('- "Hotel, Beta"')
        )
        motel_line = next(
            line for line in logistics.splitlines() if line.startswith('- "Motel, Beta"')
        )
        self.assertIn("days=1,2", hotel_line)
        self.assertIn("eligible=yes", hotel_line)
        self.assertIn("min_nights=3", motel_line)
        self.assertIn("days=none", motel_line)
        self.assertIn("eligible=no", motel_line)

    def test_prompt_and_budget_use_evaluator_first_duplicate_entity(self):
        records = reference_records()
        restaurant_record = next(
            record
            for record in records
            if record["Description"] == "Restaurants in Beta"
        )
        restaurant_record["Content"] = (
            "Name Average Cost Cuisines Aggregate Rating City\n"
            "Burger King 98 Pizza 4.0 Beta\n"
            "Burger King 13 Indian 4.1 Beta\n"
            "Cafe 20 Indian 4.2 Beta\n"
            "Diner 15 American 4.3 Beta"
        )
        item = fixture_item()
        item["reference_information"] = repr(records)
        item["reference_records"] = records
        item["local_constraint"] = {
            **item["local_constraint"],
            "cuisine": ["Indian"],
        }

        contract = build_role_budget_contract(item)
        self.assertTrue(contract.feasible)
        # The evaluator maps the copyable value to the first Burger King row.
        # Three distinct meals therefore cost 98 + 20 + 15, not 13 + 20 + 15.
        self.assertEqual(contract.experience_floor_cents, 13_300)
        experience = get_single_turn_formatters(num_agents=2)[1](item)
        burger_lines = [
            line
            for line in experience.splitlines()
            if line.startswith('- "Burger King, Beta"')
        ]
        self.assertEqual(len(burger_lines), 1)
        self.assertIn("team_cost_per_meal=98.00", burger_lines[0])
        self.assertIn("cuisines=Pizza", burger_lines[0])
        self.assertNotIn("team_cost_per_meal=13.00", experience)

    def test_role_prompts_explain_reference_free_contract(self):
        item = fixture_item()
        prompts = [
            formatter(item) for formatter in get_single_turn_formatters(num_agents=2)
        ]
        self.assertIn("LOGISTICS AND FEASIBILITY", prompts[0])
        self.assertIn("DAILY EXPERIENCE", prompts[1])
        self.assertIn("A travel day requires matching transportation", prompts[0])
        self.assertIn('as "Name, City"', prompts[1])
        self.assertIn('"assignments" must contain exactly 9 objects', prompts[0])
        self.assertIn('"assignments" must contain exactly 12 objects', prompts[1])
        self.assertNotIn("annotated", prompts[0].casefold())
        self.assertNotIn("gold", prompts[1].casefold())

    def test_role_prompts_use_evaluator_exact_compact_catalogs(self):
        item = fixture_item()
        logistics, experience = [
            formatter(item) for formatter in get_single_turn_formatters(num_agents=2)
        ]

        self.assertIn(
            '"Flight Number: F100, from Alpha to Beta" | team_cost=20.00',
            logistics,
        )
        self.assertIn(
            '"Hotel, Beta" | team_cost_per_night=100.00', logistics
        )
        self.assertNotIn('"Cafe, Beta" | team_cost_per_meal=10.00', logistics)
        self.assertNotIn("Museum, Beta", logistics)

        self.assertIn(
            '"Cafe, Beta" | team_cost_per_meal=10.00', experience
        )
        self.assertIn('"Museum, Beta"', experience)
        self.assertNotIn("Hotel, Beta", experience)
        self.assertNotIn("Flight Number: F100", experience)

        for prompt in (logistics, experience):
            self.assertIn("SHARED REFERENCE-DERIVED ROUTE SCAFFOLD", prompt)
            self.assertIn(
                'day=1 date=2022-01-01 kind=move current_city="from Alpha to Beta"',
                prompt,
            )
            self.assertIn(
                'day=2 date=2022-01-02 kind=stay current_city="Beta" '
                'transportation="-"',
                prompt,
            )
            self.assertNotIn("Main-Street", prompt)
            self.assertNotIn("Park-Street", prompt)
            self.assertNotIn("Website", prompt)

    def test_compact_context_fails_closed_instead_of_dumping_raw_reference(self):
        item = fixture_item()
        item["id"] = "broken-reference"
        item["reference_information"] = "private raw table text"
        item["reference_records"] = []
        with self.assertRaisesRegex(ValueError, "broken-reference"):
            build_compact_reference_context(item, 0)

    def test_five_day_route_scaffold_is_shared_and_follows_dated_routes(self):
        records = [
            {
                "Description": f"Attractions in {city}",
                "Content": (
                    "Name Latitude Longitude Address Phone Website City\n"
                    f"Museum 1.0 2.0 Address 555 site {city}"
                ),
            }
            for city in ("Evansville", "South Bend")
        ]
        records.extend(
            {
                "Description": f"Restaurants in {city}",
                "Content": (
                    "Name Average Cost Cuisines Aggregate Rating City\n"
                    f"Cafe 10 American 4.0 {city}"
                ),
            }
            for city in ("Evansville", "South Bend")
        )
        records.extend(
            {
                "Description": f"Accommodations in {city}",
                "Content": (
                    "NAME price room type house_rules minimum nights maximum "
                    "occupancy review rate number city\n"
                    f"Hotel 100 Private room No parties 1 2 4.0 {city}"
                ),
            }
            for city in ("Evansville", "South Bend")
        )
        for number, start, end, date in (
            (100, "Key West", "Evansville", "2022-01-01"),
            (200, "Evansville", "South Bend", "2022-01-03"),
            (300, "South Bend", "Key West", "2022-01-05"),
        ):
            content = (
                "Flight Number Price DepTime ArrTime ActualElapsedTime "
                "FlightDate OriginCityName DestCityName Distance\n"
                f"F{number} 20 09:00 10:00 1 hours 0 minutes {date} "
                f"{start} {end} 100"
            )
            # An unavailable flight must still define the dated route. The
            # agent can choose a separately listed ground option for that leg.
            if number == 200:
                content = f"There is no flight from {start} to {end} on {date}."
            records.append(
                {
                    "Description": f"Flight from {start} to {end} on {date}",
                    "Content": content,
                }
            )
            if number == 200:
                records.append(
                    {
                        "Description": f"Self-driving from {start} to {end}",
                        "Content": (
                            f"self-driving, from {start} to {end}, duration: "
                            "2 hours, distance: 100 km, cost: 5"
                        ),
                    }
                )
        item = {
            "days": 5,
            "dates": [f"2022-01-0{day}" for day in range(1, 6)],
            "org": "Key West",
            "reference_information": repr(records),
        }
        logistics = build_compact_reference_context(item, 0)
        experience = build_compact_reference_context(item, 1)
        logistics_scaffold = logistics.split(
            "ROLE-SPECIFIC COMPACT REFERENCE CATALOG", 1
        )[0]
        experience_scaffold = experience.split(
            "ROLE-SPECIFIC COMPACT REFERENCE CATALOG", 1
        )[0]
        self.assertEqual(logistics_scaffold, experience_scaffold)
        for expected in (
            'day=1 date=2022-01-01 kind=move current_city="from Key West to Evansville"',
            'day=2 date=2022-01-02 kind=stay current_city="Evansville"',
            'day=3 date=2022-01-03 kind=move current_city="from Evansville to South Bend"',
            'day=4 date=2022-01-04 kind=stay current_city="South Bend"',
            'day=5 date=2022-01-05 kind=move current_city="from South Bend to Key West"',
        ):
            self.assertIn(expected, logistics_scaffold)

    def test_prompt_and_generation_prefix_start_first_owned_value(self):
        prefixes = [
            build_agent_json_prefill(agent_idx, 3) for agent_idx in range(2)
        ]
        self.assertEqual(
            prefixes[0],
            '{"agent_id": 0, "assignments": [{"day": 1, '
            '"field": "current_city", "value": "',
        )
        self.assertEqual(
            prefixes[1],
            '{"agent_id": 1, "assignments": [{"day": 1, '
            '"field": "breakfast", "value": "',
        )
        prompts = [
            formatter(fixture_item())
            for formatter in get_single_turn_formatters(num_agents=2)
        ]
        for prefix, prompt in zip(prefixes, prompts):
            self.assertIn(prefix, prompt)
            self.assertIn("do not repeat any portion of it", prompt)
            self.assertIn("already-open value string", prompt)

    def test_non_prefill_prompt_requests_a_complete_json_object(self):
        prompts = [
            formatter(fixture_item())
            for formatter in get_single_turn_formatters(
                num_agents=2, force_json_prefix=False
            )
        ]
        for agent_idx, prompt in enumerate(prompts):
            self.assertNotIn("already supplied", prompt)
            self.assertNotIn("already-open value string", prompt)
            self.assertIn(
                f'{{"agent_id": {agent_idx}, "assignments": [', prompt
            )
            self.assertIn('first generated character must be "{"', prompt)


if __name__ == "__main__":
    unittest.main()
