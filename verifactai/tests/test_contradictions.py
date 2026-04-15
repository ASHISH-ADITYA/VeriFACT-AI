"""Regression tests for obvious contradiction detection.

These claims MUST return CONTRADICTED — if any returns UNVERIFIABLE,
the verdict engine has regressed.
"""

from __future__ import annotations

from core.fact_rules import check_rules


class TestRuleBasedContradictions:
    """Test that rule-based validator catches obvious impossibilities."""

    def test_great_wall_in_south_america(self):
        v = check_rules("The Great Wall of China is located in South America.")
        assert v is not None
        assert v.rule_name == "landmark_continent"

    def test_eiffel_tower_in_germany(self):
        v = check_rules("The Eiffel Tower is in Berlin, Germany.")
        assert v is not None

    def test_taj_mahal_in_china(self):
        v = check_rules("The Taj Mahal is located in China.")
        assert v is not None

    def test_mount_everest_in_africa(self):
        v = check_rules("Mount Everest is in Africa.")
        assert v is not None

    def test_statue_of_liberty_in_france(self):
        v = check_rules("The Statue of Liberty is in Paris, France.")
        assert v is not None

    def test_colosseum_in_japan(self):
        v = check_rules("The Colosseum is in Japan.")
        assert v is not None

    def test_einstein_invented_airplane(self):
        v = check_rules("Albert Einstein invented the airplane.")
        assert v is not None
        assert v.rule_name == "person_achievement"

    def test_darwin_invented_light_bulb(self):
        v = check_rules("Charles Darwin invented the light bulb.")
        assert v is not None

    def test_newton_discovered_penicillin(self):
        v = check_rules("Isaac Newton discovered penicillin.")
        assert v is not None

    def test_tokyo_capital_of_china(self):
        v = check_rules("Tokyo is the capital of China.")
        assert v is not None

    def test_india_in_europe(self):
        v = check_rules("India is a country in Europe.")
        assert v is not None

    def test_brazil_in_africa(self):
        v = check_rules("Brazil is located in Africa.")
        assert v is not None


class TestRuleNoFalsePositives:
    """Correct claims must NOT trigger rule violations."""

    def test_eiffel_tower_in_paris(self):
        assert check_rules("The Eiffel Tower is in Paris, France.") is None

    def test_great_wall_in_china(self):
        assert check_rules("The Great Wall of China is in China.") is None

    def test_tokyo_capital_of_japan(self):
        assert check_rules("Tokyo is the capital of Japan.") is None

    def test_einstein_physics(self):
        assert check_rules("Albert Einstein worked on the theory of relativity.") is None

    def test_water_boils_100(self):
        assert check_rules("Water boils at 100 degrees Celsius at sea level.") is None

    def test_generic_statement(self):
        assert check_rules("The weather is nice today.") is None
