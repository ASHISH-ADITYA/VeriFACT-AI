"""
Rule-based factual validator for obvious impossibilities.

Catches claims that are deterministically false based on structured knowledge
(geography, dates, entity-attribute mappings) WITHOUT requiring NLI or retrieval.

This runs BEFORE the NLI pipeline and can override UNVERIFIABLE → CONTRADICTED
when a claim violates a hard factual rule.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RuleViolation:
    """A deterministic factual violation found by rule matching."""

    rule_name: str
    claim: str
    reason: str
    correct_fact: str


# ── Geographic knowledge base ────────────────────────────────────

LANDMARK_LOCATIONS: dict[str, str] = {
    "great wall of china": "China",
    "great wall": "China",
    "eiffel tower": "France",
    "statue of liberty": "United States",
    "taj mahal": "India",
    "colosseum": "Italy",
    "big ben": "United Kingdom",
    "sydney opera house": "Australia",
    "machu picchu": "Peru",
    "christ the redeemer": "Brazil",
    "pyramids of giza": "Egypt",
    "stonehenge": "United Kingdom",
    "mount rushmore": "United States",
    "leaning tower of pisa": "Italy",
    "angkor wat": "Cambodia",
    "petra": "Jordan",
    "mount fuji": "Japan",
    "mount everest": "Nepal",
    "niagara falls": "United States",
    "golden gate bridge": "United States",
    "kremlin": "Russia",
    "buckingham palace": "United Kingdom",
    "vatican": "Italy",
    "acropolis": "Greece",
    "panama canal": "Panama",
}

CAPITAL_COUNTRY: dict[str, str] = {
    "tokyo": "Japan",
    "paris": "France",
    "london": "United Kingdom",
    "berlin": "Germany",
    "beijing": "China",
    "moscow": "Russia",
    "washington": "United States",
    "new delhi": "India",
    "canberra": "Australia",
    "ottawa": "Canada",
    "rome": "Italy",
    "madrid": "Spain",
    "cairo": "Egypt",
    "brasilia": "Brazil",
    "buenos aires": "Argentina",
    "islamabad": "Pakistan",
    "nairobi": "Kenya",
    "bangkok": "Thailand",
    "hanoi": "Vietnam",
    "seoul": "South Korea",
    "pyongyang": "North Korea",
    "stockholm": "Sweden",
    "oslo": "Norway",
    "helsinki": "Finland",
    "athens": "Greece",
    "lisbon": "Portugal",
    "amsterdam": "Netherlands",
    "brussels": "Belgium",
    "vienna": "Austria",
    "warsaw": "Poland",
    "prague": "Czech Republic",
    "budapest": "Hungary",
    "dublin": "Ireland",
    "bern": "Switzerland",
}

COUNTRY_CONTINENT: dict[str, str] = {
    "china": "Asia",
    "japan": "Asia",
    "india": "Asia",
    "russia": "Europe/Asia",
    "france": "Europe",
    "germany": "Europe",
    "united kingdom": "Europe",
    "italy": "Europe",
    "spain": "Europe",
    "brazil": "South America",
    "argentina": "South America",
    "united states": "North America",
    "canada": "North America",
    "mexico": "North America",
    "australia": "Oceania",
    "egypt": "Africa",
    "south africa": "Africa",
    "kenya": "Africa",
    "nigeria": "Africa",
    "pakistan": "Asia",
    "thailand": "Asia",
    "vietnam": "Asia",
    "south korea": "Asia",
    "turkey": "Europe/Asia",
    "greece": "Europe",
    "portugal": "Europe",
    "netherlands": "Europe",
    "sweden": "Europe",
    "norway": "Europe",
    "peru": "South America",
    "colombia": "South America",
    "chile": "South America",
}

CONTINENTS = {
    "africa",
    "asia",
    "europe",
    "north america",
    "south america",
    "oceania",
    "antarctica",
}

# ── Famous person facts ──────────────────────────────────────────

PERSON_KNOWN_FOR: dict[str, list[str]] = {
    "albert einstein": ["physics", "theory of relativity", "photoelectric effect"],
    "isaac newton": ["physics", "gravity", "calculus", "optics"],
    "alexander graham bell": ["telephone"],
    "thomas edison": ["light bulb", "phonograph", "electricity"],
    "nikola tesla": ["electricity", "alternating current", "radio"],
    "marie curie": ["radioactivity", "physics", "chemistry", "polonium", "radium"],
    "charles darwin": ["evolution", "natural selection", "biology"],
    "leonardo da vinci": ["painting", "art", "mona lisa", "invention"],
    "napoleon bonaparte": ["france", "military", "emperor"],
    "mahatma gandhi": ["india", "independence", "nonviolence"],
    "william shakespeare": ["playwright", "poetry", "hamlet", "romeo"],
    "wright brothers": ["airplane", "flight"],
    "alexander fleming": ["penicillin"],
}


def check_rules(claim_text: str) -> RuleViolation | None:
    """
    Check a claim against hard factual rules.

    Returns a RuleViolation if the claim is deterministically false,
    or None if no rule applies (claim should go through NLI pipeline).
    """
    lower = claim_text.lower().strip()

    # ── Rule 1: Landmark in wrong location ────────────────────
    for landmark, correct_location in LANDMARK_LOCATIONS.items():
        if landmark in lower:
            # Check if claim places it somewhere else
            for loc_name, _loc_continent in COUNTRY_CONTINENT.items():
                if loc_name in lower and loc_name.lower() != correct_location.lower():
                    return RuleViolation(
                        rule_name="landmark_location",
                        claim=claim_text,
                        reason=f"The {landmark.title()} is in {correct_location}, not {loc_name.title()}.",
                        correct_fact=f"The {landmark.title()} is located in {correct_location}.",
                    )
            # Check if claim places it on wrong continent
            correct_continent = COUNTRY_CONTINENT.get(correct_location.lower(), "")
            for continent in CONTINENTS:
                if continent in lower and continent not in correct_continent.lower():
                    return RuleViolation(
                        rule_name="landmark_continent",
                        claim=claim_text,
                        reason=f"The {landmark.title()} is in {correct_location} ({correct_continent}), not in {continent.title()}.",
                        correct_fact=f"The {landmark.title()} is located in {correct_location}, {correct_continent}.",
                    )

    # ── Rule 2: Capital of wrong country ──────────────────────
    for capital, correct_country in CAPITAL_COUNTRY.items():
        if capital in lower and ("capital" in lower or "capital of" in lower):
            for country in COUNTRY_CONTINENT:
                if country in lower and country != correct_country.lower():
                    return RuleViolation(
                        rule_name="capital_country",
                        claim=claim_text,
                        reason=f"{capital.title()} is the capital of {correct_country}, not {country.title()}.",
                        correct_fact=f"{capital.title()} is the capital of {correct_country}.",
                    )

    # ── Rule 3: Country on wrong continent ────────────────────
    for country, correct_continent in COUNTRY_CONTINENT.items():
        if country in lower:
            for continent in CONTINENTS:
                if (
                    continent in lower
                    and continent not in correct_continent.lower()
                    and any(
                        w in lower
                        for w in [
                            "located in",
                            "is in",
                            "country in",
                            "part of",
                            "continent",
                            "situated in",
                        ]
                    )
                ):
                    return RuleViolation(
                        rule_name="country_continent",
                        claim=claim_text,
                        reason=f"{country.title()} is in {correct_continent}, not {continent.title()}.",
                        correct_fact=f"{country.title()} is located in {correct_continent}.",
                    )

    # ── Rule 4: Person did wrong thing ────────────────────────
    for person, known_for in PERSON_KNOWN_FOR.items():
        if person in lower:
            # Check if claim attributes something known for someone else
            for other_person, other_known in PERSON_KNOWN_FOR.items():
                if other_person == person:
                    continue
                for achievement in other_known:
                    if achievement in lower and (
                        "invented" in lower
                        or "discovered" in lower
                        or "created" in lower
                        or "wrote" in lower
                        or "built" in lower
                    ):
                        return RuleViolation(
                            rule_name="person_achievement",
                            claim=claim_text,
                            reason=f"{person.title()} is known for {', '.join(known_for)}, not {achievement}.",
                            correct_fact=f"The {achievement} is associated with {other_person.title()}, not {person.title()}.",
                        )

    return None
