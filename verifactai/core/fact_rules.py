"""
Rule-based factual validator for obvious impossibilities.

Catches claims that are deterministically false based on structured knowledge
(geography, dates, entity-attribute mappings) WITHOUT requiring NLI or retrieval.

This runs BEFORE the NLI pipeline and can override UNVERIFIABLE → CONTRADICTED
when a claim violates a hard factual rule.
"""

from __future__ import annotations

import re
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
    "sahara desert": "Africa",
    "sahara": "Africa",
    "amazon river": "South America",
    "nile river": "Africa",
    "nile": "Africa",
    "mississippi river": "North America",
    "mississippi": "North America",
    "danube river": "Europe",
    "ganges river": "India",
    "yangtze river": "China",
    "great barrier reef": "Australia",
    "kilimanjaro": "Tanzania",
    "mount kilimanjaro": "Tanzania",
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

# ── Historical event date ranges ─────────────────────────────────
# (event_keyword, min_year, max_year, correct_description)
HISTORICAL_EVENTS: list[tuple[str, int, int, str]] = [
    ("world war ii", 1939, 1945, "World War II lasted from 1939 to 1945"),
    ("world war i", 1914, 1918, "World War I lasted from 1914 to 1918"),
    ("french revolution", 1789, 1799, "The French Revolution was from 1789 to 1799"),
    ("moon landing", 1969, 1969, "The first moon landing was in 1969"),
    ("internet", 1960, 1995, "The internet was developed between the 1960s and 1990s"),
    ("roman empire fell", 395, 476, "The Western Roman Empire fell in 476 AD"),
    ("fall of rome", 395, 476, "The Western Roman Empire fell in 476 AD"),
]

# (person_keyword, birth_year_min, birth_year_max, death_year_min, death_year_max, desc)
PERSON_DATES: list[tuple[str, int, int, int, int, str]] = [
    ("albert einstein", 1879, 1879, 1955, 1955, "Albert Einstein lived 1879-1955"),
    ("shakespeare", 1564, 1564, 1616, 1616, "Shakespeare lived 1564-1616"),
    ("william shakespeare", 1564, 1564, 1616, 1616, "Shakespeare lived 1564-1616"),
    ("isaac newton", 1643, 1643, 1727, 1727, "Isaac Newton lived 1643-1727"),
    ("leonardo da vinci", 1452, 1452, 1519, 1519, "Leonardo da Vinci lived 1452-1519"),
    ("napoleon", 1769, 1769, 1821, 1821, "Napoleon lived 1769-1821"),
    ("marie curie", 1867, 1867, 1934, 1934, "Marie Curie lived 1867-1934"),
    ("mahatma gandhi", 1869, 1869, 1948, 1948, "Gandhi lived 1869-1948"),
    ("charles darwin", 1809, 1809, 1882, 1882, "Charles Darwin lived 1809-1882"),
]

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
            # If location IS a continent (e.g. "Africa" for Sahara), use it directly
            if not correct_continent and correct_location.lower() in CONTINENTS:
                correct_continent = correct_location
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

    # ── Rule 5: Historical event with wrong date ─────────────
    years_in_claim = [int(y) for y in re.findall(r"\b(\d{4})\b", lower)]
    if years_in_claim:
        for event_kw, min_year, max_year, correct_desc in HISTORICAL_EVENTS:
            if event_kw in lower:
                for year in years_in_claim:
                    if year < min_year - 10 or year > max_year + 10:
                        return RuleViolation(
                            rule_name="event_date",
                            claim=claim_text,
                            reason=f"The year {year} is incompatible with {event_kw}. {correct_desc}.",
                            correct_fact=correct_desc,
                        )

        # ── Rule 6: Person born/active in impossible era ─────
        for person_kw, b_min, b_max, d_min, d_max, desc in PERSON_DATES:
            if person_kw in lower:
                for year in years_in_claim:
                    if "born" in lower and (year < b_min - 5 or year > b_max + 5):
                        return RuleViolation(
                            rule_name="person_date",
                            claim=claim_text,
                            reason=f"{year} is wrong for {person_kw.title()}'s birth. {desc}.",
                            correct_fact=desc,
                        )
                    if "died" in lower and (year < d_min - 5 or year > d_max + 5):
                        return RuleViolation(
                            rule_name="person_date",
                            claim=claim_text,
                            reason=f"{year} is wrong for {person_kw.title()}'s death. {desc}.",
                            correct_fact=desc,
                        )

    # ── Rule 7: Shakespeare / person + wrong century ─────────
    centuries_map = {
        "21st century": (2001, 2100),
        "20th century": (1901, 2000),
        "19th century": (1801, 1900),
        "18th century": (1701, 1800),
        "17th century": (1601, 1700),
        "16th century": (1501, 1600),
        "15th century": (1401, 1500),
    }
    for person_kw, b_min, _b_max, _d_min, d_max, desc in PERSON_DATES:
        if person_kw in lower:
            for century_str, (c_start, c_end) in centuries_map.items():
                if century_str in lower and (
                    c_start > d_max + 50 or c_end < b_min - 50
                ):
                        return RuleViolation(
                            rule_name="person_era",
                            claim=claim_text,
                            reason=f"{person_kw.title()} lived {b_min}-{d_max}, not the {century_str}.",
                            correct_fact=desc,
                        )

    return None
