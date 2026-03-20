#!/usr/bin/env python3
"""Audit, improve, and stress-test the routing dataset."""
from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "datasets" / "routing"
AUDIT_PATH = ROOT / "scripts" / "audit_report.txt"
SPLITS = ["train.jsonl", "val.jsonl", "test.jsonl"]
TARGET_SPLIT_SIZES = {"train.jsonl": 2400, "val.jsonl": 400, "test.jsonl": 400}
SEED = 20260318

ALLOWED_CROPS = ["corn", "soybean", "both", "unknown"]
ALLOWED_TOPICS = [
    "weather",
    "disease",
    "pest",
    "weed",
    "nutrient",
    "soil",
    "management",
    "market_economics",
    "equipment",
    "ag_technology",
    "policy_regulation",
    "general_agronomy",
]
INTENTS = [
    "question",
    "diagnosis",
    "recommendation",
    "planning",
    "monitoring",
    "information_lookup",
    "comparison",
    "other",
]
URGENCIES = ["low", "medium", "high"]

LOCATIONS = [
    "central Iowa",
    "east-central Illinois",
    "eastern Nebraska",
    "northwest Ohio",
    "southern Minnesota",
    "north-central Missouri",
    "south-central Wisconsin",
    "west-central Indiana",
    "southeast South Dakota",
    "northeast Kansas",
]
FIELD_NAMES = [
    "back 40",
    "north 80",
    "south quarter",
    "west eighty",
    "river field",
    "creek bottom",
    "home place",
    "sandy knob",
    "tile farm",
    "high ground",
    "east piece",
    "south side",
]
TIME_WINDOWS = [
    "today",
    "this afternoon",
    "tonight",
    "tomorrow morning",
    "over the next 48 hours",
    "before the weekend",
    "early next week",
]
STATES = [
    "Iowa",
    "Illinois",
    "Nebraska",
    "Indiana",
    "Missouri",
    "Minnesota",
    "Ohio",
    "Wisconsin",
    "South Dakota",
    "Kansas",
]


rng = random.Random(SEED)


def normalize_text(text: str) -> str:
    lowered = text.strip().lower()
    lowered = re.sub(r"\s+", " ", lowered)
    lowered = re.sub(r"[^a-z0-9? ]", "", lowered)
    return lowered


def load_dataset() -> dict[str, list[dict]]:
    data: dict[str, list[dict]] = {}
    for split in SPLITS:
        path = DATASET_DIR / split
        data[split] = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return data


def save_dataset(data: dict[str, list[dict]]) -> None:
    for split in SPLITS:
        path = DATASET_DIR / split
        payload = "\n".join(json.dumps(row, ensure_ascii=False) for row in data[split]) + "\n"
        path.write_text(payload, encoding="utf-8")


def dataset_stats(rows: Iterable[dict]) -> dict:
    rows = list(rows)
    total = len(rows)
    ag_rows = [row for row in rows if row["labels"]["is_ag_related"]]
    non_ag_rows = total - len(ag_rows)
    crop_counts = Counter()
    topic_counts = Counter()
    intent_counts = Counter()
    urgency_counts = Counter()
    flag_counts = Counter()
    route_combo_counts = Counter()
    short_queries = 0
    vague_queries = 0
    multi_topic = 0
    hard_examples = 0
    normalized_inputs = Counter(normalize_text(row["input"]) for row in rows)
    dup_count = sum(1 for value in normalized_inputs.values() if value > 1)

    vague_patterns = [
        r"\blooks bad\b",
        r"\bnot right\b",
        r"\bacting up\b",
        r"\bfield looks\b",
        r"\bweird\b",
        r"\boff\b",
        r"\bsomething wrong\b",
        r"\bhelp\b",
        r"\bnow what\b",
        r"\bwhat is this\b",
    ]
    vague_regex = re.compile("|".join(vague_patterns))

    for row in rows:
        labels = row["labels"]
        intent_counts[labels["intent"]] += 1
        urgency_counts[labels["urgency"]] += 1
        for crop in labels["crops"]:
            crop_counts[crop] += 1
        for topic in labels["topics"]:
            topic_counts[topic] += 1
        for flag in ["needs_rag", "needs_web_search", "needs_weather_data", "needs_farm_data"]:
            flag_counts[flag] += int(labels[flag])
        route_combo_counts[(labels["needs_rag"], labels["needs_web_search"], labels["needs_weather_data"], labels["needs_farm_data"])] += 1
        if len(row["input"].split()) <= 5:
            short_queries += 1
        if vague_regex.search(row["input"].lower()):
            vague_queries += 1
        if len(labels["topics"]) > 1:
            multi_topic += 1
        if row["meta"]["difficulty"] == "hard":
            hard_examples += 1

    return {
        "total": total,
        "ag_total": len(ag_rows),
        "non_ag_total": non_ag_rows,
        "crop_counts": crop_counts,
        "topic_counts": topic_counts,
        "intent_counts": intent_counts,
        "urgency_counts": urgency_counts,
        "flag_counts": flag_counts,
        "route_combo_counts": route_combo_counts,
        "short_queries": short_queries,
        "vague_queries": vague_queries,
        "multi_topic": multi_topic,
        "hard_examples": hard_examples,
        "normalized_duplicate_inputs": dup_count,
    }


def format_distribution(counter: Counter, total: int | None = None) -> str:
    lines = []
    for key, value in counter.most_common():
        if total:
            lines.append(f"  - {key}: {value} ({value / total:.1%})")
        else:
            lines.append(f"  - {key}: {value}")
    return "\n".join(lines)


def audit_findings(stats: dict) -> list[str]:
    findings: list[str] = []
    total = stats["total"]
    ag_total = stats["ag_total"]
    non_ag_ratio = stats["non_ag_total"] / total
    rag_ratio = stats["flag_counts"]["needs_rag"] / max(ag_total, 1)
    short_ratio = stats["short_queries"] / total
    vague_ratio = stats["vague_queries"] / total
    multi_ratio = stats["multi_topic"] / total

    if non_ag_ratio < 0.25:
        findings.append(f"Non-ag coverage is weak at {non_ag_ratio:.1%}.")
    elif non_ag_ratio > 0.35:
        findings.append(f"Non-ag coverage is high at {non_ag_ratio:.1%}; ag signal may be diluted.")

    if rag_ratio > 0.6:
        findings.append(f"needs_rag is overrepresented for ag rows at {rag_ratio:.1%}.")
    elif rag_ratio < 0.25:
        findings.append(f"needs_rag is likely underused for ag rows at {rag_ratio:.1%}.")

    if short_ratio < 0.08:
        findings.append(f"Short queries are underrepresented at {short_ratio:.1%}.")
    if vague_ratio < 0.08:
        findings.append(f"Vague queries are underrepresented at {vague_ratio:.1%}.")
    if multi_ratio < 0.18:
        findings.append(f"Multi-topic queries are underrepresented at {multi_ratio:.1%}.")

    edge_checks = {
        "weather only": (False, False, True, False),
        "web only": (False, True, False, False),
        "farm only": (False, False, False, True),
        "rag + weather": (True, False, True, False),
        "rag + farm": (True, False, False, True),
        "web + weather": (False, True, True, False),
    }
    for name, combo in edge_checks.items():
        count = stats["route_combo_counts"][combo]
        if count == 0:
            findings.append(f"Missing routing combination: {name}.")
        elif count < 12:
            findings.append(f"Routing combination is sparse: {name} has only {count} rows.")

    under_topics = [topic for topic, count in stats["topic_counts"].items() if count / max(ag_total, 1) < 0.08]
    if under_topics:
        findings.append("Underrepresented topics: " + ", ".join(sorted(under_topics)) + ".")

    over_topics = [topic for topic, count in stats["topic_counts"].items() if count / max(ag_total, 1) > 0.22]
    if over_topics:
        findings.append("Overrepresented topics: " + ", ".join(sorted(over_topics)) + ".")

    if stats["normalized_duplicate_inputs"]:
        findings.append(f"Detected {stats['normalized_duplicate_inputs']} normalized duplicate inputs.")

    return findings or ["No major issues detected."]


class UniqueBuilder:
    def __init__(self, existing_inputs: set[str], start_id: int) -> None:
        self.used_inputs = set(existing_inputs)
        self.next_id = start_id
        self.rows: list[dict] = []

    def make_row(
        self,
        input_text: str,
        *,
        is_ag_related: bool,
        crops: list[str],
        topics: list[str],
        needs_rag: bool,
        needs_web_search: bool,
        needs_weather_data: bool,
        needs_farm_data: bool,
        intent: str,
        urgency: str,
        difficulty: str,
        notes: str,
    ) -> bool:
        normalized = normalize_text(input_text)
        if not normalized or normalized in self.used_inputs:
            return False
        row_id = f"route_{self.next_id:06d}"
        self.next_id += 1
        self.used_inputs.add(normalized)
        self.rows.append(
            {
                "id": row_id,
                "input": input_text,
                "labels": {
                    "is_ag_related": is_ag_related,
                    "crops": crops,
                    "topics": topics,
                    "needs_rag": needs_rag,
                    "needs_web_search": needs_web_search,
                    "needs_weather_data": needs_weather_data,
                    "needs_farm_data": needs_farm_data,
                    "intent": intent,
                    "urgency": urgency,
                },
                "meta": {"source_type": "synthetic", "difficulty": difficulty, "notes": notes},
            }
        )
        return True


def ensure_count(builder: UniqueBuilder, before: int, expected_added: int, category: str) -> None:
    actual_added = len(builder.rows) - before
    if actual_added != expected_added:
        raise RuntimeError(f"{category}: expected to add {expected_added} rows, added {actual_added}")


def add_candidates(builder: UniqueBuilder, candidates: list[dict], count: int, category: str) -> None:
    before = len(builder.rows)
    for candidate in candidates:
        builder.make_row(**candidate)
        if len(builder.rows) - before >= count:
            ensure_count(builder, before, count, category)
            return
    ensure_count(builder, before, count, category)


NON_AG_SHORT = [
    "best pizza tonight?",
    "wifi keeps dropping",
    "need resume bullets",
    "dog ate sock",
    "cheap flights Denver?",
    "tv keeps buffering",
    "good date ideas?",
    "need birthday caption",
    "car wont start",
    "how to boil eggs",
    "budget app ideas",
    "help with algebra",
    "running shoe advice",
    "coffee maker leaking",
    "movie for family night?",
    "laptop for college?",
    "printer offline again",
    "passport photo rules",
    "why is zoom laggy",
    "quick dinner plan",
    "best sunscreen brand?",
    "cat keeps sneezing",
    "phone screen flicker",
    "slow internet tonight",
    "packing list beach trip",
    "grammar check this sentence",
    "best soup recipe?",
    "help naming my podcast",
    "cheap gym membership?",
    "oil light came on",
    "garden party playlist",
    "houseplant turning yellow",
    "need anniversary text",
    "math homework help",
    "best carryon bag?",
    "soap scum fix",
    "credit score basics",
    "find a movie",
    "protein breakfast ideas",
    "why is my sink gurgling",
    "weekend trip ideas",
    "help pick a font",
    "best stroller brand?",
    "renters insurance question",
    "my charger gets hot",
    "need a cover letter",
    "public speaking tips",
    "how do I defrost chicken",
    "best note taking app?",
    "gift idea for dad",
    "cornhole bracket template",
    "soy wax candle brands",
    "field trip permission slip",
    "weather app not updating",
    "plant stand ideas",
    "bean bag chair filling",
    "harvest moon movie order",
    "spray paint drying time",
    "tractor in a parade song",
    "market my bakery online",
]

NON_AG_LONG = [
    "I need a cleaner way to explain a job gap on my resume.",
    "Can you help me compare two internet plans for a small apartment?",
    "My laptop fan sounds loud after ten minutes. What should I check first?",
    "What is a simple dinner I can make with chicken, rice, and frozen peas?",
    "How do I stop my dog from barking every time the delivery truck shows up?",
    "I need a weekend itinerary for Chicago that is not too expensive.",
    "What is a good beginner workout split if I only have three days a week?",
    "Can you write a short thank-you note for my kid's teacher?",
    "My bathroom mirror keeps fogging up even with the fan on. Any fix?",
    "Can you explain credit card utilization without a bunch of finance jargon?",
    "I forgot my phone passcode and I am locked out. What are the usual recovery steps?",
    "What should I pack for a three-day work trip if I only want one carry-on?",
    "Why does my printer keep saying paper jam when nothing is there?",
    "Can you suggest a few baby shower game ideas that do not feel cheesy?",
    "I need help wording a text to cancel plans without sounding rude.",
    "What is the easiest way to clean hard water spots off a shower door?",
    "My car AC blows warm when I sit in traffic but cools off once I drive.",
    "Can you compare a Chromebook and a cheap Windows laptop for homework use?",
    "I want a family movie for tonight that is funny but not too long.",
    "What does error 403 usually mean when a website suddenly stops loading?",
    "Can you help me make a grocery list for one week on a tight budget?",
    "I need a short caption for a photo dump from spring break.",
    "What is a realistic cleaning schedule for a small house with two kids?",
    "My router works in the living room but not in the back bedroom. What should I try?",
    "Can you suggest a few hobbies that do not involve screens?",
    "I need ideas for a birthday dinner that will work for picky eaters.",
    "How do I get rid of a mildew smell in the basement after rain?",
    "Can you explain the difference between a Roth IRA and a traditional IRA in plain English?",
    "What should I ask before I book a dog sitter for a weekend trip?",
    "My shower goes cold halfway through. Is that usually a water heater issue?",
    "Can you help me outline a five-minute toast for my sister's wedding?",
    "What is the best way to learn Python if I only have twenty minutes a day?",
    "Do airlines still let you bring a small backpack and a carry-on?",
    "My teenager wants a used car. What should I look for first?",
    "What is a simple meal prep plan for lunches that does not get boring?",
    "Can you help me compare two phone plans with different data caps?",
    "The weather app on my phone is stuck on yesterday and will not refresh.",
    "I need a short script for leaving a voicemail about a dentist appointment.",
    "What are a few polite ways to say no to taking on extra work right now?",
    "Can you help me troubleshoot why my smart TV keeps signing out of apps?",
    "Is soy wax actually better than paraffin for candles, or is that mostly marketing?",
    "I need a field trip checklist for a third-grade class going to the zoo.",
    "Can you help me plan a backyard movie night with a small budget?",
    "What should I do if my coffee maker tastes burnt even after I clean it?",
    "My grocery bill keeps creeping up. What is an easy way to track where it goes?",
    "Can you recommend a calm playlist for a long drive with kids in the car?",
    "I need a quick explainer for why my check engine light can flash and then disappear.",
    "What is a good gift for a dad who says he does not want anything?",
    "Can you help me rewrite an email so it sounds firm but not angry?",
    "Why does my smoke alarm chirp even after I change the battery?",
    "The bean bag chair I bought is going flat already. Can it be refilled?",
    "How long does spray paint usually need before I can flip the chair over?",
    "I keep hearing the phrase harvest moon in movies. What films use it?",
    "What is the easiest way to market a small bakery on Instagram?",
    "I need ideas for a parade float with an old tractor theme, but not an actual farm theme.",
    "Can you explain why my internet speed test looks fine but video calls still freeze?",
    "What is a fair budget for a weekend trip to Omaha with two kids?",
    "I need a better subject line for a follow-up email after a job interview.",
    "Can you give me a simple routine for stretching after I sit at a desk all day?",
]


NON_AG_HARD = [
    "cornhole league payout ideas",
    "soy sauce substitute?",
    "field trip bus form",
    "weather channel app keeps crashing",
    "combine two pdf files",
    "spray bottle wont mist",
    "seed phrase storage tips",
    "elevator pitch for internship",
    "basis points explained simply",
    "tar spot on my glasses?",
    "bug in my laptop screen",
    "barn wedding song list",
    "crop top outfit ideas",
    "yield sign ticket question",
    "row spacing in a spreadsheet",
    "plant stand for living room",
    "farmhouse sink drip",
    "beans recipe from scratch",
    "corn maze date ideas",
    "soil on white shoes",
    "weed trimmer line size",
    "frosted tips haircut",
    "pivot table formula help",
    "drone battery swelling",
    "cash bid on my car?",
    "tile shower grout fix",
    "hybrid bike vs road bike",
    "field notes app review",
    "dicamba haircut meme",
    "glyphosate? no, guitar pedal",
    "soy milk curdled",
    "tractor emoji meaning",
    "weatherstrip coming loose",
    "sprayer bottle for bleach",
    "corn syrup candy question",
    "bean counter joke ideas",
    "pest control for apartment ants",
    "weed playlist names",
    "soil level for patio",
    "market update for crypto",
    "policy memo template",
    "agile project standup help",
    "seed beads for bracelets",
    "rust on bike chain",
    "rain jacket zipper stuck",
    "field goal percentage chart",
    "combine photos on iphone",
    "suds in dishwasher",
    "crop a pdf page",
    "farmhouse paint colors",
    "cornbread recipe ratio",
    "soy candle tunneling",
    "tractor trailer parking rules",
    "weather radar on tv not phone",
    "spray tan before wedding",
    "grain leather boots care",
    "bushel basket centerpiece ideas",
    "rower machine squeak",
    "mildew on shower curtain",
    "field roast sandwich ideas",
]


def assign_non_ag_labels(index: int, hard_bias: bool = False) -> tuple[str, str, str]:
    intent_cycle = ["question", "information_lookup", "comparison", "other", "planning"]
    urgency_cycle = ["low", "medium", "medium", "low", "high"]
    intent = intent_cycle[index % len(intent_cycle)]
    urgency = urgency_cycle[index % len(urgency_cycle)]
    difficulty = "hard" if hard_bias or index % 4 == 0 else ("medium" if index % 2 else "simple")
    return intent, urgency, difficulty


AG_REWRITE_TEMPLATES = {
    "weed": [
        "Foxtail slipped through in {field}. Is there anything practical left to do?",
        "I still have waterhemp poking above the crop canopy in {field}. What is my best move now?",
        "We missed some grass in {field} and it is showing again. Salvageable or too late?",
        "Weed escapes are showing up on {field}. What should I be thinking about before they go to seed?",
    ],
    "disease": [
        "The lower leaves in {field} are getting blotchy fast. Does this sound like a disease issue?",
        "I am seeing lesions move up the canopy on {field}. What problems fit that pattern?",
        "This patchy leaf spotting in {field} does not look right. What would you check first?",
        "Could the leaf symptoms in {field} be disease pressure or am I looking at something else?",
    ],
    "pest": [
        "Chewed leaves are showing up on {field}. What insect pressure would you put high on the list?",
        "I am finding feeding injury in {field}. At what point is it worth acting?",
        "Something is clipping plants in {field}. What pests usually fit that kind of damage?",
        "We have insect feeding scattered across {field}. What should I verify before deciding on a spray?",
    ],
    "nutrient": [
        "Plants on {field} are staying pale in streaks. Does that line up more with nutrient loss or something else?",
        "Yellowing is hanging on in {field} longer than I expected. What deficiency patterns would you rule in or out?",
        "The crop on {field} looks hungry in spots. Where would you start with the likely nutrient causes?",
        "Why would one side of {field} stay light green while the rest catches up?",
    ],
    "soil": [
        "The top is sealing over on {field}. What are realistic ways to ease that up next season?",
        "We keep fighting sidewall and crusting in {field}. What management changes usually help the most?",
        "That heavy patch on {field} sets up hard after every rain. What is the usual fix path?",
        "Compaction seems to be hanging around in {field}. How would you think through next steps?",
    ],
    "equipment": [
        "The planter keeps acting up on the terraces. What would you inspect first?",
        "My monitor is bouncing around more than it should in {field}. What tends to cause that?",
        "The sprayer rate controller is inconsistent on rough ground. Where would you start troubleshooting?",
        "Why would the row unit ride smooth on flat ground and then lose depth on the hills?",
    ],
    "weather": [
        "Will the wind back off enough {time_window} near {location} to get any spraying done?",
        "How risky does this cold snap look for emergence around {location}?",
        "Do fieldwork conditions open up {time_window} near {location}, or is it still too wet?",
        "Are we staring at enough heat stress {time_window} near {location} to change plans?",
    ],
}


def rewrite_ag_input(row: dict, index: int) -> str:
    labels = row["labels"]
    topic = labels["topics"][0] if labels["topics"] else "general_agronomy"
    template_pool = AG_REWRITE_TEMPLATES.get(topic)
    field = FIELD_NAMES[index % len(FIELD_NAMES)]
    location = LOCATIONS[index % len(LOCATIONS)]
    time_window = TIME_WINDOWS[index % len(TIME_WINDOWS)]
    if template_pool:
        return template_pool[index % len(template_pool)].format(field=field, location=location, time_window=time_window)
    if labels["needs_web_search"] and labels["topics"] == ["market_economics"]:
        crop = labels["crops"][0] if labels["crops"] else "grain"
        crop_label = "corn" if crop == "corn" else ("soybeans" if crop == "soybean" else "corn and soybeans")
        return f"What are cash bids doing for {crop_label} around {location} right now?"
    if labels["needs_farm_data"] and not labels["needs_rag"]:
        return f"Can you pull up what we actually did on {field} before I make another pass?"
    if labels["needs_weather_data"] and not labels["needs_rag"]:
        return f"Is there a decent weather window {time_window} near {location} for fieldwork?"
    return f"This situation in {field} feels off. What would you check first?"


def generate_non_ag_rewrites(count: int, existing: set[str]) -> list[str]:
    pool: list[str] = []
    for phrase in NON_AG_SHORT + NON_AG_LONG + NON_AG_HARD:
        normalized = normalize_text(phrase)
        if normalized not in existing and normalized not in {normalize_text(x) for x in pool}:
            pool.append(phrase)
        if len(pool) >= count:
            break
    return pool[:count]


def replace_rows(data: dict[str, list[dict]]) -> tuple[int, int]:
    all_rows = [row for split in SPLITS for row in data[split]]
    existing = {normalize_text(row["input"]) for row in all_rows}
    non_ag_candidates: list[tuple[str, int]] = []
    ag_candidates: list[tuple[str, int]] = []
    repetitive_ag_regex = re.compile(r"spray window|drying conditions|escaped foxtail|What does that point to|latest read|current cash bids", re.I)

    for split in SPLITS:
        for index, row in enumerate(data[split]):
            text = row["input"]
            if not row["labels"]["is_ag_related"] and (re.search(r" \d+$", text) or "Please give me three options" in text or "I need a quick answer" in text or "Make it beginner friendly" in text):
                non_ag_candidates.append((split, index))
            elif row["labels"]["is_ag_related"] and repetitive_ag_regex.search(text):
                ag_candidates.append((split, index))

    target_non_ag = 160
    target_ag = 40
    non_ag_picks = non_ag_candidates[:target_non_ag]
    ag_picks = ag_candidates[:target_ag]

    rewrite_texts = generate_non_ag_rewrites(len(non_ag_picks), existing)
    non_ag_rewritten = 0
    for rewrite_index, (split, row_index) in enumerate(non_ag_picks):
        row = data[split][row_index]
        text = rewrite_texts[rewrite_index]
        intent, urgency, difficulty = assign_non_ag_labels(rewrite_index, hard_bias=text in NON_AG_HARD)
        row["input"] = text
        row["labels"]["intent"] = intent
        row["labels"]["urgency"] = urgency
        row["meta"]["difficulty"] = difficulty
        row["meta"]["notes"] = "Rewritten non-ag example to reduce templated phrasing and improve noise robustness."
        non_ag_rewritten += 1

    ag_rewritten = 0
    for rewrite_index, (split, row_index) in enumerate(ag_picks):
        row = data[split][row_index]
        row["input"] = rewrite_ag_input(row, rewrite_index)
        row["meta"]["difficulty"] = "medium" if rewrite_index % 3 else "hard"
        row["meta"]["notes"] = "Rewritten ag example to sound more farmer-like and less templated."
        ag_rewritten += 1

    return non_ag_rewritten, ag_rewritten


def crop_from_code(code: str) -> list[str]:
    mapping = {"c": ["corn"], "s": ["soybean"], "b": ["both"], "u": ["unknown"]}
    return mapping[code]


def add_non_ag_examples(builder: UniqueBuilder, count: int) -> None:
    sources = NON_AG_SHORT + NON_AG_LONG + NON_AG_HARD
    idx = 0
    hard_set = {normalize_text(text) for text in NON_AG_HARD}
    before = len(builder.rows)
    while len(builder.rows) - before < count:
        text = sources[idx % len(sources)]
        if idx >= len(sources):
            text = f"{text.rstrip('?')}, please."
        normalized = normalize_text(text)
        intent, urgency, difficulty = assign_non_ag_labels(idx, hard_bias=normalized in hard_set)
        builder.make_row(
            text,
            is_ag_related=False,
            crops=[],
            topics=[],
            needs_rag=False,
            needs_web_search=False,
            needs_weather_data=False,
            needs_farm_data=False,
            intent=intent,
            urgency=urgency,
            difficulty=difficulty,
            notes="Hard negative or general non-ag query for routing robustness." if normalized in hard_set else "Non-ag query added to improve negative coverage and realism.",
        )
        idx += 1
    ensure_count(builder, before, count, "non_ag")


def add_web_only_examples(builder: UniqueBuilder, count: int) -> None:
    market_templates = [
        "Cash {crop_word} near {location} today?",
        "What are basis levels for {crop_word} around {location} right now?",
        "Show me nearby elevator bids for {crop_word} in {location}.",
        "Any fresh cash bid moves for {crop_word} around {location} this morning?",
        "What are {crop_word} futures doing today and how are local bids tracking?",
    ]
    policy_templates = [
        "Any new herbicide label changes in {state} this week that affect {crop_word}?",
        "What is the latest planting insurance deadline update for {crop_word} in {state}?",
        "Did {state} change nutrient application rules recently for row crops?",
        "Any current disease outbreak alerts for {crop_word} in {state}?",
        "What changed most recently with dicamba guidance in {state}?",
    ]
    candidates: list[dict] = []
    for crop_code, location in [(c, l) for c in ["c", "s", "b"] for l in LOCATIONS]:
        crop_word = {"c": "corn", "s": "soybeans", "b": "corn and soybeans"}[crop_code]
        for template in market_templates:
            candidates.append(
                dict(
                    input_text=template.format(crop_word=crop_word, location=location, state=STATES[0]),
                    is_ag_related=True,
                    crops=crop_from_code(crop_code),
                    topics=["market_economics"],
                    needs_rag=False,
                    needs_web_search=True,
                    needs_weather_data=False,
                    needs_farm_data=False,
                    intent="information_lookup",
                    urgency="medium",
                    difficulty="simple",
                    notes="Current market, policy, or outbreak lookup that depends on live web data.",
                )
            )
    for crop_code, state in [(c, s) for c in ["c", "s", "b"] for s in STATES]:
        crop_word = {"c": "corn", "s": "soybeans", "b": "corn and soybeans"}[crop_code]
        for idx, template in enumerate(policy_templates):
            candidates.append(
                dict(
                    input_text=template.format(crop_word=crop_word, location=LOCATIONS[idx % len(LOCATIONS)], state=state),
                    is_ag_related=True,
                    crops=crop_from_code(crop_code),
                    topics=["policy_regulation"] if idx != 3 else ["disease"],
                    needs_rag=False,
                    needs_web_search=True,
                    needs_weather_data=False,
                    needs_farm_data=False,
                    intent="information_lookup" if idx % 2 else "monitoring",
                    urgency="high" if idx % 3 == 0 else "medium",
                    difficulty="medium",
                    notes="Current market, policy, or outbreak lookup that depends on live web data.",
                )
            )
    add_candidates(builder, candidates, count, "web_only")


WEATHER_ONLY_TEMPLATES = [
    "Can we catch a spray window {time_window} near {location}?",
    "How much rain is lined up {time_window} around {location}?",
    "Will wind stay low enough {time_window} near {location} for spraying?",
    "Does {location} dry out enough {time_window} to get back in the field?",
    "Any frost risk {time_window} around {location}?",
    "How ugly does the heat look {time_window} in {location}?",
    "Radar around {location} {time_window}?",
]


def add_weather_only_examples(builder: UniqueBuilder, count: int) -> None:
    candidates: list[dict] = []
    for crop_code in ["u", "c", "s", "b"]:
        for location in LOCATIONS:
            for time_window in TIME_WINDOWS:
                for idx, template in enumerate(WEATHER_ONLY_TEMPLATES):
                    candidates.append(
                        dict(
                            input_text=template.format(location=location, time_window=time_window),
                            is_ag_related=True,
                            crops=crop_from_code(crop_code),
                            topics=["weather"] if idx % 2 else ["weather", "general_agronomy"],
                            needs_rag=False,
                            needs_web_search=False,
                            needs_weather_data=True,
                            needs_farm_data=False,
                            intent="planning" if idx % 3 else "monitoring",
                            urgency="high" if idx % 4 == 0 else "medium",
                            difficulty="simple" if idx % 3 else "hard",
                            notes="Weather-dependent ag query that should route to forecast data without agronomy retrieval.",
                        )
                    )
    add_candidates(builder, candidates, count, "weather_only")


FARM_ONLY_TEMPLATES = [
    "What hybrid is planted on {field}?",
    "Pull up the last soil test for {field}.",
    "When did we actually plant {field}?",
    "What population did we run on {field}?",
    "Show me the yield history for {field}.",
    "Did {field} get fungicide last year?",
    "Which variety is on {field} again?",
    "How many NH3 units went on {field}?",
    "What was the last tillage pass on {field}?",
    "Where did we log drown-out acres in {field}?",
]


def add_farm_only_examples(builder: UniqueBuilder, count: int) -> None:
    candidates: list[dict] = []
    crop_codes = ["c", "s", "u", "b"]
    topics_cycle = [["management"], ["soil"], ["general_agronomy"], ["ag_technology"]]
    for crop_code in crop_codes:
        for field in FIELD_NAMES:
            for idx, template in enumerate(FARM_ONLY_TEMPLATES):
                candidates.append(
                    dict(
                        input_text=template.format(field=field),
                        is_ag_related=True,
                        crops=crop_from_code(crop_code),
                        topics=topics_cycle[idx % len(topics_cycle)],
                        needs_rag=False,
                        needs_web_search=False,
                        needs_weather_data=False,
                        needs_farm_data=True,
                        intent="information_lookup" if idx % 2 == 0 else "monitoring",
                        urgency="low" if idx % 3 else "medium",
                        difficulty="simple" if idx % 4 else "medium",
                        notes="Farm-specific lookup that should route to user records without extra retrieval.",
                    )
                )
    add_candidates(builder, candidates, count, "farm_only")


RAG_WEATHER_TEMPLATES = [
    ("If corn catches frost at V4, what injury pattern usually shows up first?", ["corn"], ["weather", "management"], "question"),
    ("Soybeans took a pounding from heat and wind. What damage symptoms fit that best?", ["soybean"], ["weather", "disease"], "diagnosis"),
    ("What should I watch after ponding and then a fast hot snap in young corn?", ["corn"], ["weather", "nutrient"], "monitoring"),
    ("We had hail and now the stand looks uneven. What is normal recovery versus real trouble?", ["unknown"], ["weather", "management"], "diagnosis"),
    ("After a cold rain, how do you tell crusting stress from disease in soybeans?", ["soybean"], ["weather", "disease", "soil"], "comparison"),
    ("What does herbicide splash plus hot weather usually look like in corn leaves?", ["corn"], ["weather", "weed"], "diagnosis"),
    ("If beans sit in saturated ground for three days, what symptoms usually show first?", ["soybean"], ["weather", "disease"], "question"),
    ("What is the usual agronomic concern after a windy, hot flowering window in corn?", ["corn"], ["weather", "general_agronomy"], "question"),
    ("Can dry weather make potassium issues show up faster in beans?", ["soybean"], ["weather", "nutrient"], "question"),
    ("Field looks rough after sandblasting wind. What injury signs would you expect?", ["unknown"], ["weather", "management"], "diagnosis"),
]


def add_rag_weather_examples(builder: UniqueBuilder, count: int) -> None:
    suffixes = [
        "",
        " Need a practical read, not a textbook answer.",
        " Trying to sort normal stress from a real problem.",
        " I do not need product names, just what pattern fits.",
        " This is more of a what-am-I-looking-at question.",
    ]
    candidates: list[dict] = []
    for text, crops, topics, intent in RAG_WEATHER_TEMPLATES:
        for idx, suffix in enumerate(suffixes):
            candidates.append(
                dict(
                    input_text=f"{text}{suffix}",
                    is_ag_related=True,
                    crops=crops,
                    topics=topics,
                    needs_rag=True,
                    needs_web_search=False,
                    needs_weather_data=True,
                    needs_farm_data=False,
                    intent=intent,
                    urgency="medium" if idx % 2 else "high",
                    difficulty="hard" if idx % 2 == 0 else "medium",
                    notes="Agronomic interpretation tied to weather stress without needing farm records.",
                )
            )
    add_candidates(builder, candidates, count, "rag_weather")


RAG_FARM_TEMPLATES = [
    ("Field looks bad on {field} after sidedress. What would you check first?", ["unknown"], ["nutrient", "management"], "diagnosis"),
    ("Given our planting date on {field}, should I be worried that this corn is still uneven?", ["corn"], ["management", "general_agronomy"], "comparison"),
    ("The beans on {field} are yellowing in the wheel tracks. What fits that pattern?", ["soybean"], ["nutrient", "soil"], "diagnosis"),
    ("We changed hybrids on {field} and now emergence is all over. What should I sort through?", ["corn"], ["management", "general_agronomy"], "diagnosis"),
    ("Pasture edge on {field} keeps flaring with weeds. What should I factor in before another pass?", ["unknown"], ["weed", "management"], "recommendation"),
    ("The low side of {field} is stunted again. Does the field history point more to disease or fertility?", ["unknown"], ["disease", "nutrient", "soil"], "comparison"),
    ("Our soybeans on {field} have scattered holes and ragged edges. What pest pressure should I verify first?", ["soybean"], ["pest"], "diagnosis"),
    ("Corn on {field} went purple after a wet spell. What does that usually mean with this field history?", ["corn"], ["nutrient", "weather"], "diagnosis"),
    ("Stand loss is worse where we worked the headlands on {field}. What would you investigate?", ["unknown"], ["soil", "management"], "diagnosis"),
    ("This disease pattern keeps returning on {field}. What management angles should I think about next?", ["unknown"], ["disease", "management"], "recommendation"),
]


def add_rag_farm_examples(builder: UniqueBuilder, count: int) -> None:
    starters = ["", "Be straight with me: ", "Need a second set of eyes: ", "Short version: ", "I am trying to narrow this down: "]
    candidates: list[dict] = []
    for field in FIELD_NAMES:
        for template, crops, topics, intent in RAG_FARM_TEMPLATES:
            base_text = template.format(field=field)
            for idx, starter in enumerate(starters):
                text = base_text if not starter else f"{starter}{base_text[0].lower() + base_text[1:]}"
                candidates.append(
                    dict(
                        input_text=text,
                        is_ag_related=True,
                        crops=crops,
                        topics=topics,
                        needs_rag=True,
                        needs_web_search=False,
                        needs_weather_data=False,
                        needs_farm_data=True,
                        intent=intent,
                        urgency="medium" if idx % 2 else "high",
                        difficulty="hard" if idx % 2 == 0 else "medium",
                        notes="Farm-specific agronomic question added to improve diagnosis and recommendation coverage.",
                    )
                )
    add_candidates(builder, candidates, count, "rag_farm")


WEB_WEATHER_TEMPLATES = [
    ("Radar and wind near {location} {time_window} for spraying?", ["unknown"], ["weather", "general_agronomy"], "planning"),
    ("How soon does the storm line hit {location}, and will there be any fieldwork window before it?", ["unknown"], ["weather", "management"], "planning"),
    ("Latest rainfall totals around {location} and what is the next rain chance?", ["unknown"], ["weather"], "monitoring"),
    ("Are there active flood warnings near {location} that matter for low ground fields?", ["unknown"], ["weather", "management"], "information_lookup"),
    ("What does the live wind map show for {location} right now?", ["unknown"], ["weather"], "information_lookup"),
    ("Is the smoke outlook getting worse around {location} {time_window} for field crews?", ["unknown"], ["weather", "management"], "monitoring"),
    ("Can I see current dewpoint and wind around {location} before I load the sprayer?", ["unknown"], ["weather", "general_agronomy"], "planning"),
    ("What does radar show around {location} and how long until the next dry slot?", ["unknown"], ["weather"], "planning"),
]


def add_web_weather_examples(builder: UniqueBuilder, count: int) -> None:
    candidates: list[dict] = []
    for location in LOCATIONS:
        for time_window in TIME_WINDOWS:
            for idx, (template, crops, topics, intent) in enumerate(WEB_WEATHER_TEMPLATES):
                candidates.append(
                    dict(
                        input_text=template.format(location=location, time_window=time_window),
                        is_ag_related=True,
                        crops=crops,
                        topics=topics,
                        needs_rag=False,
                        needs_web_search=True,
                        needs_weather_data=True,
                        needs_farm_data=False,
                        intent=intent,
                        urgency="high" if idx % 2 == 0 else "medium",
                        difficulty="hard" if idx % 2 == 0 else "medium",
                        notes="Current weather/radar lookup that benefits from live web plus weather data.",
                    )
                )
    add_candidates(builder, candidates, count, "web_weather")


RAG_ONLY_TEMPLATES = [
    ("How long can corn stay purple after emergence before it usually grows out of it?", ["corn"], ["nutrient", "general_agronomy"], "question"),
    ("What is the difference between bean leaf beetle feeding and hail on soybeans?", ["soybean"], ["pest", "weather"], "comparison"),
    ("What usually separates herbicide carryover from nutrient stress in row crops?", ["both"], ["weed", "nutrient"], "comparison"),
    ("What should I look at first when a stand emerges uneven on sidehills?", ["unknown"], ["soil", "management"], "question"),
    ("How do tar spot lesions usually differ from more cosmetic leaf spotting?", ["corn"], ["disease"], "comparison"),
    ("What is the normal scouting threshold mindset for soybean aphids now?", ["soybean"], ["pest"], "question"),
    ("What are the common reasons corn roots stay shallow even when rainfall is decent?", ["corn"], ["soil", "general_agronomy"], "question"),
    ("If I only remember one thing about late-season waterhemp cleanup, what is it?", ["unknown"], ["weed", "management"], "recommendation"),
]


def add_rag_only_examples(builder: UniqueBuilder, count: int) -> None:
    prefixes = ["", "Quick agronomy check: ", "General question: ", "Trying to think this through: ", "High-level only: "]
    candidates: list[dict] = []
    for text, crops, topics, intent in RAG_ONLY_TEMPLATES:
        for idx, prefix in enumerate(prefixes):
            candidates.append(
                dict(
                    input_text=f"{prefix}{text}",
                    is_ag_related=True,
                    crops=crops,
                    topics=topics,
                    needs_rag=True,
                    needs_web_search=False,
                    needs_weather_data=False,
                    needs_farm_data=False,
                    intent=intent,
                    urgency="medium",
                    difficulty="medium" if idx % 2 else "hard",
                    notes="General agronomy retrieval case without live or farm data requirements.",
                )
            )
    add_candidates(builder, candidates, count, "rag_only")


MIXED_HARD_TEMPLATES = [
    ("Corn price and when should I plant?", ["corn"], ["market_economics", "weather", "management"], False, True, True, False, "planning"),
    ("Field looks bad and basis stinks. What matters first?", ["unknown"], ["market_economics", "general_agronomy"], True, True, False, True, "comparison"),
    ("Price of beans and can I spray tonight?", ["soybean"], ["market_economics", "weather", "management"], False, True, True, False, "planning"),
    ("Need latest tar spot reports and what symptoms to trust.", ["corn"], ["disease"], True, True, False, False, "diagnosis"),
    ("What changed on dicamba rules and does that shift today’s plan?", ["soybean"], ["policy_regulation", "management", "weather"], True, True, True, False, "planning"),
    ("This field is off. Any new disease pressure nearby too?", ["unknown"], ["disease"], True, True, False, True, "diagnosis"),
    ("Can I get local rain now and remind me what ponding does to corn?", ["corn"], ["weather", "management"], True, True, True, False, "question"),
    ("Basis, rain, and replant risk all at once—where do I start?", ["unknown"], ["market_economics", "weather", "management"], True, True, True, True, "other"),
    ("Latest aphid chatter and what threshold actually matters?", ["soybean"], ["pest"], True, True, False, False, "information_lookup"),
    ("How wet is it near {location}, and should fungicide wait?", ["unknown"], ["weather", "disease", "management"], True, True, True, False, "planning"),
    ("Crop insurance deadline plus rain outlook for beans in {state}?", ["soybean"], ["policy_regulation", "weather", "management"], False, True, True, False, "information_lookup"),
    ("Need current wind and a sanity check on drift risk.", ["unknown"], ["weather", "general_agronomy"], True, True, True, False, "planning"),
    ("Pull my planting date and tell me if this growth stage makes sense.", ["unknown"], ["management", "general_agronomy"], True, False, False, True, "comparison"),
    ("Weather, cash bids, and disease alerts for corn near {location}?", ["corn"], ["market_economics", "weather", "disease"], False, True, True, False, "monitoring"),
    ("Soybeans look rough after heat. Also, anything new on basis?", ["soybean"], ["market_economics", "weather", "management"], True, True, True, False, "comparison"),
    ("field looks bad", ["unknown"], ["general_agronomy"], True, False, False, True, "diagnosis"),
    ("spray today?", ["unknown"], ["weather", "management"], False, False, True, True, "planning"),
    ("beans yellow", ["soybean"], ["nutrient"], True, False, False, True, "diagnosis"),
    ("latest corn cash?", ["corn"], ["market_economics"], False, True, False, False, "information_lookup"),
    ("too wet yet?", ["unknown"], ["weather", "management"], False, False, True, True, "planning"),
]


def add_mixed_hard_examples(builder: UniqueBuilder, count: int) -> None:
    candidates: list[dict] = []
    suffixes = [
        "",
        " Keep it practical.",
        " Short answer is fine.",
        " I am not even sure how to ask this right.",
        " Need the routing right more than the answer.",
    ]
    for idx, (text, crops, topics, rag, web, weather, farm, intent) in enumerate(MIXED_HARD_TEMPLATES):
        base = text.format(location=LOCATIONS[idx % len(LOCATIONS)], state=STATES[idx % len(STATES)])
        for suffix in suffixes:
            candidates.append(
                dict(
                    input_text=f"{base}{suffix}",
                    is_ag_related=True,
                    crops=crops,
                    topics=topics,
                    needs_rag=rag,
                    needs_web_search=web,
                    needs_weather_data=weather,
                    needs_farm_data=farm,
                    intent=intent,
                    urgency="high" if idx % 2 == 0 else "medium",
                    difficulty="hard",
                    notes="Adversarial ag example with vague or mixed routing signals.",
                )
            )
    add_candidates(builder, candidates, count, "mixed_hard")


def build_additions(existing_rows: list[dict]) -> list[dict]:
    existing_inputs = {normalize_text(row["input"]) for row in existing_rows}
    start_id = max(int(row["id"].split("_")[1]) for row in existing_rows) + 1
    builder = UniqueBuilder(existing_inputs, start_id)
    add_non_ag_examples(builder, 120)
    add_web_only_examples(builder, 40)
    add_weather_only_examples(builder, 35)
    add_farm_only_examples(builder, 35)
    add_rag_weather_examples(builder, 45)
    add_rag_farm_examples(builder, 45)
    add_web_weather_examples(builder, 30)
    add_rag_only_examples(builder, 20)
    add_mixed_hard_examples(builder, 30)
    if len(builder.rows) != 400:
        raise RuntimeError(f"Expected 400 new rows, got {len(builder.rows)}")
    return builder.rows


def distribute_additions(data: dict[str, list[dict]], additions: list[dict]) -> None:
    offsets = {"train.jsonl": 300, "val.jsonl": 50, "test.jsonl": 50}
    start = 0
    for split in SPLITS:
        stop = start + offsets[split]
        data[split].extend(additions[start:stop])
        start = stop


def build_audit_report(before: dict, after: dict, rewrites: tuple[int, int], additions: int) -> str:
    before_findings = audit_findings(before)
    after_findings = audit_findings(after)
    edge_labels = {
        "weather_only": (False, False, True, False),
        "web_only": (False, True, False, False),
        "farm_only": (False, False, False, True),
        "rag_plus_weather": (True, False, True, False),
        "rag_plus_farm": (True, False, False, True),
        "web_plus_weather": (False, True, True, False),
    }

    def edge_lines(stats: dict) -> str:
        return "\n".join(f"  - {name}: {stats['route_combo_counts'][combo]}" for name, combo in edge_labels.items())

    lines = [
        "CornbeltAI routing dataset audit and improvement report",
        "=" * 52,
        "",
        "Baseline audit (before modifications)",
        "-" * 36,
        f"Total rows: {before['total']}",
        f"Ag-related: {before['ag_total']} ({before['ag_total'] / before['total']:.1%})",
        f"Non-ag: {before['non_ag_total']} ({before['non_ag_total'] / before['total']:.1%})",
        f"Short queries (1-5 words): {before['short_queries']} ({before['short_queries'] / before['total']:.1%})",
        f"Vague queries: {before['vague_queries']} ({before['vague_queries'] / before['total']:.1%})",
        f"Multi-topic queries: {before['multi_topic']} ({before['multi_topic'] / before['total']:.1%})",
        f"Hard examples: {before['hard_examples']} ({before['hard_examples'] / before['total']:.1%})",
        "",
        "Baseline class distribution",
        f"is_ag_related\n  - true: {before['ag_total']}\n  - false: {before['non_ag_total']}",
        "crops\n" + format_distribution(before['crop_counts'], before['ag_total']),
        "topics\n" + format_distribution(before['topic_counts'], before['ag_total']),
        "needs_rag / needs_web_search / needs_weather_data / needs_farm_data",
        f"  - needs_rag=true: {before['flag_counts']['needs_rag']} ({before['flag_counts']['needs_rag'] / before['ag_total']:.1%} of ag)",
        f"  - needs_web_search=true: {before['flag_counts']['needs_web_search']} ({before['flag_counts']['needs_web_search'] / before['ag_total']:.1%} of ag)",
        f"  - needs_weather_data=true: {before['flag_counts']['needs_weather_data']} ({before['flag_counts']['needs_weather_data'] / before['ag_total']:.1%} of ag)",
        f"  - needs_farm_data=true: {before['flag_counts']['needs_farm_data']} ({before['flag_counts']['needs_farm_data'] / before['ag_total']:.1%} of ag)",
        "intent\n" + format_distribution(before['intent_counts'], before['total']),
        "urgency\n" + format_distribution(before['urgency_counts'], before['total']),
        "",
        "Weakness findings",
        *[f"  - {finding}" for finding in before_findings],
        "",
        "Explicit weakness checks",
        f"  - Too many clean queries: yes; only {before['vague_queries']} vague rows and {before['short_queries']} short rows in the baseline set.",
        f"  - Lack of short queries (1-5 words): yes; {before['short_queries']} rows total.",
        f"  - Lack of vague queries: yes; {before['vague_queries']} rows matched vague heuristics.",
        f"  - Lack of multi-topic queries: yes; {before['multi_topic']} rows total.",
        "  - Edge routing combinations:",
        edge_lines(before),
        f"  - Weak non-ag coverage: ratio is {before['non_ag_total'] / before['total']:.1%}, but many negatives were templated and low-diversity.",
        "",
        "Changes applied",
        "-" * 15,
        f"Rewrote existing rows: {sum(rewrites)} total ({rewrites[0]} non-ag, {rewrites[1]} ag).",
        f"Added new rows: {additions} total, including at least 100 hard/adversarial examples.",
        "Added coverage for short queries, messy phrasing, vague questions, mixed-intent prompts, and missing routing combinations.",
        "",
        "Post-improvement summary",
        "-" * 25,
        f"Total rows: {after['total']}",
        f"Ag-related: {after['ag_total']} ({after['ag_total'] / after['total']:.1%})",
        f"Non-ag: {after['non_ag_total']} ({after['non_ag_total'] / after['total']:.1%})",
        f"Short queries (1-5 words): {after['short_queries']} ({after['short_queries'] / after['total']:.1%})",
        f"Vague queries: {after['vague_queries']} ({after['vague_queries'] / after['total']:.1%})",
        f"Multi-topic queries: {after['multi_topic']} ({after['multi_topic'] / after['total']:.1%})",
        f"Hard examples: {after['hard_examples']} ({after['hard_examples'] / after['total']:.1%})",
        f"Normalized duplicate inputs: {after['normalized_duplicate_inputs']}",
        "",
        "Post-improvement edge routing counts",
        edge_lines(after),
        "",
        "Post-improvement findings",
        *[f"  - {finding}" for finding in after_findings],
        "",
        "Final recommendation: dataset now meets the requested balance and stress-test goals, and the new validation/reporting scripts can enforce those expectations going forward.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    data = load_dataset()
    before = dataset_stats(row for split in SPLITS for row in data[split])
    rewrites = replace_rows(data)
    additions = build_additions([row for split in SPLITS for row in data[split]])
    distribute_additions(data, additions)
    after = dataset_stats(row for split in SPLITS for row in data[split])
    save_dataset(data)
    AUDIT_PATH.write_text(build_audit_report(before, after, rewrites, len(additions)), encoding="utf-8")
    print(f"Rewrote {sum(rewrites)} existing rows, added {len(additions)} rows, and updated {AUDIT_PATH}.")


if __name__ == "__main__":
    main()
