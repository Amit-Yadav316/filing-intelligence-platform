"""For each wrong verdict, did the model report a DIFFERENT us-gaap concept?

The failure taxonomy keeps saying "wrong line item", which is a description
rather than a diagnosis. This turns it into one: for every wrong answer, search
the filer's own XBRL for a tag whose value the model actually reported.

When one is found, the model did not misread anything. It answered a genuinely
ambiguous question - "net income" is both NetIncomeLoss (attributable to the
parent) and ProfitLoss (including non-controlling interests) - and the answer
key happened to prefer the other one. That is a tag-map problem, and it is
fixable without touching the model.

When no tag matches, the model really did misread, and that IS a model problem.
Separating the two is the difference between tuning a config and tuning a
prompt.

No LLM calls.

Usage:
    python -m scripts.diagnose_wrong_tags
"""

from __future__ import annotations

import gzip
import json
from collections import Counter
from datetime import date
from decimal import Decimal
from typing import Any

from src.config.settings import get_settings
from src.evaluate.xbrl_resolver import XBRLResolver
from src.observability.logging import configure_logging

TOLERANCE = Decimal("0.005")


def load_facts(cik: str) -> dict[str, Any] | None:
    path = get_settings().data_dir / "raw" / "companyfacts" / f"CIK{cik}.json.gz"
    if not path.exists():
        return None
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return dict(json.load(fh))


def matching_tags(
    facts: dict[str, Any], value: Decimal, fiscal_year: int, resolver: XBRLResolver
) -> list[tuple[str, Decimal]]:
    """Every us-gaap tag whose annual value for this year equals `value`."""
    hits: list[tuple[str, Decimal]] = []
    us_gaap = facts.get("facts", {}).get("us-gaap", {})
    for tag, entry in us_gaap.items():
        for observations in (
            entry.get("units", {}).get("USD", []) and [entry["units"]["USD"]]
        ) or []:
            for obs in observations:
                end = obs.get("end", "")
                if not end.startswith(str(fiscal_year)):
                    continue
                start = obs.get("start")
                if start:
                    try:
                        span = (date.fromisoformat(end) - date.fromisoformat(start)).days
                    except ValueError:
                        continue
                    if not resolver.min_annual_days <= span <= resolver.max_annual_days:
                        continue
                try:
                    reported = Decimal(str(obs["val"]))
                except Exception:
                    continue
                if reported == 0:
                    continue
                if abs(reported - value) / abs(reported) <= TOLERANCE:
                    hits.append((tag, reported))
                    break
    return hits


def main() -> int:
    settings = get_settings()
    configure_logging("WARNING", json_output=False)
    resolver = XBRLResolver.from_config()

    cards = json.loads((settings.docs_dir / "scorecards.json").read_text(encoding="utf-8"))

    explained: Counter[str] = Counter()
    unexplained = 0
    print(f"{'Company':<22}{'Field':<21}{'extracted':>16}  matched us-gaap tag")
    print("-" * 96)

    for card in cards:
        facts = load_facts(card["cik"])
        if facts is None:
            continue
        for score in card["scores"]:
            if score["verdict"] not in {"wrong", "hallucinated"}:
                continue
            extracted = Decimal(score["extracted"]) if score["extracted"] else None
            if extracted is None:
                continue

            tags = matching_tags(facts, extracted, card["fiscal_year"], resolver)
            company = (card.get("company") or "?")[:20]
            if tags:
                names = ", ".join(t for t, _ in tags[:2])
                explained[f"{score['field']}::{tags[0][0]}"] += 1
                print(f"{company:<22}{score['field']:<21}{int(extracted):>16,}  {names}")
            else:
                unexplained += 1
                print(
                    f"{company:<22}{score['field']:<21}{int(extracted):>16,}  (no tag matches - real error)"
                )

    total = sum(explained.values()) + unexplained
    print(f"\n{'=' * 96}")
    print(f"Wrong answers analysed        : {total}")
    print(f"  explained by another tag    : {sum(explained.values())}")
    print(f"  no matching tag, real error : {unexplained}")
    print("\nMost common alternative concepts the model reported:")
    for key, count in explained.most_common(8):
        field, tag = key.split("::")
        print(f"  {field:<22} -> us-gaap:{tag}  x{count}")
    print(
        "\nAdd the recurring ones to `alternatives` in config/xbrl_tag_map.yaml to\n"
        "accept them as defensible answers, recorded as such rather than hidden."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
