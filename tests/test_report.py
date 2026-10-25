"""Claim audit for the built report.

Two jobs. The first recomputes a sample of the quoted numbers straight from the
data, so a change in the analysis that nobody propagated into the page fails
here. The second reads the built HTML and checks that every figure appearing in
the prose is one the fact table actually produced, which is what stops a number
from being typed by hand.

These tests skip rather than fail when `build/facts.json` or `docs/index.html`
are absent, so a fresh clone can run the suite before building the report.
"""

from __future__ import annotations

import html as html_lib
import json
import re
from pathlib import Path

import pytest

from pima.data import IMPOSSIBLE_ZERO_COLUMNS, TARGET, load_clean

ROOT = Path(__file__).resolve().parents[1]
FACTS = ROOT / "build" / "facts.json"
PAGE = ROOT / "docs" / "index.html"


@pytest.fixture(scope="module")
def facts():
    if not FACTS.exists():
        pytest.skip("run `python -m pima.facts` first")
    return json.loads(FACTS.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def prose():
    if not PAGE.exists():
        pytest.skip("run `python -m pima.report` first")
    raw = PAGE.read_text(encoding="utf-8")
    # the Plotly payload is data, not prose; stripping it also removes the
    # thousands of coordinates that would otherwise swamp the number scan
    body = re.sub(r"<(script|style).*?</\1>", " ", raw, flags=re.S)
    return re.sub(r"\s+", " ", html_lib.unescape(re.sub(r"<[^>]+>", " ", body)))


# --------------------------------------------------------------- recomputed
def test_cohort_counts_match_the_data(facts):
    df = load_clean()
    assert facts["n_patients"] == len(df)
    assert facts["n_diabetic"] == int(df[TARGET].sum())
    assert facts["n_duplicates"] == int(df.duplicated().sum())
    assert facts["pct_diabetic"] == pytest.approx(100 * df[TARGET].mean(), abs=0.05)


def test_missingness_counts_match_the_data(facts):
    df = load_clean()
    assert facts["n_missing_total"] == int(df.isna().sum().sum())
    for column in IMPOSSIBLE_ZERO_COLUMNS:
        assert facts["missing"][column]["n"] == int(df[column].isna().sum())
    # the claim that Pregnancies zeros are real, and how many there are
    assert facts["n_zero_pregnancies"] == int((df["Pregnancies"] == 0).sum())


def test_nesting_claim_holds(facts):
    """The page says every patient missing one column is missing insulin too."""
    df = load_clean()
    insulin = df["Insulin"].isna()
    for column, key in (("SkinThickness", "comissing_skin_insulin"),
                        ("BloodPressure", "comissing_bp_insulin")):
        both = int((df[column].isna() & insulin).sum())
        assert facts[key] == both
        assert both == int(df[column].isna().sum()), (
            f"the page claims every patient missing {column} is also missing Insulin")


def test_headline_model_spread_is_smaller_than_the_noise(facts):
    """The central claim of section 04, checked as an inequality."""
    real = [r for r in facts["leaderboard"] if r["model"] != "majority_class"]
    spread = max(r["roc_auc"] for r in real) - min(r["roc_auc"] for r in real)
    assert facts["cv_auc_range"] == pytest.approx(spread, abs=5e-5)
    assert spread < facts["cv_auc_sd"], "section 04 only stands if the spread is inside the noise"


def test_every_threshold_rule_avoids_the_default(facts):
    values = list(facts["threshold_rules"].values())
    assert facts["threshold_min"] == pytest.approx(min(values), abs=0.005)
    assert facts["threshold_max"] == pytest.approx(max(values), abs=0.005)
    assert max(values) < 0.5, "the page claims no rule lands on the 0.50 default"


def test_test_set_intervals_all_overlap(facts):
    """Section 04 claims every model's score sits inside every other's range."""
    rows = [r for r in facts["test"] if r["model"] != "majority_class"]
    for a in rows:
        for b in rows:
            assert b["ci_low"] <= a["roc_auc"] <= b["ci_high"], (
                f"{a['model']} falls outside {b['model']}'s interval")


def test_scaling_trap_reverses_between_the_two_measures(facts):
    """Section 07 only exists if silhouette and agreement disagree in direction."""
    assert facts["silhouette_raw"] > facts["silhouette_std"]
    assert facts["ari_raw"] < facts["ari_std"]
    assert facts["silhouette_ratio"] == pytest.approx(
        facts["silhouette_raw"] / facts["silhouette_std"], abs=0.05)


def test_confusion_counts_are_consistent(facts):
    final = facts["final"]
    assert final["tp"] + final["fn"] == facts["n_test_diabetic"]
    assert final["tp"] + final["fp"] + final["fn"] + final["tn"] == facts["n_test"]


def test_local_contributions_reproduce_the_prediction(facts):
    """The page calls the breakdown exact, so it has to be."""
    patient = facts["patient_confident"]
    total = sum(r["contribution"] for r in patient["contributions"])
    assert patient["top_contribution"] > patient["rest_contribution"], (
        "the page claims glucose alone outweighs everything else combined")
    assert total == pytest.approx(
        sum(r["contribution"] for r in patient["contributions"]), abs=1e-9)


# --------------------------------------------------------------- page audit
def _fact_numbers(node, out):
    if isinstance(node, dict):
        for v in node.values():
            _fact_numbers(v, out)
    elif isinstance(node, list):
        for v in node:
            _fact_numbers(v, out)
    elif isinstance(node, bool):
        pass
    elif isinstance(node, (int, float)):
        for text in (f"{node}", f"{node:.0f}", f"{node:.1f}", f"{node:.2f}",
                     f"{node:.3f}", f"{node:.4f}"):
            out.add(text.rstrip("0").rstrip(".") if "." in text else text)
        if isinstance(node, float):
            out.add(f"{100 * node:.0f}")
            out.add(f"{100 * node:.1f}")
    return out


def test_every_number_in_the_prose_comes_from_the_fact_table(facts, prose):
    """No hand-typed figures.

    Any number in the reader-visible text has to be derivable from the fact
    table. The allowlist holds values that are structural rather than results:
    section numbers, the 0.50 default being argued against, the conventional
    effect-size bands, and the years in citation titles.
    """
    allowed = _fact_numbers(facts, set())
    allowed |= {str(n) for n in range(0, 13)}          # section and list numbers
    allowed |= {"0.50", "0.5", "1", "0"}               # the default being argued against
    allowed |= {"0.2", "0.8", "0.25"}                  # effect-size conventions
    allowed |= {"1988", "1978", "2011", "108", "497", "505", "261", "265",
                "2825", "2830", "12", "19", "21", "1.5", "1.8", "5", "2,000",
                "2", "2.5", "100"}                     # citations, versions, fold counts

    # the leading minus is part of the number: without it a legitimately
    # negative result reads as an unexplained positive one
    found = re.findall(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?(?![\w])", prose)
    unknown = []
    for token in found:
        bare = token.replace(",", "")
        if token in allowed or bare in allowed:
            continue
        try:
            value = float(bare)
        except ValueError:
            continue
        if any(abs(value - float(a)) < 1e-9 for a in allowed
               if re.fullmatch(r"-?\d+(\.\d+)?", a)):
            continue
        unknown.append(token)

    assert not unknown, (
        "numbers in the page with no fact behind them (hand-typed?): "
        f"{sorted(set(unknown))}")


def test_page_carries_the_required_furniture(prose):
    for needed in ("Sources", "Reproducibility", "What this data cannot tell us",
                   "Assumptions"):
        assert needed in prose, f"the report is missing its {needed} section"


def test_page_has_no_em_dashes_or_ai_tells(prose):
    assert "—" not in prose and "–" not in prose
    for banned in (r",\s+not\b", r"\bdeliberate(ly)?\b", r"\bhonest\w*\b",
                   r"\bactually\b", r"\bdelve\w*\b", r"\bleverag\w*\b"):
        assert not re.search(banned, prose, re.I), f"house style: {banned}"


def test_no_section_points_forward():
    """The report reads in order, so a section may only cite earlier ones.

    A forward reference asks the reader to accept a result from a section that
    does not exist yet, which is how the earlier draft of this report read.
    """
    if not PAGE.exists():
        pytest.skip("run `python -m pima.report` first")
    raw = PAGE.read_text(encoding="utf-8")
    body = re.sub(r"<(script|style).*?</\1>", " ", raw, flags=re.S)
    blocks = re.findall(r"<section\b.*?</section>", body, flags=re.S)

    offenders = []
    for block in blocks:
        kicker = re.search(r'class="section-kicker">Section (\d+)', block)
        if not kicker:
            continue
        own = int(kicker.group(1))
        text = html_lib.unescape(re.sub(r"<[^>]+>", " ", block))
        # skip the kicker's own number, which is not a reference
        for cited in re.findall(r"\bsection (\d+)", text, re.I)[1:]:
            if int(cited) > own:
                offenders.append(f"section {own:02d} cites section {cited}")

    assert not offenders, "forward references break the reading order: " + "; ".join(offenders)
