"""
Deterministic pick explanations ("why does X have 86.7%?").

find_target() fuzzy-matches a free-text question to a prop pick or a game
market; render_*() turn the captured `explain` payloads (models/props.py,
picks/engine.py) into a markdown walkthrough of the actual arithmetic.
No LLM involved -- every number shown is the one the pipeline computed.
"""
import re
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Bumped whenever the UI starts depending on new symbols from this module or
# the shared model code. app_cloud.py compares this against the module already
# loaded in a hot-swapped Streamlit Cloud process and purges stale modules.
SCHEMA_VERSION = 2

from scipy.stats import norm

from models.props import STAT_MAP, prob_over_line
from picks.engine import MIN_EDGE, MODEL_WEIGHT
from analysis.confidence import get_confidence_tier

# internal stat column -> words a user might type for it
_STAT_WORDS = {
    "passing_yards":   {"passing", "pass", "yards", "yds"},
    "passing_tds":     {"passing", "pass", "tds", "touchdowns", "td"},
    "interceptions":   {"interceptions", "ints", "int", "picks"},
    "completions":     {"completions", "comps"},
    "rushing_yards":   {"rushing", "rush", "yards", "yds"},
    "rushing_tds":     {"rushing", "rush", "tds", "touchdowns", "td"},
    "receptions":      {"receptions", "recs", "catches"},
    "receiving_yards": {"receiving", "rec", "yards", "yds"},
    "receiving_tds":   {"receiving", "rec", "tds", "touchdowns", "td"},
    "rush_rec_tds":    {"rush", "rec", "tds", "touchdowns", "anytime"},
    "rush_rec_yards":  {"rush", "rec", "yards", "yds"},
    "pass_rush_yards": {"pass", "rush", "yards", "yds"},
    "total_yards":     {"total", "yards", "yds"},
    "fantasy":         {"fantasy", "points"},
    "kicking_points":  {"kicking", "points"},
    "fg_made":         {"fg", "field", "goals", "made"},
    "pat_made":        {"extra", "points", "pat"},
    "def_sacks":       {"sacks", "sack"},
    "def_tackles_total": {"tackles", "assists"},
    "def_tackles_solo": {"solo", "tackles"},
}

_COMBO_COLS = {"rush_rec_tds", "rush_rec_yards", "pass_rush_yards", "total_yards", "def_tackles_total"}

_GENERIC_TEAM_WORDS = {"new", "york", "los", "angeles", "san", "francisco",
                        "green", "bay", "tampa", "kansas", "city", "state"}


def _numbers(text: str) -> list[float]:
    return [float(n) for n in re.findall(r"\d+(?:\.\d+)?", text)]


def _score_prop(q: str, q_words: set, nums: list[float], p: dict) -> int:
    name = p["player_name"].lower()
    last = name.split()[-1]
    score = 0
    if name in q:
        score += 4
    elif last in q_words:
        score += 3
    else:
        return 0  # player name is required

    col = STAT_MAP.get(p["stat_type"])
    words = _STAT_WORDS.get(col, set())
    stat_hits = len(words & q_words)
    if stat_hits:
        score += 2 if (col not in _COMBO_COLS or stat_hits >= 2) else 1

    if p["line"] in nums:
        score += 2
    d = p["direction"]
    if (d == "More" and ({"more", "over"} & q_words)) or (d == "Less" and ({"less", "under"} & q_words)):
        score += 1
    if p.get("platform", "") in q:
        score += 1
    return score


def _match_game(q_words: set, games: list[dict]):
    best, best_hits = None, 0
    for g in games:
        words = set()
        for team in (g["home_team"], g["away_team"]):
            words |= {w for w in team.lower().split() if len(w) >= 4 and w not in _GENERIC_TEAM_WORDS}
        for team in (g["home_team"], g["away_team"]):
            parts = team.lower().split()
            words.add("".join(w[0] for w in parts))
            words.add(parts[-1][:3])
        hits = len(words & q_words)
        if hits > best_hits:
            best, best_hits = g, hits
    return best


def find_target(query: str, picks: list[dict], games: list[dict], views: dict) -> dict | None:
    """Match a question to a prop pick or a game market.

    Returns {"kind": "prop", "pick": ...} or
            {"kind": "game", "game": ..., "market": ...|None, "picks": [...], "view": ...|None}
    """
    q = query.lower()
    q_words = set(re.findall(r"[\w+\-']+", q))
    nums = _numbers(q)

    props = [p for p in picks if p["pick_type"] == "prop"]
    scored = sorted(((_score_prop(q, q_words, nums, p), p) for p in props),
                    key=lambda t: (t[0], t[1]["edge"]), reverse=True)
    prop_score, prop_pick = scored[0] if scored else (0, None)

    if prop_score >= 3:
        return {"kind": "prop", "pick": prop_pick}

    game = _match_game(q_words, games)
    if game is not None:
        if {"over", "under", "total", "totals"} & q_words:
            market = "Totals"
        elif {"spread", "cover", "covers"} & q_words:
            market = "Spread"
        elif {"ml", "moneyline", "win", "wins"} & q_words:
            market = "Moneyline"
        else:
            market = None
        g_picks = [p for p in picks if p["pick_type"] == "game" and p["home_team"] == game["home_team"]]
        return {"kind": "game", "game": game, "market": market,
                "picks": g_picks, "view": views.get(game["home_team"])}

    if prop_pick is not None and prop_score > 0:
        return {"kind": "prop", "pick": prop_pick}
    return None


# -- Renderers ------------------------------------------------------------------

def render_prop(p: dict) -> str:
    e = p.get("explain") or {}
    if not e:
        return "That pick predates the explain feature -- hit Refresh to rebuild picks."

    ot = p.get("odds_type", "standard")
    ot_label = {"goblin": " . goblin line", "demon": " . demon line"}.get(ot, "")
    lines = [f"**{p['player_name']} -- {p['stat_type']} {p['direction']} {p['line']}** "
             f"({p['platform']}{ot_label}) -> model **{p['model_prob']:.1%}**\n"]

    # 1. baseline
    if e["form_source"] == "blended":
        lines.append(
            f"1. **Baseline** -- 55% x last-4-game average ({p.get('recent_rate')}) + 45% x season average "
            f"({p.get('season_rate')}) = **{e['base_rate']}**")
    else:
        lines.append(f"1. **Baseline** -- season average **{e['base_rate']}** over {e['n_games']} games "
                     f"(not enough recent data to weight form)")

    # 2. opponent adjustment -- EPA-based pass/rush split for offensive stats,
    # opponent's sacks-allowed rate for Sacks (a pass rusher's matchup is the
    # opposing O-line, not their own team's defense), no adjustment at all
    # for kicking/tackles (see models/props.py's SACK_ADJUSTED_STATS comment).
    opp = e.get("opponent") or "an unresolved opponent"
    split = e.get("def_split")
    if split == "blend":
        lines.append(f"2. **Opponent** -- {opp}'s pass+rush defense EPA/play blended "
                     f"-> production scaled x{e['def_adj']:.3f}")
    elif split == "sacks_allowed":
        sacks = e.get("def_epa")
        sacks_str = f"{sacks:.2f}/game" if isinstance(sacks, (int, float)) else "n/a"
        lines.append(f"2. **Opponent** -- {opp}'s offense allows {sacks_str} sacks "
                     f"-> production scaled x{e['def_adj']:.3f}")
    elif split in ("pass", "rush"):
        epa = e.get("def_epa")
        epa_str = f"{epa:.3f}" if isinstance(epa, (int, float)) else "n/a"
        lines.append(f"2. **Opponent** -- {opp}'s {split}-defense EPA/play allowed = {epa_str} "
                     f"-> production scaled x{e['def_adj']:.3f}")
    elif e.get("def_adj") is None:
        lines.append("2. **Opponent** -- no adjustment resolved (opponent unknown)")
    else:
        lines.append("2. **Opponent** -- no opponent adjustment for this stat (kicking/tackles volume "
                     "tracks pace, not opponent quality, in this model)")

    # 3. pace
    if e.get("pace_factor") is not None:
        lines.append(f"3. **Pace** -- expected play-volume vs league average -> x{e['pace_factor']:.3f}")
    else:
        lines.append("3. **Pace** -- no pace adjustment for this stat")

    lines.append(f"4. **Expected value: {e['expected']}** vs the line **{p['line']}**")

    # 5. distribution
    if e["dist"] == "normal":
        lines.append(
            f"5. **Probability** -- normal distribution using {p['player_name'].split()[-1]}'s own game-to-game "
            f"std **{e['std']:.2f}**: P(more than {p['line']}) = {e['p_more']:.1%}, "
            f"P(less) = {1 - e['p_more']:.1%} -> pick **{p['direction']}** at **{p['model_prob']:.1%}**")
    elif e["dist"] == "poisson":
        lines.append(
            f"5. **Probability** -- Poisson(mu={e['expected']}) for a low-count stat: "
            f"P(more than {p['line']}) = {e['p_more']:.1%} -> pick **{p['direction']}** at **{p['model_prob']:.1%}**")
    else:
        lines.append(
            f"5. **Probability** -- no reliable per-player std, heuristic normal (sigma = 35% of expected): "
            f"P(more) = {e['p_more']:.1%} -> pick **{p['direction']}** at **{p['model_prob']:.1%}**")

    lines.append(
        f"6. **Edge** -- {p['model_prob']:.1%} model vs **{e['breakeven']:.1%}** break-even "
        f"(a 2-pick 3x slip needs each leg at sqrt(1/3) ~= 57.7%) = **{p['edge']:+.1%}** "
        f"-> {p.get('confidence_tier', '--')} confidence")

    if ot == "goblin":
        lines.append("\n(warning) Goblin (lowered) lines pay *less* than the standard 3x when in a slip, "
                     "so this edge is somewhat overstated.")
    elif ot == "demon":
        lines.append("\n(warning) Demon (raised) lines pay *more* than the standard 3x, "
                     "so this edge is somewhat understated.")
    return "\n".join(lines)


def _game_numbers(picks: list[dict], view: dict | None) -> dict | None:
    """Prefer a pick's explain payload; fall back to the model-view dict."""
    for p in picks:
        if p.get("explain"):
            return p["explain"]
    if view is not None:
        return {
            "raw_diff": view.get("raw_diff"), "raw_total": view.get("raw_total"),
            "market_diff": -view["spread_line"] if view.get("spread_line") is not None else None,
            "market_total": view.get("total_line"),
            "anch_diff": view["diff"], "anch_total": view["total"],
            "spread_std": view["spread_std"], "totals_std": view["totals_std"],
            "home_win_prob": view["home_p"], "model_weight": MODEL_WEIGHT,
            "home_qb_out": view.get("home_qb_out", False), "away_qb_out": view.get("away_qb_out", False),
        }
    return None


def render_game(game: dict, market: str | None, picks: list[dict], view: dict | None) -> str:
    home, away = game["home_team"], game["away_team"]
    e = _game_numbers(picks, view)
    if e is None:
        return (f"I can see **{away} @ {home}** on the schedule, but the model has no view yet -- "
                f"likely missing game logs or odds. Try Refresh.")

    out = [f"**{away} @ {home}**\n"]
    if e.get("home_qb_out") or e.get("away_qb_out"):
        flagged = [t for t, out_ in ((home, e.get("home_qb_out")), (away, e.get("away_qb_out"))) if out_]
        out.append(f"(!) Starting QB flagged Out/Doubtful+ for: {', '.join(flagged)}\n")

    markets = [market] if market else ["Moneyline", "Spread", "Totals"]

    fav, marg = (home, e["anch_diff"]) if e["anch_diff"] >= 0 else (away, -e["anch_diff"])
    raw_fav, raw_marg = (home, e["raw_diff"]) if e["raw_diff"] >= 0 else (away, -e["raw_diff"])

    for m in markets:
        pick = next((p for p in picks if p["market"] == m), None)
        has_line = e.get("market_diff") is not None if m in ("Moneyline", "Spread") else e.get("market_total") is not None
        if m in ("Moneyline", "Spread"):
            out.append(f"**{m}**")
            out.append(f"- XGBoost spread model (raw): {raw_fav} by {raw_marg:.1f}")
            if e.get("market_diff") is not None:
                mfav = home if e["market_diff"] >= 0 else away
                out.append(f"- Market line implies {mfav} by {abs(e['market_diff']):.1f} -> anchored "
                           f"{e['model_weight']:.0%} model / {1 - e['model_weight']:.0%} market = **{fav} by {marg:.1f}**")
            else:
                out.append("- No market spread posted -- raw model value used unanchored")
            if m == "Moneyline":
                out.append(f"- Win probability = Phi(margin {e['anch_diff']:+.1f} / calibration std {e['spread_std']:.1f}) "
                           f"= **{e['home_win_prob']:.1%} {home}** / {1 - e['home_win_prob']:.1%} {away}")
            elif e.get("market_diff") is not None:
                spread_line = -e["market_diff"]
                p_cover = float(norm.cdf((e["anch_diff"] + spread_line) / e["spread_std"]))
                side, pc = (f"{home} {spread_line:+.1f}", p_cover) if p_cover >= 0.5 else (f"{away} {-spread_line:+.1f}", 1 - p_cover)
                out.append(f"- Cover probability = Phi((margin {e['anch_diff']:+.1f} + line {spread_line:+.1f}) / "
                           f"std {e['spread_std']:.1f}) -> **{side} covers {pc:.1%}**")
        else:
            out.append("**Totals**")
            out.append(f"- XGBoost totals model (raw): **{e['raw_total']:.1f}** combined points")
            if e.get("market_total") is not None:
                out.append(f"- Market line {e['market_total']:.1f} -> anchored {e['model_weight']:.0%} model / "
                           f"{1 - e['model_weight']:.0%} market = **{e['anch_total']:.1f}**")
                p_over = float(1 - norm.cdf(e["market_total"], e["anch_total"], e["totals_std"]))
                ou, po = ("Over", p_over) if p_over >= 0.5 else ("Under", 1 - p_over)
                out.append(f"- P = Phi with calibration std {e['totals_std']:.1f} -> **{ou} {e['market_total']:.1f} at {po:.1%}**")
            else:
                out.append("- No market total posted")

        if pick is not None:
            odds = f" `{int(pick['best_odds']):+d}`" if pick.get("best_odds") else ""
            out.append(f"- Pick: **{pick['selection']}**{odds} @ {pick['best_platform']} -- "
                       f"model {pick['model_prob']:.1%} vs de-vigged implied {pick['implied_prob']:.1%} "
                       f"= edge **{pick['edge']:+.1%}** ({pick['confidence_tier']})")
        elif not has_line:
            out.append("- No bet -- no odds fetched for this market yet (hit Refresh for current lines).")
        else:
            out.append(f"- Pass -- edge vs the de-vigged book price is below the {MIN_EDGE:.0%} minimum. "
                       f"A confident probability isn't a bet if the payout already reflects it.")
        out.append("")
    return "\n".join(out)


NO_MATCH_MSG = (
    "I couldn't match that to a pick or game. Try including the **player's name** "
    "(e.g. *why does Saquon Barkley Rush Yards More 78.5 have 93%?*) or a **team + market** "
    "(e.g. *Panthers @ Cardinals over 44.5*). Note: props are only explainable "
    "if they appear in this week's pick list."
)

NO_CONTEXT_MSG = (
    "I'm not sure which pick that refers to -- ask about a specific pick first "
    "(e.g. *why does Josh Allen Pass Yards More 223.5 have 85%?*), then follow up with "
    "*what about a line of 240.5?*"
)


# -- What-if recomputations -----------------------------------------------------

def _edge_verdict(prob: float, breakeven: float) -> str:
    edge = prob - breakeven
    if edge >= MIN_EDGE:
        return f"edge **{edge:+.1%}** -> would be a **{get_confidence_tier(edge)}** pick"
    if edge > 0:
        return f"edge **{edge:+.1%}** -> below the {MIN_EDGE:.0%} minimum, would be a **pass**"
    return f"edge **{edge:+.1%}** -> negative EV, would be a **pass**"


def whatif_line(p: dict, new_line: float) -> str:
    e = p.get("explain") or {}
    if not e:
        return "That pick predates the explain feature -- hit Refresh to rebuild picks."
    col = STAT_MAP.get(p["stat_type"])
    p_more = prob_over_line(e["expected"], new_line, col, std=e.get("std"))
    d, prob = ("More", p_more) if p_more >= 0.5 else ("Less", 1.0 - p_more)
    return "\n".join([
        f"**What-if: {p['player_name']} {p['stat_type']}** at a line of **{new_line}** "
        f"(actual line {p['line']}):",
        f"- Same expected value **{e['expected']}** and std {e.get('std', '--')}",
        f"- P(More {new_line}) = **{p_more:.1%}**, P(Less) = {1 - p_more:.1%} -> model side: **{d}** at {prob:.1%}",
        f"- vs {e['breakeven']:.1%} break-even: {_edge_verdict(prob, e['breakeven'])}",
    ])


def flip_direction(p: dict) -> str:
    e = p.get("explain") or {}
    if not e:
        return "That pick predates the explain feature -- hit Refresh to rebuild picks."
    other = "Less" if p["direction"] == "More" else "More"
    other_prob = 1.0 - e["p_more"] if other == "Less" else e["p_more"]
    lines = [
        f"**{p['player_name']} {p['stat_type']} {other} {p['line']}** (the other side):",
        f"- P({other}) = 1 - {p['model_prob']:.1%} = **{other_prob:.1%}**",
        f"- vs {e['breakeven']:.1%} break-even: {_edge_verdict(other_prob, e['breakeven'])}",
    ]
    if p.get("odds_type") in ("goblin", "demon"):
        lines.append(f"- (warning) This is a {p['odds_type']} line -- only **More** is offered on the platform, "
                     f"so {other} isn't actually playable here.")
    return "\n".join(lines)


def compare_player(q_words: set, q: str, nums: list[float], props: list[dict]) -> str | None:
    matched = [p for p in props if p["player_name"].lower() in q
               or p["player_name"].lower().split()[-1] in q_words]
    if not matched:
        return None
    stat_words = {w for col, ws in _STAT_WORDS.items() for w in ws}
    if q_words & stat_words:
        by_stat = [p for p in matched
                   if _STAT_WORDS.get(STAT_MAP.get(p["stat_type"]), set()) & q_words]
        matched = by_stat or matched
    matched.sort(key=lambda p: (p["stat_type"], -p["edge"]))
    name = matched[0]["player_name"]
    out = [f"**{name} -- all current picks:**", ""]
    for p in matched[:12]:
        ot = {"goblin": " (goblin)", "demon": " (demon)"}.get(p.get("odds_type", ""), "")
        out.append(f"- **{p['stat_type']} {p['direction']} {p['line']}**{ot} ({p['platform']}) -- "
                   f"model {p['model_prob']:.1%}, edge {p['edge']:+.1%} ({p['confidence_tier']})")
    best = max(matched, key=lambda p: p["edge"])
    out.append(f"\nBiggest edge: **{best['stat_type']} {best['direction']} {best['line']}** "
               f"@ {best['platform']} at {best['edge']:+.1%}. "
               f"Note: different lines on different platforms are different bets -- check the line, not just the edge.")
    return "\n".join(out)


def superlative(q_words: set, props: list[dict]) -> str | None:
    if {"safest", "surest"} & q_words:
        key, label = (lambda p: p["model_prob"]), "highest model probability"
    elif {"best", "top", "biggest", "highest", "strongest"} & q_words:
        key, label = (lambda p: p["edge"]), "biggest edge"
    else:
        return None
    pool = props
    hit_cols = {col for col, ws in _STAT_WORDS.items() if ws & q_words}
    if hit_cols:
        pool = [p for p in props if STAT_MAP.get(p["stat_type"]) in hit_cols] or props
    if not pool:
        return None
    ranked = sorted(pool, key=key, reverse=True)[:5]
    out = [f"**Top picks by {label}:**", ""]
    for i, p in enumerate(ranked, 1):
        ot = {"goblin": " (goblin)", "demon": " (demon)"}.get(p.get("odds_type", ""), "")
        out.append(f"{i}. **{p['player_name']} {p['stat_type']} {p['direction']} {p['line']}**{ot} "
                   f"({p['platform']}) -- model {p['model_prob']:.1%}, edge {p['edge']:+.1%}")
    out.append("\nAsk *why* on any of these for the full breakdown.")
    return "\n".join(out)


# -- Entry point ------------------------------------------------------------------

def answer_question(query: str, picks: list[dict], games: list[dict], views: dict, ctx: dict) -> str:
    """Route a question to a what-if, comparison, ranking, or explanation.

    ctx is a mutable dict (e.g. st.session_state) used to remember the last
    prop pick so bare follow-ups ("what about a line of 240.5?") resolve.
    """
    q = query.lower()
    q_words = set(re.findall(r"[\w+\-']+", q))
    # numbers, excluding percentages ("91.3%") which are never lines
    nums = [float(n) for n in re.findall(r"(\d+(?:\.\d+)?)\s*(?!%)", q)
            if not re.search(re.escape(n) + r"\s*%", q)]
    props = [p for p in picks if p["pick_type"] == "prop"]

    def resolve_pick() -> dict | None:
        scored = sorted(((_score_prop(q, q_words, nums, p), p) for p in props),
                        key=lambda t: (t[0], t[1]["edge"]), reverse=True)
        if scored and scored[0][0] >= 3:
            return scored[0][1]
        return ctx.get("ask_last_pick")

    # 1. compare a player's picks across platforms/stats
    if {"compare", "vs", "versus"} & q_words:
        ans = compare_player(q_words, q, nums, props)
        if ans:
            return ans

    # 2. safest / best rankings
    ans = superlative(q_words, props)
    if ans:
        return ans

    # 3. line what-if -- explicit "line of 22.5", or a bare number in a follow-up phrase
    followup = any(t in q for t in ("what if", "what about", "instead", "how about"))
    m = re.search(r"line\s+(?:of\s+|at\s+|was\s+|were\s+)?(\d+(?:\.\d+)?)", q)
    if m or (followup and len(nums) == 1 and not (q_words & {"more", "less", "over", "under"})):
        pick = resolve_pick()
        if pick is None:
            return NO_CONTEXT_MSG
        ctx["ask_last_pick"] = pick
        return whatif_line(pick, float(m.group(1)) if m else nums[0])

    # 4. direction flip on the pick in context
    if followup and (q_words & {"more", "less", "over", "under", "flip", "opposite"}) and not nums:
        pick = ctx.get("ask_last_pick")
        if pick is None:
            return NO_CONTEXT_MSG
        return flip_direction(pick)

    # 5. plain explanation
    target = find_target(query, picks, games, views)
    if target is None:
        return NO_MATCH_MSG
    if target["kind"] == "prop":
        ctx["ask_last_pick"] = target["pick"]
        return render_prop(target["pick"])
    return render_game(target["game"], target["market"], target["picks"], target["view"])
