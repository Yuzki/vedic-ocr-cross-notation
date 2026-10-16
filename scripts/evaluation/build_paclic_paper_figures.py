"""Shared helpers retained for the PACLIC v2 release."""

from __future__ import annotations


import argparse


import json


import math


import sys


from collections.abc import Sequence


from pathlib import Path


from typing import Any, Optional


from . import aggregate_accent_ratio_multiseed as agg


from .build_paclic_accent_transfer_analysis import ERROR_CLASSES, PLATEAU_RATIOS


_AXIS_B = "results"


MULTISEED_DIR = "results/_maingrid_source"


DATA_EFFICIENCY_DIR = "results/_data_efficiency_source"


COLOR_RV = "#2a78d6"  # blue   — RV test / RV->MS / average / baseline


COLOR_MS = "#eb6834"  # orange — MS test / MS->RV / worst-case / +minority


MIX_COLORS: dict[float, str] = {
    1.0: "#184f95",  # blue ramp, dark   — RV-only
    0.75: "#3987e5",  # blue ramp, mid
    0.5: "#7d7b74",  # neutral midpoint  — balanced 50:50
    0.25: "#eb6834",  # orange ramp, mid
    0.0: "#a5380f",  # orange ramp, dark — MS-only
}


MIX_MARKERS: dict[float, str] = {1.0: "o", 0.75: "^", 0.5: "D", 0.25: "v", 0.0: "s"}


MixLinestyle = str | tuple[int, tuple[int, ...]]


MIX_LINESTYLES: dict[float, MixLinestyle] = {
    1.0: "-",
    0.75: (0, (5, 1)),
    0.5: (0, (1, 1)),
    0.25: (0, (3, 1, 1, 1)),
    0.0: "--",
}


INK_PRIMARY = "#0b0b0b"


INK_SECONDARY = "#52514e"


GRID = "#e1e0d9"


BASELINE = "#c3c2b7"


def _stat(d: Optional[dict[str, Any]]) -> Optional[agg.Stat]:
    """Rebuild a :class:`Stat` from its committed ``as_dict`` form (or ``None``)."""
    if d is None:
        return None
    return agg.Stat(**agg._stat_from_dict(d))


def _mean(d: Optional[dict[str, Any]]) -> float:
    st = _stat(d)
    return st.mean if st is not None and st.mean is not None else math.nan


def _hw(d: Optional[dict[str, Any]]) -> float:
    st = _stat(d)
    return (
        st.ci95_halfwidth if st is not None and st.ci95_halfwidth is not None else 0.0
    )


def _ratio_key(rv_ratio: float) -> str:
    """The ``per_ratio`` JSON key for a mix ratio (matches ``f'{rv:g}'``)."""
    return f"{rv_ratio:g}"


def _mix_label(rv_ratio: float) -> str:
    return f"{int(round(rv_ratio * 100))}:{int(round((1 - rv_ratio) * 100))}"


FONT_SCALE: float = 1.45


def _fs(base: float) -> float:
    """A base point size scaled for print."""
    return round(base * FONT_SCALE, 2)


def _seed_caption(grid: dict[str, Any]) -> str:
    """Caption for a seed-aggregated panel — the count, never the seed values.

    The individual seed numbers are provenance, not something a paper figure's
    reader can use, and they cost space that the print size needs. ``n`` stays
    DERIVED from the grid rather than written as a literal: it comes from
    ``coverage["seeds_present"]``, which is what prevents a caption claiming more
    seeds than the bundle actually has.
    """
    seeds = grid.get("seeds", list(agg.SEEDS))
    return f"mean ± 95% CI (n={len(seeds)})"


def _get_plt() -> Any:
    """Return a pyplot configured with the Agg backend and the paper style."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": _fs(10),
            "axes.titlesize": _fs(12),
            "axes.titleweight": "bold",
            "axes.labelsize": _fs(10),
            "axes.labelcolor": INK_PRIMARY,
            "axes.edgecolor": BASELINE,
            "axes.linewidth": 0.8,
            "axes.grid": False,
            "xtick.color": INK_SECONDARY,
            "ytick.color": INK_SECONDARY,
            "xtick.labelsize": _fs(9),
            "ytick.labelsize": _fs(9),
            "text.color": INK_PRIMARY,
            "legend.fontsize": _fs(9),
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "pdf.fonttype": 42,  # embed editable TrueType text (true vector)
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    return plt


def _style_axis(ax: Any, *, ylabel: str = "", xlabel: str = "") -> None:
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    if ylabel:
        ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)


def _save(fig: Any, out_dir: Path, stem: str, *, make_pdf: bool, dpi: int) -> list[str]:
    """Write the PNG (>=300 dpi) and, unless suppressed, the vector PDF.

    Metadata is pinned (PNG ``Software`` fixed, PDF ``CreationDate`` cleared) so
    the committed figures regenerate byte-identically across runs.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    png = out_dir / f"{stem}.png"
    fig.savefig(png, dpi=dpi, metadata={"Software": "paclic-paper-figures"})
    written.append(png.name)
    if make_pdf:
        pdf = out_dir / f"{stem}.pdf"
        fig.savefig(pdf, metadata={"CreationDate": None})
        written.append(pdf.name)
    return written


def _bar_value_labels(ax: Any, bars: Any, values: list[float], fmt: str) -> None:
    for rect, val in zip(bars, values):
        if not math.isfinite(val):
            continue
        ax.annotate(
            fmt.format(val),
            xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=_fs(8),
            color=INK_PRIMARY,
        )


def fig_per_ratio_performance(
    multiseed: dict[str, Any], out_dir: Path, *, make_pdf: bool, dpi: int
) -> dict[str, Any]:
    plt = _get_plt()
    per_ratio = multiseed["per_ratio"]
    ratios = list(agg.RV_RATIOS)
    x = list(range(len(ratios)))
    labels = [_mix_label(r) for r in ratios]
    panels = [
        ("(a) Character error rate", "rv_cer", "ms_cer", "CER  (↓ better)", 3),
        ("(b) Word error rate", "rv_wer", "ms_wer", "WER  (↓ better)", 3),
        ("(c) Accent F1", "rv_accent_f1", "ms_accent_f1", "accent F1  (↑ better)", 3),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.0))
    for ax, (title, rv_key, ms_key, ylabel, _dg) in zip(axes, panels):
        for key, color, label, ls, mk in (
            (rv_key, COLOR_RV, "RV test", "-", "o"),
            (ms_key, COLOR_MS, "MS test", "--", "s"),
        ):
            means = [_mean(per_ratio[_ratio_key(r)][key]) for r in ratios]
            hw = [_hw(per_ratio[_ratio_key(r)][key]) for r in ratios]
            ax.errorbar(
                x,
                means,
                yerr=hw,
                color=color,
                marker=mk,
                markersize=6,
                markeredgecolor="white",
                markeredgewidth=0.6,
                linewidth=2.0,
                linestyle=ls,
                capsize=3,
                label=label,
            )
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        _style_axis(ax, ylabel=ylabel, xlabel="training mix (RV:MS)")
        ax.set_ylim(bottom=0)
        ax.legend(loc="best")
    fig.suptitle(
        f"Per-mix-ratio OCR performance on the RV and MS test sets — "
        f"{_seed_caption(multiseed['grid'])}",
        fontsize=_fs(12.5),
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    written = _save(
        fig, out_dir, "fig1_per_ratio_performance", make_pdf=make_pdf, dpi=dpi
    )
    plt.close(fig)
    return {
        "stem": "fig1_per_ratio_performance",
        "title": "Main per-mix-ratio performance",
        "shows": (
            "CER, WER, and accent F1 on the RV-test and MS-test sets across the "
            "five RV:MS training mixes (100:0 … 0:100), mean ± 95% CI over "
            "n=3 seeds. The headline result: how each notation's accuracy trades off "
            "as the training mix shifts."
        ),
        "files": written,
        "sources": [
            {
                "file": f"{MULTISEED_DIR}/aggregate.json",
                "fields": [
                    "per_ratio.<ratio>.rv_cer / ms_cer",
                    "per_ratio.<ratio>.rv_wer / ms_wer",
                    "per_ratio.<ratio>.rv_accent_f1 / ms_accent_f1",
                ],
            },
            {"file": f"{MULTISEED_DIR}/tables/per_ratio.csv"},
        ],
    }


def fig_retention_adaptation(
    multiseed: dict[str, Any], out_dir: Path, *, make_pdf: bool, dpi: int
) -> dict[str, Any]:
    plt = _get_plt()
    ra = multiseed["retention_adaptation"]
    width = 0.36
    # (scenario label, contrast block). a = baseline (single notation), b = +25% minority.
    retention = [
        ("RV notation\n(100:0 → 75:25)", ra["rv_primary_add_ms"]["rv_retention_cer"]),
        ("MS notation\n(0:100 → 25:75)", ra["ms_primary_add_rv"]["ms_retention_cer"]),
    ]
    adaptation = [
        (
            "MS added to RV\n(100:0 → 75:25)",
            ra["rv_primary_add_ms"]["ms_adaptation_accent_f1"],
        ),
        (
            "RV added to MS\n(0:100 → 25:75)",
            ra["ms_primary_add_rv"]["rv_adaptation_accent_f1"],
        ),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6))
    for ax, title, ylabel, group, fmt in (
        (
            axes[0],
            "(a) Retention",
            "CER  (↓ better)",
            retention,
            "{:.3f}",
        ),
        (
            axes[1],
            "(b) Adaptation",
            "accent F1  (↑ better)",
            adaptation,
            "{:.3f}",
        ),
    ):
        x = list(range(len(group)))
        before = [_mean(c["a"]) for _lbl, c in group]
        before_hw = [_hw(c["a"]) for _lbl, c in group]
        after = [_mean(c["b"]) for _lbl, c in group]
        after_hw = [_hw(c["b"]) for _lbl, c in group]
        b0 = ax.bar(
            [xi - width / 2 for xi in x],
            before,
            width,
            yerr=before_hw,
            capsize=3,
            color=COLOR_RV,
            edgecolor="white",
            linewidth=1.2,
            label="single-notation baseline",
            error_kw={"ecolor": INK_SECONDARY, "elinewidth": 1.0},
        )
        b1 = ax.bar(
            [xi + width / 2 for xi in x],
            after,
            width,
            yerr=after_hw,
            capsize=3,
            color=COLOR_MS,
            edgecolor="white",
            linewidth=1.2,
            hatch="///",
            label="+25% minority notation",
            error_kw={"ecolor": INK_SECONDARY, "elinewidth": 1.0},
        )
        _bar_value_labels(ax, b0, before, fmt)
        _bar_value_labels(ax, b1, after, fmt)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([lbl for lbl, _c in group])
        _style_axis(ax, ylabel=ylabel)
        # Headroom so the Δ labels sit inside the axes, clear of the panel title.
        panel_max = max(
            max(m + h for m, h in zip(before, before_hw)),
            max(m + h for m, h in zip(after, after_hw)),
        )
        ax.set_ylim(0, panel_max * 1.32)
        # Δ = committed seed-paired mean difference, placed at a fixed height.
        for xi, (_lbl, c) in zip(x, group):
            ax.text(
                xi,
                0.92,
                f"Δ {_mean(c['diff']):+.3f}",
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=_fs(8.5),
                fontweight="bold",
                color=INK_PRIMARY,
            )
    leg_handles, leg_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        leg_handles,
        leg_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.95),
        ncol=2,
        fontsize=_fs(9),
    )
    fig.suptitle(
        f"Adding a 25% minority notation — {_seed_caption(multiseed['grid'])}",
        fontsize=_fs(12.5),
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    written = _save(
        fig, out_dir, "fig4_retention_adaptation", make_pdf=make_pdf, dpi=dpi
    )
    plt.close(fig)
    return {
        "stem": "fig4_retention_adaptation",
        "title": "Retention & adaptation (+25% minority)",
        "shows": (
            "Paired before/after when a 25% minority notation is added to a "
            "single-notation baseline: (a) the primary notation's CER barely moves "
            "(retention — no catastrophic forgetting); (b) the added minority's "
            "accent F1 jumps sharply (adaptation). Δ is the committed seed-paired "
            "mean difference; mean ± 95% CI, n=3 seeds."
        ),
        "files": written,
        "sources": [
            {
                "file": f"{MULTISEED_DIR}/aggregate.json",
                "fields": [
                    "retention_adaptation.rv_primary_add_ms.{rv_retention_cer,ms_adaptation_accent_f1}.{a,b,diff}",
                    "retention_adaptation.ms_primary_add_rv.{ms_retention_cer,rv_adaptation_accent_f1}.{a,b,diff}",
                ],
            },
            {"file": f"{MULTISEED_DIR}/tables/retention_adaptation.csv"},
        ],
    }


DATA_EFFICIENCY_SUPTITLE: str = (
    "Data efficiency by training mix — base letters transfer across notation, "
    "accent marks do not"
)


def fig_data_efficiency(
    data_efficiency: dict[str, Any],
    out_dir: Path,
    *,
    make_pdf: bool,
    dpi: int,
    rv_ratios: Sequence[float] = agg.RV_RATIOS,
    suptitle: str = DATA_EFFICIENCY_SUPTITLE,
) -> dict[str, Any]:
    """Per-mix data-efficiency curves: all-symbol CER, then the MS test's channels.

    Four single-axis panels (never a shared/dual y across different units). The
    x axis is the training total on a log scale, because the four totals are
    near-geometric (1042 / 2605 / 5210 / 10420) and a linear axis would crowd the
    low-data end where the curves actually separate.

    ``rv_ratios`` and ``suptitle`` exist for the **v2** data-efficiency study, which
    trains three mixes at two totals rather than five at four: the defaults reproduce
    the v1 figure exactly, and a caller with a different ratio set must say so rather
    than ``KeyError`` on a series that was never trained. ``suptitle`` must be
    overridden by any caller that is not plotting v1, because the default asserts a v1
    conclusion (:data:`DATA_EFFICIENCY_SUPTITLE`).
    """
    plt = _get_plt()
    curves = data_efficiency["curves"]
    ratios = list(rv_ratios)
    # (a)/(b) the aggregate gap; (c)/(d) split the SAME test set into its two
    # channels, which is what carries the title: base letters cross notation
    # freely (100:0 is the *best* series in (c)), accent marks do not (100:0 is
    # the worst in (d), and no amount of data lifts it to the in-notation band).
    panels = (
        ("(a) RV test", "rv_cer", "CER  (↓ better)"),
        ("(b) MS test", "ms_cer", "CER  (↓ better)"),
        (
            "(c) MS test, base only",
            "ms_accent_stripped_cer",
            "base-letter CER  (↓ better)",
        ),
        ("(d) MS test, accents", "ms_accent_f1", "accent F1  (↑ better)"),
    )
    fig, axes = plt.subplots(1, 4, figsize=(16.4, 4.0))
    totals: list[int] = []
    for ax, (title, metric, ylabel) in zip(axes, panels):
        for rv_ratio in ratios:
            series = curves[_ratio_key(rv_ratio)][metric]
            xs = [int(p["total_train_samples"]) for p in series]
            means = [float(p["mean"]) for p in series]
            hw = [float(p["ci95_halfwidth"]) for p in series]
            totals = xs or totals
            ax.errorbar(
                xs,
                means,
                yerr=hw,
                color=MIX_COLORS[rv_ratio],
                marker=MIX_MARKERS[rv_ratio],
                markersize=6,
                markeredgecolor="white",
                markeredgewidth=0.6,
                linewidth=2.0,
                linestyle=MIX_LINESTYLES[rv_ratio],
                capsize=3,
                label=_mix_label(rv_ratio),
            )
        ax.set_xscale("log")
        ax.set_xticks(totals)
        ax.set_xticklabels(
            [f"{t}\n({round(100 * t / max(totals))}%)" for t in totals],
            fontsize=_fs(8.5),
        )
        ax.minorticks_off()
        ax.set_title(title)
        _style_axis(ax, ylabel=ylabel, xlabel="training total (parallel keys)")
        ax.set_ylim(bottom=0)
    # Figure-level, not inside panel (a): at print type size an in-axes legend
    # covered the curves it was labelling. Same placement fig4 uses.
    leg_handles, leg_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        leg_handles,
        leg_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=len(leg_labels),
        title="training mix (RV:MS)",
        fontsize=_fs(9),
        title_fontsize=_fs(9),
    )
    fig.suptitle(suptitle, fontsize=_fs(12.5), fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    written = _save(fig, out_dir, "fig6_data_efficiency", make_pdf=make_pdf, dpi=dpi)
    plt.close(fig)
    return {
        "stem": "fig6_data_efficiency",
        "title": "Mix-ratio x data-efficiency curves",
        "shows": (
            "Four panels against the training total (10/25/50/100% of the 10,420 "
            "shared parallel keys) for each of the five RV:MS mixes, mean ± 95% CI "
            "over n=3 seeds. (a)/(b) the aggregate picture: on all-symbol CER the "
            "wrong-notation mix at 100% of the data stays worse than a "
            "right-notation mix at 10% of it (RV test 0.271 at 0:100/100% vs 0.098 "
            "at 100:0/10%; MS test 0.166 at 100:0/100% vs 0.065 at 0:100/10%). "
            "(c)/(d) split that same MS test set into its two channels, which is "
            "where the penalty is actually located. In (c), base letters only, the "
            "cross-notation series is not penalised but is the *best* one — RV-only "
            "training reaches 0.028 at 100%, below every MS-inclusive mix "
            "(0.032-0.034). In (d), accent marks only, the same series sits at "
            "0.221-0.317 against an in-notation band of 0.946-0.955 at 100%. Its "
            "trend is NOT monotone (0.280 at 10%, 0.221 at 25%, 0.243 at 50%, 0.317 "
            "at 100%), and the 10%-to-100% CIs overlap, so the defensible "
            "improvement claim is 25% -> 100% (0.221 -> 0.317, disjoint CIs) — a "
            "real gain that still leaves the channel ~3x below the in-notation "
            "level. So the cross-notation cost is accent-borne, not general — which "
            "is what motivates the minimum-target-notation-ratio reading."
        ),
        "files": written,
        "sources": [
            {
                "file": f"{DATA_EFFICIENCY_DIR}/aggregate.json",
                # Derived from `panels` so the declared provenance cannot drift
                # from what the figure actually reads — the hand-written list here
                # named rv_accent_f1 (never plotted) and omitted both accent
                # channels that are.
                "fields": [
                    f"curves.<ratio>.{metric}[].mean / ci95_halfwidth"
                    for _title, metric, _ylabel in panels
                ]
                + ["curves.<ratio>.<metric>[].total_train_samples"],
            },
            {"file": f"{DATA_EFFICIENCY_DIR}/tables/curve.csv"},
        ],
    }

