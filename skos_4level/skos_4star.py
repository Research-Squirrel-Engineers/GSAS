#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SKOS 4-Level Mapping (SKOS-Plus)
================================

This script defines a **4-level mapping refinement** inspired by the 7-star logic,
using the keywords:

- high
- medium
- low
- dubious

Model
-----
Property-only RDF (TBox). The 4 levels are represented as owl:ObjectProperty and
placed as sub-properties of:

- skos:relatedMatch

(Reason: keeps a clean SKOS backbone and avoids redefining SKOS core properties;
the 4-level layer is a refinement "under relatedMatch", comparable to the 7-star layer.)

Numeric logic
-------------
We reuse the same normalised saturating exponential function as the 7-star layer:

    d(s) = (1 - exp(-k*(s-1)/6)) / (1 - exp(-k))    for s in [1..7], k>0

We then derive the 4 levels from star bins (default mapping):

- dubious = mean(d(1), d(2))
- low     = mean(d(3), d(4))
- medium  = d(5)
- high    = mean(d(6), d(7))

Each level property is annotated with:
- skosplus:degreeOfConnection  (0..1)

Plot
----
Plots the underlying 7-star curve (1..7) and marks the 4 level points.
Each point is plotted at an "equivalent star level" s* (fractional x) such that:

    d(s*) = D(level)

This ensures all points lie ON the curve.

Inputs
------
- None

Outputs (written next to this script by default)
------------------------------------------------
- skos_4level_mapping.ttl
- skos_4level_degrees.csv
- skos_4level_degree_plot.jpg  (300 DPI)

Namespace
---------
- skosplus: https://w3id.org/skos-plus/

Usage
-----
Run from within the folder:

    python skos_4level.py

Optional arguments:

    python skos_4level.py --help
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import matplotlib.pyplot as plt

from rdflib import Graph, Literal, Namespace
from rdflib.namespace import OWL, RDF, RDFS, SKOS, XSD


# -----------------------------
# CONFIG
# -----------------------------

SKOSPLUS_NS = "https://w3id.org/skos-plus/"

OUT_TTL = "skos_4level_mapping.ttl"
OUT_CSV = "skos_4level_degrees.csv"
OUT_PLOT = "skos_4level_degree_plot.jpg"

DEFAULT_K = 2.0
EXPORT_DPI = 300


# -----------------------------
# DEGREE FUNCTION
# -----------------------------


def degree_of_connection(star: float, k: float) -> float:
    """
    Normalised saturating exponential from 0..1 for star in [1..7].

    d(s) = (1 - exp(-k*(s-1)/6)) / (1 - exp(-k))
    """
    s = float(star)
    if not (1.0 <= s <= 7.0):
        raise ValueError("star must be in [1..7]")
    if k <= 0:
        raise ValueError("k must be > 0")

    x = (s - 1.0) / 6.0
    num = 1.0 - math.exp(-k * x)
    den = 1.0 - math.exp(-k)
    d = num / den if den != 0 else 0.0
    return max(0.0, min(1.0, d))


def equivalent_star_for_degree(y: float, k: float) -> float:
    """
    Invert d(s) to get s* in [1..7] such that d(s*) = y.

    Derived from:
      y = (1 - exp(-k*(s-1)/6)) / (1 - exp(-k))

    => s* = 1 + (6/k) * ( -ln(1 - y*(1 - e^{-k})) )
    """
    y = float(y)
    if not (0.0 <= y <= 1.0):
        raise ValueError("y must be in [0..1]")
    if k <= 0:
        raise ValueError("k must be > 0")

    if y <= 0.0:
        return 1.0
    if y >= 1.0:
        return 7.0

    term = 1.0 - y * (1.0 - math.exp(-k))
    term = max(1e-15, min(1.0, term))  # numeric safety
    s_star = 1.0 + (6.0 / k) * (-math.log(term))
    return max(1.0, min(7.0, s_star))


# -----------------------------
# 4-LEVEL DEFINITIONS
# -----------------------------


def four_level_bins() -> Dict[str, List[int]]:
    """
    Default mapping from 4 levels to 7-star bins.
    """
    return {
        "dubious": [1, 2],
        "low": [3, 4],
        "medium": [5],
        "high": [6, 7],
    }


def level_degree(level: str, stars: List[int], k: float) -> float:
    """Compute the level degree as mean of d(star) over its bin."""
    vals = [degree_of_connection(s, k) for s in stars]
    return sum(vals) / len(vals)


# -----------------------------
# RDF HELPERS
# -----------------------------


def bind_common_prefixes(g: Graph) -> Namespace:
    """Bind common prefixes for stable Turtle output and return the SKOSPLUS namespace."""
    skosplus = Namespace(SKOSPLUS_NS)
    g.bind("skos", SKOS)
    g.bind("rdfs", RDFS)
    g.bind("owl", OWL)
    g.bind("xsd", XSD)
    g.bind("skosplus", skosplus)
    return skosplus


def ensure_skos_backbone(g: Graph) -> None:
    """
    Make the SKOS mapping backbone explicit for Protégé:
    - declare skos:Concept as OWL class
    - declare mapping properties as owl:ObjectProperty with domain/range skos:Concept
    """
    g.add((SKOS.Concept, RDF.type, OWL.Class))
    for p in (SKOS.relatedMatch, SKOS.closeMatch, SKOS.exactMatch):
        g.add((p, RDF.type, OWL.ObjectProperty))
        g.add((p, RDFS.domain, SKOS.Concept))
        g.add((p, RDFS.range, SKOS.Concept))


def declare_annotation_property(g: Graph, prop, label_en: str, comment_en: str) -> None:
    """Declare an annotation property with label/comment."""
    g.add((prop, RDF.type, OWL.AnnotationProperty))
    g.add((prop, RDFS.label, Literal(label_en, lang="en")))
    g.add((prop, RDFS.comment, Literal(comment_en, lang="en")))


# -----------------------------
# RDF BUILD
# -----------------------------


def build_rdf(k: float, outdir: Path) -> pd.DataFrame:
    """
    Build the RDF graph and write:
    - OUT_TTL
    - OUT_CSV

    Returns the degrees table as DataFrame.
    """
    g = Graph()
    skosplus = bind_common_prefixes(g)
    ensure_skos_backbone(g)

    # Stable numeric property used in other layers
    DEG = skosplus.degreeOfConnection
    declare_annotation_property(
        g,
        DEG,
        "degree of connection (0–1)",
        "Numeric degree derived from the SKOS-Plus 7-star exponential function.",
    )

    # Optional: keep bin info as annotation (merge-safe)
    BIN = skosplus.starBin
    declare_annotation_property(
        g,
        BIN,
        "star bin",
        "Internal explanatory star bin used to derive a 4-level degree (annotation only).",
    )

    bins = four_level_bins()

    rows = []
    for level, stars in bins.items():
        # URI localname: relatedMatchHigh / relatedMatchMedium / ...
        local = "relatedMatch" + level.capitalize()
        p = skosplus[local]

        d = level_degree(level, stars, k)

        g.add((p, RDF.type, OWL.ObjectProperty))
        g.add((p, RDFS.subPropertyOf, SKOS.relatedMatch))
        g.add((p, RDFS.domain, SKOS.Concept))
        g.add((p, RDFS.range, SKOS.Concept))

        g.add((p, RDFS.label, Literal(f"relatedMatch {level}", lang="en")))
        g.add((p, DEG, Literal(d, datatype=XSD.decimal)))
        g.add((p, BIN, Literal(",".join(str(s) for s in stars), lang="en")))

        rows.append(
            {
                "level": level,
                "property": str(p),
                "degree_of_connection": float(d),
                "note": f"mean(d({','.join(str(s) for s in stars)}))",
                "stars": ",".join(str(s) for s in stars),
                "k": float(k),
            }
        )

    df = pd.DataFrame(rows).sort_values("degree_of_connection").reset_index(drop=True)

    outdir.mkdir(parents=True, exist_ok=True)
    g.serialize(destination=str(outdir / OUT_TTL), format="turtle")
    df.to_csv(outdir / OUT_CSV, index=False, encoding="utf-8")
    return df


# -----------------------------
# PLOT
# -----------------------------


def plot_degrees(df: pd.DataFrame, outdir: Path) -> Path:
    """
    Plot the underlying 7-star curve and mark the 4 level points.

    All points are plotted at equivalent star levels s* so they lie ON the curve.
    Labels on points: D (dubious), L (low), M (medium), H (high).
    """
    k = float(df["k"].iloc[0])

    stars = list(range(1, 8))
    curve = [degree_of_connection(s, k) for s in stars]

    # Extract degrees
    def get(level: str) -> float:
        return float(df.loc[df["level"] == level, "degree_of_connection"].iloc[0])

    y_d = get("dubious")
    y_l = get("low")
    y_m = get("medium")
    y_h = get("high")

    x_d = equivalent_star_for_degree(y_d, k)
    x_l = equivalent_star_for_degree(y_l, k)
    x_m = equivalent_star_for_degree(y_m, k)
    x_h = equivalent_star_for_degree(y_h, k)

    plt.figure(figsize=(10, 4))
    plt.plot(stars, curve, marker="o")  # curve
    plt.plot([x_d, x_l, x_m, x_h], [y_d, y_l, y_m, y_h], marker="o", linestyle="")

    # Minimal labels
    plt.annotate(
        "D",
        xy=(x_d, y_d),
        xytext=(0, 8),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
    )
    plt.annotate(
        "L",
        xy=(x_l, y_l),
        xytext=(0, 8),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
    )
    plt.annotate(
        "M",
        xy=(x_m, y_m),
        xytext=(0, 8),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
    )
    plt.annotate(
        "H",
        xy=(x_h, y_h),
        xytext=(0, -12),
        textcoords="offset points",
        ha="center",
        va="top",
        fontsize=11,
        fontweight="bold",
    )

    plt.xticks(stars)
    plt.ylim(0, 1.05)
    plt.xlabel("Underlying 7-star level (for numeric logic)")
    plt.ylabel("Degree of Connection (0–1)")
    plt.title("SKOS 4-level degrees (derived from 7-star bins)")

    out_file = outdir / OUT_PLOT
    plt.savefig(out_file, dpi=EXPORT_DPI, bbox_inches="tight")
    plt.close()
    return out_file


# -----------------------------
# MAIN
# -----------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SKOS 4-level mapping derived from 7-star exponential degrees."
    )
    parser.add_argument(
        "--k",
        type=float,
        default=DEFAULT_K,
        help=f"Steepness k (>0). Default: {DEFAULT_K}",
    )
    parser.add_argument(
        "--outdir",
        default=".",
        help="Output dir (relative to script dir). Default: .",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.k <= 0:
        raise ValueError("--k must be > 0")

    script_dir = Path(__file__).resolve().parent
    outdir = (script_dir / args.outdir).resolve()

    df = build_rdf(args.k, outdir)
    plot_file = plot_degrees(df, outdir)

    print("Done.")
    print(f"- RDF (Turtle): {outdir / OUT_TTL}")
    print(f"- CSV:          {outdir / OUT_CSV}")
    print(f"- Plot:         {plot_file}")
    print()
    print(df.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
