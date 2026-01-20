#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Minimal SKOS degrees (ONLY 3 SKOS mapping properties) + Option C plot
--------------------------------------------------------------------
RDF output contains ONLY:
- skos:relatedMatch
- skos:closeMatch
- skos:exactMatch

These three properties are annotated with:
- skosplus:degreeOfConnection (owl:AnnotationProperty)

Numeric logic uses the same 7-star exponential function:
d(s) = (1 - exp(-k*(s-1)/6)) / (1 - exp(-k))    for s in {1..7}, k>0

Mapping to the 3 SKOS properties:
- exactMatch   -> d(7)
- closeMatch   -> d(6)
- relatedMatch -> mean(d(1..5))  (represents the whole related-family)

Plot (Option C):
- The relatedMatch value is plotted at an "equivalent star level" s* in [1..7]
  such that d(s*) = mean(d(1..5)).
  This ensures the relatedMatch point lies ON the curve (fractional x).

Outputs (written next to this script by default):
- skos_minimal_degrees.ttl
- skos_minimal_degrees.csv
- skos_minimal_degree_plot.jpg (300 DPI)
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from rdflib import Graph, Namespace, Literal
from rdflib.namespace import RDF, RDFS, OWL, XSD, SKOS


SKOSPLUS_NS = "https://w3id.org/skos-plus/"

OUT_TTL = "skos_minimal_degrees.ttl"
OUT_CSV = "skos_minimal_degrees.csv"
OUT_PLOT = "skos_minimal_degree_plot.jpg"

DEFAULT_K = 2.0


def degree_of_connection(star: float, k: float) -> float:
    """
    Normalised saturating exponential from 0..1 for star in [1..7].

    d(s) = (1 - exp(-k*(s-1)/6)) / (1 - exp(-k))
    """
    if not (1.0 <= float(star) <= 7.0):
        raise ValueError("star must be in [1..7]")
    if k <= 0:
        raise ValueError("k must be > 0")

    x = (float(star) - 1.0) / 6.0
    num = 1.0 - math.exp(-k * x)
    den = 1.0 - math.exp(-k)
    d = num / den if den != 0 else 0.0
    return max(0.0, min(1.0, d))


def mean15_related_degree(k: float) -> float:
    """relatedMatch degree as mean(d(1..5))."""
    vals = [degree_of_connection(s, k) for s in range(1, 6)]
    return sum(vals) / len(vals)


def equivalent_star_for_degree(y: float, k: float) -> float:
    """
    Option C: invert d(s) to get s* in [1..7] such that d(s*) = y.

    s* = 1 + (6/k) * ( -ln(1 - y*(1 - e^{-k})) )
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


def build_rdf(k: float, outdir: Path) -> pd.DataFrame:
    SKOSPLUS = Namespace(SKOSPLUS_NS)

    g = Graph()
    g.bind("skos", SKOS)
    g.bind("rdfs", RDFS)
    g.bind("owl", OWL)
    g.bind("xsd", XSD)
    g.bind("skosplus", SKOSPLUS)

    # TBox: explicit class so we can attach domain/range
    g.add((SKOS.Concept, RDF.type, OWL.Class))

    # Ensure the 3 SKOS mapping relations are shown as object properties
    for p in (SKOS.relatedMatch, SKOS.closeMatch, SKOS.exactMatch):
        g.add((p, RDF.type, OWL.ObjectProperty))
        g.add((p, RDFS.domain, SKOS.Concept))
        g.add((p, RDFS.range, SKOS.Concept))

    # Annotation property for numeric value (avoids OWL2 punning / individuals)
    DEG = SKOSPLUS.degreeOfConnection
    g.add((DEG, RDF.type, OWL.AnnotationProperty))
    g.add((DEG, RDFS.label, Literal("degree of connection (0–1)", lang="en")))
    g.add(
        (
            DEG,
            RDFS.comment,
            Literal(
                "Numeric degree derived from the SKOS-Plus 7-star exponential function.",
                lang="en",
            ),
        )
    )

    # Compute degrees
    d_exact = degree_of_connection(7, k)
    d_close = degree_of_connection(6, k)
    d_related = mean15_related_degree(k)

    # Annotate ONLY the 3 SKOS properties
    g.add((SKOS.exactMatch, DEG, Literal(d_exact, datatype=XSD.decimal)))
    g.add((SKOS.closeMatch, DEG, Literal(d_close, datatype=XSD.decimal)))
    g.add((SKOS.relatedMatch, DEG, Literal(d_related, datatype=XSD.decimal)))

    # Labels (optional)
    g.add((SKOS.exactMatch, RDFS.label, Literal("SKOS exactMatch", lang="en")))
    g.add((SKOS.closeMatch, RDFS.label, Literal("SKOS closeMatch", lang="en")))
    g.add(
        (
            SKOS.relatedMatch,
            RDFS.label,
            Literal("SKOS relatedMatch (mean(d(1..5)))", lang="en"),
        )
    )

    df = pd.DataFrame(
        [
            {
                "property": str(SKOS.relatedMatch),
                "degree_of_connection": float(d_related),
                "note": "mean(d(1..5))",
                "k": float(k),
            },
            {
                "property": str(SKOS.closeMatch),
                "degree_of_connection": float(d_close),
                "note": "d(6)",
                "k": float(k),
            },
            {
                "property": str(SKOS.exactMatch),
                "degree_of_connection": float(d_exact),
                "note": "d(7)",
                "k": float(k),
            },
        ]
    )

    outdir.mkdir(parents=True, exist_ok=True)
    g.serialize(destination=str(outdir / OUT_TTL), format="turtle")
    df.to_csv(outdir / OUT_CSV, index=False, encoding="utf-8")

    return df


def plot_degrees(df: pd.DataFrame, outdir: Path) -> Path:
    """
    Plot the underlying 7-star curve and mark the 3 SKOS points.
    Option C for relatedMatch: plot at equivalent star level s* so the point lies ON the curve.
    Includes a band for the related-family (d(1)..d(5)).
    """
    k = float(df["k"].iloc[0])

    # Underlying curve
    stars = list(range(1, 8))
    curve = [degree_of_connection(s, k) for s in stars]

    # Band for relatedMatch (range of stars 1..5)
    # rel_vals = [degree_of_connection(s, k) for s in range(1, 6)]
    # rel_min = min(rel_vals)
    # rel_max = max(rel_vals)

    # y values from df
    y_related = float(
        df.loc[df["note"] == "mean(d(1..5))", "degree_of_connection"].iloc[0]
    )
    y_close = float(df.loc[df["note"] == "d(6)", "degree_of_connection"].iloc[0])
    y_exact = float(df.loc[df["note"] == "d(7)", "degree_of_connection"].iloc[0])

    # Option C: compute equivalent star level for related
    x_related = equivalent_star_for_degree(y_related, k)

    # Points (note: related is fractional x)
    points_x = [x_related, 6.0, 7.0]
    points_y = [y_related, y_close, y_exact]

    plt.figure()
    plt.plot(stars, curve, marker="o")
    plt.plot(points_x, points_y, marker="o", linestyle="")

    # --- Minimal labels on the orange points only (R / C / E) ---

    plt.annotate(
        "R",
        xy=(x_related, y_related),
        xytext=(0, 8),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
    )

    plt.annotate(
        "C",
        xy=(6.0, y_close),
        xytext=(0, 8),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
    )

    plt.annotate(
        "E",
        xy=(7.0, y_exact),
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
    plt.title(
        "SKOS minimal degrees derived from 7-star exponential function (mean15 + equivalent star)"
    )

    # Draw the relatedMatch band as a vertical anchor at x=3
    # plt.vlines(3, rel_min, rel_max, linewidth=4, alpha=0.6)

    # Label for the relatedMatch point (moved further left + slightly down)

    # --- Labels for the orange points (related / close / exact) ---

    out_file = outdir / OUT_PLOT
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    return out_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Minimal SKOS degrees (3 properties only, mean15 + Option C plot)."
    )
    parser.add_argument(
        "--k",
        type=float,
        default=DEFAULT_K,
        help=f"Steepness k (>0). Default: {DEFAULT_K}",
    )
    parser.add_argument(
        "--outdir", default=".", help="Output dir (relative to script dir). Default: ."
    )

    args = parser.parse_args()
    if args.k <= 0:
        raise ValueError("--k must be > 0")

    script_dir = Path(__file__).resolve().parent
    outdir = (script_dir / args.outdir).resolve()

    df = build_rdf(args.k, outdir)
    plot_file = plot_degrees(df, outdir)

    # Print a useful small summary, including the equivalent star
    y_related = float(
        df.loc[df["note"] == "mean(d(1..5))", "degree_of_connection"].iloc[0]
    )
    s_star = equivalent_star_for_degree(y_related, args.k)

    print("Done.")
    print(f"- RDF (Turtle): {outdir / OUT_TTL}")
    print(f"- CSV:          {outdir / OUT_CSV}")
    print(f"- Plot:         {plot_file}")
    print(
        f"- relatedMatch: mean(d(1..5))={y_related:.6f} -> equivalent star s*={s_star:.4f}"
    )
    print()
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
