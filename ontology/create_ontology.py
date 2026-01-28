#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create merged SKOS-Plus Ontology
===============================

This script merges the three ontology layers of the project into a single,
consolidated Turtle file:

1. Minimal SKOS degrees
2. 7-star SKOS refinement
3. Empirical perceptions-of-probability mapping

All input files are **property-only RDF (TBox)** and share a common SKOS backbone
and namespace conventions. The merge therefore consists of a simple graph union.

Folder layout (fixed)
---------------------
root/
 ├─ ontology/
 │   └─ create_ontology.py        <-- this script
 ├─ skos/
 │   └─ skos_minimal_degrees.ttl
 ├─ skos_7star/
 │   └─ skos_7star_mapping.ttl
 └─ skos_perceptions/
     └─ skos_perceptions_mapping.ttl

Output
------
- merged_ontology.ttl : merged ontology (Turtle)

Notes
-----
- Uses rdflib to parse each Turtle file and add triples to a single graph.
- Re-binds common prefixes for a clean, readable serialisation.
- Optionally scans for (subject, predicate) pairs with multiple objects, which
  may indicate accidental re-definitions across layers (not always an error).

Namespace
---------
- skosplus: https://w3id.org/skos-plus/

Usage
-----
Run from within the ontology/ folder or from anywhere:

    python create_ontology.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Set, Tuple

from rdflib import Graph, URIRef
from rdflib.namespace import RDF, RDFS, OWL, XSD, SKOS, DCTERMS


# -----------------------------
# PATHS / LAYOUT
# -----------------------------


def repo_root_from_this_file() -> Path:
    """
    Return the repository root directory.

    Assumes this script is located at:
        root/ontology/create_ontology.py
    """
    return Path(__file__).resolve().parents[1]


def ttl_input_paths(root: Path) -> Tuple[Path, Path, Path]:
    """Return the expected input TTL paths in fixed locations."""
    p_minimal = root / "skos" / "skos_minimal_degrees.ttl"
    p_7star = root / "skos_7star" / "skos_7star_mapping.ttl"
    p_perc = root / "skos_perceptions" / "skos_perceptions_mapping.ttl"
    p_4level = root / "skos_4level" / "skos_4level_mapping.ttl"
    return p_minimal, p_7star, p_perc, p_4level


# -----------------------------
# RDF HELPERS
# -----------------------------


def bind_common_prefixes(g: Graph) -> None:
    """
    Bind common prefixes to improve Turtle readability.
    Must match bindings used in the individual modules.
    """
    g.bind("rdf", RDF)
    g.bind("rdfs", RDFS)
    g.bind("owl", OWL)
    g.bind("xsd", XSD)
    g.bind("skos", SKOS)
    g.bind("dct", DCTERMS)
    g.bind("skosplus", "https://w3id.org/skos-plus/")


def parse_into(target: Graph, ttl_file: Path) -> int:
    """
    Parse a Turtle file into the target graph.

    Returns
    -------
    int
        Number of triples added.
    """
    before = len(target)
    target.parse(str(ttl_file), format="turtle")
    return len(target) - before


# -----------------------------
# DIAGNOSTICS
# -----------------------------


def find_potential_conflicts(g: Graph) -> Dict[Tuple[str, str], Set[str]]:
    """
    Identify (subject, predicate) pairs that have multiple distinct objects.

    This is not automatically an error:
    - multiple rdfs:label values (different languages) are normal
    - multiple annotations can be intentional

    However, the scan is useful for spotting unintended re-definitions.

    Returns
    -------
    dict
        (subject, predicate) -> set(objects)
    """
    spo_map: Dict[Tuple[str, str], Set[str]] = {}

    for s, p, o in g:
        if isinstance(s, URIRef) and isinstance(p, URIRef):
            key = (str(s), str(p))
            spo_map.setdefault(key, set()).add(str(o))

    return {k: v for k, v in spo_map.items() if len(v) > 1}


# -----------------------------
# MAIN
# -----------------------------


def main() -> int:
    root = repo_root_from_this_file()
    in_minimal, in_7star, in_perc, in_4level = ttl_input_paths(root)

    missing = [p for p in (in_minimal, in_7star, in_perc, in_4level) if not p.exists()]
    if missing:
        print("ERROR: Missing input TTL file(s):", file=sys.stderr)
        for p in missing:
            print(f" - {p}", file=sys.stderr)
        return 1

    out_file = (root / "ontology" / "merged_ontology.ttl").resolve()

    merged = Graph()
    bind_common_prefixes(merged)

    # Merge layers
    added_min = parse_into(merged, in_minimal)
    added_7s = parse_into(merged, in_7star)
    added_perc = parse_into(merged, in_perc)
    added_4l = parse_into(merged, in_4level)

    # Optional diagnostics
    conflicts = find_potential_conflicts(merged)

    merged.serialize(destination=str(out_file), format="turtle")

    # -----------------------------
    # SUMMARY OUTPUT
    # -----------------------------
    print("Merged ontology created.")
    print(f"Repository root: {root}")
    print("Inputs:")
    print(f" - {in_minimal}   (+{added_min} triples)")
    print(f" - {in_7star}     (+{added_7s} triples)")
    print(f" - {in_perc}      (+{added_perc} triples)")
    print(f" - {in_4level}   (+{added_4l} triples)")
    print("Output:")
    print(f" - {out_file}")
    print(f"Total triples: {len(merged)}")

    if conflicts:
        print(
            "\nPotential (subject, predicate) -> multiple objects situations detected:"
        )
        print(f"Count: {len(conflicts)}")
        shown = 0
        for (s, p), objs in conflicts.items():
            print(f"\n- {s}\n  {p}")
            for o in sorted(objs)[:10]:
                print(f"   -> {o}")
            if len(objs) > 10:
                print(f"   ... (+{len(objs) - 10} more)")
            shown += 1
            if shown >= 10:
                print("\n(Showing first 10 only.)")
                break
    else:
        print("\nNo (subject, predicate) multi-object situations detected.")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
