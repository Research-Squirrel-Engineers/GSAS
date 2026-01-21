# fuzzy-wobbly-semantic-alignment

This repository provides a **reproducible, ontology-driven framework** for modelling
*graded semantic alignment* in SKOS using **numeric degrees of connection**.

It combines three complementary layers:

1. **Minimal SKOS mapping degrees** (`exactMatch`, `closeMatch`, `relatedMatch`)
2. **7-star SKOS mapping refinement** (methodological layer)
3. **Empirical perceptions of probability** (language-based, empirical layer)

All layers are published as **property-only RDF (TBox)**, are fully mergeable into
a single ontology, and are accompanied by CSV tables and plots for transparency
and reproducibility.

This repository is intended to be cited together with a **working paper**
that explains the modelling decisions, mathematical background, and empirical basis
of the approach.

---

## Repository structure

```text
fuzzy-wobbly-semantic-alignment/
│
├─ ontology/
│  ├─ create_ontology.py
│  └─ merged_ontology.ttl
│
├─ skos/
│  ├─ skos.py
│  ├─ skos_minimal_degrees.ttl
│  ├─ skos_minimal_degrees.csv
│  └─ skos_minimal_degree_plot.jpg
│
├─ skos_7star/
│  ├─ skos_7star.py
│  ├─ probly.csv
│  ├─ skos_7star_mapping.ttl
│  ├─ skos_7star_degrees.csv
│  └─ skos_7star_degree_plot.jpg
│
├─ skos_perceptions/
│  ├─ skos_perceptions.py
│  ├─ probly.csv
│  ├─ skos_perceptions_mapping.ttl
│  ├─ skos_perceptions_stats.csv
│  └─ skos_perceptions_degree_plot.jpg
│
├─ .gitignore
└─ README.md
```

---

## Design principles

Across all scripts and outputs, the following principles apply:

- **SKOS-first modelling**  
  `skos:relatedMatch`, `skos:closeMatch`, and `skos:exactMatch` form the stable semantic backbone.

- **Property-only RDF (TBox)**  
  No individuals are created. All modelling is done using
  `owl:ObjectProperty` and `owl:AnnotationProperty`.

- **Single custom namespace**  
  All extensions use: <https://w3id.org/skos-plus/> ensuring stable, consistent URIs across all layers.

- **Numeric justification**  
Each mapping relation is annotated with a numeric
`skosplus:degreeOfConnection` in the range `[0,1]`.

- **Reproducibility & transparency**  
Every RDF file is accompanied by:
- a CSV table with numeric values
- a plot visualising the mapping logic

---

## skos/skos.py — Minimal SKOS degree model

### Purpose

This script defines a **baseline numeric interpretation** of SKOS mapping relations
*without* any 7-star or perception-based logic.

It answers the question:

> How can `skos:exactMatch`, `skos:closeMatch`, and `skos:relatedMatch`
> be grounded in numeric degrees of connection?

### What the script does

- Assigns a numeric degree of connection to:
- `skos:exactMatch`
- `skos:closeMatch`
- `skos:relatedMatch`
- Uses a simple, monotonic mapping in the range `[0,1]`
- Produces RDF, CSV, and a plot

### Outputs

- `skos_minimal_degrees.ttl`
- `skos_minimal_degrees.csv`
- `skos_minimal_degree_plot.jpg`

This layer acts as the **semantic anchor** for all further extensions.

---

## skos_7star/skos_7star.py — 7-star SKOS mapping refinement

### Purpose

This script introduces a **7-star refinement** of SKOS mappings that allows
`skos:relatedMatch` to be expressed at multiple granular levels.

It answers the question:

> How can SKOS mappings be refined without breaking SKOS semantics?

### What the script does

- Introduces five sub-properties of `skos:relatedMatch`:
  - very weak
  - weak
  - moderate
  - strong
  - very strong
- Maps:
  - star level 6 → `skos:closeMatch`
  - star level 7 → `skos:exactMatch`
- Computes numeric degrees using a **normalised, saturating exponential function**
- Annotates each relation with:
  - star level
  - degree of connection
  - an **internal explanatory perception band**

### Important modelling decision

The *perception band* is modelled as an **annotation property**, not as part of the
object-property hierarchy.  
This preserves a clean SKOS-first structure while keeping the internal logic documented.

### Outputs

- `skos_7star_mapping.ttl`
- `skos_7star_degrees.csv`
- `skos_7star_degree_plot.jpg`

---

## skos_perceptions/skos_perceptions.py — Empirical perceptions mapping

### Purpose

This script models **natural language probability phrases** (e.g. *Highly Likely*,
*Almost Certainly*) as SKOS-compatible mapping relations based on empirical data.

It answers the question:

> How do people interpret probability phrases, and how can this be integrated
> into SKOS mappings?

### Data source

- `probly.csv`  
Derived from the study published at:
<https://github.com/zonination/perceptions>

### What the script does

- Treats each perception phrase as a **mapping property**
- Computes:
  - median
  - quartiles
  - interquartile range (IQR)
  - Tukey whiskers
- Uses the **median (normalised to 0–1)** as the degree of connection
- Assigns each phrase as a sub-property of:
  - `skos:exactMatch`
  - `skos:closeMatch`
  - `skos:relatedMatch`
based on numeric thresholds
- Produces a plot showing:
  - the 17 empirical medians
  - a smooth approximation curve for generalisation

### Outputs

- `skos_perceptions_mapping.ttl`
- `skos_perceptions_stats.csv`
- `skos_perceptions_degree_plot.jpg`

---

## ontology/create_ontology.py — Merging all layers

### Purpose

This script merges all RDF layers into a **single coherent ontology**.

It answers the question:

> Can minimal SKOS mappings, 7-star refinement, and empirical perceptions
> coexist without semantic inconsistency?

### What the script does

- Loads:
- `skos_minimal_degrees.ttl`
- `skos_7star_mapping.ttl`
- `skos_perceptions_mapping.ttl`
- Merges them into one RDF graph
- Relies on shared URIs, namespaces, and the SKOS backbone

### Output

- `merged_ontology.ttl`

This file can be:
- opened directly in Protégé
- published as an ontology artefact
- referenced from academic publications

---

## License

MIT License