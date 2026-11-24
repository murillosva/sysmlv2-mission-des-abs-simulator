# `scripts/data/` — Pinned Data Snapshots

Versioned input data used by the generators in `scripts/` and by the tests. Do not edit
these files by hand; regenerate them from the primary source if it changes. Their SHA-256
values below refer to the bytes stored in git (LF line endings, enforced by
`.gitattributes`).

## `setores_zida_27.json`

The 27-sector generating polygon of the stochastic incursion process, with the sector
method adapted from Miranda (2025). Each sector carries the exact polyline of its border arc
(`arco_lonlat`, `arco_s_km`) besides its end points, chord and azimuth. The field names are
in Portuguese, as in the source snapshot.

- **Source:** Natural Earth 50m, Brazil–Colombia and Brazil–Peru borders, clipped to the
  area of operations.
- **SHA-256:** `753c70a02ed472b75a5894e7826cc2c482d5b4048a3f83ed84c834cd270bed88`

| Consumer | Use |
|---|---|
| `scripts/gen_arc_block.py` | Writes the `border_sectors` block, with the `arc` field, of `configs/zida_shield_default.yaml` |
| `tests/statistical/test_miranda_v2_adherence.py` | Statistical adherence test of the incursion generator |

## `brazil_ne50m_mainland.json`

Exterior ring of mainland Brazil (the largest part of the MultiPolygon), the territory
reference of the inside-territory tests of the border geometry audit; 1,648 vertices,
about 60 KB.

- **Source:** feature `Brazil` of `ne_50m_admin_0_countries.geojson` (Natural Earth 50m,
  nvkelso/natural-earth-vector), the same base as the 27 sectors.
- **SHA-256 of the source:** `3e458fc036ad0a66411f2c1e6cac49c5d7bfb81cb1123bc513b22511a2b7fdeb`
- **SHA-256 of this file:** `c0c934374822bebc9c5b71b8473f16718f4fdcca282a1c80c47263e5e756cfb7`
- **Why vendored:** it avoids `shapely` as a dependency; point-in-polygon is a ray-casting
  test in numpy (`points_in_ring`). It is a generalized world-scale representation, not a
  legal boundary demarcation ([`docs/assumptions.md`](../../docs/assumptions.md) §4a).

| Consumer | Use |
|---|---|
| `scripts/border_geometry_audit.py` | Geometric audit of the generating polygon (chord versus arc, compression of ω, inward heading) |
| `tests/unit/test_border_geometry_audit.py` | Known-answer tests of the geometric primitives |
