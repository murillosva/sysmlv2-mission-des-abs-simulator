"""HTML dashboard renderer of the consolidated report.

The **presentation** owner of the analysis layer: turns the consolidated
``report.json`` contract into a self-contained, offline, byte-deterministic HTML
dashboard (tabs per response + a sortable master table). ``jinja2`` is imported
**lazily** (optional ``[analysis]`` extra) and figures arrive as **PNG bytes** —
this module never imports ``matplotlib`` (the sole matplotlib owner is
``report.py``). It **only formats the contract** and never computes a number:
``render_html`` is a pure function of ``(report, figures_png)``.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

__all__ = ["render_html", "write_html"]

_INSTALL_HINT = (
    "jinja2 is required for the HTML report; install the analysis extra: "
    'pip install -e ".[analysis]"'
)

# Vendored, byte-deterministic template (no CDN, no timestamp).
# CSS + a tiny sortable-table script are inlined so the file opens offline.
_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Consolidated report ({{ report.study.id }})</title>
<style>
body{font-family:Segoe UI,Arial,sans-serif;margin:0;background:#f4f6f8;color:#1b2733}
header{background:#1b2733;color:#fff;padding:16px 24px}
header h1{margin:0;font-size:20px}
.wrap{max-width:1100px;margin:0 auto;padding:16px 24px}
.card{background:#fff;border:1px solid #d7dde3;border-radius:6px;padding:16px;margin:14px 0}
.validity{border-left:6px solid #e0a500;background:#fffbf0}
.summary{border-left:6px solid #2e7d32}
h2{font-size:16px;margin:0 0 8px}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{border:1px solid #d7dde3;padding:6px 8px;text-align:left}
th{background:#eef2f5;cursor:pointer;user-select:none}
tr:nth-child(even) td{background:#fafbfc}
.sig-yes{color:#2e7d32;font-weight:bold}
.sig-no{color:#b23b3b}
.tabbtn{background:#eef2f5;border:1px solid #d7dde3;border-bottom:none;padding:6px 12px;cursor:pointer;margin-right:2px}
.tabbtn.active{background:#fff;font-weight:bold}
.tab{display:none}
.tab.active{display:block}
.caveat{color:#7a5c00;font-style:italic}
img{max-width:100%;height:auto;border:1px solid #d7dde3;border-radius:4px;margin:6px 0}
code{background:#eef2f5;padding:1px 4px;border-radius:3px}
.small{color:#5a6b7b;font-size:12px}
</style>
</head>
<body>
<header><h1>Consolidated report &mdash; {{ report.study.id }}
<span class="small">(field_version {{ report.field_version }})</span></h1></header>
<div class="wrap">

<div class="card validity">
<h2>Validity / scope</h2>
<p>Factors:
{%- for f, r in report.validity.factor_ranges|dictsort %}
<code>{{ f }} [{{ r.min|fmt }}, {{ r.max|fmt }}]</code>
{%- endfor %}</p>
<p>design_points = {{ report.validity.n_design_points }} |
n_reps = {{ report.validity.n_reps }} | max_arrivals = {{ report.validity.max_arrivals }}</p>
<p>seeds:
{%- for k, v in report.validity.seeds|dictsort %} {{ k }}={{ v }}{% endfor %}</p>
<ul>{% for c in report.validity.caveats %}<li class="caveat">{{ c }}</li>{% endfor %}</ul>
<p class="small">{{ report.validity.byte_determinism_note }}</p>
</div>

<div class="card summary">
<h2>Decision summary</h2>
<p><b>Primary:</b> {{ report.decision_summary.primary_response }}</p>
<p>{{ report.decision_summary.headline.text }}</p>
<p>{{ report.decision_summary.dominant_factor.text }}</p>
<p><b>Evidence:</b> {{ report.decision_summary.evidence.text }}</p>
<p class="small">{{ report.decision_summary.note }}</p>
</div>

<div class="card">
<h2>Decision index <span class="small">[{{ report.decision_index.label }}]</span></h2>
<p class="small">{{ report.decision_index.criterion }}</p>
<table><thead><tr><th>rank</th><th>AoR</th><th>effect</th><th>significant</th>
<th>cost (annotation)</th><th>note</th></tr></thead><tbody>
{% for e in report.decision_index.ranking %}
<tr><td>{{ e.rank }}</td><td>{{ e.aor }}</td><td>{{ e.effect|fmt }}</td>
<td class="{{ 'sig-yes' if e.significant else 'sig-no' }}">{{ e.significant }}</td>
<td>{% for k, v in e.cost|dictsort %}{{ k }}={{ v|fmt }} {% endfor %}</td>
<td>{{ e.note }}</td></tr>
{% endfor %}
</tbody></table>
</div>

<div class="card">
<h2>Master table</h2>
<table id="master" class="sortable"><thead><tr>
<th>response</th><th>config</th><th>aor</th><th>effect</th><th>CI</th>
<th>p</th><th>test</th><th>significant</th><th>n</th></tr></thead><tbody>
{% for r in report.master_table %}
<tr><td>{{ r.response }}</td><td>{{ r.config }}</td><td>{{ r.aor }}</td>
<td>{{ r.effect|fmt }}</td>
<td>{% if r.ci_low is none %}-{% else %}[{{ r.ci_low|fmt }}, {{ r.ci_high|fmt }}] ({{ r.ci_method }}){% endif %}</td>
<td>{{ r.p_value|fmtp }}</td><td>{{ r.test }}</td>
<td class="{{ 'sig-yes' if r.significant else 'sig-no' }}">{{ r.significant }}</td>
<td>{{ r.n_pairs }}</td></tr>
{% endfor %}
</tbody></table>
</div>

<div class="card">
<h2>Responses</h2>
<div>{% for b in report.responses %}<span class="tabbtn{{ ' active' if loop.first else '' }}"
onclick="showTab('resp{{ loop.index }}',this)">{{ b.response }}</span>{% endfor %}</div>
{% for b in report.responses %}
<div class="tab{{ ' active' if loop.first else '' }}" id="resp{{ loop.index }}">
<p>{{ b.insight.text }}</p>
<p>{{ b.dominant_factor.text }}</p>
<p class="small">outliers = {{ b.outlier_count }}
{% if b.fit %}| best fit: {{ b.fit.name }} (ks={{ b.fit.ks_statistic|fmt }}){% endif %}</p>
<table><thead><tr><th>factor</th><th>feature score</th></tr></thead><tbody>
{% for fs in b.feature_scores %}<tr><td>{{ fs.factor }}</td><td>{{ fs.score|fmt }}</td></tr>{% endfor %}
</tbody></table>
</div>
{% endfor %}
</div>

<div class="card">
<h2>Relationships (factor x gain)</h2>
<table class="sortable"><thead><tr><th>response</th><th>factor</th><th>pearson</th>
<th>spearman</th><th>kendall</th><th>slope</th><th>r2</th><th>strength</th></tr></thead><tbody>
{% for rel in report.relationships %}
<tr><td>{{ rel.response }}</td><td>{{ rel.factor }}</td><td>{{ rel.pearson|fmt }}</td>
<td>{{ rel.spearman|fmt }}</td><td>{{ rel.kendall|fmt }}</td><td>{{ rel.slope|fmt }}</td>
<td>{{ rel.r2|fmt }}</td><td>{{ rel.strength }}</td></tr>
{% endfor %}
</tbody></table>
<p class="caveat">{{ causality }}</p>
</div>

<div class="card">
<h2>ANOVA by factor levels</h2>
<table><thead><tr><th>response</th><th>factor</th><th>levels</th><th>test</th><th>p</th></tr></thead><tbody>
{% for a in report.anova_by_levels %}
<tr><td>{{ a.response }}</td><td>{{ a.factor }}</td><td>{{ a.n_levels }}</td>
<td>{{ a.test }}</td><td>{{ a.p_value|fmtp }}</td></tr>
{% endfor %}
</tbody></table>
</div>

{% if images %}
<div class="card">
<h2>Figures</h2>
{% for key, b64 in images %}
<div><b>{{ key }}</b><br><img alt="{{ key }}" src="data:image/png;base64,{{ b64 }}"></div>
{% endfor %}
</div>
{% endif %}

</div>
<script>
function showTab(id,btn){
 var card=btn.parentNode.parentNode;
 card.querySelectorAll('.tab').forEach(function(t){t.classList.remove('active');});
 card.querySelectorAll('.tabbtn').forEach(function(b){b.classList.remove('active');});
 document.getElementById(id).classList.add('active');btn.classList.add('active');
}
document.querySelectorAll('table.sortable th').forEach(function(th,idx){
 th.addEventListener('click',function(){
  var tb=th.closest('table').querySelector('tbody');
  var rows=Array.prototype.slice.call(tb.rows);
  var asc=th.dataset.asc!=='true';th.dataset.asc=asc;
  rows.sort(function(a,b){
   var x=a.cells[idx].innerText,y=b.cells[idx].innerText;
   var nx=parseFloat(x),ny=parseFloat(y);
   if(!isNaN(nx)&&!isNaN(ny)){return asc?nx-ny:ny-nx;}
   return asc?x.localeCompare(y):y.localeCompare(x);
  });
  rows.forEach(function(r){tb.appendChild(r);});
 });
});
</script>
</body>
</html>
"""


def _fmt(x: Any) -> str:
    return "-" if x is None else f"{float(x):.4f}"


def _fmtp(x: Any) -> str:
    return "-" if x is None else f"{float(x):.4g}"


def render_html(report: dict[str, Any], figures_png: dict[str, bytes]) -> str:
    """Render the consolidated ``report.json`` as a self-contained HTML string.

    Pure function of ``(report, figures_png)``: no datafarm, no
    recomputation. Figures are embedded as base64 PNG ``<img>`` (offline, no CDN).
    Deterministic: sorted image keys, fixed template, no timestamp.
    """
    try:
        from jinja2 import Environment
    except ImportError as exc:  # pragma: no cover — exercised only without the extra
        raise ImportError(_INSTALL_HINT) from exc

    env = Environment(autoescape=True)
    env.filters["fmt"] = _fmt
    env.filters["fmtp"] = _fmtp
    images = [
        (key, base64.b64encode(figures_png[key]).decode("ascii"))
        for key in sorted(figures_png)
    ]
    from cuas_sim.analysis.stats import CAUSALITY_CAVEAT

    template = env.from_string(_TEMPLATE)
    return template.render(report=report, images=images, causality=CAUSALITY_CAVEAT)


def write_html(
    report: dict[str, Any], out_dir: Path, figures_png: dict[str, bytes]
) -> None:
    """Write ``report.html`` to ``out_dir`` (thin I/O over :func:`render_html`)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.html").write_text(
        render_html(report, figures_png), encoding="utf-8"
    )
