"""Self-contained HTML report. No external assets, inline SVG + CSS.

The report is part of the deliverable, not decoration: the challenge grades
'explainable risk assessment -- indicate which layers, weights or characteristics
appear suspicious'. So every finding is shown with its location and the raw
features that produced it, and recovered payloads are shown as a hexdump.
"""
from __future__ import annotations
import html, os

BAND_HEX = {"clean": "#1a9850", "review": "#d9a400",
            "suspicious": "#d1590f", "likely-compromised": "#c1121f"}


def _gauge(risk, band):
    color = BAND_HEX.get(band, "#888")
    angle = 180 * risk / 100.0
    import math
    x = 100 + 90 * math.cos(math.radians(180 - angle))
    y = 100 - 90 * math.sin(math.radians(180 - angle))
    return f'''<svg width="220" height="130" viewBox="0 0 200 120">
      <path d="M10,100 A90,90 0 0,1 190,100" fill="none" stroke="#eee" stroke-width="16"/>
      <path d="M10,100 A90,90 0 0,1 {x:.1f},{y:.1f}" fill="none" stroke="{color}" stroke-width="16" stroke-linecap="round"/>
      <text x="100" y="92" text-anchor="middle" font-size="34" font-weight="700" fill="{color}">{risk}</text>
      <text x="100" y="112" text-anchor="middle" font-size="12" fill="#666">/ 100</text>
    </svg>'''


def write_report(result, out_path: str):
    band = result.band
    color = BAND_HEX.get(band, "#888")
    findings = []
    for e in result.top:
        if e.score <= 0:
            continue
        loc = html.escape(e.location.describe()) if e.location else ""
        feats = "".join(
            f"<tr><td>{html.escape(str(k))}</td><td>{html.escape(str(v))}</td></tr>"
            for k, v in e.features.items() if k != "hexdump")
        hexdump = e.features.get("hexdump", "")
        hexblock = (f'<pre class="hex">{html.escape(hexdump)}</pre>'
                    if hexdump else "")
        findings.append(f'''
        <div class="finding tier-{e.tier_hint}">
          <div class="fhead"><span class="badge">{e.detector}</span>
            <span class="tier">tier {e.tier_hint}</span>
            <span class="floc">{loc}</span></div>
          <p>{html.escape(e.explanation)}</p>
          {hexblock}
          <details><summary>raw features</summary>
            <table class="feat">{feats}</table></details>
        </div>''')
    findings_html = "\n".join(findings) or "<p>No positive findings.</p>"

    doc = f'''<!doctype html><html><head><meta charset="utf-8">
<title>ModelSentry report: {html.escape(os.path.basename(result.path))}</title>
<style>
 body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:860px;
   margin:24px auto;color:#1a1a1a;padding:0 16px}}
 header{{display:flex;align-items:center;gap:24px;border-bottom:3px solid {color};
   padding-bottom:12px}}
 h1{{font-size:20px;margin:0}} .sub{{color:#666;font-size:13px}}
 .band{{display:inline-block;padding:3px 10px;border-radius:12px;color:#fff;
   background:{color};font-weight:600;font-size:13px}}
 .summary{{background:#f7f7f9;border-left:4px solid {color};padding:12px 16px;
   margin:18px 0;border-radius:4px}}
 .finding{{border:1px solid #e2e2e6;border-radius:8px;padding:14px 16px;margin:12px 0}}
 .finding.tier-E4{{border-left:5px solid #c1121f}}
 .finding.tier-E3{{border-left:5px solid #d1590f}}
 .finding.tier-E2{{border-left:5px solid #d9a400}}
 .finding.tier-E1{{border-left:5px solid #999}}
 .fhead{{display:flex;gap:10px;align-items:center;font-size:13px;margin-bottom:6px}}
 .badge{{background:#222;color:#fff;padding:2px 8px;border-radius:4px;font-weight:600}}
 .tier{{color:{color};font-weight:700}} .floc{{color:#666;font-family:monospace}}
 pre.hex{{background:#0d1117;color:#c9d1d9;padding:12px;border-radius:6px;
   overflow-x:auto;font-size:12px;line-height:1.45}}
 table.feat{{font-size:12px;border-collapse:collapse;margin-top:6px}}
 table.feat td{{border:1px solid #eee;padding:3px 8px}}
 details summary{{cursor:pointer;color:#555;font-size:12px}}
 footer{{color:#999;font-size:11px;margin-top:30px;border-top:1px solid #eee;padding-top:10px}}
</style></head><body>
<header>{_gauge(result.risk, band)}
 <div><h1>ModelSentry pre-deployment scan</h1>
  <div class="sub">{html.escape(result.path)}</div>
  <div style="margin-top:8px"><span class="band">{band}</span>
   &nbsp; evidence tier <b>{result.tier}</b></div></div>
</header>
<div class="summary">{html.escape(result.summary)}</div>
<h2 style="font-size:16px">Findings</h2>
{findings_html}
<footer>ModelSentry &middot; static pre-deployment analysis. A high score indicates
statistically anomalous encoded data consistent with a hidden payload and elevated
supply-chain risk; it is not a claim that the model will execute malware unless an
extraction/execution vector (tier E4) is also reported.</footer>
</body></html>'''
    with open(out_path, "w") as f:
        f.write(doc)
    return out_path
