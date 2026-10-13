"""Render RESULTS.md as a standalone HTML page (results.html) for sharing.

Handles exactly the markdown RESULTS.md uses: #/##/### headings, paragraphs, "- " bullets,
pipe tables, `code`, **bold**, *italic*. Rebuild after make_results_md.py:
  python make_results_html.py
"""
import html
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
SRC, OUT = os.path.join(HERE, "RESULTS.md"), os.path.join(HERE, "results.html")


def inline(t):
    t = html.escape(t, quote=False)
    t = re.sub(r"`([^`]+)`", r"<code>\1</code>", t)
    t = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", t)
    t = re.sub(r"(?<![\w*])\*([^*\s][^*]*)\*(?![\w*])", r"<em>\1</em>", t)
    return t


def family(key):
    k = key.strip()
    if k.startswith("GRPO"):
        return "grpo"
    if k.startswith("PPO"):
        return "ppo"
    if k.startswith("DS_") or k.startswith("noDS_"):
        return "dapo"
    if k == "base":
        return "base"
    return ""


NUMERIC = re.compile(r"^[+-]?\d+(\.\d+)?$")


def table(rows):
    head = [c.strip() for c in rows[0].strip().strip("|").split("|")]
    body = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows[2:]]
    key_col = next((i for i, h in enumerate(head) if h in ("key", "model")), None)
    out = ['<div class="tbl"><table><thead><tr>']
    out += [f"<th>{inline(h)}</th>" for h in head]
    out.append("</tr></thead><tbody>")
    for r in body:
        fam = family(r[key_col]) if key_col is not None and key_col < len(r) else ""
        out.append(f'<tr class="{fam}">' if fam else "<tr>")
        for i, c in enumerate(r):
            cls = ' class="num"' if NUMERIC.match(c) else ""
            if i == key_col and fam:
                c_html = f'<span class="chip {fam}">{fam.upper() if fam != "base" else "BASE"}</span>{inline(c)}'
            else:
                c_html = inline(c)
            out.append(f"<td{cls}>{c_html}</td>")
        out.append("</tr>")
    out.append("</tbody></table></div>")
    return "".join(out)


def render(md):
    lines = md.split("\n")
    out, toc, i, para, items = [], [], 0, [], []

    def flush():
        nonlocal para, items
        if para:
            out.append(f"<p>{inline(' '.join(para))}</p>")
            para = []
        if items:
            out.append("<ul>" + "".join(f"<li>{inline(x)}</li>" for x in items) + "</ul>")
            items = []

    title = "Results"
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("|"):
            flush()
            blk = []
            while i < len(lines) and lines[i].startswith("|"):
                blk.append(lines[i]); i += 1
            out.append(table(blk))
            continue
        m = re.match(r"^(#{1,3}) (.*)$", ln)
        if m:
            flush()
            lvl, txt = len(m.group(1)), m.group(2).strip()
            if lvl == 1:
                title = txt
            else:
                sid = re.sub(r"[^a-z0-9]+", "-", txt.lower()).strip("-")
                if lvl == 2:
                    toc.append((sid, txt))
                out.append(f'<h{lvl} id="{sid}">{inline(txt)}</h{lvl}>')
        elif ln.startswith("- "):
            if para:
                flush()
            items.append(ln[2:])
        elif not ln.strip():
            flush()
        else:
            if items:
                flush()
            para.append(ln.strip())
        i += 1
    flush()
    return title, toc, "\n".join(out)


CSS = """
:root{--bg:#f6f7f8;--surface:#ffffff;--ink:#1c2127;--muted:#5a6470;--rule:#d9dee4;--head:#eef1f4;
--accent:#1d6a86;--dapo:#1d6a86;--grpo:#2c7a4b;--ppo:#9a5a12;--base:#6b7480;
--dapo-bg:#e3f0f5;--grpo-bg:#e4f2e9;--ppo-bg:#f6ecdf;--base-bg:#eceef1}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--bg:#111417;--surface:#181c20;
--ink:#e4e8ec;--muted:#9aa4ae;--rule:#2b3137;--head:#20252a;--accent:#63b4d2;--dapo:#63b4d2;--grpo:#6cc493;
--ppo:#e0a05a;--base:#a3acb6;--dapo-bg:#16303a;--grpo-bg:#18311f;--ppo-bg:#3a2a17;--base-bg:#262b30}}
:root[data-theme="dark"]{color-scheme:dark;--bg:#111417;--surface:#181c20;--ink:#e4e8ec;--muted:#9aa4ae;
--rule:#2b3137;--head:#20252a;--accent:#63b4d2;--dapo:#63b4d2;--grpo:#6cc493;--ppo:#e0a05a;--base:#a3acb6;
--dapo-bg:#16303a;--grpo-bg:#18311f;--ppo-bg:#3a2a17;--base-bg:#262b30}
body{background:var(--bg);color:var(--ink);font:15px/1.6 "IBM Plex Sans",system-ui,-apple-system,"Segoe UI",sans-serif;
padding-inline:20px;padding-block:32px 64px}
.wrap{max-width:1240px;margin:0 auto}
header{border-bottom:2px solid var(--ink);padding-bottom:18px;margin-bottom:8px}
.eyebrow{font:600 12px/1 "IBM Plex Mono",ui-monospace,monospace;letter-spacing:.08em;text-transform:uppercase;color:var(--accent)}
h1{font:600 clamp(26px,3.4vw,36px)/1.15 "IBM Plex Serif",Georgia,serif;margin:10px 0 8px;text-wrap:balance}
h2{font:600 22px/1.25 "IBM Plex Serif",Georgia,serif;margin:44px 0 10px;padding-top:12px;border-top:1px solid var(--rule);text-wrap:balance}
h3{font:600 16px/1.3 "IBM Plex Sans",sans-serif;margin:28px 0 8px}
p,li{max-width:78ch}
p{margin:8px 0}
ul{padding-left:20px;margin:8px 0}
li{margin:5px 0}
code{font:13px/1.4 "IBM Plex Mono",ui-monospace,monospace;background:var(--head);padding:1px 5px;border-radius:3px;overflow-wrap:anywhere}
strong{font-weight:600}
.lede{color:var(--muted);max-width:78ch}
nav{display:flex;flex-wrap:wrap;gap:6px 18px;margin:14px 0 4px;font-size:13.5px}
nav a{color:var(--accent);text-decoration:none;border-bottom:1px solid transparent}
nav a:hover,nav a:focus-visible{border-bottom-color:var(--accent);outline:none}
.legend{display:flex;flex-wrap:wrap;gap:8px 14px;align-items:center;font-size:13px;color:var(--muted);margin:10px 0}
.tbl{overflow-x:auto;margin:10px 0 18px;border:1px solid var(--rule);border-radius:4px;background:var(--surface)}
table{border-collapse:collapse;font-size:13px;width:100%}
th,td{padding:6px 10px;border-bottom:1px solid var(--rule);text-align:left;vertical-align:top}
th{background:var(--head);font-weight:600;white-space:nowrap;position:sticky;top:0}
td.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
tbody tr:hover td{background:var(--head)}
tr.dapo td:first-child{box-shadow:inset 3px 0 0 var(--dapo)}
tr.grpo td:first-child{box-shadow:inset 3px 0 0 var(--grpo)}
tr.ppo td:first-child{box-shadow:inset 3px 0 0 var(--ppo)}
tr.base td:first-child{box-shadow:inset 3px 0 0 var(--base)}
.chip{display:inline-block;font:600 10px/1 "IBM Plex Mono",monospace;letter-spacing:.06em;padding:3px 5px;border-radius:3px;margin-right:7px;vertical-align:1px}
.chip.dapo{color:var(--dapo);background:var(--dapo-bg)}.chip.grpo{color:var(--grpo);background:var(--grpo-bg)}
.chip.ppo{color:var(--ppo);background:var(--ppo-bg)}.chip.base{color:var(--base);background:var(--base-bg)}
@media (max-width:600px){body{font-size:14.5px}table{font-size:12px}th,td{padding:5px 7px}}
"""


def main():
    title, toc, body = render(open(SRC).read())
    nav = "".join(f'<a href="#{sid}">{inline(t)}</a>' for sid, t in toc)
    page = f"""<title>Countdown RL Transfer Results</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:ital,wght@0,400;0,600;1,400&family=IBM+Plex+Serif:wght@600&display=swap">
<style>{CSS}</style>
<div class="wrap">
<header>
<div class="eyebrow">Qwen2.5-1.5B-Instruct &middot; trained on Countdown &middot; GSM8K / MATH / BBH / MMLU-Pro</div>
<h1>{inline(title)}</h1>
<p class="lede">Every evaluated checkpoint with its training parameters (read from wandb configs and training logs, not run names) and every downstream result. Accuracies are in %, greedy decoding, 8192 output tokens unless marked otherwise.</p>
<nav aria-label="Sections">{nav}</nav>
<div class="legend"><span class="chip dapo">DAPO</span><span class="chip grpo">GRPO</span><span class="chip ppo">PPO</span><span class="chip base">BASE</span> row markers show the training family</div>
</header>
{body}
</div>
"""
    open(OUT, "w").write(page)
    print("wrote", OUT, f"({len(page) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
