#!/usr/bin/env python3
"""Assemble the docs/ course into a single Kindle-friendly PDF with a cover image.

   .venv/bin/python tools/build_pdf.py

No LaTeX engine is available, so inline/display math ($...$, $$...$$) is converted
to a Unicode approximation that reads well on an e-reader.
"""
import os, re, html
from fpdf import FPDF
from markdown_it import MarkdownIt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = os.path.join(ROOT, "docs")
COVER = os.path.join(ROOT, "imgs", "rocket_route66.png")
ROCKET = os.path.join(ROOT, "imgs", "the_real_rocket.jpg")
OUT = os.path.join(ROOT, "F2K_CUDA_course.pdf")
FONTDIR = "/usr/share/fonts/truetype/dejavu"

ORDER = (["README.md"] + [f"{i:02d}-*.md" for i in range(29)] +
         ["A-glossary.md", "B-bug-museum.md", "C-build-reproduce.md"])

# ---------------------------------------------------------------- math → unicode
SUP = str.maketrans("0123456789+-=()n2ijk", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿ²ⁱʲᵏ")
SUB = str.maketrans("0123456789+-=()aeoxhklmnpst", "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑₒₓₕₖₗₘₙₚₛₜ")
GREEK = {r"\alpha":"α",r"\beta":"β",r"\gamma":"γ",r"\delta":"δ",r"\epsilon":"ε",
    r"\zeta":"ζ",r"\eta":"η",r"\theta":"θ",r"\kappa":"κ",r"\lambda":"λ",r"\mu":"μ",
    r"\nu":"ν",r"\xi":"ξ",r"\pi":"π",r"\rho":"ρ",r"\sigma":"σ",r"\tau":"τ",
    r"\phi":"φ",r"\psi":"ψ",r"\omega":"ω",r"\Gamma":"Γ",r"\Delta":"Δ",r"\Theta":"Θ",
    r"\Lambda":"Λ",r"\Sigma":"Σ",r"\Phi":"Φ",r"\Psi":"Ψ",r"\Omega":"Ω"}
OPS = {r"\cdot":"·",r"\cdots":"⋯",r"\ldots":"…",r"\dots":"…",r"\times":"×",
    r"\approx":"≈",r"\neq":"≠",r"\leq":"≤",r"\le":"≤",r"\geq":"≥",r"\ge":"≥",
    r"\in":"∈",r"\notin":"∉",r"\subset":"⊂",r"\sum":"Σ",r"\prod":"∏",r"\int":"∫",
    r"\infty":"∞",r"\partial":"∂",r"\nabla":"∇",r"\mapsto":"↦",r"\to":"→",
    r"\Rightarrow":"⇒",r"\rightarrow":"→",r"\odot":"⊙",r"\otimes":"⊗",r"\oplus":"⊕",
    r"\langle":"⟨",r"\rangle":"⟩",r"\|":"‖",r"\pm":"±",r"\propto":"∝",r"\sim":"~",
    r"\blacksquare":"■",r"\forall":"∀",r"\exists":"∃",r"\circ":"∘",r"\star":"⋆",
    r"\ge":"≥",r"\equiv":"≡",r"\ll":"≪",r"\gg":"≫"}
BB = {r"\mathbb{R}":"ℝ",r"\mathbb{E}":"E",r"\mathbb{N}":"ℕ",r"\mathbb{Z}":"ℤ",
    r"\mathcal{N}":"N",r"\mathcal{L}":"L",r"\mathcal{U}":"U",r"\mathbf{I}":"I",
    r"\mathbf{x}":"x"}
SERIF_FIX = {"≪":"<<","≫":">>","∎":"■","𝐈":"I","𝒩":"N","≳":"≥","ℒ":"L","𝔼":"E","𝒰":"U"}
MONO_FIX  = {"≪":"<<","≫":">>","ℓ":"l","∥":"‖","≳":"≥","𝐈":"I"}

def _supsub(s):
    s = re.sub(r"\^\{([^{}]*)\}", lambda m: m.group(1).translate(SUP)
               if all(c in "0123456789+-=()nij2k" for c in m.group(1)) else "^("+m.group(1)+")", s)
    s = re.sub(r"\^(\w)", lambda m: m.group(1).translate(SUP)
               if m.group(1) in "0123456789nij2k" else "^"+m.group(1), s)
    s = re.sub(r"_\{([^{}]*)\}", lambda m: m.group(1).translate(SUB)
               if all(c in "0123456789+-=()aeoxhklmnpst" for c in m.group(1)) else "_"+m.group(1), s)
    s = re.sub(r"_(\w)", lambda m: m.group(1).translate(SUB)
               if m.group(1) in "0123456789aeoxhklmnpst" else "_"+m.group(1), s)
    return s

ACC = {"bar":"̄","hat":"̂","tilde":"̃","vec":"⃗","dot":"̇"}

def _grp(s, i):
    """s[i]=='{'; return (inner, index_after_closing_brace) with brace matching."""
    depth=0
    for j in range(i, len(s)):
        if s[j]=="{": depth+=1
        elif s[j]=="}":
            depth-=1
            if depth==0: return s[i+1:j], j+1
    return s[i+1:], len(s)

def _arg(s, i):
    """Read one argument at s[i]: a {group} or a single token; return (text, next_i)."""
    if i<len(s) and s[i]=="{": return _grp(s,i)
    if i<len(s): return s[i], i+1
    return "", i

def _passes(s):
    # brace-commands that just unwrap or relabel their single argument
    UNWRAP = ("text","operatorname","mathrm","mathbf","mathcal","mathbb","boxed","underbrace")
    out=""; i=0
    while i < len(s):
        if s[i]=="\\":
            m=re.match(r"\\([a-zA-Z]+)", s[i:])
            name=m.group(1) if m else ""
            nxt=i+(m.end() if m else 1)
            if name in UNWRAP:
                a,k=_arg(s,nxt); out+=a; i=k; continue
            if name in ACC:
                a,k=_arg(s,nxt); out+=a+ACC[name]; i=k; continue
            if name=="sqrt":
                a,k=_arg(s,nxt); out+=("√("+a+")" if len(a)>1 else "√"+a); i=k; continue
            if name=="frac":
                n,k=_arg(s,nxt); d,k=_arg(s,k); out+="("+n+")/("+d+")"; i=k; continue
        out+=s[i]; i+=1
    return out

def latex_to_unicode(s):
    s = s.replace("\n"," ")
    s = re.sub(r"\\(left|right|big|Big|bigg|Bigg|displaystyle|,|;|:|!|quad|qquad)", " ", s)
    s = s.replace(r"\\", "  ").replace("&", " ")
    s = s.replace(r"\mid"," | ")
    for k in sorted(GREEK, key=len, reverse=True): s = s.replace(k,GREEK[k])  # greek before accents/frac
    for _ in range(6):                              # resolve nested frac/sqrt/accents/unwraps
        ns=_passes(s)
        if ns==s: break
        s=ns
    for k in sorted(OPS, key=len, reverse=True): s = s.replace(k,OPS[k])  # longest first: \infty before \in
    s = s.replace(r"\{","{").replace(r"\}","}").replace(r"\%","%").replace(r"\#","#")
    s = _supsub(s)
    s = re.sub(r"\\([a-zA-Z]+)", r"\1", s)          # strip any leftover \cmd
    s = re.sub(r"[{}]", "", s)
    return re.sub(r"  +", " ", s).strip()

SENT = "␟"   # marks a converted display-equation paragraph (for centering)

def preprocess(text):
    """Convert $$...$$ display math to a one-line sentinel paragraph BEFORE markdown,
       protecting code so '$' and line-leading +/- inside math can't confuse the parser."""
    fences=[]; inls=[]
    text=re.sub(r"```.*?```", lambda m:(fences.append(m.group(0)) or f"\x00F{len(fences)-1}\x00"),
                text, flags=re.S)
    text=re.sub(r"`[^`]*`", lambda m:(inls.append(m.group(0)) or f"\x00I{len(inls)-1}\x00"), text)
    text=re.sub(r"\$\$(.+?)\$\$",
                lambda m: f"\n\n{SENT}{latex_to_unicode(m.group(1))}\n\n", text, flags=re.S)
    # inline $...$ MUST be converted before markdown, or the underscores/asterisks
    # inside the math get parsed as emphasis and split the span across runs.
    # [^$]+? (not [^$\n]) so math that wraps across source lines still matches.
    text=re.sub(r"\$([^$]+?)\$", lambda m: latex_to_unicode(m.group(1).replace("\n"," ")), text)
    text=re.sub(r"\x00I(\d+)\x00", lambda m: inls[int(m.group(1))], text)
    text=re.sub(r"\x00F(\d+)\x00", lambda m: fences[int(m.group(1))], text)
    return text

def demath(text):
    text = re.sub(r"\$\$(.+?)\$\$", lambda m: latex_to_unicode(m.group(1)), text, flags=re.S)
    text = re.sub(r"\$([^$]+)\$", lambda m: latex_to_unicode(m.group(1)), text)
    return text

# ---------------------------------------------------------------- PDF
FRONT = 4   # cover, dedication, foreword, contents — unnumbered front matter

class Book(FPDF):
    def footer(self):
        if self.page_no() <= FRONT: return
        self.set_y(-12); self.set_font("Sans","",8); self.set_text_color(140)
        self.cell(0, 8, str(self.page_no()-FRONT), align="C")
        self.set_text_color(0)

BODY=10.5; LH=5.3
pdf = Book(format=(150,210), unit="mm")
pdf.set_margins(16,16,16); pdf.set_auto_page_break(True, 16)
for nm,st,f in [("Serif","","DejaVuSerif.ttf"),("Serif","B","DejaVuSerif-Bold.ttf"),
                ("Serif","I","DejaVuSerif-Italic.ttf"),("Serif","BI","DejaVuSerif-BoldItalic.ttf"),
                ("Sans","","DejaVuSans.ttf"),("Sans","B","DejaVuSans-Bold.ttf"),
                ("Sans","I","DejaVuSans-Oblique.ttf"),("Sans","BI","DejaVuSans-BoldOblique.ttf"),
                ("Mono","","DejaVuSansMono.ttf"),("Mono","B","DejaVuSansMono-Bold.ttf")]:
    pdf.add_font(nm, st, os.path.join(FONTDIR,f))

md = MarkdownIt("commonmark").enable("table")

def runs_from(tok):
    """Flatten an inline token's children into (style,text) runs (style: '', B, I, CODE)."""
    out=[]; style=""
    for c in (tok.children or []):
        if c.type=="text": out.append((style, c.content))
        elif c.type=="code_inline": out.append(("CODE", c.content))
        elif c.type in ("strong_open",): style="B"
        elif c.type in ("strong_close",): style=""
        elif c.type in ("em_open",): style="I"
        elif c.type in ("em_close",): style=""
        elif c.type=="softbreak": out.append((style," "))
        elif c.type=="hardbreak": out.append((style,"\n"))
        elif c.type in ("link_open","link_close","image"): pass
        else:
            if c.content: out.append((style,c.content))
    return out

def fix_glyphs(s, mono=False):
    for k,v in (MONO_FIX if mono else SERIF_FIX).items(): s = s.replace(k,v)
    return s

def write_runs(rl, size=BODY, color=(0,0,0)):
    pdf.set_text_color(*color)
    for style,text in rl:
        text = fix_glyphs(text, mono=True) if style=="CODE" else fix_glyphs(demath(text))
        if style=="CODE":
            pdf.set_font("Mono","",size-1.5)
        elif style=="B": pdf.set_font("Sans","B",size)
        elif style=="I": pdf.set_font("Sans","I",size)
        else: pdf.set_font("Sans","",size)
        for j,part in enumerate(text.split("\n")):
            if j>0: pdf.ln(LH)
            if part: pdf.write(LH, part)
    pdf.set_text_color(0)

def plain(tok):  # for tables / headings: flatten to demath'd plain text
    return fix_glyphs(demath("".join(c.content for c in (tok.children or []) if c.type in ("text","code_inline","softbreak") or c.content)).replace("\n"," "))

def code_block(text):
    body = fix_glyphs(text.rstrip("\n"), mono=True)
    lines = body.split("\n")
    avail = pdf.w - pdf.l_margin - pdf.r_margin
    size = 7.2
    pdf.set_font("Mono","",size)
    widest = max((pdf.get_string_width(ln) for ln in lines), default=0.0)
    if widest > avail*0.97:                   # shrink to fit the widest line — no wrap
        size = max(4.8, size * avail*0.97 / widest)   # (slack so the widest line can't wrap)
    pdf.ln(1.5); pdf.set_font("Mono","",size); pdf.set_fill_color(244,244,246)
    pdf.set_draw_color(225); pdf.set_text_color(20)
    pdf.multi_cell(0, 3.5*size/7.2, body, border=0, fill=True, new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0); pdf.ln(1.8)

FIRST_H1=[True]

def render(tokens):
    i=0; list_stack=[]
    while i < len(tokens):
        t=tokens[i]
        if t.type=="heading_open":
            lvl=int(t.tag[1]); inl=tokens[i+1]
            if lvl==1:
                # the first chapter reuses the fresh page left by the ToC placeholder
                if FIRST_H1[0]: FIRST_H1[0]=False
                else: pdf.add_page()
                name=plain(inl)
                if name.startswith("F2K_CUDA"): name="Introduction"
                pdf.start_section(name)            # PDF outline + contents entry
                pdf.set_font("Sans","B",17)
                pdf.set_text_color(20,20,90); pdf.multi_cell(0,8,plain(inl),new_x="LMARGIN",new_y="NEXT")
                pdf.set_text_color(0); pdf.ln(3)
            else:
                sz={2:13.5,3:11.5,4:10.5,5:10,6:10}[lvl]
                pdf.ln(2.5); pdf.set_font("Sans","B",sz); pdf.set_text_color(30,30,60)
                pdf.multi_cell(0, sz*0.42, plain(inl), new_x="LMARGIN", new_y="NEXT")
                pdf.set_text_color(0); pdf.ln(1.2)
            i+=3; continue
        if t.type=="paragraph_open":
            runs=runs_from(tokens[i+1])
            raw="".join(x for _,x in runs)
            if SENT in raw:   # display equation (already converted in preprocess) → center
                expr=fix_glyphs(raw.replace(SENT,"").strip())
                pdf.ln(1.2); pdf.set_font("Sans","I",BODY); pdf.set_text_color(15,15,15)
                pdf.multi_cell(0, LH, expr, align="C", new_x="LMARGIN", new_y="NEXT")
                pdf.set_text_color(0); pdf.ln(1.8); i+=3; continue
            write_runs(runs); pdf.ln(LH); pdf.ln(1.6); i+=3; continue
        if t.type=="fence":
            code_block(t.content); i+=1; continue
        if t.type=="hr":
            pdf.ln(1); pdf.set_draw_color(200); y=pdf.get_y()
            pdf.line(pdf.l_margin, y, pdf.w-pdf.r_margin, y); pdf.ln(3); i+=1; continue
        if t.type=="blockquote_open":
            j=i+1; depth=1
            while j<len(tokens) and depth>0:
                if tokens[j].type=="blockquote_open": depth+=1
                if tokens[j].type=="blockquote_close": depth-=1
                j+=1
            inner=tokens[i+1:j-1]
            pdf.set_left_margin(pdf.l_margin+5); pdf.set_x(pdf.l_margin)
            pdf.set_fill_color(248,248,240)
            for k in range(len(inner)):
                if inner[k].type=="paragraph_open":
                    write_runs(runs_from(inner[k+1]), size=BODY-0.5, color=(70,70,70)); pdf.ln(LH); pdf.ln(1)
            pdf.set_left_margin(pdf.l_margin-5); pdf.set_x(pdf.l_margin); pdf.ln(1); i=j; continue
        if t.type in ("bullet_list_open","ordered_list_open"):
            list_stack.append([t.type, 0]); i+=1; continue
        if t.type in ("bullet_list_close","ordered_list_close"):
            list_stack.pop(); pdf.ln(1.5); i+=1; continue
        if t.type=="list_item_open":
            list_stack[-1][1]+=1
            indent=4*len(list_stack)
            pdf.set_x(pdf.l_margin+indent)
            bullet=("• " if list_stack[-1][0]=="bullet_list_open" else f"{list_stack[-1][1]}. ")
            pdf.set_font("Sans","",BODY); pdf.write(LH, bullet)
            # render the item's paragraph inline-ish
            j=i+1
            while j<len(tokens) and tokens[j].type!="list_item_close":
                if tokens[j].type=="paragraph_open":
                    write_runs(runs_from(tokens[j+1])); j+=3
                elif tokens[j].type=="inline":
                    write_runs(runs_from(tokens[j])); j+=1
                else: j+=1
            pdf.ln(LH); i=j+1; continue
        if t.type=="table_open":
            # collect rows
            rows=[]; j=i+1; header=None
            while j<len(tokens) and tokens[j].type!="table_close":
                if tokens[j].type=="tr_open":
                    cells=[]; k=j+1
                    while tokens[k].type!="tr_close":
                        if tokens[k].type in ("th_open","td_open"):
                            cells.append(plain(tokens[k+1])); k+=3
                        else: k+=1
                    if header is None and any(tokens[m].type=="th_open" for m in range(j,k)):
                        header=cells
                    else: rows.append(cells)
                    j=k
                j+=1
            pdf.ln(1); pdf.set_font("Sans","",7.8)
            try:
                with pdf.table(first_row_as_headings=True, line_height=4.2,
                               headings_style={"bold":True,"fill_color":(225,228,240)}) as table:
                    if header:
                        r=table.row()
                        for c in header: r.cell(c)
                    for row in rows:
                        r=table.row()
                        for c in row: r.cell(c)
            except Exception:
                if header: pdf.set_font("Mono","B",7); pdf.multi_cell(0,3.6," | ".join(header),new_x="LMARGIN",new_y="NEXT")
                pdf.set_font("Mono","",7)
                for row in rows: pdf.multi_cell(0,3.6," | ".join(row),new_x="LMARGIN",new_y="NEXT")
            pdf.ln(2.5); i=j+1; continue
        i+=1

# ---------------------------------------------------------------- cover + build
pdf.add_page()
from PIL import Image
iw,ih = Image.open(COVER).size
W = pdf.w - 2*pdf.l_margin
pdf.ln(6)
pdf.set_font("Sans","B",22); pdf.set_text_color(25,25,80)
pdf.multi_cell(0,10,"F2K_CUDA", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.set_font("Sans","",12.5); pdf.set_text_color(60,60,60)
pdf.multi_cell(0,6,"From Diffusion Theory to a\nBlackwell GPU Implementation", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.ln(2.5)
pdf.set_font("Serif","I",12); pdf.set_text_color(80,80,80)
pdf.multi_cell(0,6,"by Chris Hebert and Claude", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.ln(3.5)
imgw = 96
pdf.image(COVER, x=(pdf.w-imgw)/2, y=pdf.get_y(), w=imgw, h=imgw*ih/iw)
pdf.set_y(pdf.get_y()+imgw*ih/iw+6)
pdf.set_font("Serif","I",11); pdf.set_text_color(90,90,90)
pdf.multi_cell(0,5.5,"A long-form course on how FLUX.2-klein works and how it is\n"
                     "implemented, from scratch, in C++/CUDA/cuDNN for an NVIDIA GB10.",
                     align="C", new_x="LMARGIN", new_y="NEXT")
pdf.set_text_color(0)

# ---- dedication (page 2) ----
pdf.add_page()
pdf.ln(70)
pdf.set_font("Serif","I",13.5); pdf.set_text_color(55,55,55)
pdf.multi_cell(0,8,"Dedicated to Mr Bojangles", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.ln(2)
pdf.set_font("Serif","I",11.5); pdf.set_text_color(115,115,115)
pdf.multi_cell(0,7,"a dog like no other", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.set_text_color(0)

# ---- foreword (page 3) ----
pdf.add_page()
pdf.ln(10)
pdf.set_font("Sans","B",16); pdf.set_text_color(25,25,80)
pdf.multi_cell(0,10,"Foreword", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.ln(4)
# Rocket, in person
_rw=52; _riw,_rih=Image.open(ROCKET).size; _rh=_rw*_rih/_riw
_rx=(pdf.w-_rw)/2; _ry=pdf.get_y()
pdf.image(ROCKET, x=_rx, y=_ry, w=_rw, h=_rh)
pdf.set_draw_color(170); pdf.set_line_width(0.3); pdf.rect(_rx,_ry,_rw,_rh)
pdf.set_y(_ry+_rh+3)
pdf.set_font("Serif","I",10.5); pdf.set_text_color(120,120,120)
pdf.multi_cell(0,6,"by Rocket", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.ln(22)
pdf.set_font("Serif","",14); pdf.set_text_color(40,40,40)
pdf.multi_cell(0,9,"woof woof grrrrr woof", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.set_text_color(0)

# ---- contents (page 4); rendered at output() once page numbers are known ----
def render_toc(pdf, outline):
    pdf.set_left_margin(16); pdf.set_right_margin(16)
    pdf.set_xy(16, 18)
    W = pdf.w - 32
    pdf.set_font("Sans","B",16); pdf.set_text_color(25,25,80)
    pdf.cell(W, 10, "Contents", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)
    for s in outline:
        if s.level != 0: continue
        pdf.set_x(16)
        pdf.set_font("Sans","",9); pdf.set_text_color(30,30,30)
        pdf.cell(W-12, 4.7, s.name, border=0)
        pdf.set_text_color(120,120,120)
        pdf.cell(12, 4.7, str(s.page_number - FRONT), border=0, align="R",
                 new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0)

pdf.add_page()
pdf.insert_toc_placeholder(render_toc, pages=1, reset_page_indices=False)
# insert_toc_placeholder breaks to a fresh page; the first chapter reuses it (FIRST_H1)

import glob
files=[]
for pat in ORDER:
    m=sorted(glob.glob(os.path.join(DOCS,pat)))
    files += m
seen=set(); files=[f for f in files if not (f in seen or seen.add(f))]
print(f"assembling {len(files)} documents")
for f in files:
    text=preprocess(open(f,encoding="utf-8").read())
    render(md.parse(text))

pdf.output(OUT)
print("wrote", OUT, f"({os.path.getsize(OUT)/1e6:.1f} MB, {pdf.page_no()-1} pages)")
