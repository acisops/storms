import re
import shutil
from datetime import date
from pathlib import Path

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "ACIS Storms"
copyright = "2026, CXC ACIS Operations"
author = "CXC ACIS Operations"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["myst_parser", "sphinx.ext.mathjax"]
myst_enable_extensions = ["dollarmath", "linkify"]

templates_path = ["_templates"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# Storm memos are formatted like standalone print memos (their own
# letterhead, own figure numbering) rather than reference-doc pages, so the
# primary (left-hand) site-navigation sidebar is turned off for them via
# html_sidebars below. Pages that match nothing here just keep the theme's
# normal sidebar, so no "**" default entry is needed.
html_theme_options = {}

html_sidebars = {
    "storm_memos/index": [],
    "storm_memos/JAN1926/JAN1926_memo": [],
}

# {raw} html blocks are opaque to myst-parser's normal substitution pass
# (their content bypasses markdown parsing entirely), so the {{ today }}
# MyST substitution syntax never reaches it. Do a plain text substitution
# on the raw source instead, before Sphinx parses it at all.
BUILD_DATE = date.today().strftime("%B %d, %Y")

# Each figure's <figcaption> (in its HTML raw block) is the one spot in the
# doc where its caption text is actually written out; everywhere else that
# needs it (its {only} latex {figure} block) just says {{CAPTION:xxx}} and
# this pass fills in whatever was captured from the <figcaption>. A caption
# may itself reference another figure via {{FIGREF:xxx}}, which gets
# expanded (after the {{CAPTION:xxx}} copies are filled in, so copies pick
# it up too) to a plain "Figure N" for LaTeX, or an in-page link for HTML.
CAPTION_DEF_RE = re.compile(
    r"<figcaption[^>]*>Figure \{\{FIGNUM:([a-z0-9_]+)\}\}: (.*?)</figcaption>"
)

# Figure order is read off the (figure_xxx)= anchors, and every
# {{FIGNUM:xxx}} token is replaced with that figure's 1-based position.
# Reordering, adding, or removing a (figure_xxx)= block renumbers
# everything on the next build.
FIGURE_ANCHOR_RE = re.compile(r"^\(figure_([a-z0-9_]+)\)=$", re.MULTILINE)
FIGNUM_RE = re.compile(r"\{\{FIGNUM:([a-z0-9_]+)\}\}")
CAPTION_RE = re.compile(r"\{\{CAPTION:([a-z0-9_]+)\}\}")
FIGREF_RE = re.compile(r"\{\{FIGREF:([a-z0-9_]+)\}\}")


def _substitute_tokens(app, _docname, source):
    text = source[0]
    text = text.replace("{{BUILD_DATE}}", BUILD_DATE)

    captions = dict(CAPTION_DEF_RE.findall(text))
    text = CAPTION_RE.sub(
        lambda m: "Figure {{FIGNUM:" + m.group(1) + "}}: " + captions[m.group(1)],
        text,
    )

    is_html = app.builder.name == "html"

    def _figref(m):
        name = m.group(1)
        if is_html:
            anchor_id = "figure-" + name.replace("_", "-")
            return '<a href="#' + anchor_id + '">{{FIGNUM:' + name + "}}</a>"
        return "{{FIGNUM:" + name + "}}"

    text = FIGREF_RE.sub(_figref, text)

    numbers = {name: i + 1 for i, name in enumerate(FIGURE_ANCHOR_RE.findall(text))}
    text = FIGNUM_RE.sub(lambda m: str(numbers.get(m.group(1), "?")), text)
    source[0] = text


# html_extra_path (below) always copies an entry's *contents* into the
# build root, dropping the entry's own leading path component - it has no
# way to preserve a nested relative path like "storm_memos/older_memos/"
# for a directory that lives right alongside other real doc sources
# (storm_memos/index.rst, storm_memos/JAN1926/). So instead of
# html_extra_path, older_memos/*.pdf is copied here, once the rest of the
# HTML build is done, straight from source to the matching path under the
# output directory.
def _copy_older_memos(app, exc):
    if exc is not None or app.builder.name != "html":
        return
    src_dir = Path(app.srcdir) / "storm_memos" / "older_memos"
    if not src_dir.is_dir():
        return
    out_dir = Path(app.outdir) / "storm_memos" / "older_memos"
    out_dir.mkdir(parents=True, exist_ok=True)
    for pdf in src_dir.glob("*.pdf"):
        shutil.copy2(pdf, out_dir / pdf.name)


def setup(app):
    app.connect("source-read", _substitute_tokens)
    app.connect("build-finished", _copy_older_memos)


# NOTE: root_doc is left at its default ("index") here - this conf.py now
# covers the whole "ACIS Storms" doc site (index.rst, command_line.rst,
# storm_memos/**), not just a single storm memo.
#
# Both ".rst" and ".md" need to be listed explicitly: setting this to a
# dict replaces Sphinx's default source_suffix mapping rather than adding
# to it, so leaving ".rst" out would stop index.rst/command_line.rst from
# being recognized as documents at all.
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}

exclude_patterns = ["_build", "_sphinx_build", "old", "**/_build"]

# _static/plotly.min.js and _images/cxc_logo.png are shared by every storm
# memo (each memo's <script src="../../plotly.min.js"> / <img
# src="../../cxc_logo.png"> reaches up to this directory), rather than each
# memo under storm_memos/ carrying its own multi-MB copy. html_extra_path
# copies them flat into the build root (alongside index.html), regardless
# of their _static/_images source subdirectory, since they're loaded via
# plain <script>/<img> src rather than a proper MyST image/raw-file
# reference that Sphinx's asset collector would otherwise pick up on its
# own - hence the memo's links being "../../plotly.min.js" and
# "../../cxc_logo.png", not "../../_static/..." or "../../_images/...".
#
# storm_memos/JAN1926/JAN1926_memo.pdf is the "download PDF" version linked
# from that memo's html page (also via a plain <a href>, so it needs the
# same explicit treatment, and also lands flat at the build root). It's a
# build artifact of `make latexpdf`, kept checked into source and copied
# back into place by that Makefile target rather than generated fresh on
# every html build, so `make html` alone doesn't require a LaTeX
# toolchain. Run `make latexpdf` first (or after editing the memo/plots)
# to refresh it before `make html`.
# storm_memos/older_memos/*.pdf is handled separately, by the
# build-finished hook (_copy_older_memos, above) rather than listed here -
# html_extra_path can't preserve that nested relative path (see the hook's
# comment for why).
html_extra_path = [
    "_static/plotly.min.js",
    "_images/cxc_logo.png",
    "storm_memos/JAN1926/JAN1926_memo.pdf",
]

html_theme = "pydata_sphinx_theme"

# PDF output: raw HTML blocks (the interactive plotly embeds and the
# letterhead) are automatically dropped by the LaTeX writer, so each memo
# source also has an {only} latex block that builds the equivalent
# letterhead/header using the CXC memox/cxc_letterhead LaTeX packages
# (memox.sty, cxc_letterhead.sty) instead of hand-rolled LaTeX.
#
# cxc_letterhead.sty hardcodes a `cxc_logo.eps` graphic, which requires
# pdflatex (xelatex can't include raw EPS). cxc_logo.eps is a one-time
# ImageMagick conversion of cxc_logo.png; cxc_logo-eps-converted-to.pdf is
# a pre-run of the `epstopdf` package's conversion so pdflatex can use it
# without needing -shell-escape at build time.
#
# Only the memo(s) below get their own separate PDF - index.rst/
# command_line.rst are HTML-only. latex_elements (below) applies to every
# entry in latex_documents, so if the main package docs ever get a PDF
# too, the letterhead preamble/maketitle/tableofcontents suppression here
# would need to move to a per-document theme override instead of staying
# global.
#
# To add another memo's PDF, add another
# (docname, "<name>.tex", "<Title>", "<Author>", "howto") tuple below.
latex_engine = "pdflatex"
latex_documents = [
    (
        "storm_memos/JAN1926/JAN1926_memo",
        "JAN1926_memo.tex",
        "Chandra Storm Memo — January 2026",
        "John ZuHone",
        "howto",
    ),
]
latex_elements = {
    "papersize": "letterpaper",
    "pointsize": "11pt",
    # Suppress the auto-generated "Contents" page; the HTML build still
    # gets its page TOC from the pydata theme's sidebar, independently
    # of this template hook.
    "tableofcontents": "",
    # Suppress the auto-generated title/author/date block above our own
    # letterhead (built via \CXCletterhead/\memond in the {only} latex
    # block below).
    "maketitle": "",
    "preamble": r"""
\usepackage[addr=cfa]{cxc_letterhead}
\usepackage{memox}
\usepackage{epstopdf}
""",
}

# Not part of the doctree (referenced only from the {raw} latex letterhead
# block and the packages' own internals), so they need to be listed
# explicitly to get copied into the LaTeX build directory. Like
# html_extra_path, this flattens each file into the build root regardless
# of its _static/_images source subdirectory - cxc_letterhead.sty's own
# \includegraphics{cxc_logo.eps} (a bare filename, no path) relies on that.
latex_additional_files = [
    "_static/memox.sty",
    "_static/cxc_letterhead.sty",
    "_images/cxc_logo.eps",
    "_images/cxc_logo-eps-converted-to.pdf",
]
