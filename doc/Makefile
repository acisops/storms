# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Storm memos with their own standalone PDF (see conf.py's latex_documents),
# and the source-tree path each one's compiled PDF gets copied back to so
# the html build's "Download PDF version" link (html_extra_path in
# conf.py) has something current to serve. Add a line here for each new
# memo added to latex_documents.
MEMO_PDFS = JAN1926_memo.pdf:storm_memos/JAN1926/JAN1926_memo.pdf

# Generates each storm memo's plot fragments/images (ace_p3.html,
# ace_p3.png, etc.) from its .yml parameter file, via storms'
# make_storm_plots (see /doc/source/command_line.rst). Discovered by
# globbing storm_memos/*/*.yml, so adding a new memo just means dropping
# its <name>.yml and <NAME>_memo.md into a new storm_memos/<NAME>/
# directory - no Makefile changes needed here. make_storm_plots writes
# its output files into the current directory, hence the per-memo `cd`,
# and always writes its own plotly.min.js there too, which is removed
# afterward since every memo shares the one copy in source/_static
# instead (see conf.py's html_extra_path).
#
# Skips a memo directory whose figures were already generated (checking
# txings.png, the last file make_storm_plots writes, as a stand-in for
# "this memo's figures are all here") rather than regenerating every time,
# since that means re-fetching ACE/GOES/HRC data over the network. Delete
# that memo's *.png/*.html (or the whole directory's worth of them) to
# force a re-run.
figures:
	@for yml in $(SOURCEDIR)/storm_memos/*/*.yml; do \
		dir=$$(dirname "$$yml"); \
		if [ -f "$$dir/txings.png" ]; then \
			echo "Figures already exist in $$dir, skipping."; \
			continue; \
		fi; \
		echo "Generating figures for $$dir from $$(basename $$yml)..."; \
		(cd "$$dir" && make_storm_plots "$$(basename $$yml)") || exit 1; \
		rm -f "$$dir/plotly.min.js"; \
	done

# Builds every memo's PDF via Sphinx's own latexpdf target, then copies
# each one back into the source tree (see MEMO_PDFS above) so a later
# `make html` picks up the current version. Explicit target: takes
# precedence over the catch-all pattern rule below for this one name.
latexpdf: Makefile
	@$(SPHINXBUILD) -M latexpdf "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@for pair in $(MEMO_PDFS); do \
		built=$${pair%%:*}; dest=$${pair#*:}; \
		echo "Copying $(BUILDDIR)/latex/$$built -> $(SOURCEDIR)/$$dest"; \
		cp "$(BUILDDIR)/latex/$$built" "$(SOURCEDIR)/$$dest"; \
	done

.PHONY: help figures latexpdf Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
