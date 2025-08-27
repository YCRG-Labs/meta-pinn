# Enhanced Makefile for ML Research Pipeline Documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD  ?= sphinx-build
SOURCEDIR    = .
BUILDDIR     = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile clean-all doctest linkcheck coverage api-docs

# Enhanced clean target
clean-all:
	@echo "Removing everything under '$(BUILDDIR)'..."
	@rm -rf $(BUILDDIR)/*
	@echo "Removing auto-generated API documentation..."
	@find . -name "*rst" -path "./api/*" -delete 2>/dev/null || true

# Run doctest to verify code examples
doctest:
	@echo "Running doctest to verify code examples..."
	@$(SPHINXBUILD) -b doctest "$(SOURCEDIR)" "$(BUILDDIR)/doctest" $(SPHINXOPTS) $(O)
	@echo "Running custom docstring tests..."
	@cd .. && python docs/doctest_examples.py

# Check for broken links
linkcheck:
	@echo "Checking for broken links..."
	@$(SPHINXBUILD) -b linkcheck "$(SOURCEDIR)" "$(BUILDDIR)/linkcheck" $(SPHINXOPTS) $(O)

# Generate coverage report for documentation
coverage:
	@echo "Generating documentation coverage report..."
	@$(SPHINXBUILD) -b coverage "$(SOURCEDIR)" "$(BUILDDIR)/coverage" $(SPHINXOPTS) $(O)
	@echo "Coverage report available at $(BUILDDIR)/coverage/python.txt"

# Auto-generate API documentation
api-docs:
	@echo "Auto-generating API documentation..."
	@sphinx-apidoc -f -o api ../ml_research_pipeline
	@echo "API documentation generated in api/ directory"

# Build HTML with all checks
html-full: clean-all api-docs
	@echo "Building full HTML documentation with all checks..."
	@$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) $(O) -W --keep-going
	@echo "Running doctest..."
	@$(SPHINXBUILD) -b doctest "$(SOURCEDIR)" "$(BUILDDIR)/doctest" $(SPHINXOPTS) $(O)
	@echo "Full documentation build completed!"

# Build PDF documentation
pdf: latex
	@echo "Building PDF documentation..."
	@cd $(BUILDDIR)/latex && make all-pdf
	@echo "PDF documentation available at $(BUILDDIR)/latex/mlresearchpipeline.pdf"

# Development build (faster, no warnings as errors)
dev:
	@echo "Building documentation for development..."
	@$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) $(O)

# Live reload for development
live:
	@echo "Starting live reload server..."
	@sphinx-autobuild "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) $(O)

# Catch-all target: route all unknown targets to Sphinx using the new
# "make" command.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)