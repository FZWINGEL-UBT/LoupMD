SOURCES := main.tex methods.tex related-work.tex references.bib
LATEX := pdflatex -no-shell-escape -halt-on-error -file-line-error -interaction=nonstopmode

.PHONY: pdf
pdf: build/main.pdf

build/main.pdf: $(SOURCES) Makefile
	mkdir -p build
	cp $(SOURCES) build/
	cd build && $(LATEX) main.tex && bibtex main && $(LATEX) main.tex && $(LATEX) main.tex
