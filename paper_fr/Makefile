LATEX = pdflatex
BIBTEX = bibtex
PROJECT = kmamba

all: $(PROJECT).pdf

$(PROJECT).pdf: $(PROJECT).tex $(PROJECT).bib
	$(LATEX) $(PROJECT)
	$(BIBTEX) $(PROJECT)
	$(LATEX) $(PROJECT)
	$(LATEX) $(PROJECT)

clean:
	rm -f *.aux *.log *.out *.bbl *.blg *.pdf *.toc *.lof *.lot

.PHONY: all clean