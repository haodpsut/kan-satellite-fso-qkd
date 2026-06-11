MAIN = main
LATEX = pdflatex -interaction=nonstopmode -halt-on-error
BIB = bibtex

.PHONY: all clean figures

all: $(MAIN).pdf

figures:
	python figures/make_figs.py

$(MAIN).pdf: $(MAIN).tex sections/*.tex refs.bib figures/*.tex figures/*.pdf
	$(LATEX) $(MAIN)
	$(BIB) $(MAIN)
	$(LATEX) $(MAIN)
	$(LATEX) $(MAIN)

clean:
	rm -f $(MAIN).aux $(MAIN).log $(MAIN).bbl $(MAIN).blg \
	      $(MAIN).out $(MAIN).toc $(MAIN).lof $(MAIN).lot

distclean: clean
	rm -f $(MAIN).pdf
