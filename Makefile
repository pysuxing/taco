LATEX = pdflatex

FIGURES = figures/gemm.pdf

main.pdf: main.tex intro.tex background.tex scp.tex figs $(FIGURES)
	$(LATEX) $<

figs:
	make -C figures

t: main.pdf
	$(LATEX) main

clean:
	rm -f *.aux *.log *~
