PAPER=IMC

all: paper

paper:
	pdflatex -shell-escape $(PAPER)
	bibtex $(PAPER)
	pdflatex  -shell-escape $(PAPER)
	pdflatex  -shell-escape $(PAPER)

fast: 
	pdflatex -shell-escape $(PAPER)

clean: 
	rm -rf *~ *.log *.aux *.bbl *.blg $(PAPER).ps *.dvi *.bcf *.run.xml


