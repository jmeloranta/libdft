all: libdft-manual.pdf libdft-manual.html

libgrid-manual.html:gendoc libdft-manual.tex
	latex2html -dir html/ libdft-manual.tex

libdft-manual.pdf: gendoc libdft-manual.tex
	$(shell ./gendoc ../src/classical.c > classical-functions.tex)
	$(shell ./gendoc ../src/common.c > common-functions.tex)
	pdflatex libdft-manual

gendoc: gendoc.o
gendoc.o: gendoc.c

clean:
	-rm *.aux *.log *~ *.synctex.gz *.pdf
