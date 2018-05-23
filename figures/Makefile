FIGURES = gemm.pdf

all: $(FIGURES) sec1flag sec2flag papiflag privbflag partflag workloadflag

gemm.pdf: gemm.py
	python3 $<
	pdfcrop $@ $@

sec1flag: sec1.py
	python3 $<
	pdfcrop sec1-4.pdf sec1-4.pdf
	pdfcrop sec1-8.pdf sec1-8.pdf
	pdfcrop sec1-16.pdf sec1-16.pdf
	pdfcrop sec1-32.pdf sec1-32.pdf
	touch sec1flag
sec2flag: sec2.py
	python3 $<
	pdfcrop sec2-4.pdf sec2-4.pdf
	pdfcrop sec2-8.pdf sec2-8.pdf
	pdfcrop sec2-16.pdf sec2-16.pdf
	pdfcrop sec2-32.pdf sec2-32.pdf
	pdfcrop sec2-64.pdf sec2-64.pdf
	touch sec2flag
papiflag: papi.py
	python3 $<
	pdfcrop papi-4.pdf papi-4.pdf
	pdfcrop papi-8.pdf papi-8.pdf
	pdfcrop papi-16.pdf papi-16.pdf
	pdfcrop papi-32.pdf papi-32.pdf
	pdfcrop papi-64.pdf papi-64.pdf
	touch papiflag
privbflag: privb.py
	python3 $<
	pdfcrop privb-ate.pdf privb-ate.pdf
	pdfcrop privb-papi.pdf privb-papi.pdf
	pdfcrop privb-breakdown.pdf privb-breakdown.pdf
	pdfcrop strategy-conventional.pdf strategy-conventional.pdf
	pdfcrop strategy-full.pdf strategy-full.pdf
	pdfcrop strategy-partial.pdf strategy-partial.pdf
	touch privbflag

partflag: partitions.py
	python3 $<
	pdfcrop wpart.pdf wpart.pdf
	pdfcrop spart.pdf spart.pdf
	touch partflag

workloadflag: workload.py
	python3 $<
	pdfcrop workload.pdf workload.pdf
	touch workloadflag
