images: q1-errors.png q2-errors.png

q1.data:
	python3 knn.py >q1.data
q2.data:
	python3 q2.2.py >q2.data

q1-errors.png: q1.data q1-plot.py
	python2 q1-plot.py
q2-errors.png: q2.data q2.2-plot.py
	python2 q2.2-plot.py

zip:
	tar -czf code.tar.gz *.py *.data p2-data/*.csv Makefile
