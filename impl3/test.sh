set -eu
python test.py q1-model/q1-lr0.100000-epoch20
python test.py q2-model/q2-lr0.010000-epoch15
python test.py q3-model/q3-d0.1-m0.4-wd0.001-epoch5 --dropout=0.1
python test.py q4-model/q4-lr0.100000-epoch18
