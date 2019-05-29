set -eu
for k in {2..10}; do
    #for seed in {1..10}; do
    #    python kmeans.py --seed=$seed $k
    #done
    seq 1 10 | parallel -j 5 python kmeans.py --seed='{}' "$k"
done
