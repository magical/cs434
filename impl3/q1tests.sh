for lr in 0.1 0.001 0.0001; do
    python -u q1.py $lr 2>&1  | tee q1-lr$lr.log
done
