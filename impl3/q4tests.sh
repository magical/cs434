for lr in 0.1 0.01; do
    python -u q4.py $lr 2>&1  | tee q4-lr$lr.log
done
