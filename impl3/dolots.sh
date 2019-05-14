for dropout in 0.1 0.2 0.3 0.4; do
    for momentum in 0.1 0.2 0.3 0.4; do
        for wd in 0.01 0.001 0.0001 0.00001; do
            python -u q3.py --epochs=5 $dropout $momentum $wd 2>&1  | tee q3-d$dropout-m$momentum-wd$wd.log
        done
    done
done
