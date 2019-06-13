python3 randomforest2.py --noclass features103_train.txt features103_test.txt >features103_pred1.txt
python3 randomforest2.py --noclass featuresall_train.txt featuresall_test.txt >featuresall_pred1.txt

python3 randomforest2_weighted.py --noclass features103_train.txt features103_test.txt >features103_pred2.txt
python3 randomforest2_weighted.py --noclass featuresall_train.txt featuresall_test.txt >featuresall_pred2.txt

python3 randomforest2.py --noclass features103_train.txt features103_test.txt >features103_pred3.txt
python3 randomforest2.py --noclass featuresall_train.txt featuresall_test.txt >featuresall_pred3.txt
