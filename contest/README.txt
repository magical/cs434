This is my code for the CS 434 competition, Spring 2019.

randomforest2.py and randomforest2_weighted.py implement random forest
classification with unweighted and weighted samples, respectively.
resample.py samples data points from a text file. it is unused.

requires python3 and the sklearn library.

both scripts accept the same arguments:
the first argument is the training data set
the second argument is the testing data set

the input format is assumed to be the same as the data provided for the contest;
namely, the first column is a name, the second column is the class label, and the
remaining columns are features.

you can use a (labeled) training data set as the testing data set if you want.
if the testing data set does not have a class label, you must pass the --noclass flag.

examples:

    python3 randomforest2_weighted.py --noclass features103_train.txt features103_test.txt > features103_pred2.txt

    python3 resample.py >resampled.txt
    python3 randomforest2.py resampled.txt features103_train.txt > features103_pred1.txt


-Andrew Ekstedt
