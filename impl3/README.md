This code requires Python 3. You will also need to have PyTorch and matplotlib installed.

    virtualenv -p python3 ~/torch
    cd ~/torch
    bin/pip install --upgrade pip
    bin/pip install https://download.pytorch.org/whl/cpu/torch-1.1.0-cp35-cp35m-linux_x86_64.whl
    bin/pip install torchvision

The CIFAR dataset will automatically be downloaded to ./data when the code is run.

Code for training the neural networks is in q1.py through q4.py.

    bin/python q1.py [lr]
    bin/python q2.py [lr]
    bin/python q3.py [dropout] [momentum] [weight_decay]
    bin/python q4.py [lr]

q3.py additionally supports the arguments --epoch=N to specify the number of
epochs and --lr=N to specify the learning rate.

Running any of these scripts will dump model data into the directory qN-model,
where N matches to the number in the qN.py script. The model is dumped after each epoch, and the filename contains the parameters you supplied on the command line as well as the epoch number.

They will also drop files named qN-$parameters.data into the qN-model
directories. The data files contain the average testing loss and validation set
accuracy, for turning into graphs. Run plot.py to generate a graph.

    bin/python plot.py q1-model/q1-lr0.1.data
    # generates q1-lr0.1.png


The subplots.py script can be used to generate plots with more than one run's data.
It creates one image with subplots for each data file listed on the command line.

    bin/python plot.py q3-model/q3-*.data
    # generates q3-something.png


Finally, test.py can be used to run any model against the training set.
It guesses which network to use based on the qN substring in the model path.
e.g.

    % bin/python test.py q1-model/q1-lr0.1-epoch10
    Using PyTorch version: 1.1.0  Device: cpu
    Using Q1 network
    Q1(
      (fc1): Linear(in_features=3072, out_features=100, bias=True)
      (fc3): Linear(in_features=100, out_features=10, bias=True)
    )
    Test set: Average loss: 1.4439, Accuracy: 4883/10000 (49%)


