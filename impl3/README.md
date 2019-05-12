This code requires Python 3. You will also need to have PyTorch and matplotlib installed.

    virtualenv -p python3 ~/torch
    cd ~/torch
    bin/pip install --upgrade pip
    bin/pip install https://download.pytorch.org/whl/cpu/torch-1.1.0-cp35-cp35m-linux_x86_64.whl
    bin/pip install torchvision

The CIFAR dataset will automatically be downloaded to ./data when the code is run.
