#! /bin/sh

# Download the MNIST database from Yann LeCun's website, output expected names
wget -q -O - "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz" | gzip -d -c - > train-images &
wget -q -O - "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz" | gzip -d -c - > train-labels &
wget -q -O - "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz" | gzip -d -c - > test-images &
wget -q -O - "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz" | gzip -d -c - > test-labels &
echo "* Please wait for the background downloads to complete *"
