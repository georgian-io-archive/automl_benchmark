#!/usr/bin/env bash

#Config matplotlib backend via matplotlibrc
export MATPLOTLIBRC=/root/automl_benchmark/benchmark/compute/batch/scripts

#Config Java environmental variables
export _JAVA_OPTIONS='-XX:+UnlockExperimentalVMOptions -XX:+UseCGroupMemoryLimitForHeap'

#Setup RSA
aws s3 cp --recursive s3://${S3_BUCKET}/${S3_FOLDER}ssh ~/.ssh
chmod -R 400 ~/.ssh
chmod 755 ~/.ssh/batch

#Install swig
ln -s /usr/libexec/gcc/x86_64-amazon-linux/4.8.5/cc1plus /usr/local/bin/cc1plus
aws s3 cp s3://${S3_BUCKET}/${S3_FOLDER}swig.tar.gz ~/swig-3.0.12.tar.gz
tar xf ~/swig-3.0.12.tar.gz -C ~/
cd ~/swig-3.0.12
./configure --prefix=/usr --without-clisp --without-maximum-compile-warnings
make
make install
cd ~

#Clone down repository
export GIT_SSH_COMMAND="ssh -F /root/.ssh/git -o StrictHostKeyChecking=no"
git clone https://github.com/georgianpartners/automl_benchmark.git
cd  automl_benchmark

#Setup virtual environment
python3 -m venv automl_benchmark
source automl_benchmark/bin/activate


#Install requirements
pip install -r batch-requirements.txt
pip install -r pre-requirements.txt
pip install -r requirements.txt

#Download data
aws s3 cp s3://${S3_BUCKET}/${S3_FOLDER}tests.dat ./

#Execute benchmark
timeout ${TIME} python -m benchmark.compute.batch.batch_wrapper > logs.out 2>&1

aws s3 cp logs.out s3://${S3_BUCKET}/$(cat status)/$(date +%Y%m%d%H%M%S).log

exit 0
