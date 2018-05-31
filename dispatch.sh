#!/usr/bin/env bash

#Setup RSA
aws s3 cp --recursive s3://georgianpartners-auto-ml-data/ssh ~/.ssh
chmod -R 400 ~/.ssh
chmod 755 ~/.ssh/config

#Install swig
ln -s /usr/libexec/gcc/x86_64-amazon-linux/4.8.5/cc1plus /usr/local/bin/cc1plus
curl -o /tmp/swig-3.0.12.tar.gz -L http://downloads.sourceforge.net/swig/swig-3.0.12.tar.gz 
tar -zxvf /tmp/swig-3.0.12.tar.gz -C /tmp
cd /tmp/swig-3.0.12 
./configure --prefix=/usr --without-clisp --without-maximum-compile-warnings
make
make install
cd ~

#Clone down repository
export GIT_SSH_COMMAND="ssh -F /root/.ssh/config -o StrictHostKeyChecking=no"
git clone repo:georgianpartners/automl_benchmark automl
cd  automl

#Setup virtual environment
python3 -m venv automl
source automl/bin/activate

mkdir data

pip install numpy
pip install cython
pip install -r requirements.txt

mkdir lib
cd lib
git clone https://github.com/openml/openml-python openml-python
cd openml-python
pip install ./
cd ../..

#Download data
python get_data.py

#Run a test
tail data/197.csv

#Run computation
python compute.py

#Publish results to s3
aws s3 cp *.csv s3://georgianpartners-auto-ml-data/

exit 0
