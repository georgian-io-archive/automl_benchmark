#!/usr/bin/env bash

#Setup swap space
sed -e 's/\s*\([\+0-9a-zA-Z]*\).*/\1/' << EOF | fdisk /dev/xvdb
  n # new partition
  p # primary partition
  1 # partition number 1
    # default - start at beginning of disk 
    # default - go to end of disk
  t # change type
 82 # set type to swap
  w # write the partition table
EOF

mkswap /dev/xvdb1
swapon /dev/xvdb1

yum -y install openssh openssh-clients unzip aws-cli git python36 python36-devel
yum -y groupinstall 'Development Tools'
yum -y install wget java-1.8.0-openjdk antlr-tool autoconf boost-devel expat-devel libcurl-devel gcc-c++ pcre-devel

#Setup RSA
aws s3 cp --recursive s3://${S3_BUCKET}/${S3_FOLDER}ssh /root/.ssh
chmod -R 400 /root/.ssh
chmod 755 /root/.ssh/git

#Install swig
export PATH=$PATH:/usr/local/bin
ln -s /usr/libexec/gcc/x86_64-amazon-linux/4.8.5/cc1plus /usr/local/bin/cc1plus
aws s3 cp s3://${S3_BUCKET}/${S3_FOLDER}swig.tar.gz ~/swig-3.0.12.tar.gz
tar xf /root/swig-3.0.12.tar.gz -C /root/
cd /root/swig-3.0.12 
./configure --prefix=/usr --without-clisp --without-maximum-compile-warnings
make
make install
cd /root/

#Clone down repository
export GIT_SSH_COMMAND="ssh -F /root/.ssh/git -o StrictHostKeyChecking=no"
git clone repo:georgianpartners/automl_benchmark automl_benchmark
cd  automl_benchmark

#Setup virtual environment
python3 -m venv automl_benchmark
source automl_benchmark/bin/activate


#Install requirements
pip install --upgrade pip
pip install -r batch-requirements.txt
pip install -r pre-requirements.txt
pip install -r requirements.txt


