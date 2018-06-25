FROM amazonlinux:latest

RUN yum -y install openssh openssh-clients unzip aws-cli git python36 python36-devel
RUN yum -y groupinstall 'Development Tools'
RUN yum -y install wget java-1.8.0-openjdk antlr-tool autoconf boost-devel expat-devel libcurl-devel gcc-c++ pcre-devel
ADD batch_job.sh /usr/local/bin/batch_job.sh
WORKDIR /root
USER root

ENTRYPOINT ["/usr/local/bin/batch_job.sh"]
