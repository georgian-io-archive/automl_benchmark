
import boto3
import threading
import subprocess

from config import load_config
from dispatcher import Dispatcher, AutoMLMethods

@AutoMLMethods('h2o')
class BareDispatch(Dispatcher):

    @staticmethod
    def provision_instances(num, s3_bucket):
        """Provisions spot EC2 instances with H2O AMI"""
        ec2 = boto3.resource('ec2')
        instances = ec2.create_instances(ImageId='ami-14c5486b',InstanceType='c4.xlarge',MinCount=num,MaxCount=num,
                                         LaunchTemplate={'LaunchTemplateId':'lt-0837f52ac031b2719'})
        ips = [i.public_ip_address for i in instances]
        prov = []
        for ip in ips: prov.append(subprocess.Popen('ssh -F ssh/config ec2-user@' + ip + ' "sudo S3_BUCKET=' + s3_bucket  +  ' bash -s" < provision_baremetal.sh'))
        codes = [p.wait() for p in prov]
        return instances, ips

    @staticmethod
    def chunk(seq, num):
        avg = len(seq) / float(num)
        out = []
        last = 0.0

        while last < len(seq):
            out.append(seq[int(last):int(last + avg)])
            last += avg

        return out

    @staticmethod
    def dispatch(test, ip, s3_bucket, s3_folder):
        for t in tests:
            p = subprocess.Popen('ssh -F ssh.config ec2-user@' + ip + ' "sudo TASK="' + str(test) + '" bash -s" < baremetal_job.sh')
            p.wait()

    @classmethod
    def process(cls,tests):
        """Main function to schedule h2o jobs"""

        #Load config
        config = load_config()
        s3_bucket = config["s3_bucket_root"]
        s3_folder = config["s3_folder"]

        s3 = boto3.resource('s3')
        bucket = s3.Bucket(s3_bucket)

        instances, ips = cls.provision_instances()
        threads = []        

        for i, c in enumerate(cls.chunk(tests, len(ips))):
            t = threading.Thread(target=dispatch, args=(c, ips[i], bucket, s3_folder))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()


        for i in instances: i.terminate()
