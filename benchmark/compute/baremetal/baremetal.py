
import boto3
import threading
import subprocess
import time
import pickle

from ...config import load_config
from ..dispatcher import Dispatcher, AutoMLMethods

@AutoMLMethods('h2o','auto-sklearn','tpot')
class BareDispatch(Dispatcher):

    @staticmethod
    def provision_instances(num, template_id, instance_type, s3_bucket):
        """Provisions spot EC2 instances with H2O AMI"""
        ec2 = boto3.resource('ec2')
        instances = ec2.create_instances(ImageId='ami-14c5486b',InstanceType=instance_type,MinCount=num,MaxCount=num,
                                         LaunchTemplate={'LaunchTemplateId':template_id},
                                         InstanceMarketOptions={'MarketType':'spot','SpotOptions':{'SpotInstanceType':'one-time'}})
        ips = []
        for i in instances:
            ip = None
            while i.public_ip_address == None: i.reload()
            ips.append(i.public_ip_address)
        print("Provisioning Servers...")
        time.sleep(60)
        prov = []
        for ip in ips: prov.append(subprocess.Popen('ssh -F benchmark/config/ssh/baremetal ec2-user@' + ip + ' "sudo S3_BUCKET=' + s3_bucket  +  ' bash -s" < benchmark/compute/baremetal/scripts/provision_baremetal.sh', shell=True))
        codes = [p.wait() for p in prov]
        print("Servers successfully provisioned")
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
    def dispatch(tests, ip, s3_bucket, bucket_name, s3_folder):
        ssh_cmd = 'ssh -F ssh/baremetal ' + ip
        exec_cmd = 'nohup bash /root/automl_benchmark/benchmark/compute/baremental/scripts/baremetal_job.sh > logs.out 2>&1'
        s3_cmd = 'sudo S3_BUCKET=' + bucket_name + ' S3_FOLDER=' + s3_folder
        task_cmds = ' && '.join([s3_cmd + ' TASK=' + str(t).replace('\'','').replace(' ','') + ' ' + exec_cmd for t in tests])
        cmd = ssh_cmd + ' "' + task_cmds + '"'
        print(cmd)
        p = subprocess.Popen(cmd, shell=True)
        

    @classmethod
    def process(cls, tests):
        """Main function to schedule h2o jobs"""

        if not tests:
            return

        #Load config
        config = load_config()
        s3_bucket = config["s3_bucket_root"]
        s3_folder = config["s3_folder"]
        cluster_size = min(config["cluster_size"], len(tests))
        template_id = config["ec2_template"]
        cluster_type = config["cluster_type"]

        s3 = boto3.resource('s3')
        bucket = s3.Bucket(s3_bucket)

        instances, ips = cls.provision_instances(cluster_size, template_id, cluster_type, s3_bucket)
        threads = []

        for i, c in enumerate(cls.chunk(tests, len(ips))):
            t = threading.Thread(target=cls.dispatch, args=(c, ips[i], bucket, s3_bucket, s3_folder))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        with open("running.dat", "wb") as f:
            pickle.dump([i.instance_id for i in instances], f)
