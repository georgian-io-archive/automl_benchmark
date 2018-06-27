import boto3

from tqdm import tqdm

from .benchmark import generate_tests
from ..config import load_config

class TempFile:
    """Escrow to hold AWS S3 file object"""

    data = bytes()
    def write(self, b):
        self.data += b

    def getbytes(self):
        return self.data

def delete_file(fid): 
    s3 = boto3.resource('s3')
    cfg = load_config()
    s3_bucket = cfg["s3_bucket_root"]
    s3_folder = cfg["s3_folder"]
    s3.Bucket(s3_bucket).delete_objects(Delete={'Objects':[{'Key':s3_folder+ "out/results" + str(fid) + ".csv"}]})

def download_data():
    s3 = boto3.resource('s3')
    cfg = load_config()
    s3_bucket = cfg["s3_bucket_root"]
    s3_folder = cfg["s3_folder"]

    RANGE = len(generate_tests())

    with open("compiled_results.csv", "wb") as f:
        f.write('ID,MODEL,DATASET_ID,TYPE,SEED,MSE,R2_SCORE,LOGLOSS,F1_SCORE\n'.encode('utf-8'))
        for c in tqdm(range(RANGE)):
            temp = TempFile()
            last_pos = f.tell()
            try:
                f.write((str(c) + ",").encode("utf-8"))
                s3.Bucket(s3_bucket).download_fileobj(s3_folder + "out/results" + str(c) + ".csv",  temp)
                f.write(temp.getbytes())
                #delete_file(c)
            except Exception as e:
                pass
                f.seek(last_pos)
