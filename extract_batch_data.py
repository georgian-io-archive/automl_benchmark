#!/usr/bin/env python

import boto3

from config import load_config

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

def extract():
    s3 = boto3.resource('s3')
    cfg = load_config()
    s3_bucket = cfg["s3_bucket_root"]
    s3_folder = cfg["s3_folder"]
  
    count = 0

    with open("compiled_results.csv", "wb") as f:
        f.write('ID,MODEL,DATASET_ID,TYPE,SEED,RMSE,R2_SCORE,LOGLOSS,F1_SCORE\n')
        while count < 5200:
            temp = TempFile()
            print(count)
            last_pos = f.tell()
            try:
                f.write((str(count) + ",").encode("utf-8"))
                s3.Bucket(s3_bucket).download_fileobj(s3_folder + "out/results" + str(count) + ".csv",  temp)
                f.write(temp.getbytes())
                #delete_file(count)
            except Exception as e:
                pass
                f.seek(last_pos)
            count += 1
     

if __name__ == '__main__':
    extract()
