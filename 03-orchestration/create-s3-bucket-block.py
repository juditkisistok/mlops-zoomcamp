from time import sleep
from prefect_aws import S3Bucket, AwsCredentials
from dotenv import load_dotenv
import os

# load environment variables from .env file
load_dotenv()

def create_aws_creds_block():
    """
    Create an AWS credentials block.
    """
    aws_creds = AwsCredentials(
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )
    # save the AWS credentials block - overwrite if it already exists
    aws_creds.save(name="aws-creds", overwrite=True)

def create_s3_bucket_block():
    """
    Create an S3 bucket block.
    """
    # import the AWS credentials
    aws_creds = AwsCredentials.load("aws-creds")
    
    # create the S3 bucket block
    s3_bucket = S3Bucket(
        bucket_name="mlops-bucket-orchestration",
        credentials=aws_creds,
    )
    s3_bucket.save(name="s3-bucket-mlops", overwrite=True)

if __name__ == "__main__":
    create_aws_creds_block()
    sleep(5)
    create_s3_bucket_block()
    # we might also need to register the blocks so the server knows about them:
    # prefect block register -m prefect_aws 
