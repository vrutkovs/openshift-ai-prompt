import os
import sys
DEVICE=os.environ.get("OPENJOURNEY_DEVICE", "CPU")
STEPS=os.environ.get("OPENJOURNEY_STEPS", 20)
PROMPT=os.environ.get("PROMPT", 'A person in red fedora, in style of picasso')
RESULT_FILENAME=os.environ.get("RESULT_FILENAME", 'foobar.png')

print(f"Generating picture for '{PROMPT}' using {STEPS} steps")

AWS_ACCESS_KEY=os.environ.get("AWS_ACCESS_KEY")
AWS_ACCESS_SECRET=os.environ.get("AWS_ACCESS_SECRET")
S3_BUCKET_NAME=os.environ.get("S3_BUCKET_NAME")

from diffusers import AutoPipelineForText2Image
from diffusers import DPMSolverMultistepScheduler
import torch

pipeline = AutoPipelineForText2Image.from_pretrained(
	'./model',
    torch_dtype=torch.float16, variant="fp16",
    use_safetensors=True,
).to(DEVICE)

import random
rand = random.randrange(100000)
generator = torch.Generator(DEVICE).manual_seed(rand)
final_image = pipeline(PROMPT, generator=generator).images[0]

import tempfile
_, file_extension = os.path.splitext(RESULT_FILENAME)
temp = tempfile.NamedTemporaryFile(suffix=file_extension)
print(f"Saving picture to {temp.name}")
final_image.save(temp.name)

if not AWS_ACCESS_KEY or not AWS_ACCESS_SECRET or not S3_BUCKET_NAME:
    sys.exit(0)

S3_LOCATION = 'http://{}.s3.amazonaws.com'.format(S3_BUCKET_NAME)
print(f"Uploading {temp.name} to {S3_LOCATION}")
import boto3, botocore

s3 = boto3.client(
   "s3",
   aws_access_key_id=AWS_ACCESS_KEY,
   aws_secret_access_key=AWS_ACCESS_SECRET
)
s3.upload_fileobj(
    temp,
    S3_BUCKET_NAME,
    RESULT_FILENAME,
    ExtraArgs={
        "ACL": "public-read",
        "ContentType": "image/png",
    }
)
print(f"Uploaded to {S3_LOCATION}/{RESULT_FILENAME}")
