import os
import sys

MODEL_BASEDIR=os.path.abspath('/opt/app-root/src/models')

DEVICE=os.environ.get("OPENJOURNEY_DEVICE", "CPU")
STEPS=os.environ.get("OPENJOURNEY_STEPS", 20)
PROMPT=os.environ.get("PROMPT", 'A person in red fedora, in style of picasso')
RESULT_FILENAME=os.environ.get("RESULT_FILENAME", 'foobar.png')

print(f"Generating picture for '{PROMPT}' with {STEPS} steps")
MODEL=os.environ.get("OPENJOURNEY_MODEL", "xlbase")
model_path = os.path.join(MODEL_BASEDIR, MODEL)
print(f"Using model from '{MODEL}'")

AWS_ACCESS_KEY=os.environ.get("AWS_ACCESS_KEY")
AWS_ACCESS_SECRET=os.environ.get("AWS_ACCESS_SECRET")
S3_BUCKET_NAME=os.environ.get("S3_BUCKET_NAME")

from diffusers import AutoPipelineForText2Image
from diffusers import DPMSolverMultistepScheduler
import torch

pipeline = AutoPipelineForText2Image.from_pretrained(
    pretrained_model_or_path = os.path.join(MODEL_BASEDIR, MODEL),
    torch_dtype=torch.float16, variant="fp16",
    use_safetensors=True,
)

ADAPTER_PATHS=os.environ.get("ADAPTER_PATHS")
ADAPTER_NAMES=os.environ.get("ADAPTER_NAMES")
ADAPTER_WEIGHTS=os.environ.get("ADAPTER_WEIGHTS")
if ADAPTER_PATHS and ADAPTER_NAMES and ADAPTER_WEIGHTS:
    paths = ADAPTER_PATHS.split(',')
    names = ADAPTER_NAMES.split(',')
    weights = [float(x) for x in ADAPTER_WEIGHTS.split(',')]
    if len(paths) > 0 and len(names) > 0 and len(weights) > 0 and \
       len(paths) == len(paths) == len(weights):
        for idx, p in enumerate(paths):
            abspath = os.path.abspath(os.path.join(MODEL_BASEDIR, p))
            pipeline.load_lora_weights(abspath, adapter_name=names[idx])
        pipeline.set_adapters(names, adapter_weights=weights)

pipeline.to(device="cuda", dtype=torch.float16)

import random
rand = random.randrange(100000)
generator = torch.Generator(DEVICE).manual_seed(rand)
final_image = pipeline(
    PROMPT,
    negative_prompt="ugly, deformed, disfigured, poor details, bad anatomy",
    generator=generator
).images[0]

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
        "ContentType": "image/png",
    }
)
print(f"Uploaded to {S3_LOCATION}/{RESULT_FILENAME}")
