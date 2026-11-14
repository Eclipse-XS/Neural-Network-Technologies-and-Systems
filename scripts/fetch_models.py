"""Download immutable Hugging Face snapshots and verify weight SHA-256 values."""
import argparse, hashlib, json
from pathlib import Path
from huggingface_hub import snapshot_download
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('model',choices=['llava','diffusion','minilm','e5','ministral']);p.add_argument('--fp16',action='store_true');a=p.parse_args()
keys={'llava':'llava-hf/llava-interleave-qwen-0.5b-hf','diffusion':'stable-diffusion-v1-5/stable-diffusion-v1-5','minilm':'sentence-transformers/all-MiniLM-L6-v2','e5':'intfloat/multilingual-e5-small','ministral':'lmstudio-community/Ministral-3-3B-Instruct-2512-GGUF'}
item=next(x for x in json.loads((ROOT/'models.lock.json').read_text()) if x['repo_id']==keys[a.model])
patterns=['*.json','*.txt','*.jinja','*.model','*.tiktoken','*.safetensors']
ignore=['onnx/*','openvino/*','flax/*','*.onnx','*.bin','*.msgpack','*.h5']
if a.model=='ministral':
 patterns=['*Q4_K_M.gguf']
if a.fp16 and a.model=='diffusion':
 patterns=['*.json','*.txt','*.jinja','*.model','*.fp16.safetensors']
root=Path(snapshot_download(item['repo_id'],revision=item['revision'],allow_patterns=patterns,ignore_patterns=ignore))
required = ['model.safetensors']
if a.model=='diffusion':
 suffix='.fp16.safetensors' if a.fp16 else '.safetensors'
 required=['unet/diffusion_pytorch_model'+suffix,'vae/diffusion_pytorch_model'+suffix,'text_encoder/model'+suffix,'safety_checker/model'+suffix]
if a.model=='ministral':required=['Ministral-3-3B-Instruct-2512-Q4_K_M.gguf']
missing=[name for name in required if not (root/name).is_file()]
if missing:raise RuntimeError('Incomplete model snapshot: '+', '.join(missing))
count=0
for name,expected in item['lfs_sha256'].items():
 path=root/name
 if path.is_file():
  h=hashlib.sha256()
  with path.open('rb') as f:
   for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
  if h.hexdigest()!=expected:raise RuntimeError('Checksum mismatch: '+name)
  count+=1
if count==0:raise RuntimeError('No verified model weights')
print(item['repo_id'],item['revision'],'verified weights:',count)

print("Snapshot:",root)
