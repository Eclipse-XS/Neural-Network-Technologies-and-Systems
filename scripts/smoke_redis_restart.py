"""Run the live Lab 7 smoke against an owned Redis container and test AOF restart.

Requires the prepared Lab 7 interpreter, Docker image and LM Studio. No existing
container or volume is restarted or deleted. The temporary volume is removed.
"""
import json,os,socket,subprocess,sys,time,uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
name='nn-repro-redis-'+uuid.uuid4().hex[:12]
volume=name+'-data'
image=json.loads((ROOT/'services.lock.json').read_text())['redis']['image']
def docker(*args):
 return subprocess.check_output(['docker',*args],text=True,stderr=subprocess.STDOUT,timeout=90).strip()
# Docker creates the unique volume explicitly so ownership is reviewable.
with socket.socket() as reservation:
 reservation.bind(('127.0.0.1',0));port=reservation.getsockname()[1]
created=False;started=False
try:
 docker('volume','create','--label','codex.repro=true',volume);created=True
 docker('run','-d','--name',name,'--label','codex.repro=true','-p',f'127.0.0.1:{port}:6379','--mount',f'type=volume,source={volume},target=/data','-e','REDIS_ARGS=--appendonly yes',image);started=True
 mapping=json.loads(docker('inspect',name))[0]['NetworkSettings']['Ports']['6379/tcp'][0]
 env=os.environ.copy();env['REDIS_URL']='redis://127.0.0.1:'+mapping['HostPort'];env['SMOKE_REDIS_CONTAINER']=name
 deadline=time.monotonic()+30
 while True:
  try:
   if docker('exec',name,'redis-cli','ping')=='PONG':break
  except subprocess.CalledProcessError:pass
  if time.monotonic()>deadline:raise RuntimeError('Owned Redis container did not become ready')
  time.sleep(.25)
 result=subprocess.run([sys.executable,'-u',str(ROOT/'scripts/smoke.py'),'--lab','7'],env=env,timeout=600)
 if result.returncode:
  print(docker('logs',name),flush=True)
  raise SystemExit(result.returncode)
 print('PASS isolated Redis restart and Lab 7 end-to-end workflow')
finally:
 if started:
  docker('stop',name);docker('rm',name)
 if created:docker('volume','rm',volume)
