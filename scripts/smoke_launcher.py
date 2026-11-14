"""Windows live launcher/restart check. Terminates only its own child process tree."""
import importlib.util,os,shutil,subprocess,sys,time,urllib.request
from pathlib import Path
lab=Path(__file__).resolve().parents[1]/'labs/lab07_final_project'
spec=importlib.util.spec_from_file_location('startup',lab/'scripts/startup.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
def ready():
 try:
  with urllib.request.urlopen('http://127.0.0.1:8503/_stcore/health',timeout=2) as r:return r.status==200 and r.read()==b'ok'
 except Exception:return False
if ready():raise SystemExit('Port 8503 already hosts an application; stop it before this isolated launcher test')
logs=lab/'outputs/smoke';logs.mkdir(parents=True,exist_ok=True)
os.environ['REDIS_INDEX_NAME']='lab07_repro_launcher_idx'
os.environ['REDIS_KEY_PREFIX']='lab07:repro:launcher'
os.environ['SESSION_DB']=str(logs/'launcher.sqlite')
for iteration in range(2):
 job=mod.ChildJob();child=None
 try:
  with (logs/f'launcher-{iteration}.log').open('w',encoding='utf-8') as log:
   child=subprocess.Popen([shutil.which('pwsh') or 'powershell','-NoProfile','-ExecutionPolicy','Bypass','-File',str(lab/'run.ps1'),'-NoBrowser','-Port','8503'],cwd=os.environ.get('TEMP',str(lab)),stdout=log,stderr=subprocess.STDOUT)
   job.assign(child)
   deadline=time.monotonic()+240
   while not ready():
    if child.poll() is not None or time.monotonic()>deadline:raise RuntimeError((logs/f'launcher-{iteration}.log').read_text())
    time.sleep(.5)
   import sqlite3
   with sqlite3.connect(logs/'launcher.sqlite') as db:
    if iteration==0:
     db.execute("INSERT OR IGNORE INTO sessions VALUES ('restart-smoke','Restart check','2026-11-14','2026-11-14')")
    else:
     assert db.execute("SELECT title FROM sessions WHERE session_id='restart-smoke'").fetchone()[0]=='Restart check'
   print('READY',iteration,flush=True)
   duplicate=subprocess.run([sys.executable,str(lab/'scripts/startup.py'),'--no-browser','--port','8503'],capture_output=True,text=True,timeout=60)
   assert duplicate.returncode!=0 and 'already running' in duplicate.stderr,(duplicate.stdout,duplicate.stderr)
 finally:
  job.close()
  if child is not None:child.wait(timeout=15)
 deadline=time.monotonic()+15
 while ready() and time.monotonic()<deadline:time.sleep(.25)
 assert not ready(),'Owned Streamlit child survived shutdown'
print('PASS launcher, duplicate rejection, restart and child cleanup')
