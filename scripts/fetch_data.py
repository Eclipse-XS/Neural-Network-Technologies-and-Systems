"""Fetch versioned source archives; refuse changed data instead of accepting drift."""
from pathlib import Path
import csv,hashlib,io,json,urllib.request,zipfile
ROOT=Path(__file__).resolve().parents[1]
for item in json.loads((ROOT/'datasets.lock.json').read_text()):
 with urllib.request.urlopen(item['url'],timeout=120) as response:blob=response.read()
 if hashlib.sha256(blob).hexdigest()!=item['archive_sha256']:raise RuntimeError('Archive checksum mismatch: '+item['source'])
 archive=zipfile.ZipFile(io.BytesIO(blob))
 for name,expected in item['files'].items():
  filename=Path(name).name
  member=next(x for x in archive.namelist() if Path(x).name==filename)
  data=archive.read(member)
  if item['key']=='sms':
   rows=list(csv.reader(io.StringIO(data.decode('latin-1'))));stream=io.StringIO(newline='');writer=csv.writer(stream,lineterminator='\n');writer.writerow(['Category','Message']);writer.writerows([row[:2] for row in rows[1:] if len(row)>=2]);data=stream.getvalue().encode('utf-8')
  if hashlib.sha256(data).hexdigest()!=expected:raise RuntimeError('File checksum mismatch: '+filename)
  path=ROOT/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
  print('VERIFIED',name)
