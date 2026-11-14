"""Live smoke tests. No synthetic embeddings, mocked LLMs, or tiny substitute models."""
import argparse, asyncio, json, os, sys, tempfile, uuid
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
LABS = sorted(p for p in (ROOT / 'labs').glob('lab*') if p.is_dir()) if (ROOT/'labs').exists() else sorted(p for p in ROOT.glob('lab*') if p.is_dir())
p = argparse.ArgumentParser(); p.add_argument('--lab', type=int, choices=range(1,8), required=True)
a = p.parse_args(); lab = LABS[a.lab-1]
os.chdir(lab)

def check_inputs():
 import hashlib
 for name, expected in json.loads((ROOT/'inputs.sha256.json').read_text()).items():
  if not name.startswith(str(lab.relative_to(ROOT)).replace('\\','/')+'/'): continue
  path=ROOT/name; b=path.read_bytes()
  if path.suffix in ['.txt','.md','.csv','.json']: b=b.replace(b'\r\n',b'\n')
  assert hashlib.sha256(b).hexdigest()==expected, name

def live_chat():
 messages=[{'role':'user','content':'What is 2 + 2? Answer with the number only.'}]
 if (ROOT/'labs').exists():
  from labs.lab02_llm_api.src.client import create_client,request_chat_completion
  with create_client(base_url=os.getenv('LM_STUDIO_BASE_URL','http://localhost:1234/v1'),timeout=180) as client:
   text=request_chat_completion(client,model=os.getenv('LM_STUDIO_MODEL','mistralai/ministral-3-3b'),messages=messages,temperature=0,max_tokens=256)
 else:
  sys.path.insert(0,str(ROOT/'lab2_vllm'))
  from cli import VLLMClient,AppConfig
  client=VLLMClient(AppConfig(base_url=os.getenv('OLLAMA_BASE_URL','http://127.0.0.1:11434/v1'),model=os.getenv('OLLAMA_MODEL','qwen3:4b-studio'),temperature=0,max_tokens=256))
  client.verify_model();text=client.complete(messages)
 import re
 text=re.sub(r'<think>.*?</think>', '', text, flags=re.S).strip()
 assert text and not text.startswith('<think>') and '4' in text,text
 print('LIVE_CHAT',text)

def copilot():
 import importlib.util
 spec=importlib.util.spec_from_file_location('lab_startup',lab/'scripts/startup.py')
 startup=importlib.util.module_from_spec(spec);spec.loader.exec_module(startup)
 startup.load_environment()
 from fastapi.testclient import TestClient
 from knowledge_copilot.api.app import create_app
 from knowledge_copilot.api import dependencies as dep
 from knowledge_copilot.infrastructure.sqlite_repo import SQLiteRepository
 from knowledge_copilot.infrastructure.llm_client import probe_lm_studio_readiness
 probe=probe_lm_studio_readiness(os.getenv('LM_STUDIO_BASE_URL','http://localhost:1234/v1'),os.getenv('LM_STUDIO_MODEL') or None,timeout=180)
 assert probe.is_ready, probe
 os.environ['LM_STUDIO_MODEL']=probe.model_id
 with tempfile.TemporaryDirectory() as temp:
  # Unique test namespace protects existing Redis and SQLite application data.
  os.environ['SESSION_DB']=str(Path(temp)/'sessions.sqlite')
  os.environ['REDIS_INDEX_NAME']='smoke_'+uuid.uuid4().hex[:12]
  os.environ['REDIS_KEY_PREFIX']=os.environ['REDIS_INDEX_NAME']
  with TestClient(create_app()) as client:
   content=b'# Smoke persistence\n\nThe recovery code for the smoke system is ORBIT-731. Redis stores vectors and SQLite stores session history.'
   upload=client.post('/api/v1/documents',files={'file':('smoke.md',content,'text/markdown')})
   assert upload.status_code==201, upload.text
   doc=upload.json(); assert doc['chunk_count']>0,doc
   # Exercise text and PDF extraction through the actual multipart endpoint.
   import fitz
   pdf=fitz.open();page=pdf.new_page();page.insert_text((72,72),'PDF upload smoke: SQLite persists document metadata and Redis stores search vectors.')
   pdf_bytes=pdf.tobytes();pdf.close()
   extra_docs=[]
   for filename,body,mime in [('smoke.txt',b'Text upload smoke: persistent storage uses SQLite and Redis.','text/plain'),('smoke.pdf',pdf_bytes,'application/pdf')]:
    result=client.post('/api/v1/documents',files={'file':(filename,body,mime)})
    assert result.status_code==201 and result.json()['chunk_count']>0,result.text
    extra_docs.append(result.json()['document_id'])
   container=os.getenv('SMOKE_REDIS_CONTAINER')
   if container:
    import subprocess,time
    assert container.startswith('nn-repro-redis-'),'Restart only the isolated smoke container'
    labels=json.loads(subprocess.check_output(['docker','inspect','--format','{{json .Config.Labels}}',container],text=True))
    assert labels.get('codex.repro')=='true','Container is not owned by this smoke harness'
    redis=dep.get_redis_adapter();before=redis.get_client().dbsize()
    subprocess.run(['docker','restart',container],check=True,capture_output=True,timeout=60)
    deadline=time.monotonic()+30
    while not redis.ping():
     if time.monotonic()>deadline:raise RuntimeError('Redis did not recover after restart')
     time.sleep(.25)
    indexes=redis.get_client().execute_command('FT._LIST')
    assert redis.index_name.encode() in indexes and redis.get_client().dbsize()==before
    print('REDIS_RESTART_PERSISTENCE',before,flush=True)

   answer=client.post('/api/v1/chat',json={'query':'What is the recovery code in smoke.md?','retrieval_mode':'hybrid'})
   assert answer.status_code==200,answer.text
   data=answer.json(); assert 'ORBIT-731' in data['answer'],data
   assert data['evidence'] and data['citations'],data
   session=data['session_id']; print('CHAT',data['answer'])
   # Open a new repository instance to exercise disk persistence.
   persisted=SQLiteRepository(Path(temp)/'sessions.sqlite')
   assert persisted.get_session(session) is not None
   assert client.get('/api/v1/sessions/'+session).status_code==200
   assert client.post('/api/v1/documents/'+doc['document_id']+'/reindex').status_code==200
   for identifier in [doc['document_id'],*extra_docs]:
    assert client.delete('/api/v1/documents/'+identifier).status_code==204
  from streamlit.testing.v1 import AppTest
  ui=AppTest.from_file(str(lab/'src/knowledge_copilot/ui/streamlit_app.py')).run(timeout=180)
  assert not ui.exception,str(ui.exception)
  # The test owns only this index and its keys.
  redis=dep.get_redis_adapter(); redis.get_client().execute_command('FT.DROPINDEX',redis.index_name,'DD')

def transformer():
 import torch
 torch.set_num_threads(2)
 torch.manual_seed(42)
 if (ROOT/'labs').exists():
  from labs.lab01_transformer.src.classification.data import inspect_data,clean_data,split_data,create_loaders
  from labs.lab01_transformer.src.generation.data import inspect_corpus
  from labs.lab01_transformer.src.common.text import Vocabulary
  frame,summary=inspect_data(lab/'data/raw/spam.csv');splits=split_data(clean_data(frame))
  assert len(frame)==5572 and len(inspect_corpus(lab/'data/raw/bbc-text.csv'))==2225
  vocabulary=Vocabulary.build(splits['train'].Message,min_frequency=1,max_size=24)
  loaders=create_loaders(splits,vocabulary,max_len=8,batch_size=4)
  real_x,real_y=next(iter(loaders['train']));assert real_x.shape[0]==4
  print('REAL_DATA',summary['rows'],'BBC',2225)
  from labs.lab01_transformer.src.classification.model import SpamClassifier
  from labs.lab01_transformer.src.generation.model import CausalLanguageModel
  for model,classes in [(SpamClassifier(24,8,d_model=16,nhead=2,dim_feedforward=32),2),(CausalLanguageModel(24,8,d_model=16,nhead=2,dim_feedforward=32),24)]:
   x=real_x if classes==2 else torch.randint(4,24,(4,8)); y=model(x); target=real_y if classes==2 else torch.randint(classes,y.shape[:-1]); loss=torch.nn.functional.cross_entropy(y.reshape(-1,classes),target.reshape(-1)); loss.backward()
   opt=torch.optim.Adam(model.parameters(),lr=.01); opt.step(); assert torch.isfinite(loss)
   print('TRAIN_STEP',type(model).__name__,float(loss.detach()))
 else:
  # Execute the real model class definitions stored in the preserved notebooks.
  import ast, math
  sys.path.insert(0,str(lab)); import utils
  namespace={'torch':torch,'nn':torch.nn,'math':math, **vars(utils)}
  found=0
  for path in sorted((lab/'notebooks').glob('*.ipynb')):
   for cell in json.loads(path.read_text(encoding='utf-8'))['cells']:
    if cell['cell_type']!='code': continue
    try: tree=ast.parse(''.join(cell['source']))
    except SyntaxError: continue
    for node in tree.body:
     if isinstance(node,ast.ClassDef):
      exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
      print('MODEL_CLASS',node.name); found+=1
  import pandas as pd
  frame=pd.read_parquet(lab/'data/raw/enron_train.parquet'); assert {'text','label'}<=set(frame.columns)
  vocab=utils.Vocabulary.build_from_texts(frame['text'].head(32).tolist(),min_freq=1)
  assert len(vocab)>4 and utils.tokenize_text(frame.iloc[0]['text'])
  assert found>=2
  for name,classes in [('TransformerClassifier',2),('TransformerDecoderLM',24)]:
   model=namespace[name](24,d_model=16,nhead=2,dim_feedforward=32,max_seq_len=8)
   x=torch.randint(4,24,(4,8));y=model(x);target=torch.randint(classes,y.shape[:-1])
   loss=torch.nn.functional.cross_entropy(y.reshape(-1,classes),target.reshape(-1));loss.backward();torch.optim.Adam(model.parameters()).step()
   assert torch.isfinite(loss);print('TRAIN_STEP',name,float(loss.detach()))
  print('REAL_DATA',len(frame),'VOCAB',len(vocab))

def rag():
 import torch
 torch.set_num_threads(2)
 with tempfile.TemporaryDirectory() as temp:
  if (ROOT/'labs').exists():
   from labs.lab03_rag.src.data import audit_corpus,load_csv_documents
   from labs.lab03_rag.src.documents import build_documents
   from labs.lab03_rag.src.embeddings import create_embeddings
   from labs.lab03_rag.src.redis_store import build_store
   data=lab/'data/raw/csv';audit=audit_corpus(data);assert sum(x['rows'] for x in audit)==185
   docs=build_documents(load_csv_documents(data));emb=create_embeddings(Path(temp),local_files_only=False)
   store,info=build_store(docs,emb,os.getenv('REDIS_URL','redis://localhost:6379'))
   results=store.similarity_search('sports shoes',k=3);assert results and all(x.page_content for x in results)
   print('REDIS_RETRIEVAL',info['document_count'],results[0].page_content[:160])
  else:
   from langchain_core.documents import Document
   from langchain_text_splitters import RecursiveCharacterTextSplitter
   from langchain_huggingface import HuggingFaceEmbeddings
   from langchain_chroma import Chroma
   docs=[Document(page_content=x.read_text(encoding='utf-8'),metadata={'source':x.name}) for x in (lab/'data/knowledge_base').glob('*') if x.is_file()]
   chunks=RecursiveCharacterTextSplitter(chunk_size=700,chunk_overlap=100).split_documents(docs)
   emb=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',model_kwargs={'device':'cpu','revision':'1110a243fdf4706b3f48f1d95db1a4f5529b4d41'})
   store=Chroma.from_documents(chunks,emb,persist_directory=temp,collection_name='smoke')
   assert store.similarity_search('model monitoring',k=3)
   again=Chroma(persist_directory=temp,embedding_function=emb,collection_name='smoke');assert again.similarity_search('model monitoring',k=1)
   print('CHROMA_RETRIEVAL',len(chunks))
 live_chat()

def vision():
 from PIL import Image
 import torch
 torch.set_num_threads(2)
 if a.lab==4:
  if (ROOT/'labs').exists():
   from labs.lab04_stable_diffusion.src.pipeline import load_pipeline
   pipe,info=load_pipeline()
   image=pipe('a photograph of a red apple on a wooden table',num_inference_steps=8,width=512,height=512,generator=torch.Generator('cpu').manual_seed(42)).images[0]
  else:
   from diffusers import StableDiffusionImg2ImgPipeline
   pipe=StableDiffusionImg2ImgPipeline.from_pretrained('stable-diffusion-v1-5/stable-diffusion-v1-5',revision='451f4fe16113bff5a5d2269ed5ad43b0592e9a14',torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,variant='fp16' if torch.cuda.is_available() else None,use_safetensors=True)
   if torch.cuda.is_available():pipe.enable_model_cpu_offload()
   image=pipe('a watercolor lighthouse',image=Image.open(lab/'inputs/01_coastal_lighthouse.png').convert('RGB'),num_inference_steps=8,strength=.8,generator=torch.Generator('cpu').manual_seed(42)).images[0]
  assert image.size==(512,512) and any(high>low for low,high in image.convert('RGB').getextrema()), 'Empty or safety-blocked generation'
  out=lab/'outputs/smoke';out.mkdir(parents=True,exist_ok=True);image.save(out/'generation.png')
 else:
  if (ROOT/'labs').exists():
   from labs.lab05_multimodal.src.model import load_pipeline
   pipe,info=load_pipeline();image=Image.open(lab/'assets/input/primary_image.png').convert('RGB')
   result=pipe(text=[{'role':'user','content':[{'type':'image','image':image},{'type':'text','text':'Describe the image in one sentence.'}]}],return_full_text=False,generate_kwargs={'max_new_tokens':32,'do_sample':False})
   from labs.lab05_multimodal.src.inference import extract_answer
   text=extract_answer(result)
  else:
   from transformers import AutoProcessor,LlavaForConditionalGeneration
   device='cuda' if torch.cuda.is_available() else 'cpu';rev='1090956dd1c79bc93ae98dcf395590369435ec91';model_id='llava-hf/llava-interleave-qwen-0.5b-hf'
   processor=AutoProcessor.from_pretrained(model_id,revision=rev);model=LlavaForConditionalGeneration.from_pretrained(model_id,revision=rev,torch_dtype=torch.float16 if device=='cuda' else torch.float32).to(device)
   image=Image.open(lab/'inputs/01_coastal_lighthouse.png').convert('RGB')
   inputs=processor(text='<|im_start|>user\n<image>\nDescribe this image.<|im_end|>\n<|im_start|>assistant\n',images=image,return_tensors='pt').to(device,dtype=model.dtype)
   with torch.inference_mode():ids=model.generate(**inputs,max_new_tokens=32,do_sample=False)
   text=processor.decode(ids[0][inputs.input_ids.shape[1]:],skip_special_tokens=True)
  assert text.strip();print('VISION',text)
 print('DEVICE',torch.__version__,torch.cuda.is_available())

def agent():
 async def run():
  if (ROOT/'labs').exists():
   from labs.lab06_agents.src.config import Settings
   from labs.lab06_agents.src.agent import connect,create_coursework_agent
   from agents import Runner,RunConfig
   client,settings=await connect(Settings())
   try:
    result=await Runner.run(create_coursework_agent(client,settings),'Use the calculate tool to compute 17 * 23.',max_turns=5,run_config=RunConfig(tracing_disabled=True))
    assert '391' in str(result.final_output),result.final_output
    assert any(getattr(x,'type','')=='tool_call_item' for x in result.new_items)
    print('AGENT',result.final_output)
   finally:await client.close()
  else:
   # Run the real local tools defined in the lab notebook with the real agent.
   import ast
   namespace={'json':json,'Path':Path,'DATA_DIR':lab/'data'}
   notebook=json.loads((lab/'notebooks/01_agent_tools_and_memory.ipynb').read_text(encoding='utf-8'))
   for cell in notebook['cells']:
    if cell['cell_type']!='code':continue
    try:tree=ast.parse(''.join(cell['source']))
    except SyntaxError:continue
    for node in tree.body:
     if isinstance(node,ast.FunctionDef) and node.name in ['query_model_registry','list_degraded_models','consult_incident_runbook','calculate_slo_margin']:
      exec(compile(ast.Module(body=[node],type_ignores=[]),'lab6-tools','exec'),namespace)
   from llama_index.llms.openai_like import OpenAILike
   from llama_index.core.agent.workflow import AgentWorkflow,ToolCallResult
   llm=OpenAILike(model=os.getenv('OLLAMA_MODEL','qwen3:4b-studio'),api_base=os.getenv('OLLAMA_BASE_URL','http://127.0.0.1:11434/v1'),api_key='ollama',is_chat_model=True,is_function_calling_model=True,context_window=4096)
   flow=AgentWorkflow.from_tools_or_functions([namespace['query_model_registry'],namespace['calculate_slo_margin']],llm=llm,system_prompt='Use the tools to answer. /no_think')
   handler=flow.run(user_msg='Use query_model_registry for fraud_detector_v2 and report its status.');calls=[]
   async for event in handler.stream_events():
    if isinstance(event,ToolCallResult):calls.append(event.tool_name)
   response=await handler;assert 'query_model_registry' in calls and str(response).strip(),(response,calls);print('AGENT',str(response))
 asyncio.run(run())

check_inputs()
if a.lab==1: transformer()
elif a.lab==2: live_chat()
elif a.lab==7: copilot()
elif a.lab==3: rag()
elif a.lab in [4,5]: vision()
elif a.lab==6: agent()
else:
 # Execute real notebooks with their full workflows; the runner saves partial output on errors.
 import subprocess
 subprocess.run([sys.executable,str(ROOT/'scripts/run_notebooks.py'),str(lab)],check=True)
print('PASS',lab.name)
