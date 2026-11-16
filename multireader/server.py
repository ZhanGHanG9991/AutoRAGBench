"""Bounded vLLM lifecycle, independent from the build-stage generator configuration."""
import json,os,pathlib,secrets,signal,socket,subprocess,time
import httpx
from .ports import choose_port_base

# Keep the same spelling as configs/pilot.env and the build sandbox. Resolving
# the WORK symlink changes the bind destination and breaks frozen absolute paths.
ROOT=pathlib.Path(os.environ.get('RAGENG_ROOT') or pathlib.Path(__file__).absolute().parents[1])

def config():
    # reader paths in configs/readers.json are relative to <repository>/models (a symlink is fine)
    cfg=json.loads((ROOT/'configs/readers.json').read_text())
    for r in cfg['readers'].values():r['path']=str(ROOT/'models'/r['path'])
    return cfg

def free_base():
    return choose_port_base(os.environ.get('SLURM_JOB_ID','0'))

class Server:
    def __init__(self,reader,logs,retrieval=False):
        self.reader=reader;self.profile=config()['readers'][reader];self.logs=pathlib.Path(logs)
        self.logs.mkdir(parents=True,exist_ok=True);self.base=free_base();self.key=secrets.token_hex(24);self.proc=None
        self.retrieval=retrieval

    @property
    def url(self):return f'http://127.0.0.1:{self.base}'

    def __enter__(self):
        self.started=time.time()
        if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Run on an allocated GPU through Slurm')
        env={k:v for k,v in os.environ.items() if k.lower() not in ['http_proxy','https_proxy','all_proxy']}
        env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',VLLM_USE_FLASHINFER_SAMPLER='0',
                   HF_HOME=str(ROOT/'envs/.hf-home'),VLLM_CACHE_ROOT=str(ROOT/'envs/.vllm-cache'),
                   PATH=str(ROOT/'envs/vllm/bin')+':'+env['PATH'])
        cmd=[str(ROOT/'envs/vllm/bin/vllm'),'serve',self.profile['path'],'--served-model-name','generator',
             '--host','127.0.0.1','--port',str(self.base),'--api-key',self.key,'--dtype','bfloat16',
             '--max-model-len','32768','--max-num-seqs','16','--gpu-memory-utilization',
             '0.35' if self.retrieval else '0.55','--generation-config','vllm']
        if self.profile['multimodal']:cmd+=['--limit-mm-per-prompt','{"image":0,"video":0,"audio":0}']
        self.log=(self.logs/'vllm.log').open('a')
        self.proc=subprocess.Popen(cmd,env=env,stdout=self.log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            with httpx.Client(timeout=3,trust_env=False) as c:
                for _ in range(400):
                    if self.proc.poll() is not None:raise RuntimeError('vLLM exited before ready; inspect vllm.log')
                    try:
                        if c.get(self.url+'/health').status_code==200:
                            self.startup_seconds=time.time()-self.started
                            return self
                    except httpx.HTTPError:pass
                    time.sleep(2)
            raise TimeoutError('vLLM startup deadline')
        except BaseException:
            self.__exit__(None,None,None);raise

    def __exit__(self,*args):
        if self.proc is not None:
            try:
                os.killpg(self.proc.pid,signal.SIGTERM);self.proc.wait(timeout=30)
            except ProcessLookupError:pass
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid,signal.SIGKILL);self.proc.wait(timeout=10)
            self.log.close();time.sleep(3)

    def gateway_env(self):
        return {**os.environ,'VLLM_API_KEY':self.key,'RAGENG_PORT_BASE':str(self.base),
                'RAGENG_PORT_JOB':os.environ['SLURM_JOB_ID'],'RAGENG_DATA_VERSION':'formal-v2'}
