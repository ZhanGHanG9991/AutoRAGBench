import ast,json,pathlib,tempfile,unittest
from types import SimpleNamespace
from .contract import validate_passages,verify_corpus,sha,messages,clean_answer,pack_passages,pack_for_readers

ROOT=pathlib.Path(__file__).resolve().parents[1]

class EvidenceTests(unittest.TestCase):
    def test_schema_rejects_invalid_and_strips_extra_fields(self):
        for value in [None,{},[{'doc_id':1,'text':'a'}],[{'doc_id':'a','text':' '}]]:
            with self.assertRaises(ValueError):validate_passages(value)
        self.assertEqual(validate_passages([{'doc_id':'d','text':'x','answer':'injection'}]),[{'doc_id':'d','title':'','text':'x'}])

    def test_fabricated_evidence_and_missing_doc_rejected(self):
        rows=[{'passages':[{'doc_id':'d','title':'T','text':x}]} for x in ['a b','fabricated']]
        rows.append({'passages':[{'doc_id':'missing','title':'T','text':'a'}]})
        with tempfile.TemporaryDirectory() as d:
            p=pathlib.Path(d)/'corpus.jsonl';p.write_text(json.dumps({'doc_id':'d','title':'T','text':'a  b\nc'}))
            result=verify_corpus(rows,p)
        self.assertEqual(result['invalid_rows'],2)
        self.assertFalse(rows[0].get('error'));self.assertEqual(rows[1]['passages'],[])

    def test_line_level_evidence(self):
        corpus=[{'doc_id':'d','title':'T','text':'x1\nx2  a\nx3'},{'doc_id':'e','title':'U','text':'y1\ny2'}]
        def run(scope):
            rows=[{'passages':[{'doc_id':'d','title':'T','text':x}]} for x in ['x3\nx1','x2 a\n\nx1','y1\nx1','x1\nz','x1 x3']]
            rows.append({'passages':[{'doc_id':'d','title':'U','text':'x1'}]})
            rows.append({'passages':[{'doc_id':'d','title':'T','text':'x2  a\nx3'}]})
            with tempfile.TemporaryDirectory() as d:
                p=pathlib.Path(d)/'corpus.jsonl';p.write_text('\n'.join(json.dumps(c) for c in corpus))
                result=verify_corpus(rows,p,line_scope=scope)
            return [not r.get('error') for r in rows],result
        self.assertEqual(run(None)[0],[False,False,False,False,False,False,True])
        self.assertEqual(run('claimed')[0],[True,True,False,False,False,False,True])
        ok,result=run('any')
        self.assertEqual(ok,[True,True,True,False,False,False,True])
        self.assertEqual((result['line_level_passages'],result['bad_passages'],result['invalid_rows']),(3,3,3))

    def test_hash_changes_when_order_or_text_changes(self):
        self.assertNotEqual(sha(['a','b']),sha(['b','a']))
        self.assertNotEqual(sha({'text':'a'}),sha({'text':'b'}))
        self.assertEqual(sha({'a':1,'b':2}),sha({'b':2,'a':1}))

    def test_prompt_and_cleaning_preserve_multiline_answer(self):
        self.assertEqual(messages('Q',[],'yes or no')[0]['role'],'user')
        self.assertEqual(clean_answer('Answer: Alice\nBob'),'Alice\nBob')

    def test_common_prefix_budget(self):
        ps=[{'doc_id':'a','title':'T','text':'123456789'},{'doc_id':'b','title':'','text':'abcdef'}]
        self.assertEqual(pack_passages(ps,20),[{'doc_id':'a','title':'T','text':'123'}])
        self.assertEqual(pack_passages(ps,100),ps)

    def test_panel_context_limit_is_shared(self):
        class Tokenizer:
            def __init__(self,factor):self.factor=factor
            def encode(self,text,**kwargs):return [0]*len(text)
            def apply_chat_template(self,msg,return_dict=True,**kwargs):
                tokens=[0]*(len(msg[0]['content'])*self.factor)
                return {'input_ids':tokens,'attention_mask':[1]*len(tokens)} if return_dict else tokens
        qwen=Tokenizer(1);other=Tokenizer(10)
        ps=[{'doc_id':'d','title':'T','text':'x'*20000}]
        common=pack_for_readers('Q',ps,'short',qwen,[qwen,other])
        self.assertLess(len(common[0]['text']),4000)
        for tok in [qwen,other]:self.assertLessEqual(len(tok.apply_chat_template(messages('Q',common,'short'),return_dict=False))+512,32768)

    def test_retrieve_does_not_generate_final_answer(self):
        for baseline in ['naive_bm25','reference_rag']:
            source=(ROOT/'baselines'/baseline/'submission/server.py').read_text()
            fun=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='retrieve')
            fun.decorator_list=[];calls=[]
            r=SimpleNamespace(search=lambda *a,**k:[0],fmt=lambda i:'context',doc_of=[0],doc_list=['d'],store=SimpleNamespace(get=lambda i:('T','a b')))
            env={'R':r,'TASK':{'multi_hop':True},'FOLLOWUP':'{q} {ctx}',
                 'generate':lambda p,**k:calls.append(k['max_tokens']) or 'followup', 'clean_answer':lambda x:x}
            exec(compile(ast.Module(body=[fun],type_ignores=[]),'retrieval','exec'),env)
            self.assertEqual(env['retrieve']({'question':'Q'})['passages'],[{'doc_id':'d','title':'T','text':'a b'}])
            self.assertEqual(calls,[] if baseline=='naive_bm25' else [32])

    def test_batch_packing_preserves_scalar_prefix_and_binary_search(self):
        class Tokenizer:
            def __init__(self,factor=1,nonmonotonic=False):self.factor=factor;self.nonmonotonic=nonmonotonic
            def encode(self,text,**kwargs):return [0]*len(text)
            def __call__(self,texts,**kwargs):return {'input_ids':[self.encode(text) for text in texts]}
            def apply_chat_template(self,msg,return_dict=True,**kwargs):
                if msg and isinstance(msg[0],list):
                    ids=[self.apply_chat_template(m,return_dict=False,**kwargs) for m in msg]
                else:
                    text=msg[0]['content'];n=len(text)*self.factor
                    if self.nonmonotonic and '[2] TRIPWIRE' in text and '[3] SAFE' not in text:n=40000
                    ids=[0]*n
                return {'input_ids':ids} if return_dict else ids
        cases=[
            ([{'doc_id':str(i),'title':'T','text':'x'*500} for i in range(64)],Tokenizer(10)),
            ([{'doc_id':str(i),'title':'T','text':'x'*2000} for i in range(12)],Tokenizer()),
            ([{'doc_id':str(i),'title':'T','text':'small'} for i in range(3)],Tokenizer()),
            ([{'doc_id':str(i),'title':title,'text':'x'} for i,title in enumerate(['FIRST','TRIPWIRE','SAFE'])],Tokenizer(nonmonotonic=True)),
            ([],Tokenizer()),
        ]
        for passages,other in cases:
            with self.subTest(passages=len(passages),factor=other.factor,nonmonotonic=other.nonmonotonic):
                qwen=Tokenizer();panel=[qwen,other]
                expected=pack_for_readers('Q',passages,'short',qwen,panel)
                self.assertEqual(pack_for_readers('Q',passages,'short',qwen,panel,batch=True),expected)
        for batch in [False,True]:
            with self.assertRaises(ValueError):pack_for_readers('x'*40000,[],'short',Tokenizer(),[Tokenizer()],batch=batch)

if __name__=='__main__':unittest.main()
