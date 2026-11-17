                                                                                    
import importlib
import json
import os
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from magra_rag import ann, ann_catalog, cli
from magra_benchmark.protocol import runtime_identity


class PureANNCLIContracts(unittest.TestCase):
    def test_exact_default_does_not_acquire_ANN_configuration(self):
        args=cli.build_parser().parse_args(["run","--bundle","unused","--out","unused"])
        self.assertEqual(args.dense_backend,"exact");self.assertIsNone(args.ann_catalog)
        baseline={"generation":False,"model":None,"dense_model":None,"dense_index_hash":None,
                  "reranker_model":None,"max_tokens":None,"reasoning_effort":None}
        self.assertEqual(runtime_identity(baseline),baseline)
        self.assertEqual(runtime_identity({**baseline,"dense_backend_hash":"a"*64})["dense_backend_hash"],"a"*64)

    def test_fresh_model_guard_failure_prevents_constructor_and_closes_runtime(self):
        from magra_rag.retrieval import RetrievalConfig
        from unittest.mock import Mock
        runtime=SimpleNamespace(guard=Mock(side_effect=RuntimeError("Fresh model allocation refused")),close=Mock())
        args=SimpleNamespace(dense_backend="hnsw",_ann_description={"catalog":{"runtime":{}}},dense_model="unused",dense_revision="a"*40)
        with (patch("magra_rag.ann_catalog.LoadedRuntime",return_value=runtime),
              patch("magra_rag.dense.DenseRetriever",side_effect=AssertionError("Constructor must remain untouched")) as constructor):
            with self.assertRaisesRegex(RuntimeError,"Fresh model allocation refused"):
                cli._retriever(SimpleNamespace(budget=object()),RetrievalConfig(strategy="dense"),args)
        runtime.guard.assert_called_once_with(ram_reserve=3_500_000_000)
        constructor.assert_not_called();runtime.close.assert_called_once_with()

    def test_generic_oversample_is_bounded_and_filters_keep_exact_fallback(self):
        class StopBeforeSQL(Exception):pass
        requests=[];backend=object.__new__(ann.FaissHNSWDenseRetriever)
        backend._fresh=lambda:None;backend._scope=lambda source,prefix:None
        backend.config=SimpleNamespace(dimensions=2,unit_tolerance=.0001)
        backend.manifest={"raw_sql":{"document_count":200}};backend.internal_candidates=128
        backend.encoder=lambda _: [1.,0.]
        backend.np=SimpleNamespace(frombuffer=lambda *a,**kw:SimpleNamespace(reshape=lambda *a:"query-matrix"))
        def searching(matrix,count):requests.append(count);raise StopBeforeSQL
        backend.index=SimpleNamespace(search=searching)
        with self.assertRaises(StopBeforeSQL):backend.search("public question",k=50)
        self.assertEqual(requests,[128])
        backend.manifest["raw_sql"]["document_count"]=3
        with self.assertRaises(StopBeforeSQL):backend.search("public question",k=50)
        self.assertEqual(requests,[128,3])
        calls=[];backend.exact=SimpleNamespace(search=lambda question,**kw:calls.append(kw) or [{"id":"owned:b"}])
        result=backend.search("public question",k=1,source="owned",allowed_ids=["owned:b"],exclude_ids=["owned:a"])
        self.assertEqual(result[0]["retriever"],"dense_exact_filter_fallback")
        self.assertEqual(calls[0]["allowed_ids"],["owned:b"]);self.assertEqual(calls[0]["exclude_ids"],["owned:a"])
        self.assertEqual(requests,[128,3])

    def test_explicit_scope_routing_includes_global_scope_and_rejects_mismatch(self):
        router=object.__new__(ann_catalog.RoutedHNSWDenseRetriever)
        global_backend=object();owned_backend=object()
        router.backends={(None,None):global_backend,("owned","manual:"):owned_backend}
        self.assertIs(router._route(None,None),global_backend)
        self.assertIs(router._route("owned","manual:"),owned_backend)
        with self.assertRaisesRegex(ValueError,"exact query source/prefix"):router._route("owned",None)

    def test_thread_check_is_cached_and_fails_on_actual_pool_or_Torch_drift(self):
        runtime=object.__new__(ann_catalog.LoadedRuntime)
        runtime.controller=SimpleNamespace(info=lambda:[{"num_threads":1}]);runtime.torch=None
        self.assertEqual(runtime.check(),[{"num_threads":1}])
        runtime.controller=SimpleNamespace(info=lambda:[{"num_threads":2}])
        with self.assertRaisesRegex(ValueError,"thread1"):runtime.check()
        runtime.controller=SimpleNamespace(info=lambda:[])
        runtime.torch=SimpleNamespace(get_num_threads=lambda:4,get_num_interop_threads=lambda:1)
        with self.assertRaisesRegex(ValueError,"Torch runtime"):runtime.check()


NATIVE=os.environ.get("MAGRA_ANN_NATIVE_TESTS")=="1" and bool(os.environ.get("MAGRA_ANN_RUNTIME_CONFIG"))


@unittest.skipUnless(NATIVE,"Explicit MAGRA_ANN_NATIVE_TESTS=1 and local MAGRA_ANN_RUNTIME_CONFIG required")
class NativeANNCLIIntegration(unittest.TestCase):
    def setUp(self):
        from magra_rag.storage import DocumentStore
        from magra_rag.dense import DEFAULT_DENSE_MODEL,DenseRetriever,retrieval_model_identity
        from magra_benchmark.splits import make_splits,write_split_bundle
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        self.data=Path(temporary.name);self.revision="a"*40;self.model=DEFAULT_DENSE_MODEL
        self.model_id=retrieval_model_identity(self.model,self.revision);self.real_dense=DenseRetriever
        self.configuration=ann_catalog.read_runtime(os.environ["MAGRA_ANN_RUNTIME_CONFIG"])
        self.runtime_path=self.data/"runtime.json";self.runtime_path.write_text(json.dumps(self.configuration),encoding="utf-8")
        self.sql=self.data/"corpus.sqlite"
        with DocumentStore(self.sql) as store:
            store.ingest([{"id":identity,"title":"Owned fixture","text":"Owned manual calibration constants.","metadata":{"source":source}}
                for identity,source in [("owned:a","owned"),("owned:b","owned"),("other:c","other"),("other:d","other")]])
            with store.connection:
                for ordinal,identity in enumerate(["owned:a","owned:b","other:c","other:d"]):
                    content=store.connection.execute("SELECT content_hash FROM documents WHERE id=?",(identity,)).fetchone()[0]
                    vector=[1.]+[0.]*383 if ordinal<2 else [0.,1.]+[0.]*382
                    store.connection.execute("INSERT INTO vectors VALUES(?,?,?,?,?)",(identity,self.model_id,content,384,struct.pack("<384f",*vector)))
        self.bundle=self.data/"bundle"
        rows=[{"id":"fixture:dev","question":"Which instrument measures the quartz divisor?","source":"owned","corpus_source":"owned","domain":"manuals","official_split":"dev","relevance":{"owned:a":1}},
              {"id":"fixture:eval","question":"Where did the coastal navigation survey begin?","source":"owned","corpus_source":"owned","domain":"manuals","official_split":"test","relevance":{"owned:b":1}}]
        write_split_bundle(self.bundle,make_splits(rows))

    def invoke(self,*arguments):
        args=cli.build_parser().parse_args(["--data-root",str(self.data),*[str(a) for a in arguments]])
        return args.func(args)

    def build(self,source=None,name="ann"):
        args=["ann-build","--dense-model",self.model,"--dense-revision",self.revision,"--ann-runtime",self.runtime_path,
              "--max-index-bytes","4000000","--out",self.data/name]
        if source is not None:args.extend(["--source",source])
        return self.invoke(*args)

    def catalog(self,built,source=None,name="catalog.json"):
        value={"format":"magra-hnsw-catalog-v1","runtime":self.configuration,"ef_search":512,"internal_candidates":128,
               "artifacts":[{"path":built["artifact"],"artifact_hash":built["artifact_hash"],"source":source,"id_prefix":None}]}
        target=self.data/name;target.write_text(json.dumps(value),encoding="utf-8");return target

    def encoder_factory(self,store,**kwargs):
                                                                           
                                                                           
        dense=object.__new__(self.real_dense);dense.store=store;dense.model_id=self.model_id;dense.is_qwen=False;dense.cpu_threads=4
        dense.np=importlib.import_module("numpy")
        dense.encoder=SimpleNamespace(encode=lambda texts,**kw:dense.np.asarray([[1.]+[0.]*383 for _ in texts],dtype="<f4"))
        return dense

    def serve(self,output,*,catalog=None,split="dev",strategy="dense"):
        args=["run","--bundle",self.bundle,"--split",split,"--strategy",strategy,"--dense-model",self.model,
              "--dense-revision",self.revision,"--out",output]
        if catalog is not None:args.extend(["--dense-backend","hnsw","--ann-catalog",catalog])
        with patch("magra_rag.dense.DenseRetriever",side_effect=self.encoder_factory):return self.invoke(*args)

    def test_real_build_is_readonly_global_artifact_and_corrupt_file_rejected(self):
        before=self.sql.read_bytes()
        with patch("magra_rag.dense.DenseRetriever",side_effect=AssertionError("Building SQL vectors must not load a model")):
            built=self.build()
        self.assertEqual(self.sql.read_bytes(),before);self.assertTrue(built["canonical_SQL_readonly"])
        self.assertFalse(any(Path(str(self.sql)+s).exists() for s in ("-wal","-shm","-journal")))
        catalog=self.catalog(built)
        description=ann_catalog.describe_catalog(catalog,cli.ResourceBudget(self.data),model_id=self.model_id,
            queries=[{"id":"global","question":"Owned question","corpus_source":None}])
        self.assertEqual(len(description["entries"]),1)
        with self.assertRaisesRegex(ValueError,"exact query source/prefix"):
            ann_catalog.describe_catalog(catalog,cli.ResourceBudget(self.data),model_id=self.model_id,
                queries=[{"id":"scoped","question":"Owned question","corpus_source":"owned"}])
        scoped=self.build("owned","scoped");scoped_catalog=self.catalog(scoped,"owned","scoped-catalog.json")
        (self.data/"scoped/index.faiss").write_bytes((self.data/"scoped/index.faiss").read_bytes()+b"corrupt")
        output=self.data/"corrupt.jsonl"
        with self.assertRaisesRegex(ValueError,"checksum mismatch"):self.serve(output,catalog=scoped_catalog)
        self.assertFalse(output.exists());self.assertFalse(Path(str(output)+".manifest.json").exists())

    def test_real_cli_roundtrip_deterministic_ties_and_frozen_search_policy(self):
        from dataclasses import asdict
        from magra_rag.retrieval import RetrievalConfig
        built=self.build("owned");catalog=self.catalog(built,"owned")
        outputs={}
        for name,strategy in [("baseline","bm25"),("candidate","dense")]:
            output=self.data/(name+".jsonl");self.serve(output,catalog=catalog if name=="candidate" else None,strategy=strategy)
            metadata=json.loads(Path(str(output)+".manifest.json").read_text(encoding="utf-8"));self.assertEqual(metadata["status"],"complete")
            if name=="candidate":
                record=json.loads(output.read_text(encoding="utf-8"));self.assertEqual(record["ranked_doc_ids"],["owned:a","owned:b"])
                self.assertEqual(metadata["dense_runtime_profile"]["Torch_intraop_threads"],1)
                self.assertFalse(metadata["dense_runtime_profile"]["physical_affinity_enforced"])
            else:self.assertNotIn("dense_backend_hash",metadata)
            report=self.data/(name+".report.json");self.invoke("evaluate","--bundle",self.bundle,"--predictions",output,"--out",report)
            config=self.data/(name+".config.json");config.write_text(json.dumps(asdict(RetrievalConfig(strategy=strategy))),encoding="utf-8")
            outputs[name]=(report,config)
        freeze=self.data/"freeze.json"
        self.invoke("freeze","--bundle",self.bundle,"--dev-reports",*[n+"="+str(v[0]) for n,v in outputs.items()],
                    "--configs",*[n+"="+str(v[1]) for n,v in outputs.items()],"--baselines","baseline","--candidate","candidate","--out",freeze)
        value=json.loads(catalog.read_text(encoding="utf-8"));value["internal_candidates"]=129;catalog.write_text(json.dumps(value),encoding="utf-8")
        output=self.data/"changed-eval.jsonl"
        with patch("magra_rag.dense.DenseRetriever",side_effect=AssertionError("Frozen runtime mismatch must precede model loading")):
            with self.assertRaisesRegex(ValueError,"settings differ"):
                self.invoke("run","--bundle",self.bundle,"--split","eval","--strategy","dense","--dense-model",self.model,
                    "--dense-revision",self.revision,"--dense-backend","hnsw","--ann-catalog",catalog,"--freeze",freeze,"--out",output)
        self.assertFalse(output.exists())


if __name__=="__main__":unittest.main()
