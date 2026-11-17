                                                                                    
import ast
from contextlib import contextmanager, ExitStack
import copy
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
REV="a"*40
RAG="_magra_cuda_routing_tests_rag"
BENCH="_magra_cuda_routing_tests_benchmark"


class RouteImports(ast.NodeTransformer):
    def visit_ImportFrom(self,node):
        if node.module:
            node.module=node.module.replace("magra_rag",RAG,1) if node.module.startswith("magra_rag") else node.module
            node.module=node.module.replace("magra_benchmark",BENCH,1) if node.module.startswith("magra_benchmark") else node.module
        return node
    def visit_Constant(self,node):
        if isinstance(node.value,str):
            node.value=node.value.replace("magra_rag.",RAG+".").replace("magra_benchmark.",BENCH+".")
        return node


class GPU:
    def __init__(self,description):
        self.runtime_metadata=copy.deepcopy(description);self.model_id=description["reranker_model"]
        self.closed=False;self.transitions=0;self.cpu_threads=1
    def close(self):
        if not self.closed:self.transitions+=1;self.closed=True
    def resource_observations(self):return {"closed":self.closed,"sampled_peaks":{"GPU_bytes":123},"own_PID_VRAM_bytes":None}


class CudaRoutingTests(unittest.TestCase):
    def setUp(self):
        modules={}
        for name,path in ((RAG,ROOT/"magra_rag"),(BENCH,ROOT/"magra_benchmark")):
            module=ModuleType(name);module.__path__=[str(path)];modules[name]=module
        for name in ("torch","transformers","numpy","pynvml","psutil"):modules[name]=None
        scope=patch.dict(sys.modules,modules);scope.start();self.addCleanup(scope.stop)
        def load(name,path):
            spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
            sys.modules[name]=module
            exec(compile(RouteImports().visit(ast.parse(path.read_text(encoding="utf-8"))),str(path),"exec"),module.__dict__)
            parent,short=name.rsplit(".",1);setattr(sys.modules[parent],short,module)
            return module
        load(RAG+".cuda_resources",ROOT/"magra_rag/cuda_resources.py")
        self.q=load(RAG+".qwen",ROOT/"magra_rag/qwen.py")
        self.d=load(RAG+".dense",ROOT/"magra_rag/dense.py")
        self.p=load(BENCH+".protocol",ROOT/"magra_benchmark/protocol.py")
        self.c=load(RAG+".cli",ROOT/"magra_rag/cli.py")
        self.integrity=ModuleType(BENCH+".integrity");self.integrity.audit_corpus=Mock(return_value={"fictional":True})
        self.tuning=ModuleType(BENCH+".tuning");self.tuning.tune_retrieval=Mock()
        sys.modules[self.integrity.__name__]=self.integrity;sys.modules[self.tuning.__name__]=self.tuning
        versions=patch.object(self.d.importlib.metadata,"version",return_value="1.0-fixture")
        versions.start();self.addCleanup(versions.stop)
        self.description=self.d.reranker_runtime_metadata(self.d.QWEN_RERANKER_MODEL,REV,device="cuda:0")
        self.gpu=GPU(self.description)
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);self.root=Path(temp.name)

    def test_lazy_descriptor_and_unchanged_encoder_identity(self):
        self.assertEqual(self.description["reranker_execution_hash"],self.p.json_hash(self.description["reranker_execution"]))
        cpu=self.d.reranker_runtime_metadata(self.d.QWEN_RERANKER_MODEL,REV)
        self.assertEqual(cpu,{"reranker_model":self.d.retrieval_model_identity(self.d.QWEN_RERANKER_MODEL,REV)})
        embedding_before=self.d.retrieval_model_identity(self.d.QWEN_DENSE_MODEL,REV)
        self.assertIsNone(sys.modules["torch"])
        with patch.object(self.d.importlib.metadata,"version",return_value="2.0-fixture"):
            changed=self.d.reranker_runtime_metadata(self.d.QWEN_RERANKER_MODEL,REV,device="cuda:0")
        self.assertNotEqual(changed["reranker_execution_hash"],self.description["reranker_execution_hash"])
        self.assertEqual(embedding_before,self.d.retrieval_model_identity(self.d.QWEN_DENSE_MODEL,REV))

    def test_missing_dependency_and_wrong_role_fail_before_model_loading(self):
        with patch.object(self.q,"QwenReranker") as factory:
            for model in (self.d.DEFAULT_RERANKER_MODEL,self.d.QWEN_DENSE_MODEL):
                with self.assertRaises(ValueError):self.d.CrossEncoderReranker(None,model=model,revision=REV,device="cuda:0")
            with patch.object(self.d.importlib.metadata,"version",side_effect=self.d.importlib.metadata.PackageNotFoundError("nvidia-ml-py")):
                with self.assertRaisesRegex(RuntimeError,"nvidia-ml-py"):
                    self.d.CrossEncoderReranker(None,model=self.d.QWEN_RERANKER_MODEL,revision=REV,device="cuda:0")
                self.d.reranker_runtime_metadata(self.d.DEFAULT_RERANKER_MODEL,REV)
            factory.assert_not_called()
        with self.assertRaisesRegex(ValueError,"CPU"):
            self.d.DenseRetriever(None,model=self.d.QWEN_DENSE_MODEL,revision=REV,device="cuda:0")

    def test_GPU_dispatch_identity_resources_and_CPU_legacy_kwargs(self):
        native=Mock(model_id=self.description["reranker_model"],cpu_threads=1)
        with patch.object(self.q,"QwenReranker",return_value=native) as factory:
            value=self.d.CrossEncoderReranker(None,model=self.d.QWEN_RERANKER_MODEL,revision=REV,device="cuda:0")
            factory.assert_called_once_with(None,self.d.QWEN_RERANKER_MODEL,REV,device="cuda:0")
            self.assertEqual(value.runtime_metadata,self.description)
            value.rerank("q",[]);native.rerank.assert_called_once_with("q",[])
            self.assertIs(value.resource_observations(),native.resource_observations.return_value)
            value.close();native.close.assert_called_once_with()
        native=Mock(model_id=self.d.retrieval_model_identity(self.d.QWEN_RERANKER_MODEL,REV),cpu_threads=4)
        with patch.object(self.q,"QwenReranker",return_value=native) as factory:
            value=self.d.CrossEncoderReranker(None,model=self.d.QWEN_RERANKER_MODEL,revision=REV)
            factory.assert_called_once_with(None,self.d.QWEN_RERANKER_MODEL,REV)
            value.close();native.close.assert_not_called();self.assertIsNone(value.resource_observations())

    def test_protocol_requires_role_full_execution_and_exact_package_policy_hash(self):
        identity=self.description["reranker_model"]
        self.p._require_pinned_retrieval_model(identity,"invented",role="reranker_model",metadata=self.description)
        for role,metadata in (("dense_model",self.description),("reranker_model",{}),
                             ("reranker_model",{**self.description,"reranker_execution_hash":"0"*64})):
            with self.subTest(role=role),self.assertRaises(ValueError):
                self.p._require_pinned_retrieval_model(identity,"invented",role=role,metadata=metadata)
        bad=copy.deepcopy(self.description);bad["reranker_execution"]["policy"]["max_length"]=1024
        bad["reranker_execution_hash"]=self.p.json_hash(bad["reranker_execution"])
        with self.assertRaises(ValueError):self.p._require_pinned_retrieval_model(identity,"invented",role="reranker_model",metadata=bad)
        cuda_encoder=identity.replace("Qwen3-Reranker","Qwen3-Embedding")
        with self.assertRaises(ValueError):self.p._require_pinned_retrieval_model(cuda_encoder,"invented",role="reranker_model",metadata=self.description)
        self.assertNotIn("reranker_execution_hash",self.p.runtime_identity({"reranker_model":None}))
        compact=self.p.runtime_identity(self.description)
        self.assertEqual(compact,self.p.runtime_identity({**self.description,"reranker_resources":{"GPU_bytes":999}}))
        self.assertNotIn("reranker_execution",compact)

    def test_device_flag_only_run_tune_and_inactive_rejection(self):
        parser=self.c.build_parser()
        self.assertEqual(parser.parse_args(["run","--bundle","fixture","--out","fixture"]).reranker_device,"cpu")
        self.assertEqual(parser.parse_args(["tune","--bundle","fixture","--out","fixture","--reranker-device","cuda:0"]).reranker_device,"cuda:0")
        with patch("sys.stderr",io.StringIO()),self.assertRaises(SystemExit):
            parser.parse_args(["doctor","--reranker-device","cuda:0"])
        args=SimpleNamespace(reranker_device="cuda:0")
        with patch.object(self.d,"DenseRetriever") as dense,self.assertRaisesRegex(ValueError,"active"):
            self.c._retriever(None,self.c.RetrievalConfig(strategy="bm25"),args)
        dense.assert_not_called()

    def test_retriever_construction_error_closes_already_loaded_GPU(self):
        args=SimpleNamespace(reranker_device="cuda:0",reranker_model=self.d.QWEN_RERANKER_MODEL,reranker_revision=REV)
        config=self.c.RetrievalConfig(strategy="bm25",use_reranker=True)
        with patch.object(self.d,"CrossEncoderReranker",return_value=self.gpu),patch.object(self.c,"Retriever",side_effect=RuntimeError("fictional wrapper failure")):
            with self.assertRaisesRegex(RuntimeError,"wrapper failure"):
                self.c._retriever(None,config,args)
        self.assertTrue(self.gpu.closed);self.assertEqual(self.gpu.transitions,1)

    def test_faulted_GPU_close_still_attempts_ANN_runtime_close(self):
        runtime=SimpleNamespace(guard=Mock(),refresh=Mock(),close=Mock())
        ann=ModuleType(RAG+".ann_catalog")
        ann.LoadedRuntime=Mock(return_value=runtime)
        ann.RoutedHNSWDenseRetriever=Mock(side_effect=RuntimeError("fictional ANN setup failure"))
        args=SimpleNamespace(reranker_device="cuda:0",reranker_model=self.d.QWEN_RERANKER_MODEL,reranker_revision=REV,
            dense_backend="hnsw",_ann_description={"catalog":{"runtime":{}}},
            dense_model=self.d.DEFAULT_DENSE_MODEL,dense_revision=REV)
        store=SimpleNamespace(budget=object())
        config=self.c.RetrievalConfig(strategy="hybrid",use_reranker=True)
        with patch.dict(sys.modules,{ann.__name__:ann}),patch.object(self.d,"DenseRetriever",return_value=SimpleNamespace(cpu_threads=4)),\
             patch.object(self.d,"CrossEncoderReranker",return_value=self.gpu),patch.object(self.gpu,"close",side_effect=RuntimeError("fictional GPU close failure")) as close:
            with self.assertRaisesRegex(RuntimeError,"GPU close failure"):
                self.c._retriever(store,config,args)
            close.assert_called_once_with()
        ann.RoutedHNSWDenseRetriever.assert_called_once()
        runtime.close.assert_called_once_with()

    @contextmanager
    def cli_fixture(self,command="run"):
        argv=[command,"--bundle","fictional","--out",str(self.root/(command+".out")),
              "--reranker-model",self.d.QWEN_RERANKER_MODEL,"--reranker-revision",REV,"--reranker-device","cuda:0"]
        if command=="tune":argv.append("--reranker")
        args=self.c.build_parser().parse_args(argv)
        budget=SimpleNamespace(contains=lambda path:Path(path),check=Mock(return_value={}))
        store=SimpleNamespace(corpus_hash=Mock(return_value="fictional-corpus"),budget=budget)
        @contextmanager
        def opened(*_,**__):yield store
        engine=Mock();engine.run.return_value={"query_id":"fictional:q","ranked_doc_ids":[]}
        with ExitStack() as stack:
            for name,value in {"_budget":lambda _:budget,"_store":opened,"_config":lambda _:self.c.RetrievalConfig(strategy="bm25",use_reranker=True),
                "read_split_bundle":Mock(return_value=[{"id":"fictional:q","question":"Fictional question","split":"dev"}]),
                "sha256_file":Mock(return_value="fictional-sha"),"_dense_fingerprint":Mock(return_value=None),
                "environment":Mock(return_value={"code_hash":"fictional"}),"MagraEngine":Mock(return_value=engine)}.items():
                stack.enter_context(patch.object(self.c,name,value))
            yield args,store,engine

    def test_prelocked_runtime_failure_precedes_GPU_load(self):
        with self.cli_fixture() as (args,store,engine):
            args.split="eval";args.freeze="fictional-freeze"
            with patch.object(self.c,"verify_protocol",side_effect=ValueError("fictional locked mismatch")) as verify,patch.object(self.c,"_retriever") as factory:
                with self.assertRaisesRegex(ValueError,"locked mismatch"):self.c.cmd_run(args)
                factory.assert_not_called()
            self.assertEqual(verify.call_args.kwargs["runtime"]["reranker_execution_hash"],self.description["reranker_execution_hash"])

    def test_run_error_closes_GPU_and_keeps_running_manifest(self):
        with self.cli_fixture() as (args,store,engine):
            engine.run.side_effect=RuntimeError("fictional engine failure")
            retriever=SimpleNamespace(dense=None,reranker=self.gpu)
            with patch.object(self.c,"_retriever",return_value=retriever),self.assertRaisesRegex(RuntimeError,"engine failure"):
                self.c.cmd_run(args)
            self.assertTrue(self.gpu.closed);self.assertEqual(self.gpu.transitions,1)
            self.assertEqual(json.loads(Path(str(args.out)+".manifest.json").read_text(encoding="utf-8"))["status"],"running")

    def test_run_success_uses_same_identity_and_records_closed_resources(self):
        with self.cli_fixture() as (args,store,engine):
            args.split="eval";args.freeze="fictional-freeze"
            retriever=SimpleNamespace(dense=None,reranker=self.gpu)
            with patch.object(self.c,"_retriever",return_value=retriever),patch.object(self.c,"verify_protocol") as verify:
                self.c.cmd_run(args)
            metadata=json.loads(Path(str(args.out)+".manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(self.p.runtime_identity(metadata),verify.call_args.kwargs["runtime"])
            self.assertEqual(metadata["reranker_execution"],self.description["reranker_execution"])
            self.assertTrue(metadata["reranker_resources"]["closed"]);self.assertEqual(self.gpu.transitions,1)

    def test_tune_error_closes_GPU(self):
        with self.cli_fixture("tune") as (args,store,engine):
            self.tuning.tune_retrieval.side_effect=RuntimeError("fictional tuner failure")
            with patch.object(self.d,"CrossEncoderReranker",return_value=self.gpu),self.assertRaisesRegex(RuntimeError,"tuner failure"):
                self.c.cmd_tune(args)
            self.assertTrue(self.gpu.closed);self.assertEqual(self.gpu.transitions,1)

    def test_tune_metadata_hashes_reflect_only_profiles_using_GPU(self):
        with self.cli_fixture("tune") as (args,store,engine):
            reports={"used":{"run_metadata":{"reranker_model":self.description["reranker_model"]}},"unused":{"run_metadata":{"reranker_model":None}}}
            manifest={"backend_inventory":{},"report_hashes":{},"manifest_hash":"old"}
            result={"development_reports":reports,"configs":{"used":{},"unused":{}},"baseline_selection":{},"manifest":manifest}
            self.tuning.tune_retrieval.return_value=result
            with patch.object(self.d,"CrossEncoderReranker",return_value=self.gpu):self.c.cmd_tune(args)
            self.assertEqual(reports["used"]["run_metadata"]["reranker_execution_hash"],self.description["reranker_execution_hash"])
            self.assertNotIn("reranker_execution_hash",reports["unused"]["run_metadata"])
            self.assertEqual(manifest["report_hashes"],{name:self.p.json_hash(report) for name,report in reports.items()})
            self.assertEqual(manifest["manifest_hash"],self.p.json_hash({key:value for key,value in manifest.items() if key!="manifest_hash"}))
            self.assertTrue(self.gpu.closed);self.assertEqual(self.gpu.transitions,1)


if __name__=="__main__":unittest.main()
