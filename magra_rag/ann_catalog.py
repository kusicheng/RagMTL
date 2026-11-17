                                                                                     
from contextlib import suppress
import importlib
import json
import math
import os
from pathlib import Path
import re
import sys
from .ann import (FAISS_VERSION, FAISS_WHEEL_HASHES, HNSWConfig, ResourceGuard,
                  FaissHNSWDenseRetriever, _hash, _sha, _runtime, _inside)
from .index_integrity import effective_query_scope

CATALOG_LIMIT = 262_144
THREAD_PROFILE = {"FAISS_threads":1, "Torch_intraop_threads":1,
                  "Torch_interop_threads":1, "numerical_pool_threads":1, "device":"cpu"}


def _json_file(path):
    if Path(path).stat().st_size > CATALOG_LIMIT:raise ValueError("ANN metadata256KB cap exceeded")
    with Path(path).open("rb") as stream:raw=stream.read(CATALOG_LIMIT+1)
    if len(raw)>CATALOG_LIMIT:raise ValueError("ANN metadata256KB cap exceeded")
    return json.loads(raw,parse_constant=lambda _:(_ for _ in ()).throw(ValueError("Nonfinite ANN metadata")))


def _runtime_config(value):
    if not isinstance(value,dict) or set(value)!={"site","pin","python_files"} or not isinstance(value["site"],str):raise ValueError("Explicit ANN site/native/wrapper pin configuration required")
    site=Path(value["site"]).resolve();pin=value["pin"];files=value["python_files"]
    if (not site.is_dir() or not isinstance(pin,dict) or set(pin)!={"version","wheel_sha256","binary_files"}
            or pin["version"]!=FAISS_VERSION or not isinstance(pin["wheel_sha256"],str) or pin["wheel_sha256"] not in FAISS_WHEEL_HASHES
            or not isinstance(pin["binary_files"],list) or not 1<=len(pin["binary_files"])<=64
            or not isinstance(files,dict) or not 1<=len(files)<=128):raise ValueError("Bounded supported ANN runtime required")
    if not {"faiss/__init__.py","faiss/loader.py","faiss/swigfaiss.py"}<=set(files):raise ValueError("Generic FAISS wrapper pins required")
    for name,expected in files.items():
        if not isinstance(name,str) or not isinstance(expected,dict):raise ValueError("Malformed wrapper pin")
        original=site/name;target=original.resolve()
        if (not isinstance(name,str) or not name.startswith("faiss/") or not name.endswith(".py")
                or not target.is_relative_to(site) or original.is_symlink() or any(parent.is_symlink() for parent in original.parents if parent!=site)
                or set(expected)!={"bytes","sha256"} or type(expected["bytes"])is not int or not 0<=expected["bytes"]<=16_000_000
                or not isinstance(expected["sha256"],str) or not re.fullmatch(r"[0-9a-f]{64}",expected["sha256"])):raise ValueError("Exact bounded local wrapper pin required")
    for item in pin["binary_files"]:
        if (not isinstance(item,dict) or set(item)!={"name","path","sha256"}
                or not isinstance(item["path"],str) or not Path(item["path"]).resolve().is_relative_to(site)
                or not isinstance(item["sha256"],str) or not re.fullmatch(r"[0-9a-f]{64}",item["sha256"])):raise ValueError("Native ANN file must belong to declared site")
    return site


def read_runtime(path):
    value=_json_file(path);_runtime_config(value);return value


def describe_catalog(path,budget,*,model_id,queries):
    pass                                                                               
    path=_inside(budget,path,file=True);catalog=_json_file(path)
    required={"format","runtime","ef_search","internal_candidates","artifacts"}
    if not isinstance(catalog,dict) or set(catalog)!=required or catalog["format"]!="magra-hnsw-catalog-v1":raise ValueError("Explicit ANN catalog required")
    _runtime_config(catalog["runtime"])
    for key,low,high in (("ef_search",1,65536),("internal_candidates",0,4096)):
        if type(catalog[key])is not int or not low<=catalog[key]<=high:raise ValueError("Invalid ANN search configuration")
    if not isinstance(catalog["artifacts"],list) or not 1<=len(catalog["artifacts"])<=128:raise ValueError("Bounded explicit ANN artifact list required")
    entries=[];scopes=set()
    for item in catalog["artifacts"]:
        if not isinstance(item,dict) or set(item)!={"path","artifact_hash","source","id_prefix"}:raise ValueError("Exact published ANN artifact entry required")
        folder=_inside(budget,item["path"]);manifest=_json_file(_inside(budget,folder/"manifest.json",file=True))
        actual=_hash({k:v for k,v in manifest.items() if k!="artifact_hash"})
        if (manifest.get("format")!="magra-hnsw-flat-v1" or manifest.get("status")!="complete"
                or actual!=item["artifact_hash"] or manifest.get("artifact_hash")!=actual
                or manifest.get("model_id")!=model_id):raise ValueError("Published ANN identity/model differs")
        config=HNSWConfig(**manifest["config"]);scope=(item["source"],item["id_prefix"])
        if scope!=(config.source,config.id_prefix) or scope in scopes or config.threads!=1:raise ValueError("Unique exact one-thread artifact scopes required")
        scopes.add(scope);entries.append({**item,"path":str(folder),"manifest":manifest})
    for query in queries:
        scope=effective_query_scope(query)
        if (scope["source"],scope["id_prefix"]) not in scopes:raise ValueError("No ANN artifact for exact query source/prefix scope")
    identity={"format":"magra-hnsw-backend-identity-v1","catalog_sha256":_sha(path),
              "runtime_sha256":_hash(catalog["runtime"]),"thread_profile":THREAD_PROFILE,
              "ef_search":catalog["ef_search"],"internal_candidates":catalog["internal_candidates"],
              "artifacts":[{k:e[k] for k in ("artifact_hash","source","id_prefix")} for e in entries]}
    identity["backend_hash"]=_hash(identity)
    return {"catalog_path":str(path),"identity":identity,"catalog":catalog,"entries":entries}


class LoadedRuntime:
    def __init__(self,configuration,budget,*,serving=False,guard=None):
        self.configuration=configuration
        self.guard=guard or ResourceGuard(budget);self.guard(ram_reserve=64_000_000)
        site=_runtime_config(configuration);self.limits=[];self.torch=None;self.constructor_threads=None
        self.validate_files()
        os.environ["FAISS_OPT_LEVEL"]="generic"
        if str(site) not in sys.path:sys.path.insert(0,str(site))
        try:
            self.np=importlib.import_module("numpy");self.faiss=importlib.import_module("faiss")
            self.poolctl=importlib.import_module("threadpoolctl")
            if Path(self.faiss.__file__).resolve()!=(site/"faiss/__init__.py").resolve():raise ValueError("Imported FAISS package differs from pinned site")
            if getattr(sys.modules.get("faiss.loader"),"opt_level",None)!="generic":raise ValueError("Pinned generic FAISS selector required")
            self.identity=_runtime(self.faiss,configuration["pin"])
            if serving:
                self.guard(ram_reserve=3_500_000_000)
                self.torch=importlib.import_module("torch")
                if self.torch.get_num_interop_threads()!=1:self.torch.set_num_interop_threads(1)
            self.refresh()
        except BaseException:self.close();raise

    def validate_files(self):
                                                                               
                                                                           
        configuration=self.configuration;site=_runtime_config(configuration)
        actual={p.relative_to(site).as_posix() for p in (site/"faiss").rglob("*.py")}
        if actual!=set(configuration["python_files"]):raise ValueError("Declared complete FAISS Python inventory differs")
        for name,pin in configuration["python_files"].items():
            target=site/name
            if target.stat().st_size!=pin["bytes"] or _sha(target)!=pin["sha256"]:raise ValueError("FAISS wrapper bytes differ")
        for item in configuration["pin"]["binary_files"]:
            if _sha(item["path"])!=item["sha256"]:raise ValueError("FAISS native bytes differ")
        if hasattr(self,"poolctl"):
            fresh=self.poolctl.ThreadpoolController().info()
            if len(fresh)>32 or any(p["num_threads"]!=1 for p in fresh):raise ValueError("Boundary native pool inventory differs from thread1 profile")

    def refresh(self,constructor_threads=None):
        if constructor_threads is not None:self.constructor_threads=constructor_threads
        if self.torch is not None:self.torch.set_num_threads(1)
        self.faiss.omp_set_num_threads(1)
        self.controller=self.poolctl.ThreadpoolController()
        self.limits.append(self.controller.limit(limits=1));self.check()

    def check(self):
        values=self.controller.info()
        if len(values)>32 or any(p["num_threads"]!=1 for p in values):raise ValueError("Actual ANN numerical pools must remain thread1")
        if self.torch is not None and (self.torch.get_num_threads()!=1 or self.torch.get_num_interop_threads()!=1):raise ValueError("Actual ANN Torch runtime must remain thread1")
        return values

    def profile(self):
        return {**THREAD_PROFILE,"Torch_intraop_threads":self.torch.get_num_threads() if self.torch else None,
                "Torch_interop_threads":self.torch.get_num_interop_threads() if self.torch else None,
                "constructor_reported_logical_threads":self.constructor_threads,
                "actual_pools":[{"library":Path(p["filepath"]).name,"threads":p["num_threads"],"user_api":p["user_api"],"internal_api":p["internal_api"]} for p in self.check()],
                "physical_affinity_enforced":False,"physical_affinity_policy":"Caller-managed; numerical thread1 is checked, OS affinity is not claimed.","GPU_used":False}

    def close(self):
        for limit in reversed(self.limits):
            with suppress(Exception):limit.restore_original_limits()
        self.limits=[]


class RoutedHNSWDenseRetriever:
    def __init__(self,exact,description,runtime):
        self.store,self.model_id=exact.store,exact.model_id;self.runtime=runtime;self.description=description;self.backends={}
        try:
            for entry in description["entries"]:
                self.backends[(entry["source"],entry["id_prefix"])]=FaissHNSWDenseRetriever(exact,entry["path"],
                    expected_manifest_hash=entry["artifact_hash"],faiss_module=runtime.faiss,numpy_module=runtime.np,
                    runtime_pin=description["catalog"]["runtime"]["pin"],ef_search=description["catalog"]["ef_search"],
                    internal_candidates=description["catalog"]["internal_candidates"],guard=runtime.guard)
            runtime.check()
        except BaseException:runtime.close();raise

    def _route(self,source,id_prefix):
        try:return self.backends[(source,id_prefix)]
        except KeyError:raise ValueError("No ANN artifact for exact query source/prefix scope") from None

    def search(self,question,*,k=20,allowed_ids=None,exclude_ids=None,source=None,id_prefix=None):
        self.runtime.check();backend=self._route(source,id_prefix)
        result=backend.search(question,k=k,allowed_ids=allowed_ids,exclude_ids=exclude_ids,source=source,id_prefix=id_prefix)
        self.runtime.check();return result

    def index_fingerprint(self,*,scopes):
        self.runtime.check();grouped={}
        for scope in scopes:
            key=(scope.get("source"),scope.get("id_prefix"));self._route(*key);grouped.setdefault(key,[]).append(scope)
        rows=[self.backends[key].index_fingerprint(scopes=values) for key,values in sorted(grouped.items(),key=lambda kv:repr(kv[0]))]
        value={"format":"magra-routed-hnsw-query-index-v1","backend_hash":self.description["identity"]["backend_hash"],"indexes":rows}
        self.runtime.check();return {**value,"fingerprint_hash":_hash(value)}

    def close(self):self.runtime.close()
