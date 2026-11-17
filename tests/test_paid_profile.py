                                                                                     
from contextlib import closing
import json,os,sqlite3,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from magra_rag import configuration as cfg
from magra_rag.cli import build_parser,_generation_options,_generation_settings,_judge_client,cmd_doctor
from magra_rag.generation import OpenRouterM3,SpendLedger,GenerationError
from magra_rag.resources import ResourceBudget

SECRET="synthetic-paid-key-never-log"
OLD_COST=.039254385
EVIDENCE=[{"doc_id":"toy","sentence_id":0,"text":"An invented result."}]

class PaidProfileTests(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory(prefix="paid-core-fake-")
        self.addCleanup(tmp.cleanup);self.root=Path(tmp.name)
        self.path=self.root/"paid/runs/spend.sqlite"
        self.addCleanup(patch.stopall)
        patch.object(cfg,"PAID_LEDGER_PATH",self.path).start()
        self.env=self.root/"invented.env"
        self.env.write_text("ai-model-paid-key="+SECRET+"\nOPENROUTER_API_KEY=legacy\nkey=gpu\nOPENROUTER_MODEL=old/model:free\n",encoding="utf-8")
        self.budget=ResourceBudget(self.root/"owned-run")
        initial=SpendLedger(self.path,20,budget=ResourceBudget(self.root/"paid",disk_bytes=256000000,reserve_bytes=4000000))
        reservation=initial.reserve(.05);initial.settle(reservation,OLD_COST)

    def client(self,**kw):
        options={"data_root":self.budget.root,"budget":self.budget,"profile":"paid","api_key":SECRET,"env_file":None}
        options.update(kw);return OpenRouterM3(**options)

    def rows(self):
        with closing(sqlite3.connect(self.path)) as db:
            return db.execute("SELECT id,reserved,charged,state,created_at FROM spend ORDER BY rowid").fetchall()

    def test_paid_auto_precedence_and_explicit_free_are_separate(self):
        before=dict(os.environ)
        paid=cfg.load_openrouter_settings(env_file=self.env,environ={},profile="auto")
        self.assertEqual((paid.model_id,paid.api_key,paid.profile),(cfg.PAID_MODEL,SECRET,"paid"))
        free=cfg.load_openrouter_settings(env_file=self.env,environ={},profile="auto",model_id="other/free:free")
        self.assertEqual((free.api_key,free.profile),("legacy","legacy"))
        self.assertNotIn(SECRET,repr(paid));self.assertEqual(dict(os.environ),before)

    def test_explicit_credentials_skip_file_and_implicit_paid_detection(self):
        with patch.object(cfg,"_dotenv",side_effect=AssertionError("No environment file read")):
            settings=cfg.load_openrouter_settings(api_key=SECRET,env_file="unused.env",profile="auto",
                                                  environ={"ai-model-paid-key":"unrelated"})
            self.assertEqual((settings.profile,settings.model_id,settings.api_key),("legacy",cfg.DEFAULT_MODEL,SECRET))
            paid=cfg.load_openrouter_settings(api_key=SECRET,env_file="unused.env",profile="paid",environ={})
            self.assertEqual((paid.profile,paid.model_id),("paid",cfg.PAID_MODEL))
            with patch.dict(os.environ,{"ai-model-paid-key":"unrelated"},clear=True):
                client=OpenRouterM3(data_root=self.budget.root,budget=self.budget,api_key=SECRET,env_file="unused.env")
            self.assertFalse(client.paid_profile);self.assertEqual(client.model_id,cfg.DEFAULT_MODEL)

    def test_missing_empty_or_duplicate_paid_key_never_falls_back(self):
        self.env.write_text("OPENROUTER_API_KEY=legacy\nkey=gpu\n",encoding="utf-8")
        self.assertIsNone(cfg.load_openrouter_settings(env_file=self.env,environ={},profile="paid").api_key)
        self.env.write_text("ai-model-paid-key="+SECRET+"\n",encoding="utf-8")
        self.assertIsNone(cfg.load_openrouter_settings(env_file=self.env,environ={"ai-model-paid-key":""},profile="paid").api_key)
        self.env.write_text("ai-model-paid-key=x\nai-model-paid-key=y\n",encoding="utf-8")
        with self.assertRaises(ValueError):cfg.load_openrouter_settings(env_file=self.env,environ={},profile="paid")

    def test_existing_history_and_two_clients_share_one_ledger(self):
        original=self.rows();first=self.client();second=self.client()
        self.assertEqual(self.rows(),original);self.assertEqual(first.ledger.path,second.ledger.path)
        self.assertEqual(first.ledger.summary()["accounted_usd"],OLD_COST)
        self.assertEqual((first.model_id,first.prompt_price,first.output_price,first.max_tokens,first.reasoning_effort),
                         (cfg.PAID_MODEL,.15,.60,4096,"low"))
        pending=first.ledger.reserve(.001)
        with self.assertRaises(GenerationError):second.ledger.reserve(.001)
        first.ledger.settle(pending,.0005)
        self.assertAlmostEqual(second.ledger.summary()["accounted_usd"],OLD_COST+.0005)
        self.assertEqual(self.rows()[0],original[0])

    def test_over100_other_path_wrong_prices_fail_before_SQL_or_API(self):
        for options in ({"max_total_usd":101},{"shared_ledger_path":self.root/"new.sqlite"},
                        {"prompt_usd_per_million":0},{"completion_usd_per_million":1}):
            with self.subTest(options=options),patch("magra_rag.generation.sqlite3.connect",side_effect=AssertionError("No SQL")),patch("urllib.request.urlopen",side_effect=AssertionError("No API")),self.assertRaises(ValueError):
                self.client(**options)

    def test_missing_history_never_initializes(self):
        self.path.unlink()
        with self.assertRaises(GenerationError):self.client()
        self.assertFalse(self.path.exists())

    def test_deleted_history_after_construction_is_not_recreated(self):
        client=self.client();self.path.unlink()
        with self.assertRaises(sqlite3.OperationalError):client.ledger.reserve(.001)
        self.assertFalse(self.path.exists())

    def test_lower_cumulative_cap_includes_all_prior_spend(self):
        client=self.client(max_total_usd=.04)
        with self.assertRaises(GenerationError):client.ledger.reserve(.001)
        self.assertEqual(len(self.rows()),1)

    def fake_preflight(self,client):
        client._paid_remaining=1
        client._paid_prompt_capacity=1048576
        return {"id":cfg.PAID_MODEL,"supported_parameters":["response_format","reasoning"],"reasoning":{"supported_efforts":["low"]}}

    def test_positive_invalid_charge_survives_and_unknown_stops(self):
        client=self.client()
        def completion(cost):return {"model":cfg.PAID_MODEL,"usage":{"cost":cost},"choices":[{"finish_reason":"stop","message":{"content":"{}"}}]}
        with patch.object(client,"preflight",side_effect=lambda:self.fake_preflight(client)),patch.object(client,"_json_request",return_value=completion(.00001)):
            with self.assertRaises(GenerationError):client.generate("Invented question?",EVIDENCE)
        self.assertAlmostEqual(client.ledger.summary()["accounted_usd"],OLD_COST+.00001)
        with patch.object(client,"preflight",side_effect=lambda:self.fake_preflight(client)),patch.object(client,"_json_request",return_value=completion(None)):
            with self.assertRaisesRegex(GenerationError,"unknown"):client.generate("Another toy?",EVIDENCE)
        self.assertEqual(client.ledger.summary()["unconfirmed_requests"],1)
        with self.assertRaises(GenerationError):self.client()

    def test_paid_preflight_requires_consistent_finite_key_cap(self):
        client=self.client()
        row={"id":cfg.PAID_MODEL,"context_length":1048576,"pricing":{"prompt":"0.00000015","completion":"0.00000060"},"supported_parameters":["response_format","reasoning"]}
        for data in ({"limit":None,"usage":0,"limit_remaining":20},{"limit":101,"usage":1,"limit_remaining":100},{"limit":20,"usage":1,"limit_remaining":20}):
            with self.subTest(data=data),patch.object(client,"_json_request",side_effect=[{"data":[row]},{"data":data}]),self.assertRaises(GenerationError):client.preflight()
        with patch.object(client,"_json_request",side_effect=[{"data":[row]},{"data":{"limit":20,"usage":1,"limit_remaining":19}}]):client.preflight()
        self.assertEqual(client._paid_remaining,19)

    def test_doctor_and_judge_CLI_share_paid_profile_without_key_output(self):
        parser=build_parser()
        with patch.dict(os.environ,{},clear=True):
            args=parser.parse_args(["--data-root",str(self.budget.root),"doctor","--env-file",str(self.env)])
            result=cmd_doctor(args)
            self.assertEqual((result["model"],result["monthly_max_usd"]),(cfg.PAID_MODEL,100))
            self.assertIsNone(result["cumulative_max_usd"])
            self.assertNotIn(SECRET,json.dumps(result))
            args=parser.parse_args(["--data-root",str(self.budget.root),"judge-inputs","--inputs","toy.jsonl","--inputs-sha256","0"*64,"--env-file",str(self.env),"--out","toy-output.jsonl"])
            judge=_judge_client(args,self.budget)
        self.assertEqual(judge.ledger.path,self.path.resolve());self.assertEqual(judge.ledger.summary()["accounted_usd"],OLD_COST)

    def test_run_CLI_cap_rejected_and_explicit_free_compatible(self):
        args=build_parser().parse_args(["run","--bundle","toy","--generate","--env-file",str(self.env),"--max-usd","101","--out","toy-output"])
        with patch.dict(os.environ,{},clear=True),self.assertRaises(ValueError):_generation_options(args,_generation_settings(args))
        args=build_parser().parse_args(["run","--bundle","toy","--generate","--env-file",str(self.env),"--model","example/free:free","--out","toy-output"])
        with patch.dict(os.environ,{},clear=True):options=_generation_options(args,_generation_settings(args))
        self.assertEqual((options["profile"],options["prompt_usd_per_million"],options["completion_usd_per_million"]),("legacy",0,0))

    def test_paid_judge_unknown_billing_retains_reservation(self):
        from magra_benchmark.judge import M3GroundednessJudge,JudgeError
        client=self.client();judge=M3GroundednessJudge(client)
        prediction={"answer":"Invented result","abstained":False,"citations":[{"doc_id":"toy","sentence_id":0}]}
        with patch.object(client,"preflight",side_effect=lambda:self.fake_preflight(client)),patch.object(client,"_json_request",return_value={"usage":{"cost":None}}):
            with self.assertRaises(JudgeError):judge.judge({"id":"toy-query","question":"Invented question?"},prediction,EVIDENCE)
        self.assertEqual(client.ledger.summary()["unconfirmed_requests"],1)

if __name__=="__main__":unittest.main()
