# Databricks notebook source
# MAGIC %md
# MAGIC # Lighthouse prompt registry and synthetic claim settlement evaluation
# MAGIC Synthetic records only. Select an approved Unity Gateway model service in this workspace.

# COMMAND ----------
import json, logging, os, sys
from pathlib import Path
from mlflow.genai.scorers import Correctness, ExpectationsGuidelines, Safety

PROJECT=Path.cwd()
sys.path.insert(0,str(PROJECT/"src"))
import mlflow
from lighthouse_poc.agent import UnityGatewayChatClient,ClaimSettlementAssistant
from lighthouse_poc.evaluation import MLflowEvaluationService
from lighthouse_poc.prompt_registry import MLflowPromptRegistry
from lighthouse_poc.logging_config import configure_logging

logger=configure_logging()
dbutils.widgets.text("prompt_name","main.default.claim_settlement_review")
dbutils.widgets.text("model_service","system.ai.<approved-model-service>")
dbutils.widgets.text("experiment_name","/Shared/lighthouse-claim-settlement-poc")
dbutils.widgets.text("application_version","0.1.0")
dbutils.widgets.text("prompt_version","1")
dbutils.widgets.text("register_new_prompt","false")
name=dbutils.widgets.get("prompt_name").strip()
model_service=dbutils.widgets.get("model_service").strip()
if not model_service or "<" in model_service: raise ValueError("Set model_service to an approved Unity Gateway service.")
mlflow.set_experiment(dbutils.widgets.get("experiment_name").strip())
registry=MLflowPromptRegistry(logger)

# COMMAND ----------
# To create a version, set register_new_prompt=true and supply a change note below.
# Otherwise the evaluation loads the explicitly selected version and will not create a new one.
if dbutils.widgets.get("register_new_prompt").strip().lower()=="true":
    template=(PROJECT/"prompts"/"claim_settlement_v1.txt").read_text(encoding="utf-8")
    prompt=registry.register(name,template,"Synthetic claim settlement preparation prompt",
                             {"use_case":"claim_settlement_preparation","non_decisional":"true"})
    prompt_version=prompt.version
else:
    prompt_version=dbutils.widgets.get("prompt_version").strip()
print({"prompt_name":name,"prompt_version":prompt_version})

# COMMAND ----------
with (PROJECT/"data"/"golden_eval_dataset.jsonl").open(encoding="utf-8") as f:
    records=[json.loads(line) for line in f if line.strip()]
model=UnityGatewayChatClient(model_service,logger)
assistant=ClaimSettlementAssistant(registry,model,logger,name,prompt_version,
                      dbutils.widgets.get("application_version"))
def predict_fn(case_id:str,product_segment:str,claim_narrative:str,
               submitted_documents:list,policy_or_benefit_context:str):
    return assistant.predict(case_id,product_segment,claim_narrative,submitted_documents,
                             policy_or_benefit_context)
manifest={"application_version":dbutils.widgets.get("application_version"),
          "prompt_name":name,"prompt_version":prompt_version,"model_service":model_service,
          "dataset_version":"1.0.0","scorer_version":"correctness-safety-claims-guidelines-v1"}
judge_model=os.getenv("MLFLOW_GENAI_JUDGE_DEFAULT_MODEL")
judge_options={"model":judge_model} if judge_model else {}
scorers=[Correctness(**judge_options),Safety(**judge_options),ExpectationsGuidelines(**judge_options)]
manifest["judge_model"]=judge_model or "workspace default"
result=MLflowEvaluationService(logger).run(records,predict_fn,scorers,manifest)
display(result.tables)
print("Review all cases and traces. This notebook never auto-publishes a prompt alias.")

