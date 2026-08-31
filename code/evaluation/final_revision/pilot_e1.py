import os, yaml, json, sys
os.environ["DER_LLM_STRICT"]="1"
from pathlib import Path
import numpy as np, pandas as pd
from ...llm_serving.local_lora import configure_default
from ...Multi_AI_Agent.adapter import DERSecAgentDetector
from ...simulation.feeder import StubFeeder
from ...simulation.harness import run_scenario
from ..physical_curves import sweep
from code.llm_serving import model_paths as _MP

QB=_MP.QWEN_BASE
QA="code/finetuning/results/20260519-144102-lora_qwen25_7b_local/adapter"
llm=configure_default(QB,QA,max_new_tokens=128)
assert llm._try_load(), llm._load_fail_reason
man=pd.read_csv("code/configs/ijcip_revision_r1r2_20260805/scenario_manifest.csv")
sel=man[man.scenario_id.isin(["rv13_fdi_med_medium_nom_medpen_inv",
    "rv13_spoof_med_medium_nom_medpen_inv","rv13_dos_med_medium_nom_medpen_inv",
    "rv13_replay_med_medium_nom_medpen_inv"])]
rows=[]
for mode in ("legacy_class_override","safety_projection"):
    for _,r in sel.iterrows():
        cfg=yaml.safe_load(Path(r.config_path).read_text())
        det=DERSecAgentDetector(policy_mode=mode,k_setting=1,compact_prompt=True,
                                llm_invoke_interval_s=600.0)
        det.name=f"m1q_{mode}"
        f=StubFeeder(monitored_buses=cfg["monitored_buses"],ders=cfg["ders"],
                     base_load_kw=float(cfg.get("base_load_kw",1000.0)))
        rd=run_scenario(r.config_path,det,0,
            out_root=f"code/results/ijcip_final_revision/pilot_e1/{mode}",feeder=f,
            extra_manifest={"policy_mode":mode,"llm_backend":"qwen_lora_real",
                            "llm_adapter_sha":llm.adapter_sha()})
        ph=sweep(rd,np.linspace(0,1,11)); a=ph[ph.threshold.between(.49,.51)].iloc[0]
        prop=fin=None; src=None
        for l in (rd/"decisions.jsonl").read_text().splitlines():
            d=json.loads(l).get("decision_trace")
            if d and d.get("llm_traces"):
                prop=d.get("proposed_action"); fin=d.get("final_action")
                src=(d.get("coordinator_reason") or "")[:32]
        rows.append({"mode":mode,"scenario":r.scenario_id.replace("rv13_","")[:22],
                     "proposed":prop,"executed":fin,"ens":round(float(a.ens_kwh),2),
                     "curt":round(float(a.curt_kwh),2),"src":src})
        print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv("code/results/ijcip_final_revision/pilot_e1/pilot_summary.csv",index=False)
