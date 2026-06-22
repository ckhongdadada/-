"""整理实验结果为统一格式"""
import json, os
from pathlib import Path

BASE = Path("C:/Users/28414/Desktop/机器学习—图像分类-期末汇报/output")
RESULTS = Path("C:/Users/28414/Desktop/机器学习—图像分类-期末汇报/results")
RESULTS.mkdir(exist_ok=True)

def load_agg(group, rho, base=None):
    if base is None: base = BASE
    f = base / group / f"rho{rho}" / "aggregated_results.json"
    return json.load(open(f)) if f.exists() else None

def save(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f: json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  {path}")

def extract(group, rho_list, seeds=5, base=None):
    result = {"results": {}, "full_supervised": {}}
    strategies = []
    for rho in rho_list:
        d = load_agg(group, rho, base)
        if d:
            strategies = [k for k in d if k != "full_supervision"]
            result["results"][f"rho={rho}"] = {}
            for s in strategies:
                result["results"][f"rho={rho}"][s] = {"f1": round(d[s]["final_f1_mean"], 4), "std": round(d[s].get("final_f1_std", 0), 4)}
            fs = d.get("full_supervision", {}).get("f1", 0)
            if fs: result["full_supervised"][f"rho={rho}"] = round(fs, 4)
    result["strategies"] = strategies
    result["seeds"] = seeds
    return result

# === CIFAR-10 标准AL ===
print("1. std_al")
d = extract("std_al", [1,5,10,20,50,100], seeds=5)
d["dataset"] = "cifar10"; d["experiment"] = "std_al"
save(RESULTS / "cifar10" / "std_al.json", d)

# === CIFAR-10 AL+SSL ===
print("2. al_ssl")
d = extract("al_ssl", [1,5,10,20,50,100], seeds=5)
d["dataset"] = "cifar10"; d["experiment"] = "al_ssl"
save(RESULTS / "cifar10" / "al_ssl.json", d)

# === CIFAR-10 创新AL+SSL ===
print("3. innovative_al_ssl")
d = extract("innovative_al_ssl", [1,5,10,20,50,100], seeds=5)
d["dataset"] = "cifar10"; d["experiment"] = "innovative_al_ssl"
save(RESULTS / "cifar10" / "innovative_al_ssl.json", d)

# === 消融实验 ===
print("4. ablation")
ablation = {"dataset": "cifar10", "experiment": "ablation", "results": {}}
groups = {
    "baseline": "std_al",
    "innov_al_base_ssl": "innovative_al_ssl_basic",
    "base_al_innov_ssl": "al_ssl_innovative",
    "innov_al_innov_ssl": "innovative_al_ssl"
}
for rho in [10, 50]:
    key = f"rho={rho}"
    ablation["results"][key] = {}
    for label, group in groups.items():
        d = load_agg(group, rho)
        if d:
            for s in d:
                if s != "full_supervision":
                    ablation["results"][key][f"{label}/{s}"] = round(d[s]["final_f1_mean"], 4)
save(RESULTS / "cifar10" / "ablation.json", ablation)

# === 联合分布感知 ===
print("5. joint_distribution")
joint = {"dataset": "cifar10", "experiment": "joint_distribution", "results": {}}
for cfg in ["labeled_only", "joint_r0", "joint_r3", "joint_r5", "joint_r7"]:
    for rho in [1, 5, 10, 20, 50, 100]:
        d = load_agg(cfg, rho, base=BASE / "progressive_joint")
        if d:
            key = f"rho={rho}"
            if key not in joint["results"]: joint["results"][key] = {}
            for s in d:
                if s != "full_supervision":
                    joint["results"][key][f"{cfg}/{s}"] = round(d[s]["final_f1_mean"], 4)
save(RESULTS / "cifar10" / "joint_distribution.json", joint)

# === 渐进式SSL ===
print("6. progressive_ssl")
prog = {"dataset": "cifar10", "experiment": "progressive_ssl", "results": {}}
for cfg in ["no_ssl", "base_ssl", "innov_ssl", "progressive_r3", "progressive_r5", "progressive_r7"]:
    for rho in [1, 5, 10, 20, 50, 100]:
        d = load_agg(cfg, rho, base=BASE / "progressive_ssl_full")
        if d:
            key = f"rho={rho}"
            if key not in prog["results"]: prog["results"][key] = {}
            for s in d:
                if s != "full_supervision":
                    prog["results"][key][f"{cfg}/{s}"] = round(d[s]["final_f1_mean"], 4)
save(RESULTS / "cifar10" / "progressive_ssl.json", prog)

# === LDAM ===
print("7. ldam")
ldam = json.load(open(BASE / "ldam_baseline" / "ldam_results.json"))
data = {"dataset": "cifar10", "experiment": "ldam", "results": {}}
for k, v in ldam.items():
    data["results"][k] = {"f1": round(v["mean_f1"], 4), "std": round(v.get("std_f1", 0), 4)}
save(RESULTS / "cifar10" / "ldam.json", data)

# === CB/Focal ===
print("8. cb_focal")
cb = {"dataset": "cifar10", "experiment": "cb_focal", "results": {}}
for rho in [1, 5, 10, 20, 50, 100]:
    for loss in ["cb", "focal"]:
        f = BASE / "cb_focal_baseline" / "cifar10" / f"rho{rho}" / loss / "aggregated_results.json"
        if f.exists():
            d = json.load(open(f))
            cb["results"][f"rho={rho}/{loss}"] = {s: round(d[s]["final_f1_mean"], 4) for s in d if s != "full_supervision"}
save(RESULTS / "cifar10" / "cb_focal.json", cb)

# === ResNet-18 ===
print("9. resnet18")
d = extract("resnet18_full", [1,5,10,20,50,100], seeds=3)
d["dataset"] = "cifar10"; d["experiment"] = "resnet18"
save(RESULTS / "cifar10" / "resnet18.json", d)

# === CIFAR-100 ===
print("10. cifar100")
c100 = {"dataset": "cifar100", "seeds": 1, "results": {}}
for group in ["std_al", "al_ssl", "innovative_al_ssl"]:
    for rho in [1, 10, 50]:
        d = load_agg(group, rho, base=BASE / "cifar100")
        if d:
            key = f"{group}/rho={rho}"
            c100["results"][key] = {s: round(d[s]["final_f1_mean"], 4) for s in d if s != "full_supervision"}
            fs = d.get("full_supervision", {}).get("f1", 0)
            if fs: c100["results"][key]["full_supervised"] = round(fs, 4)
save(RESULTS / "cifar100" / "all.json", c100)

# === FashionMNIST ===
print("11. fashionmnist")
fm = {"dataset": "fashionmnist", "seeds": 3, "results": {}}
for group in ["std_al", "al_ssl"]:
    for rho in [1, 5, 10, 20, 50, 100]:
        d = load_agg(group, rho, base=BASE / "fashionmnist")
        if d:
            key = f"{group}/rho={rho}"
            fm["results"][key] = {s: round(d[s]["final_f1_mean"], 4) for s in d if s != "full_supervision"}
save(RESULTS / "fashionmnist" / "all.json", fm)

# === TML ===
print("12. tml")
tml = {"dataset": "cifar10", "experiment": "tml", "results": {}}
for model in ["lr", "rf"]:
    for rho in [1, 5, 10, 20, 50, 100]:
        f = BASE / "tml_validation" / f"{model}_cifar10_rho{rho}_results.json"
        if f.exists():
            d = json.load(open(f))
            tml["results"][f"{model}/rho={rho}"] = {s: {"f1": round(v["mean_f1"], 4)} for s, v in d.items()}
save(RESULTS / "tml" / "tml.json", tml)

# === SSL对比 ===
print("13. ssl_comparison")
ssl = {"dataset": "cifar10", "experiment": "ssl_comparison", "results": {}}
for cfg in ["no_ssl", "self_training", "flexmatch"]:
    for rho in [1, 5, 10, 20, 50, 100]:
        f = BASE / "ssl_comparison" / cfg / f"rho{rho}" / "aggregated_results.json"
        if f.exists():
            d = json.load(open(f))
            ssl["results"][f"{cfg}/rho={rho}"] = {s: round(d[s]["final_f1_mean"], 4) for s in d if s != "full_supervision"}
save(RESULTS / "cifar10" / "ssl_comparison.json", ssl)

# === 全局汇总 ===
print("14. summary")
summary = {
    "generated": "2026-06-22",
    "project": "低标注预算下面向不平衡图像分类的主动学习与半监督学习联合策略研究",
    "datasets": ["cifar10", "cifar100", "fashionmnist"],
    "key_findings": {
        "best_std_al_rho50": "margin 0.2680",
        "best_innovative_rho50": "class_aware_entropy 0.3129 (+21.0%)",
        "best_joint_rho50": "joint_r5/gap_aware_entropy_ssl 0.3126",
        "ablation_al_contribution": "+16.8% (innov_al_base_ssl vs baseline)",
        "ablation_ssl_contribution": "-2.5% (base_al_innov_ssl vs baseline)",
        "progressive_ssl_rho100": "+20.5% vs base SSL",
        "resnet18_entropy_rho50": 0.4737,
        "ldam_full_rho50": 0.6049
    }
}
save(RESULTS / "summary.json", summary)

print("\nDone!")
