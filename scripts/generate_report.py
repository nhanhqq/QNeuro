#!/usr/bin/env python3
import os
import json
import glob
import time
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

ROOT = Path("/home/namphuongtran9196/intel_project/QNeuro")
OUTROOT = ROOT / "results" / "all_quantum"
CONFIG_PATH = ROOT / "configs" / "dataset_sampling_rates.json"

SEEDS = [7, 17, 27, 37, 47]
FOLDS = [0, 1, 2]

def get_datasets():
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r") as f:
            return list(json.load(f).keys())
    return [
        "2014001", "CHB-MIT", "CHSZ", "COG-BCI", "FACED",
        "ISRUC-S3", "PhysioNet-MI", "SEED", "STEW", "Sleep-EDF-20"
    ]

def get_system_stats():
    gpu_info = "N/A"
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        if res.returncode == 0:
            lines = res.stdout.strip().split("\n")
            gpu_info = " | ".join([f"🎮 {l.split(',')[0].strip()} ({l.split(',')[1].strip()}MiB / {l.split(',')[2].strip()}MiB, Util: {l.split(',')[3].strip()}%)" for l in lines if l.strip()])
    except Exception:
        pass
    
    mem_info = "N/A"
    try:
        with open("/proc/meminfo") as f:
            mem = {}
            for line in f:
                parts = line.split(":")
                if len(parts) == 2:
                    mem[parts[0].strip()] = int(parts[1].strip().split()[0])
            total_gb = mem.get("MemTotal", 0) / (1024 * 1024)
            avail_gb = mem.get("MemAvailable", 0) / (1024 * 1024)
            used_gb = total_gb - avail_gb
            mem_info = f"🧠 RAM: {used_gb:.1f}GB / {total_gb:.1f}GB (Free: {avail_gb:.1f}GB)"
    except Exception:
        pass
    return gpu_info, mem_info

def parse_all_results():
    datasets = get_datasets()
    total_expected = len(datasets) * len(SEEDS) * len(FOLDS)
    completed_jobs = 0
    running_jobs = 0
    
    # Check running processes
    running_pids = []
    try:
        ps_res = subprocess.run(["ps", "-eo", "pid,cmd"], stdout=subprocess.PIPE, text=True)
        for line in ps_res.stdout.split("\n"):
            if "train_chsz.py" in line and "python" in line:
                running_pids.append(line)
    except Exception:
        pass

    results_by_dataset = {}

    for data in datasets:
        d_dir = OUTROOT / data / "runs"
        d_results = {
            "completed_folds": 0,
            "total_folds": len(SEEDS) * len(FOLDS),
            "seed_metrics": {},
            "all_acc": [],
            "all_bacc": [],
            "all_f1": [],
            "all_loss": [],
            "active_folds": []
        }
        
        for seed in SEEDS:
            d_results["seed_metrics"][seed] = []
            for fold in FOLDS:
                run_dir = d_dir / f"quantum_fold{fold}_seed{seed}"
                target_json = run_dir / "target_selected_test.json"
                
                # Check if running
                is_running = any(f"--data {data}" in p and f"--fold {fold}" in p and f"--seed {seed}" in p for p in running_pids)
                if is_running:
                    running_jobs += 1
                    # check current epoch if epochs.csv exists
                    curr_epoch = 0
                    if (run_dir / "epochs.csv").exists():
                        try:
                            df_ep = pd.read_csv(run_dir / "epochs.csv")
                            curr_epoch = len(df_ep)
                        except Exception:
                            pass
                    d_results["active_folds"].append(f"s{seed}f{fold} (ep {curr_epoch}/100)")
                
                if target_json.exists():
                    try:
                        with open(target_json, "r") as f:
                            metrics = json.load(f)
                        completed_jobs += 1
                        d_results["completed_folds"] += 1
                        acc = metrics.get("accuracy", 0.0) * 100
                        bacc = metrics.get("balanced_accuracy", 0.0) * 100
                        f1 = metrics.get("macro_f1", 0.0) * 100
                        loss = metrics.get("loss", 0.0)
                        
                        d_results["seed_metrics"][seed].append({
                            "fold": fold,
                            "acc": acc,
                            "bacc": bacc,
                            "f1": f1,
                            "loss": loss
                        })
                        d_results["all_acc"].append(acc)
                        d_results["all_bacc"].append(bacc)
                        d_results["all_f1"].append(f1)
                        d_results["all_loss"].append(loss)
                    except Exception:
                        pass
        
        results_by_dataset[data] = d_results

    return {
        "datasets": results_by_dataset,
        "total_expected": total_expected,
        "completed_jobs": completed_jobs,
        "running_jobs": running_jobs,
        "active_processes": len(running_pids)
    }

def format_report():
    data_summary = parse_all_results()
    gpu_info, mem_info = get_system_stats()
    
    total = data_summary["total_expected"]
    done = data_summary["completed_jobs"]
    running = data_summary["running_jobs"]
    pct = (done / total) * 100 if total > 0 else 0
    
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Progress bar
    bar_len = 25
    filled_len = int(bar_len * done // total) if total > 0 else 0
    bar = "█" * filled_len + "░" * (bar_len - filled_len)
    
    lines = []
    lines.append(f"# 🚀 BÁO CÁO TIẾN ĐỘ BENCHMARK QUANTUM 10 DATASET (5 SEEDS × 3 FOLDS)")
    lines.append(f"**⏰ Thời gian cập nhật:** `{now_str}`")
    lines.append(f"**🖥️ Hệ thống:** {gpu_info} | {mem_info}")
    lines.append(f"")
    lines.append(f"### 📊 Tổng Quan Tiến Độ")
    lines.append(f"- **Tiến độ hoàn thành:** `[{bar}]` **{done}/{total} jobs** ({pct:.1f}%)")
    lines.append(f"- **Đang thực thi:** ⚡ **{running} workers** song song")
    lines.append(f"- **Còn lại:** ⏳ **{max(0, total - done - running)} jobs**")
    lines.append(f"")
    lines.append(f"---")
    lines.append(f"### 📈 Bảng Thống Kê Tổng Hợp (Tất Cả 10 Datasets)")
    lines.append(f"")
    lines.append(f"| 🏷️ Dataset | 🔄 Tiến độ | 🎯 Accuracy (Mean ± Std) | ⚖️ Bal. Acc (Mean ± Std) | 🏆 Macro F1 (Mean ± Std) | 📉 Loss | ⚡ Trạng thái |")
    lines.append(f"| :--- | :---: | :---: | :---: | :---: | :---: | :--- |")
    
    for dname, info in data_summary["datasets"].items():
        n_done = info["completed_folds"]
        n_total = info["total_folds"]
        
        if n_done == n_total:
            status_tag = "✅ **Hoàn thành**"
        elif len(info["active_folds"]) > 0:
            status_tag = f"⚡ Running: `{' '.join(info['active_folds'])}`"
        elif n_done > 0:
            status_tag = "⏳ Đang chạy các fold tiếp"
        else:
            status_tag = "💤 Chờ lượt"
            
        if len(info["all_acc"]) > 0:
            acc_str = f"**{np.mean(info['all_acc']):.2f}%** ± {np.std(info['all_acc']):.2f}"
            bacc_str = f"**{np.mean(info['all_bacc']):.2f}%** ± {np.std(info['all_bacc']):.2f}"
            f1_str = f"**{np.mean(info['all_f1']):.2f}%** ± {np.std(info['all_f1']):.2f}"
            loss_str = f"{np.mean(info['all_loss']):.4f}"
        else:
            acc_str = "—"
            bacc_str = "—"
            f1_str = "—"
            loss_str = "—"
            
        lines.append(f"| **{dname}** | `{n_done}/{n_total}` | {acc_str} | {bacc_str} | {f1_str} | {loss_str} | {status_tag} |")
        
    lines.append(f"")
    lines.append(f"---")
    lines.append(f"### 🔬 Bảng Chi Tiết Theo Seed & Fold (Các Dataset Đã/Đang Chạy)")
    lines.append(f"")
    
    for dname, info in data_summary["datasets"].items():
        if info["completed_folds"] == 0 and len(info["active_folds"]) == 0:
            continue
        lines.append(f"#### 📁 Dataset: `{dname}` (Hoàn tất {info['completed_folds']}/{info['total_folds']} Folds)")
        lines.append(f"| 🎲 Seed | Fold 0 (Acc / F1) | Fold 1 (Acc / F1) | Fold 2 (Acc / F1) | 🎯 Seed Mean Acc | 🏆 Seed Mean F1 |")
        lines.append(f"| :---: | :---: | :---: | :---: | :---: | :---: |")
        
        for seed in SEEDS:
            folds_res = {f["fold"]: f for f in info["seed_metrics"][seed]}
            f_strs = []
            seed_accs = []
            seed_f1s = []
            for fold in FOLDS:
                if fold in folds_res:
                    f_res = folds_res[fold]
                    f_strs.append(f"{f_res['acc']:.2f}% / {f_res['f1']:.2f}%")
                    seed_accs.append(f_res['acc'])
                    seed_f1s.append(f_res['f1'])
                else:
                    act = [a for a in info["active_folds"] if f"s{seed}f{fold}" in a]
                    if act:
                        f_strs.append(f"⚡ {act[0]}")
                    else:
                        f_strs.append("⏳ Chờ")
            
            s_acc_mean = f"**{np.mean(seed_accs):.2f}%**" if seed_accs else "—"
            s_f1_mean = f"**{np.mean(seed_f1s):.2f}%**" if seed_f1s else "—"
            lines.append(f"| **Seed {seed}** | {f_strs[0]} | {f_strs[1]} | {f_strs[2]} | {s_acc_mean} | {s_f1_mean} |")
        lines.append(f"")

    return "\n".join(lines), (done >= total and total > 0)

if __name__ == "__main__":
    report_text, is_done = format_report()
    print(report_text)
