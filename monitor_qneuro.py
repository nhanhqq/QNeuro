import os
import subprocess
import json
import time
import numpy as np

def get_status_report():
    datasets = {
        'seed': 15,
        'seediv': 15,
        'seedv': 16,
        'seedvii': 20
    }

    root = '/home/namphuongtran9196/intel_project/QNeuro'
    res = os.path.join(root, 'results')
    ablation_res = os.path.join(res, 'ablations')

    # Get active screen sessions
    screen_out = subprocess.run(['screen', '-ls'], stdout=subprocess.PIPE, text=True).stdout
    active_screens = set()
    scheduler_active = False
    for line in screen_out.splitlines():
        if 'qneuro_' in line and '(Detached)' in line:
            parts = line.strip().split()[0].split('.')
            if len(parts) > 1:
                active_screens.add(parts[1])
                if parts[1] == 'qneuro_scheduler':
                    scheduler_active = True

    # GPU status
    gpu_info = "N/A"
    try:
        gpu_res = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw', '--format=csv,noheader,nounits'],
            stdout=subprocess.PIPE, text=True
        ).stdout.strip()
        if gpu_res:
            parts = [p.strip() for p in gpu_res.split(',')]
            gpu_info = f"GPU Util: {parts[0]}% | VRAM: {parts[1]}/{parts[2]} MB | Temp: {parts[3]}°C | Power: {parts[4]}W"
    except Exception:
        pass

    # 1. Full Model LOSO Matrix
    matrix = {f'P{i}': {} for i in range(1, 21)}
    dataset_accs = {ds: [] for ds in datasets}
    dataset_completed_count = {ds: 0 for ds in datasets}
    dataset_running_count = {ds: 0 for ds in datasets}

    total_jobs = sum(datasets.values())
    total_completed = 0
    total_running = 0
    total_pending = 0

    for ds, count in datasets.items():
        for p in range(1, 21):
            target = f'P{p}'
            if p > count:
                matrix[target][ds] = '--'
                continue
                
            target_dir = os.path.join(res, ds, f'target_{target}')
            final_pt = os.path.join(target_dir, 'base', 'final_epoch.pt')
            metric_json = os.path.join(target_dir, 'metrics', '62ch_final_epoch.json')
            csv_file = os.path.join(target_dir, 'base', 'base_training.csv')
            item = f'{ds}_{target}'
            
            if os.path.exists(final_pt) and os.path.exists(metric_json):
                try:
                    with open(metric_json) as f:
                        data = json.load(f)
                        acc = data.get('accuracy', 0.0) * 100
                        dataset_accs[ds].append(acc)
                        matrix[target][ds] = f"🟢 {acc:.2f}%"
                except Exception:
                    matrix[target][ds] = "🟢 Done"
                dataset_completed_count[ds] += 1
                total_completed += 1
            elif f'qneuro_{item}' in active_screens:
                dataset_running_count[ds] += 1
                total_running += 1
                ep_info = ''
                if os.path.exists(csv_file):
                    try:
                        with open(csv_file) as f:
                            lines = f.readlines()
                            if len(lines) > 1:
                                cols = lines[-1].strip().split(',')
                                ep = cols[0]
                                te_acc = float(cols[7])*100 if len(cols)>7 else 0.0
                                ep_info = f"Ep {ep}/100 ({te_acc:.1f}%)"
                    except Exception:
                        pass
                matrix[target][ds] = f"🟡 {ep_info}" if ep_info else "🟡 Running"
            else:
                matrix[target][ds] = "--"
                total_pending += 1

    report = []
    report.append(f"### 📊 Bảng Thống Kê Ma Trận Task x Dataset - QNeuro ({time.strftime('%Y-%m-%d %H:%M:%S')})")
    report.append(f"- **Bộ điều phối (Scheduler)**: `{'🟢 Hoạt động (qneuro_scheduler)' if scheduler_active else '🔴 Đã dừng'}`")
    report.append(f"- **Tình trạng GPU**: `{gpu_info}`")
    report.append(f"- **Tổng tiến độ Full Model**: **{total_completed}/{total_jobs} ({total_completed/total_jobs*100:.1f}%)** Hoàn thành | **{total_running}** Đang chạy | **{total_pending}** Chờ chạy\n")

    report.append("#### 🏆 BẢNG 1: FULL MODEL (62-Channel LOSO)")
    report.append("| Task (Subject) | SEED (15 subjects) | SEED-IV (15 subjects) | SEED-V (16 subjects) | SEED-VII (20 subjects) |")
    report.append("| :--- | :---: | :---: | :---: | :---: |")

    for p in range(1, 21):
        target = f'P{p}'
        s_seed = matrix[target]['seed']
        s_seediv = matrix[target]['seediv']
        s_seedv = matrix[target]['seedv']
        s_seedvii = matrix[target]['seedvii']
        report.append(f"| **{target}** | {s_seed} | {s_seediv} | {s_seedv} | {s_seedvii} |")

    mean_std_cells = []
    for ds in ['seed', 'seediv', 'seedv', 'seedvii']:
        accs = dataset_accs[ds]
        if len(accs) > 0:
            m = np.mean(accs)
            s = np.std(accs)
            mean_std_cells.append(f"🟢 **{m:.2f} ± {s:.2f}%** ({len(accs)}/{datasets[ds]})")
        else:
            mean_std_cells.append("--")

    report.append("| :--- | :---: | :---: | :---: | :---: |")
    report.append(f"| **Mean ± Std (Đã xong)** | {mean_std_cells[0]} | {mean_std_cells[1]} | {mean_std_cells[2]} | {mean_std_cells[3]} |")
    report.append(f"| **Trạng thái tiến độ** | {dataset_completed_count['seed']}/{datasets['seed']} Xong, {dataset_running_count['seed']} Chạy | {dataset_completed_count['seediv']}/{datasets['seediv']} Xong, {dataset_running_count['seediv']} Chạy | {dataset_completed_count['seedv']}/{datasets['seedv']} Xong, {dataset_running_count['seedv']} Chạy | {dataset_completed_count['seedvii']}/{datasets['seedvii']} Xong, {dataset_running_count['seedvii']} Chạy |")

    # 2. Ablation Tracking (no_quantum, no_entropy_gate, etc.)
    ablation_variants = ['no_quantum', 'no_entropy_gate']
    has_ablation_activity = os.path.exists(ablation_res) or any('ablation' in s for s in active_screens)
    
    report.append("\n#### 🔬 BẢNG 2: ABLATION STUDIES (Các biến thể mô hình)")
    report.append("| Task (Subject) | SEED (no_quantum) | SEED (no_entropy_gate) | SEED-IV (no_quantum) | SEED-V (no_quantum) | SEED-VII (no_quantum) |")
    report.append("| :--- | :---: | :---: | :---: | :---: | :---: |")

    ab_accs = {f"{ds}_{v}": [] for ds in ['seed', 'seediv', 'seedv', 'seedvii'] for v in ['no_quantum', 'no_entropy_gate']}

    for p in range(1, 21):
        target = f'P{p}'
        row_cells = []
        cols_to_check = [
            ('seed', 'no_quantum', 15),
            ('seed', 'no_entropy_gate', 15),
            ('seediv', 'no_quantum', 15),
            ('seedv', 'no_quantum', 16),
            ('seedvii', 'no_quantum', 20),
        ]
        for ds, variant, max_p in cols_to_check:
            if p > max_p:
                row_cells.append('--')
                continue
            res_file = os.path.join(ablation_res, ds, variant, f'target_{target}', 'result.json')
            screen_tag = f"qneuro_ablation_{ds}_{variant}_{target}"
            if os.path.exists(res_file):
                try:
                    with open(res_file) as f:
                        data = json.load(f)
                        acc = data.get('accuracy', 0.0) * 100
                        ab_accs[f"{ds}_{variant}"].append(acc)
                        row_cells.append(f"🟢 {acc:.2f}%")
                except:
                    row_cells.append("🟢 Done")
            elif screen_tag in active_screens:
                row_cells.append("🟡 Running")
            else:
                row_cells.append("--")
        report.append(f"| **{target}** | " + " | ".join(row_cells) + " |")

    ab_mean_stds = []
    cols_to_check = [
        ('seed', 'no_quantum'),
        ('seed', 'no_entropy_gate'),
        ('seediv', 'no_quantum'),
        ('seedv', 'no_quantum'),
        ('seedvii', 'no_quantum'),
    ]
    for ds, variant in cols_to_check:
        accs = ab_accs[f"{ds}_{variant}"]
        if len(accs) > 0:
            m = np.mean(accs)
            s = np.std(accs)
            ab_mean_stds.append(f"🟢 **{m:.2f} ± {s:.2f}%**")
        else:
            ab_mean_stds.append("--")

    report.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
    report.append(f"| **Mean ± Std (Đã xong)** | " + " | ".join(ab_mean_stds) + " |")

    return "\n".join(report)

if __name__ == '__main__':
    print(get_status_report())
