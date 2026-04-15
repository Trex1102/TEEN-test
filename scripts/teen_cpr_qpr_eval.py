import argparse
import json
import os
import sys
import time
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataloader.data_utils import get_dataloader, set_up_datasets
from models.teen.Network import MYNET
from utils import set_seed, set_gpu


def load_args(config_path, gpu_override=None):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    args = Namespace(**cfg)
    if gpu_override is not None:
        args.gpu = str(gpu_override)
    args = set_up_datasets(args)
    return args


def find_default_checkpoint(repo_root, dataset):
    root = Path(repo_root) / "checkpoint" / dataset / "teen"
    matches = sorted(root.glob("**/session0_max_acc.pth"))
    if not matches:
        raise FileNotFoundError(f"No session0_max_acc.pth found under {root}")
    return matches[-1]


def build_model(args, checkpoint_path):
    model = MYNET(args, mode=args.base_mode)
    model = nn.DataParallel(model, list(range(args.num_gpu)))
    model = model.cuda()
    payload = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(payload["params"])
    model.eval()
    return model


def replay_incremental_sessions(model, args):
    final_session = args.sessions - 1
    final_testloader = None
    for session in range(1, args.sessions):
        train_set, trainloader, testloader = get_dataloader(args, session)
        model.module.mode = args.new_mode
        model.eval()
        trainloader.dataset.transform = testloader.dataset.transform
        class_list = np.unique(train_set.targets)
        model.module.update_fc(trainloader, class_list, session)
        model.module.soft_calibration(args, session)
        if session == final_session:
            final_testloader = testloader
    if final_testloader is None:
        raise RuntimeError("Failed to build final-session state for TEEN")
    return final_session, final_testloader


def collect_final_state(model, testloader, test_class):
    logits_all = []
    feats_all = []
    labels_all = []
    model.eval()
    with torch.no_grad():
        for data, labels in testloader:
            data = data.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            feats = model.module.encode(data)
            logits = model.module.get_logits(feats, model.module.fc.weight[:test_class])
            logits_all.append(logits)
            feats_all.append(feats)
            labels_all.append(labels)
    return (
        torch.cat(logits_all, dim=0),
        torch.cat(feats_all, dim=0),
        torch.cat(labels_all, dim=0),
    )


def accuracy_stats_from_logits(logits, labels, base_class):
    preds = logits.argmax(dim=1)
    full = float((preds == labels).float().mean().item() * 100.0)
    labels_np = labels.detach().cpu().numpy()
    preds_np = preds.detach().cpu().numpy()
    cm = confusion_matrix(labels_np, preds_np, labels=list(range(logits.size(1))), normalize="true")
    diag = np.diag(cm)
    base = float(diag[:base_class].mean() * 100.0)
    novel = float(diag[base_class:].mean() * 100.0)
    return full, base, novel


def build_base_novel_logits(probs, num_base_cls, t_base=1.0, t_novel=1.0, eps=1e-12):
    log_probs = torch.log(torch.clamp(probs, min=eps))
    return log_probs[:, :num_base_cls] / float(t_base), log_probs[:, num_base_cls:] / float(t_novel)


def mean_novel_mass_with_delta(base_logits, novel_logits, delta):
    base_score = torch.logsumexp(base_logits, dim=1)
    novel_score = torch.logsumexp(novel_logits + float(delta), dim=1)
    return torch.sigmoid(novel_score - base_score).mean()


def solve_group_delta(base_logits, novel_logits, target_novel_mass, low=-20.0, high=20.0, steps=60):
    target = max(min(float(target_novel_mass), 1.0 - 1e-6), 1e-6)
    lo = float(low)
    hi = float(high)
    for _ in range(int(steps)):
        mid = 0.5 * (lo + hi)
        mass = float(mean_novel_mass_with_delta(base_logits, novel_logits, mid).item())
        if mass < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def apply_qpr(probs, num_base_cls, t_base, t_novel, target_novel_mass):
    base_logits, novel_logits = build_base_novel_logits(probs, num_base_cls, t_base, t_novel)
    delta = solve_group_delta(base_logits, novel_logits, target_novel_mass)
    logits = torch.cat([base_logits, novel_logits + float(delta)], dim=1)
    calibrated = F.softmax(logits, dim=1)
    achieved = float(calibrated[:, num_base_cls:].sum(dim=1).mean().item())
    return calibrated, delta, achieved


def teen_cpr_once(features, baseline_probs, fc_weights, num_base_cls, alpha, mass_thr, min_count):
    feats = F.normalize(features, p=2, dim=-1)
    weights = F.normalize(fc_weights.clone(), p=2, dim=-1)
    num_cls = weights.size(0)
    top1 = baseline_probs.argmax(dim=1)
    novel_probs = baseline_probs[:, num_base_cls:]
    novel_mass = novel_probs.sum(dim=1)
    gate = (top1 >= num_base_cls) & (novel_mass >= float(mass_thr))
    top_novel = novel_probs.argmax(dim=1) + num_base_cls
    updated = weights.clone()

    stats = {
        "gated_query_count": int(gate.sum().item()),
    }

    updated_count = 0
    total_count = 0
    for class_idx in range(num_base_cls, num_cls):
        mask = gate & (top_novel == class_idx)
        count = int(mask.sum().item())
        if count < int(min_count):
            continue
        cls_w = baseline_probs[mask, class_idx]
        cls_feats = feats[mask]
        proto = F.normalize((cls_w[:, None] * cls_feats).sum(dim=0, keepdim=True), p=2, dim=-1).squeeze(0)
        mixed = F.normalize((1.0 - float(alpha)) * weights[class_idx] + float(alpha) * proto, p=2, dim=-1)
        updated[class_idx] = mixed
        updated_count += 1
        total_count += count

    stats["updated_class_count"] = int(updated_count)
    stats["mean_queries_per_updated_class"] = float(total_count / max(updated_count, 1))
    stats["updated_class_rate"] = float(updated_count / max(num_cls - num_base_cls, 1))
    return updated, stats


def logits_from_weights(model, features, weights):
    return model.module.get_logits(features, weights)


def create_output_dir(repo_root, dataset):
    out_root = Path(repo_root) / "experiments" / dataset / "teen_cpr_qpr_eval"
    out_root.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    out_dir = out_root / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def run(args):
    set_seed(args.seed)
    args.num_gpu = set_gpu(args)
    checkpoint_path = Path(args.checkpoint_path)
    config_path = Path(args.config_path)
    out_dir = create_output_dir(REPO_ROOT, args.dataset)
    start = time.time()

    teen_args = load_args(config_path, gpu_override=args.gpu)
    teen_args.num_gpu = set_gpu(teen_args)
    model = build_model(teen_args, checkpoint_path)
    final_session, testloader = replay_incremental_sessions(model, teen_args)
    test_class = teen_args.base_class + final_session * teen_args.way

    baseline_logits, query_features, query_targets = collect_final_state(model, testloader, test_class)
    baseline_probs = F.softmax(baseline_logits, dim=1)
    baseline_full, baseline_base, baseline_novel = accuracy_stats_from_logits(
        baseline_logits, query_targets, teen_args.base_class
    )

    qpr_probs, qpr_delta, qpr_mass = apply_qpr(
        baseline_probs,
        num_base_cls=teen_args.base_class,
        t_base=args.t_base,
        t_novel=args.t_novel,
        target_novel_mass=float((test_class - teen_args.base_class) / test_class),
    )
    qpr_logits = torch.log(torch.clamp(qpr_probs, min=1e-12))
    qpr_full, qpr_base, qpr_novel = accuracy_stats_from_logits(qpr_logits, query_targets, teen_args.base_class)

    fc_weights = model.module.fc.weight[:test_class].detach().clone().cuda()
    refined_weights, cpr_stats = teen_cpr_once(
        features=query_features,
        baseline_probs=baseline_probs,
        fc_weights=fc_weights,
        num_base_cls=teen_args.base_class,
        alpha=args.alpha,
        mass_thr=args.mass_thr,
        min_count=args.min_count,
    )
    cpr_logits = logits_from_weights(model, query_features, refined_weights)
    cpr_probs = F.softmax(cpr_logits, dim=1)
    cpr_full, cpr_base, cpr_novel = accuracy_stats_from_logits(cpr_logits, query_targets, teen_args.base_class)

    final_probs, final_delta, final_mass = apply_qpr(
        cpr_probs,
        num_base_cls=teen_args.base_class,
        t_base=args.t_base,
        t_novel=args.t_novel,
        target_novel_mass=float((test_class - teen_args.base_class) / test_class),
    )
    final_logits = torch.log(torch.clamp(final_probs, min=1e-12))
    final_full, final_base, final_novel = accuracy_stats_from_logits(final_logits, query_targets, teen_args.base_class)

    payload = {
        "dataset": teen_args.dataset,
        "config_path": str(config_path),
        "checkpoint_path": str(checkpoint_path),
        "runtime_sec": round(time.time() - start, 3),
        "base_class": int(teen_args.base_class),
        "num_seen_classes": int(test_class),
        "num_novel_classes": int(test_class - teen_args.base_class),
        "alpha": float(args.alpha),
        "mass_thr": float(args.mass_thr),
        "min_count": int(args.min_count),
        "t_base": float(args.t_base),
        "t_novel": float(args.t_novel),
        "baseline_full_acc": round(baseline_full, 3),
        "baseline_base_acc": round(baseline_base, 3),
        "baseline_novel_acc": round(baseline_novel, 3),
        "qpr_only_full_acc": round(qpr_full, 3),
        "qpr_only_base_acc": round(qpr_base, 3),
        "qpr_only_novel_acc": round(qpr_novel, 3),
        "qpr_only_gain": round(qpr_full - baseline_full, 3),
        "qpr_only_delta": round(float(qpr_delta), 4),
        "qpr_only_novel_mass": round(float(qpr_mass), 4),
        "cpr_full_acc": round(cpr_full, 3),
        "cpr_base_acc": round(cpr_base, 3),
        "cpr_novel_acc": round(cpr_novel, 3),
        "cpr_gain": round(cpr_full - baseline_full, 3),
        "cpr_stats": cpr_stats,
        "final_full_acc": round(final_full, 3),
        "final_base_acc": round(final_base, 3),
        "final_novel_acc": round(final_novel, 3),
        "final_gain": round(final_full - baseline_full, 3),
        "gain_over_cpr": round(final_full - cpr_full, 3),
        "final_delta": round(float(final_delta), 4),
        "final_novel_mass": round(float(final_mass), 4),
    }

    with (out_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    (out_dir / "summary.txt").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description="Post-hoc CPR+QPR evaluation for TEEN.")
    parser.add_argument("--dataset", required=True, choices=["cifar100", "mini_imagenet", "cub200"])
    parser.add_argument("--config_path", default=None)
    parser.add_argument("--checkpoint_path", default=None)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--mass_thr", type=float, default=0.3)
    parser.add_argument("--min_count", type=int, default=5)
    parser.add_argument("--t_base", type=float, default=0.75)
    parser.add_argument("--t_novel", type=float, default=0.75)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.checkpoint_path is None:
        args.checkpoint_path = str(find_default_checkpoint(REPO_ROOT, args.dataset))
    if args.config_path is None:
        args.config_path = str(Path(args.checkpoint_path).with_name("configs.json"))
    run(args)


if __name__ == "__main__":
    main()
