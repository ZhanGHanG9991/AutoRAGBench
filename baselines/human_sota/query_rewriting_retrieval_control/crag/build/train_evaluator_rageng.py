"""Fine-tune one T5-large relevance scorer on the filtered train pairs.

Matches scripts/train_evaluator.py: num_labels=1, labels mapped with
(label - 0.5) * 2, AdamW 1e-4, batch 12, 8 epochs, linear schedule,
warmup 0, max_length 512, gradient clip 1.0, fp32. The 8 epochs are split
across two jobs of 4. Job 2 loads the checkpoint and continues the same
schedule. The base is pretrained T5-large, not the author evaluator.
"""
import hashlib
import json
import os
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import TensorDataset
from transformers import T5ForSequenceClassification, T5Tokenizer, get_scheduler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from evaluator_rageng_tasks import (  # noqa: E402
    AUTHOR_EVALUATOR,
    BOUNDARY,
    CHECKPOINT,
    FILTER_VERSION,
    MODEL_ROOT,
    PAIRS,
    PAIR_STATS,
    SEED,
    STATUS,
    T5_BASE,
    refuse,
)

BATCH_SIZE = 12
EPOCHS = 8
MAX_LENGTH = 512
LEARNING_RATE = 1e-4
THREE_HOURS = 3 * 3600
PART_EPOCHS = {1: (0, 4), 2: (4, 8)}


def mapped_label(label):
    if label not in (0, 1):
        raise SystemExit(f"label {label}")
    return (int(label) - 0.5) * 2


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def epoch_orders(count, seed=SEED, epochs=EPOCHS):
    orders = []
    for epoch in range(epochs):
        rng = random.Random(seed + 1000003 * (epoch + 1))
        order = list(range(count))
        rng.shuffle(order)
        orders.append(order)
    return orders


def order_sha256(orders):
    digest = hashlib.sha256()
    for order in orders:
        digest.update(",".join(str(item) for item in order).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def capture_rng():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    # torch.load(map_location="cuda") moves the CPU generator state onto the GPU.
    # set_rng_state accepts only a CPU ByteTensor.
    cpu_state = state["torch"]
    if not torch.is_tensor(cpu_state):
        cpu_state = torch.tensor(cpu_state, dtype=torch.uint8)
    cpu_state = cpu_state.detach().cpu().contiguous().to(torch.uint8)
    torch.set_rng_state(cpu_state)
    if state.get("cuda") is not None and torch.cuda.is_available():
        cuda_states = []
        for item in state["cuda"]:
            if torch.is_tensor(item):
                # CUDAGeneratorImpl.set_state also requires a CPU ByteTensor.
                # map_location="cuda" would otherwise leave this state on the GPU.
                cuda_states.append(item.detach().cpu().contiguous().to(dtype=torch.uint8))
            else:
                cuda_states.append(torch.tensor(item, dtype=torch.uint8))
        torch.cuda.set_rng_state_all(cuda_states)


def expected_lr(global_step, total_steps, base=LEARNING_RATE):
    return base * float(total_steps - global_step) / float(total_steps)


def optim_fingerprint(optimizer):
    digest = hashlib.sha256()
    for param, state in optimizer.state.items():
        digest.update(str(tuple(param.shape)).encode())
        for key in sorted(state):
            value = state[key]
            if torch.is_tensor(value):
                digest.update(key.encode())
                array = value.detach().cpu().contiguous()
                digest.update(array.numpy().tobytes())
            else:
                digest.update(f"{key}={value}".encode())
    return digest.hexdigest()


def apply_scheduler_lr(optimizer, scheduler):
    # LambdaLR.load_state_dict restores last_epoch but leaves param group lr
    # at the value from optimizer construction, which is the initial 1e-4.
    restored = scheduler.get_last_lr()
    for group, lr_value in zip(optimizer.param_groups, restored):
        group["lr"] = lr_value
    return restored[0]


def load_pairs(require_balance=True):
    refuse(PAIRS)
    refuse(PAIR_STATS)
    stats = json.load(open(PAIR_STATS, encoding="utf-8"))
    filt = stats.get("multihop_filter") or {}
    if filt.get("version") != FILTER_VERSION:
        raise SystemExit("refusing pairs that are not the expected set")
    digest = file_sha256(PAIRS)
    # Pilot uses pairs_sha256; formal-v1 pair_stats.json uses main_sha256.
    want_sha = stats.get("pairs_sha256") or stats.get("main_sha256")
    if digest != want_sha:
        raise SystemExit("pairs file hash does not match pair stats")
    rows = [json.loads(line) for line in open(PAIRS, encoding="utf-8")]
    if any(row["task"] == "qasper" for row in rows):
        raise SystemExit("qasper pair in the training file")
    # Pilot: top-level n_balanced; formal-v1: balance.n_balanced.
    n_balanced = stats.get("n_balanced")
    if n_balanced is None:
        n_balanced = (stats.get("balance") or {}).get("n_balanced")
    if len(rows) != n_balanced:
        raise SystemExit("pair count does not match stats")
    texts = [row["text"] for row in rows]
    labels = [mapped_label(row["label"]) for row in rows]
    if not texts:
        raise SystemExit("empty training file")
    if require_balance and abs(sum(labels)) > 1e-6:
        raise SystemExit(f"unbalanced training file n={len(texts)} sum={sum(labels)}")
    return texts, labels, stats, digest


def bind_ablation():
    """Point this process at the multihop-unfiltered ablation. The main files stay put."""
    global PAIRS, PAIR_STATS, MODEL_ROOT, STATUS, CHECKPOINT, BOUNDARY, FILTER_VERSION
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(PAIRS))))
    # PAIRS is .../CorrectiveRAG/rageng/results/file.
    if os.path.basename(root) != "CorrectiveRAG":
        raise SystemExit(f"unexpected project root {root}")
    main_model = MODEL_ROOT
    PAIRS = os.path.join(root, "rageng/results/evaluator_rageng_pairs.mh_unfiltered.jsonl")
    PAIR_STATS = os.path.join(root, "rageng/results/evaluator_rageng_pair_stats.mh_unfiltered.json")
    STATUS = os.path.join(root, "rageng/results/evaluator_rageng_status.mh_unfiltered.json")
    BOUNDARY = os.path.join(root, "rageng/results/evaluator_rageng_resume_boundary.mh_unfiltered.json")
    MODEL_ROOT = os.path.join(root, "models/evaluator-rageng-mh-unfiltered")
    CHECKPOINT = os.path.join(MODEL_ROOT, "resume.pt")
    FILTER_VERSION = "mh_unfiltered_positives_v1"
    if os.path.realpath(MODEL_ROOT) == os.path.realpath(main_model):
        raise SystemExit("ablation would overwrite the main evaluator")
    if os.path.realpath(MODEL_ROOT) == os.path.realpath(AUTHOR_EVALUATOR):
        raise SystemExit("ablation would overwrite the author evaluator")


def bind_formal():
    """Point this process at formal-v1 pairs and model dir. Pilot files stay put."""
    global PAIRS, PAIR_STATS, MODEL_ROOT, STATUS, CHECKPOINT, BOUNDARY, FILTER_VERSION, SEED, THREE_HOURS
    # PAIRS defaults to .../CorrectiveRAG/rageng/results/...; climb to CorrectiveRAG.
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(PAIRS))))
    if os.path.basename(root) != "CorrectiveRAG":
        raise SystemExit(f"unexpected project root {root}")
    formal = os.path.join(root, "rageng/formal-v1")
    main_model = MODEL_ROOT
    PAIRS = os.path.join(formal, "pairs/evaluator_formal_v1_pairs.jsonl")
    PAIR_STATS = os.path.join(formal, "pairs/pair_stats.json")
    STATUS = os.path.join(formal, "results/evaluator_status.json")
    BOUNDARY = os.path.join(formal, "results/evaluator_resume_boundary.json")
    MODEL_ROOT = os.path.join(root, "models/evaluator-rageng-formal-v1")
    CHECKPOINT = os.path.join(MODEL_ROOT, "resume.pt")
    FILTER_VERSION = "formal_v1_gold_window"
    SEED = 20261019
    # Each four-epoch training part retains the six-hour time limit.
    THREE_HOURS = 6 * 3600
    if os.path.realpath(MODEL_ROOT) == os.path.realpath(main_model):
        raise SystemExit("formal would overwrite the pilot evaluator")
    if os.path.realpath(MODEL_ROOT) == os.path.realpath(AUTHOR_EVALUATOR):
        raise SystemExit("formal would overwrite the author evaluator")
    if "mh-unfiltered" in MODEL_ROOT:
        raise SystemExit("formal must not use the ablation model dir")
    print(
        f"BIND_FORMAL pairs={PAIRS} model_root={MODEL_ROOT} status={STATUS} "
        f"seed={SEED} filter={FILTER_VERSION} part_limit_sec={THREE_HOURS}",
        flush=True,
    )


def make_tensors(tokenizer, texts, labels):
    encoded = tokenizer(
        texts, padding="max_length", max_length=MAX_LENGTH, truncation=True, return_tensors="pt"
    )
    return TensorDataset(
        encoded["input_ids"], encoded["attention_mask"], torch.tensor(labels, dtype=torch.float32)
    )


def fresh_model():
    tokenizer = T5Tokenizer.from_pretrained(T5_BASE, local_files_only=True)
    model = T5ForSequenceClassification.from_pretrained(
        T5_BASE, num_labels=1, local_files_only=True
    )
    model.config.problem_type = "regression"
    return tokenizer, model


def prepare_optim(model, steps_per_epoch):
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = EPOCHS * steps_per_epoch
    scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )
    return optimizer, scheduler, total_steps


def write_status(payload):
    refuse(STATUS)
    os.makedirs(os.path.dirname(STATUS), exist_ok=True)
    temporary = STATUS + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=1)
        handle.write("\n")
    os.replace(temporary, STATUS)
    print("STATUS", payload.get("status"), flush=True)


def torch_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def save_checkpoint(payload, epoch):
    refuse(CHECKPOINT)
    os.makedirs(MODEL_ROOT, exist_ok=True)
    epoch_path = os.path.join(MODEL_ROOT, f"checkpoint-ep{epoch}.pt")
    torch.save(payload, epoch_path)
    torch.save(payload, CHECKPOINT)
    print("CHECKPOINT", epoch_path, flush=True)


def batch_slices(order):
    for start in range(0, len(order), BATCH_SIZE):
        yield order[start : start + BATCH_SIZE]


def run_epochs(
    model, dataset, orders, optimizer, scheduler, device, tokenizer, start_epoch, end_epoch,
    global_step, total_steps, pairs_sha, sec_per_step, steps_per_epoch,
):
    input_ids = dataset.tensors[0]
    mask = dataset.tensors[1]
    labels = dataset.tensors[2]
    first_resumed = global_step > 0
    for epoch in range(start_epoch, end_epoch):
        total_loss = 0.0
        seen = 0
        model.train()
        for batch_ids in batch_slices(orders[epoch]):
            index = torch.tensor(batch_ids, dtype=torch.long)
            lr_used = float(optimizer.param_groups[0]["lr"])
            batch_ids_dev = input_ids[index].to(device)
            batch_mask = mask[index].to(device)
            batch_labels = labels[index].to(device)
            model.zero_grad()
            outputs = model(input_ids=batch_ids_dev, attention_mask=batch_mask, labels=batch_labels)
            loss = outputs.loss
            if not torch.isfinite(loss):
                raise SystemExit(f"non-finite loss at epoch {epoch} step {global_step}")
            loss.mean().backward()
            value = float(loss.mean().item())
            total_loss += value
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1
            seen += 1
            lr_next = float(optimizer.param_groups[0]["lr"])
            if first_resumed:
                print(
                    f"RESUME_FIRST_STEP global_step={global_step} lr_used={lr_used:.12g} "
                    f"lr_next={lr_next:.12g}",
                    flush=True,
                )
                expected_step_lr = expected_lr(global_step - 1, total_steps)
                if abs(lr_used - expected_step_lr) > 1e-12:
                    raise SystemExit(
                        f"resumed step used lr {lr_used} expected {expected_step_lr}"
                    )
                first_resumed = False
            if global_step % 10 == 0:
                print(f"step: {global_step}  loss: {value}  lr: {lr_next:.12g}", flush=True)
        if seen != steps_per_epoch:
            raise SystemExit(f"epoch {epoch} ran {seen} steps, expected {steps_per_epoch}")
        average = total_loss / max(seen, 1)
        lr_next = float(scheduler.get_last_lr()[0])
        if abs(lr_next - expected_lr(global_step, total_steps)) > 1e-12:
            raise SystemExit(f"lr {lr_next} left the 8-epoch schedule at step {global_step}")
        if scheduler.last_epoch != global_step:
            raise SystemExit(f"scheduler last_epoch {scheduler.last_epoch} != step {global_step}")
        out = os.path.join(MODEL_ROOT, f"ep{epoch}")
        os.makedirs(out, exist_ok=True)
        model.save_pretrained(out)
        tokenizer.save_pretrained(out)
        payload = {
            "epoch_completed": epoch,
            "next_epoch": epoch + 1,
            "global_step": global_step,
            "lr_next": lr_next,
            "total_steps": total_steps,
            "steps_per_epoch": seen,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "rng": capture_rng(),
            "data_order": orders,
            "data_order_sha256": order_sha256(orders),
            "pairs_sha256": pairs_sha,
            "optim_fingerprint": optim_fingerprint(optimizer),
            "sec_per_step": sec_per_step,
            "n": int(labels.shape[0]),
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "seed": SEED,
            "filter_version": FILTER_VERSION,
        }
        save_checkpoint(payload, epoch)
        print(
            f"EPOCH_DONE epoch={epoch} global_step={global_step} avg_loss={average} "
            f"lr_next={lr_next:.12g}",
            flush=True,
        )
    return {
        "global_step": global_step,
        "lr_next": lr_next,
        "next_epoch": end_epoch,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "optim_fingerprint": optim_fingerprint(optimizer),
        "data_order_sha256": order_sha256(orders),
    }


def assert_resume(optimizer, scheduler, checkpoint, global_step, require_halfway):
    lr_value = apply_scheduler_lr(optimizer, scheduler)
    fingerprint = optim_fingerprint(optimizer)
    param_lr = float(optimizer.param_groups[0]["lr"])
    print(
        f"RESUME_AFTER global_step={global_step} lr_next={lr_value:.12g} "
        f"param_lr={param_lr:.12g} last_epoch={scheduler.last_epoch} "
        f"optim_fp={fingerprint}",
        flush=True,
    )
    if global_step != checkpoint["global_step"]:
        raise SystemExit("global step was not restored")
    if scheduler.last_epoch != global_step:
        raise SystemExit("scheduler last_epoch was not restored")
    if abs(lr_value - checkpoint["lr_next"]) > 1e-12 or abs(param_lr - lr_value) > 1e-12:
        raise SystemExit("learning rate was not restored")
    if abs(lr_value - expected_lr(global_step, checkpoint["total_steps"])) > 1e-12:
        raise SystemExit("restored learning rate is not on the 8-epoch line")
    if abs(lr_value - LEARNING_RATE) <= 1e-12:
        raise SystemExit("learning rate reset to the initial value")
    if fingerprint != checkpoint["optim_fingerprint"]:
        raise SystemExit("optimizer state fingerprint mismatch")
    if require_halfway:
        if global_step != checkpoint["total_steps"] // 2:
            raise SystemExit("part 2 did not start at the halfway step")
        if abs(lr_value - (LEARNING_RATE / 2)) > 1e-12:
            raise SystemExit(f"halfway lr {lr_value} is not half of 1e-4")
    print(
        f"RESUME_CHECK ok step_after={global_step} lr_after={lr_value:.12g} "
        f"optim_restored=1 scheduler_restored=1 data_order_match=1",
        flush=True,
    )


def train_part(part):
    formal = os.environ.get("CRAG_FORMAL") == "1"
    ablation = os.environ.get("CRAG_ABLATION") == "mh-unfiltered"
    if formal and ablation:
        raise SystemExit("refusing CRAG_FORMAL and CRAG_ABLATION together")
    if formal:
        bind_formal()
    elif ablation:
        bind_ablation()
    start_epoch, end_epoch = PART_EPOCHS[part]
    if not torch.cuda.is_available():
        raise SystemExit("refusing to train without a GPU")
    for path in (T5_BASE, PAIRS, PAIR_STATS, MODEL_ROOT, AUTHOR_EVALUATOR, STATUS, CHECKPOINT):
        refuse(path)
    if os.path.realpath(MODEL_ROOT) == os.path.realpath(AUTHOR_EVALUATOR):
        raise SystemExit("refusing to write the author evaluator")
    texts, labels, stats, pairs_sha = load_pairs(require_balance=not ablation)
    device = torch.device("cuda:0")
    if ablation:
        if len(texts) != 8350:
            raise SystemExit(f"ablation pair count {len(texts)} is not 8350")
        # Same 5064-step schedule as the filtered run. Each epoch keeps the
        # first 633 batches of a shuffle of all 8350 pairs.
        steps_per_epoch = 633
        keep = steps_per_epoch * BATCH_SIZE
        orders = [order[:keep] for order in epoch_orders(len(texts))]
        if any(len(order) != keep for order in orders):
            raise SystemExit("ablation epoch was not truncated to 7596 pairs")
    else:
        steps_per_epoch = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE
        orders = epoch_orders(len(texts))
    total_steps = EPOCHS * steps_per_epoch
    if ablation and total_steps != 5064:
        raise SystemExit(f"ablation schedule is {total_steps}, not 5064")
    checkpoint = None
    if os.path.exists(CHECKPOINT):
        checkpoint = torch_load(CHECKPOINT, device)
        if checkpoint.get("pairs_sha256") != pairs_sha:
            raise SystemExit("checkpoint was trained on a different pair file")
        if checkpoint.get("data_order") != orders:
            raise SystemExit("checkpoint data order does not match the seed")
    if part == 1 and checkpoint is not None and checkpoint["next_epoch"] >= end_epoch:
        print("PART1_ALREADY_DONE", checkpoint["next_epoch"], flush=True)
        return
    if part == 2 and checkpoint is None:
        raise SystemExit("part 2 has no checkpoint")
    sec_per_step = None if checkpoint is None else checkpoint.get("sec_per_step")
    if part == 1 and checkpoint is None:
        sec_per_step = benchmark(texts, labels, device, steps_per_epoch)
        part_steps = (end_epoch - start_epoch) * steps_per_epoch
        eta = sec_per_step * part_steps
        print(
            "ESTIMATE_SEC", round(eta, 1), "part_steps", part_steps,
            "sec_per_step", round(sec_per_step, 3), "n", len(texts),
            "total_steps", total_steps, flush=True,
        )
        if eta > THREE_HOURS:
            write_status({
                "status": "estimate_over_3h",
                "part": 1,
                "eta_sec": eta,
                "sec_per_step": sec_per_step,
                "part_steps": part_steps,
                "total_steps": total_steps,
                "n": len(texts),
                "train_started": False,
                "filter_version": FILTER_VERSION,
            })
            print("ESTIMATE_OVER_3H", round(eta / 3600, 3), "hours", flush=True)
            return
    set_seed(SEED)
    tokenizer, model = fresh_model()
    model.to(device)
    optimizer, scheduler, made_steps = prepare_optim(model, steps_per_epoch)
    if made_steps != total_steps:
        raise SystemExit("schedule length mismatch")
    dataset = make_tensors(tokenizer, texts, labels)
    global_step = 0
    resume_from = start_epoch
    if checkpoint is not None:
        resume_from = checkpoint["next_epoch"]
        if part == 2 and resume_from != start_epoch:
            raise SystemExit(f"part 2 expected next_epoch {start_epoch}, found {resume_from}")
        print(
            f"RESUME_BEFORE global_step=0 lr={float(optimizer.param_groups[0]['lr']):.12g}",
            flush=True,
        )
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        global_step = int(checkpoint["global_step"])
        assert_resume(optimizer, scheduler, checkpoint, global_step, require_halfway=(part == 2))
        restore_rng(checkpoint["rng"])
        sec_per_step = checkpoint.get("sec_per_step", sec_per_step)
    if part == 2 and sec_per_step is not None:
        remaining = (end_epoch - resume_from) * steps_per_epoch
        if sec_per_step * remaining > THREE_HOURS:
            print("ESTIMATE_OVER_3H", round(sec_per_step * remaining / 3600, 3), "hours", flush=True)
            raise SystemExit("part 2 estimate exceeds 3 hours")
    write_status({
        "status": f"running_part{part}",
        "train_started": True,
        "next_epoch": resume_from,
        "global_step": global_step,
        "n": len(texts),
        "filter_version": FILTER_VERSION,
        "pairs_sha256": pairs_sha,
    })
    print(
        "TRAIN_START", "part", part, "epoch", resume_from, "to", end_epoch,
        "global_step", global_step, "n", len(texts), flush=True,
    )
    summary = run_epochs(
        model, dataset, orders, optimizer, scheduler, device, tokenizer,
        resume_from, end_epoch, global_step, total_steps, pairs_sha, sec_per_step,
        steps_per_epoch,
    )
    if part == 1:
        if summary["next_epoch"] != 4 or summary["global_step"] != 4 * steps_per_epoch:
            raise SystemExit(
                f"part 1 ended at step {summary['global_step']} epoch {summary['next_epoch']}"
            )
        if not os.path.isfile(CHECKPOINT):
            raise SystemExit("part 1 did not write a checkpoint")
        boundary = {
            "global_step": summary["global_step"],
            "lr_next": summary["lr_next"],
            "next_epoch": summary["next_epoch"],
            "total_steps": summary["total_steps"],
            "steps_per_epoch": summary["steps_per_epoch"],
            "optim_fingerprint": summary["optim_fingerprint"],
            "data_order_sha256": summary["data_order_sha256"],
            "pairs_sha256": pairs_sha,
            "sec_per_step": sec_per_step,
            "n": len(texts),
            "filter_version": FILTER_VERSION,
        }
        with open(BOUNDARY, "w", encoding="utf-8") as handle:
            json.dump(boundary, handle, indent=1)
            handle.write("\n")
        print(
            f"BOUNDARY_END part=1 global_step={boundary['global_step']} "
            f"lr_next={boundary['lr_next']:.12g} next_epoch={boundary['next_epoch']} "
            f"optim_fp={boundary['optim_fingerprint']}",
            flush=True,
        )
        write_status({
            "status": "part1_done",
            "train_started": True,
            "global_step": boundary["global_step"],
            "lr_next": boundary["lr_next"],
            "next_epoch": boundary["next_epoch"],
            "n": len(texts),
            "filter_version": FILTER_VERSION,
            "pairs_sha256": pairs_sha,
        })
        return
    last_dir = os.path.join(MODEL_ROOT, "ep7")
    if not os.path.isfile(os.path.join(last_dir, "config.json")):
        raise SystemExit("part 2 finished without ep7")
    write_status({
        "status": "trained",
        "train_started": True,
        "model_dir": last_dir,
        "epoch": EPOCHS - 1,
        "selection": "last epoch of the 8-epoch schedule; dev was not used",
        "n": len(texts),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "base": T5_BASE,
        "seed": SEED,
        "pair_stats": PAIR_STATS,
        "pairs_sha256": pairs_sha,
        "filter_version": FILTER_VERSION,
        "positive_vs_unfiltered": stats.get("positive_vs_unfiltered"),
    })
    print("TRAIN_DONE", last_dir, flush=True)


def benchmark(texts, labels, device, steps_per_epoch):
    set_seed(SEED)
    tokenizer, model = fresh_model()
    model.to(device)
    optimizer, scheduler, total_steps = prepare_optim(model, steps_per_epoch)
    dataset = make_tensors(tokenizer, texts, labels)
    orders = epoch_orders(len(texts))
    limit = min(30, steps_per_epoch)
    torch.cuda.synchronize()
    started = time.time()
    seen = 0
    input_ids, mask, label_tensor = dataset.tensors
    model.train()
    for batch_ids in batch_slices(orders[0]):
        if seen >= limit:
            break
        index = torch.tensor(batch_ids, dtype=torch.long)
        outputs = model(
            input_ids=input_ids[index].to(device),
            attention_mask=mask[index].to(device),
            labels=label_tensor[index].to(device),
        )
        loss = outputs.loss
        if not torch.isfinite(loss):
            raise SystemExit("non-finite loss during the timing steps")
        loss.mean().backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        seen += 1
    torch.cuda.synchronize()
    elapsed = time.time() - started
    del model, optimizer, scheduler, dataset
    torch.cuda.empty_cache()
    if seen < 1 or total_steps != EPOCHS * steps_per_epoch:
        raise SystemExit("benchmark did not run")
    return elapsed / seen


def self_test():
    orders = epoch_orders(20, seed=SEED, epochs=4)
    assert orders == epoch_orders(20, seed=SEED, epochs=4)
    assert orders[0] != orders[1]
    parameter = torch.nn.Parameter(torch.zeros(3))
    optimizer = torch.optim.AdamW([parameter], lr=LEARNING_RATE)
    total = 8
    scheduler = get_scheduler(
        "linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=total
    )
    assert abs(optimizer.param_groups[0]["lr"] - LEARNING_RATE) < 1e-12
    for _ in range(4):
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    saved_lr = scheduler.get_last_lr()[0]
    saved_state = optimizer.state_dict()
    saved_sched = scheduler.state_dict()
    fingerprint = optim_fingerprint(optimizer)
    other = torch.nn.Parameter(torch.zeros(3))
    optimizer2 = torch.optim.AdamW([other], lr=LEARNING_RATE)
    scheduler2 = get_scheduler(
        "linear", optimizer=optimizer2, num_warmup_steps=0, num_training_steps=total
    )
    before = float(optimizer2.param_groups[0]["lr"])
    assert abs(before - LEARNING_RATE) < 1e-12
    optimizer2.load_state_dict(saved_state)
    scheduler2.load_state_dict(saved_sched)
    restored = apply_scheduler_lr(optimizer2, scheduler2)
    assert abs(restored - saved_lr) < 1e-12
    assert abs(restored - expected_lr(4, total)) < 1e-12
    assert abs(restored - LEARNING_RATE) > 1e-8
    assert optim_fingerprint(optimizer2) == fingerprint
    assert scheduler2.last_epoch == 4
    rng_state = capture_rng()
    broken = dict(rng_state)
    broken["torch"] = rng_state["torch"].view(torch.int8).clone()
    restore_rng(broken)
    print("TRAIN_SELF_TEST_OK", f"lr_before={before:.12g}", f"lr_after={restored:.12g}")


def parse_part(argv):
    if "--part" not in argv:
        raise SystemExit("pass --part 1 or --part 2")
    value = argv[argv.index("--part") + 1]
    if value not in ("1", "2"):
        raise SystemExit("part must be 1 or 2")
    return int(value)


def main():
    if "--self-test" in sys.argv:
        self_test()
        return
    train_part(parse_part(sys.argv))


if __name__ == "__main__":
    main()
