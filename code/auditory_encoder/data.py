"""Episode sampling and held-out evaluation for the auditory encoder."""
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from model import procrustes


class Corpus:
    def __init__(self, root, held_out=()):
        self.windows = np.load(root / "windows.npy", mmap_mode="r")
        m = np.load(root / "meta.npz", allow_pickle=True)
        self.subject = np.array([str(x) for x in m["subject"]])
        self.stimulus = np.array([str(x) for x in m["stimulus"]])
        self.dataset = np.array([str(x) for x in m["dataset"]])
        self.subjects = sorted(set(self.subject))
        self.sid = {s: i for i, s in enumerate(self.subjects)}
        self.held = set(held_out)
        self.train_subjects = [s for s in self.subjects if s not in self.held]
        self.rows = defaultdict(list)
        self.subj_stims = defaultdict(set)
        self.stim_subj = defaultdict(set)
        self.subj_ds = {}
        for i, (u, s, d) in enumerate(zip(self.subject, self.stimulus, self.dataset)):
            self.rows[(u, s)].append(i)
            self.subj_stims[u].add(s)
            self.stim_subj[(d, s)].add(u)
            self.subj_ds.setdefault(u, d)
        self.datasets = sorted(set(self.dataset))
        self.ds_stims = {d: sorted({s for (dd, s) in self.stim_subj if dd == d}) for d in self.datasets}

    def sample_episode(self, rng, dataset, subjects, k, n_sup, n_q):
        """Return support/query tensors for k subjects sharing n_sup+n_q stimuli."""
        pool = [u for u in subjects if self.subj_ds.get(u) == dataset and u in self.subj_stims]
        for _ in range(200):
            pick = list(rng.choice(pool, k, replace=False))
            common = set(self.subj_stims[pick[0]])
            for u in pick[1:]:
                common &= self.subj_stims[u]
            common = [s for s in common if (dataset, s) in self.stim_subj]
            if len(common) >= n_sup + n_q:
                break
        else:
            raise RuntimeError(f"no {k}-subject episode with {n_sup+n_q} shared stimuli in {dataset}")
        rng.shuffle(common)
        sup = common[:n_sup]; qry = common[n_sup:n_sup + n_q]
        xs, xq = [], []
        for u in pick:
            for s in sup:
                xs.append(int(rng.choice(self.rows[(u, s)])))
            for s in qry:
                xq.append(int(rng.choice(self.rows[(u, s)])))
        return {"x_sup": np.stack([self.windows[i] for i in xs]).astype(np.float32),
                "x_q": np.stack([self.windows[i] for i in xq]).astype(np.float32),
                "subj": np.array([self.sid[u] for u in pick], dtype=np.int64),
                "stim_sup": sup, "stim_q": qry, "subjects": pick, "dataset": dataset,
                "rows_sup": np.array(xs, dtype=np.int64), "rows_q": np.array(xq, dtype=np.int64)}

    def rows_for(self, subject, stimuli):
        out = []
        for s in stimuli:
            r = self.rows.get((subject, s))
            if not r:
                return None
            out.append(int(r[0]))
        return out

    def all_rows_for(self, subject, stimuli):
        out = []
        for s in stimuli:
            r = self.rows.get((subject, s), [])
            if not r:
                return None
            out.append(r)
        return out


@torch.no_grad()
def encode_rows(model, windows, rows, bs=64):
    model.eval()
    dev = next(model.parameters()).device
    out = []
    for a in range(0, len(rows), bs):
        x = torch.tensor(np.stack([windows[i] for i in rows[a:a + bs]]), dtype=torch.float32,
                         device=dev)
        z, _, _ = model(x)
        out.append(z.cpu())
    return torch.cat(out, 0)


@torch.no_grad()
def eval_subject(model, corpus, subject, stimuli, train_subjects, device="cpu", n_book=20):
    """Zero-shot and fiber-oracle 10-way accuracy for one held-out subject."""
    rows = corpus.rows_for(subject, stimuli)
    if rows is None:
        return None
    zq = encode_rows(model, corpus.windows, rows).to(device)
    # codebook: mean over training subjects, one window per stimulus
    book_subs = [u for u in train_subjects if corpus.rows_for(u, stimuli) is not None][:n_book]
    book_rows = []
    for u in book_subs:
        book_rows += corpus.rows_for(u, stimuli)
    Z = encode_rows(model, corpus.windows, book_rows).to(device)
    used = len(book_rows) // len(stimuli)
    C = F.normalize(Z.view(used, len(stimuli), -1).mean(0), dim=-1)
    y = torch.arange(len(stimuli), device=device)
    acc_zero = float(((F.normalize(zq, dim=-1) @ C.T).argmax(1) == y).float().mean())
    all_rows = corpus.all_rows_for(subject, stimuli)
    acc_or = float("nan")
    if all_rows is not None and min(len(r) for r in all_rows) >= 2:
        anchor_rows = [r[0] for r in all_rows]
        query_rows = [r[1] for r in all_rows]
        za = encode_rows(model, corpus.windows, anchor_rows).to(device)
        zt = encode_rows(model, corpus.windows, query_rows).to(device)
        R = procrustes(za, C)
        pred = (F.normalize((zt - za.mean(0)) @ R, dim=-1) @ C.T).argmax(1)
        acc_or = float((pred == y).float().mean())
    return {"zero": acc_zero, "oracle": acc_or}
