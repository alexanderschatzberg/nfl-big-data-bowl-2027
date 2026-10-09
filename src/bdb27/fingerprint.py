"""Movement fingerprint: a two-tower contrastive network across Combine and games.

One tower reads a set of a player's Combine drill clips, the other a set of the
same player's post-snap game clips. A symmetric InfoNCE loss pulls a player's
two embeddings together and pushes other players away. Negatives always come
from the same role (edge vs interior line, etc.) and draft class, so the network cannot win by
recognising position, body type or season: it has to recognise how the player
moves. Success is measured only on players held out of training.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import stats
from torch import nn

from .clips import CHANNELS
from .data import artifacts_dir, connect

N_FOLDS = 5
MIN_COMBINE_CLIPS = 3
MIN_GAME_CLIPS = 20
ROLE = {"DE": "EDGE", "OLB": "EDGE", "DT": "IDL", "NT": "IDL", "T": "OT", "G": "IOL", "C": "IOL",
        "CB": "CB", "SS": "S", "FS": "S", "WR": "WR", "TE": "TE"}
DEFAULTS = dict(dim=64, embed=32, k_combine=8, k_game=16, steps=1500, lr=2e-3, weight_decay=1e-2,
                dropout=0.1, eval_game_clips=64, eval_draws=8)


class ClipEncoder(nn.Module):
    """Dilated temporal convolutions over one clip, attention-pooled over time."""

    def __init__(self, n_tokens: int, dim: int, dropout: float):
        super().__init__()
        self.token = nn.Embedding(n_tokens, dim)
        self.stem = nn.Conv1d(len(CHANNELS), dim, 5, padding=2)
        self.blocks = nn.ModuleList(
            nn.Sequential(nn.Conv1d(dim, dim, 3, padding=2**i, dilation=2**i), nn.GroupNorm(8, dim),
                          nn.GELU(), nn.Dropout(dropout)) for i in range(4))
        self.score = nn.Linear(dim, 1)

    def forward(self, x, token):  # x: [N, T, C]
        valid = x[..., -1] > 0
        h = self.stem(x.transpose(1, 2))
        for block in self.blocks:
            h = h + block(h)
        h = h.transpose(1, 2) + self.token(token)[:, None]
        w = self.score(h).squeeze(-1).masked_fill(~valid, -1e4).softmax(-1)
        return (h * w[..., None]).sum(1)


class SetEncoder(nn.Module):
    """A player's clips -> one unit-norm embedding, via attention pooling over the set."""

    def __init__(self, n_tokens: int, dim: int, embed: int, dropout: float):
        super().__init__()
        self.clip = ClipEncoder(n_tokens, dim, dropout)
        self.seed = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.pool = nn.MultiheadAttention(dim, 4, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, embed))

    def forward(self, x, token, present):  # x: [B, K, T, C], present: [B, K]
        b, k = x.shape[:2]
        h = self.clip(x.flatten(0, 1), token.flatten()).view(b, k, -1)
        pooled, _ = self.pool(self.seed.expand(b, -1, -1), h, h, key_padding_mask=~present)
        return F.normalize(self.head(pooled.squeeze(1)), dim=-1)


class Fingerprint(nn.Module):
    def __init__(self, n_drills: int, n_roles: int, dim: int, embed: int, dropout: float):
        super().__init__()
        self.combine = SetEncoder(n_drills, dim, embed, dropout)
        self.game = SetEncoder(n_roles, dim, embed, dropout)
        self.logit_scale = nn.Parameter(torch.tensor(float(np.log(10.0))))

    def loss(self, zc, zg):
        logits = self.logit_scale.exp().clamp(max=100) * zc @ zg.T
        target = torch.arange(len(zc), device=zc.device)
        return (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target)) / 2


class Domain:
    """Clips of one domain, indexed by player."""

    def __init__(self, name: str):
        d = np.load(artifacts_dir() / f"clips_{name}.npz", allow_pickle=False)
        self.clips, self.token, self.n_tokens = d["clips"], d["token"], len(d["vocab"])
        order = np.argsort(d["nfl_id"], kind="stable")
        ids, starts = np.unique(d["nfl_id"][order], return_index=True)
        self.index = dict(zip(ids.tolist(), np.split(order, starts[1:])))

    def sample(self, players, k: int, rng, augment: bool, device, index: dict | None = None):
        """`index` restricts each player to a subset of their clips (default: all)."""
        index = index or self.index
        t, c = self.clips.shape[1:]
        x = np.zeros((len(players), k, t, c), np.float32)
        tok = np.zeros((len(players), k), np.int64)
        present = np.zeros((len(players), k), bool)
        for i, pid in enumerate(players):
            idx = index[pid]
            pick = rng.choice(idx, size=min(k, len(idx)), replace=False)
            x[i, : len(pick)], tok[i, : len(pick)], present[i, : len(pick)] = self.clips[pick], self.token[pick], True
        if augment:  # mirror image of a movement is the same movement
            flip = rng.choice([-1.0, 1.0], size=(len(players), k, 1)).astype(np.float32)
            x[..., 2] *= flip
            x[..., 3] *= flip
        return (torch.from_numpy(x).to(device), torch.from_numpy(tok).to(device),
                torch.from_numpy(present).to(device))


MIN_DRILLS = 4


def eligible_players(combine: Domain, game: Domain | None) -> pd.DataFrame:
    """Players usable for training. With `game=None`, Combine-only (needs several distinct drills)."""
    meta = connect().execute(
        "SELECT nfl_id, cr.draft_year, combine_position, nfl_position FROM combine_results cr JOIN players USING (nfl_id)"
    ).df()
    if game is None:
        ok = [p in combine.index and len(np.unique(combine.token[combine.index[p]])) >= MIN_DRILLS
              for p in meta["nfl_id"]]
    else:
        ok = [len(combine.index.get(p, ())) >= MIN_COMBINE_CLIPS and len(game.index.get(p, ())) >= MIN_GAME_CLIPS
              for p in meta["nfl_id"]]
    meta = meta[ok].reset_index(drop=True)
    # Edge rushers and interior linemen (or tackles and guards) differ in build and
    # role, so a line-up mixing them could be solved without recognising the player.
    meta["role"] = meta["nfl_position"].map(ROLE).fillna(meta["combine_position"])
    meta["group"] = meta["role"] + "_" + meta["draft_year"].astype(str)
    # Folds are stratified so every held-out line-up contains several same-group peers.
    rng = np.random.default_rng(0)
    meta["fold"] = meta.groupby("group")["nfl_id"].transform(lambda s: rng.permutation(len(s)) % N_FOLDS)
    return meta


def device() -> str:
    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def train_fold(fold: int, seed: int = 0, **overrides) -> dict:
    """Train without fold `fold`'s players, then try to re-identify them."""
    cfg = DEFAULTS | overrides
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = device()
    combine, game = Domain("combine"), Domain("game")
    meta = eligible_players(combine, game)
    train, test = meta[meta["fold"] != fold], meta[meta["fold"] == fold]
    groups = [g["nfl_id"].tolist() for _, g in train.groupby("group") if len(g) >= 4]

    model = Fingerprint(combine.n_tokens, game.n_tokens, cfg["dim"], cfg["embed"], cfg["dropout"]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, cfg["lr"], total_steps=cfg["steps"])
    model.train()
    losses = []
    for step in range(cfg["steps"]):
        players = groups[rng.integers(len(groups))]
        zc = model.combine(*combine.sample(players, cfg["k_combine"], rng, True, dev))
        zg = model.game(*game.sample(players, cfg["k_game"], rng, True, dev))
        loss = model.loss(zc, zg)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        losses.append(loss.item())

    model.eval()
    rows = []
    with torch.no_grad():
        for group, g in test.groupby("group"):
            players = g["nfl_id"].tolist()
            zc = model.combine(*combine.sample(players, 64, rng, False, dev))
            # Average several draws of game clips: a stable estimate of the game-side embedding.
            zg = torch.stack([model.game(*game.sample(players, cfg["eval_game_clips"], rng, False, dev))
                              for _ in range(cfg["eval_draws"])]).mean(0)
            sim = (zc @ F.normalize(zg, dim=-1).T).cpu().numpy()
            for i, pid in enumerate(players):
                rank = int((sim[i] > sim[i, i]).sum())  # 0 = the player's own games ranked first
                rows.append({"nfl_id": pid, "group": group, "fold": fold, "seed": seed, "lineup": len(players),
                             "rank": rank, **{f"e{j}": v for j, v in enumerate(zc[i].cpu().numpy())}})
    return {"fold": fold, "seed": seed, "config": cfg, "device": dev,
            "final_loss": float(np.mean(losses[-50:])), "rows": rows}


def _halves(combine: Domain, players, rng=None) -> tuple[list, dict, dict]:
    """Split the drill vocabulary in two, the same way for every player in the line-up.

    A per-player split would leak: a player's own halves never share a drill, but
    another player's half can, so drill overlap alone would separate them.
    Returns the players who have clips on both sides, and their two clip indexes.
    """
    drills = np.unique(np.concatenate([combine.token[combine.index[p]] for p in players]))
    first = (rng.permutation(drills) if rng is not None else drills)[::2]
    a, b = {}, {}
    for pid in players:
        idx = combine.index[pid]
        mask = np.isin(combine.token[idx], first)
        if mask.any() and not mask.all():
            a[pid], b[pid] = idx[mask], idx[~mask]
    return list(a), a, b


def train_split_fold(fold: int, seed: int = 0, **overrides) -> dict:
    """Combine-only signature: match one half of a player's drills to the other half.

    No game data and no outcome is involved, so whatever this learns is simply
    the most repeatable way a prospect moves across different drills. Held-out
    re-identification is a split-half reliability for that learned signature.
    """
    cfg = DEFAULTS | overrides
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = device()
    combine = Domain("combine")
    meta = eligible_players(combine, None)
    train, test = meta[meta["fold"] != fold], meta[meta["fold"] == fold]
    groups = [g["nfl_id"].tolist() for _, g in train.groupby("group") if len(g) >= 4]

    model = SetEncoder(combine.n_tokens, cfg["dim"], cfg["embed"], cfg["dropout"]).to(dev)
    scale = float(np.log(10.0))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, cfg["lr"], total_steps=cfg["steps"])
    model.train()
    losses = []
    for step in range(cfg["steps"]):
        players, a, b = _halves(combine, groups[rng.integers(len(groups))], rng)
        za = model(*combine.sample(players, cfg["k_combine"], rng, True, dev, a))
        zb = model(*combine.sample(players, cfg["k_combine"], rng, True, dev, b))
        logits = np.exp(scale) * za @ zb.T
        target = torch.arange(len(players), device=dev)
        loss = (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target)) / 2
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        losses.append(loss.item())

    model.eval()
    rows = []
    with torch.no_grad():
        for group, g in test.groupby("group"):
            players, a, b = _halves(combine, g["nfl_id"].tolist())
            za = model(*combine.sample(players, 64, rng, False, dev, a))
            zb = model(*combine.sample(players, 64, rng, False, dev, b))
            zall = model(*combine.sample(players, 64, rng, False, dev))
            sim = (za @ zb.T).cpu().numpy()
            for i, pid in enumerate(players):
                rows.append({"nfl_id": pid, "group": group, "fold": fold, "seed": seed, "lineup": len(players),
                             "rank": int((sim[i] > sim[i, i]).sum()),
                             **{f"e{j}": v for j, v in enumerate(zall[i].cpu().numpy())}})
    return {"fold": fold, "seed": seed, "config": cfg, "device": dev,
            "final_loss": float(np.mean(losses[-50:])), "rows": rows}


def summarize(rows: list[dict]) -> dict:
    """Held-out re-identification versus chance, over line-ups of at least three."""
    d = pd.DataFrame(rows)
    d = d[d["lineup"] >= 3]
    nrank = d["rank"] / (d["lineup"] - 1)  # 0 = perfect, 0.5 = chance
    per_player = nrank.groupby(d["nfl_id"]).mean()
    by_pos = nrank.groupby(d["group"].str.split("_").str[0]).mean()
    return {
        "n_players": int(per_player.size),
        "mean_normalized_rank": float(per_player.mean()),
        "top1_rate": float((d["rank"] == 0).mean()),
        "top1_chance": float((1 / d["lineup"]).mean()),
        "p_value": float(stats.wilcoxon(per_player - 0.5, alternative="less").pvalue),
        "mean_normalized_rank_by_role": by_pos.round(3).to_dict(),
    }


def save(results: list[dict], name: str = "fingerprint") -> dict:
    rows = [r for res in results for r in res["rows"]]
    d = pd.DataFrame(rows)
    emb = [c for c in d if c.startswith("e") and c[1:].isdigit()]
    # Each player's embedding comes only from models that never saw their games.
    d.groupby("nfl_id")[emb].mean().reset_index().to_parquet(
        artifacts_dir() / f"{name}_embeddings.parquet", index=False)
    return summarize(rows)
