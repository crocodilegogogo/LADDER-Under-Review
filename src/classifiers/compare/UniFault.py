
import os
import time
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from math import sqrt
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from utils.utils import (
    get_test_loss_acc,
    log_history,
    model_predict,
    plot_learning_history,
    save_metrics_per_cv,
    save_models,
)

UNIFAULT_TINY_URL = os.getenv("UNIFAULT_TINY_URL", "https://tinyurl.com/mtdytebs")


def _to_unifault_input(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 4:
        if x.size(2) == 1:
            return x[:, :, 0, :]
        return x.reshape(x.size(0), x.size(1) * x.size(2), x.size(3))
    if x.dim() == 3:
        return x
    if x.dim() == 2:
        return x.unsqueeze(1)
    raise ValueError(f"Unsupported input shape for UniFault adaptation: {tuple(x.shape)}")


def _download_file(url: str, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dst)
    return dst


def _validate_checkpoint(path: str) -> None:
    ckpt = torch.load(path, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise ValueError(f"Invalid UniFault checkpoint: {path}")
    _ = ckpt.get("state_dict", ckpt)


def _resolve_pretrained_checkpoint(model_type: str, pretrained_checkpoint: Optional[str]) -> Optional[str]:
    """
    Resolution order:
    1) explicit function arg
    2) UNIFAULT_PRETRAINED_CKPT
    3) if model_type == tiny and UNIFAULT_AUTO_DOWNLOAD=true:
       auto-download Tiny checkpoint into UNIFAULT_CACHE_DIR
    """
    ckpt = pretrained_checkpoint or os.getenv("UNIFAULT_PRETRAINED_CKPT")
    if ckpt:
        return ckpt

    auto_download = os.getenv("UNIFAULT_AUTO_DOWNLOAD", "true").lower() in {"1", "true", "yes"}
    if not auto_download:
        return None

    if model_type != "tiny":
        raise ValueError("Auto-download is only supported for UniFault Tiny, because the official README exposes only a Tiny download link.")

    # cache_dir = Path(os.getenv("UNIFAULT_CACHE_DIR", str(Path.home() / ".cache" / "unifault" / "tiny")))
    cache_dir = Path(os.getenv("UNIFAULT_CACHE_DIR", os.path.join(os.getcwd(), "classifiers", "compare", "UniFault", "tiny")))
    ckpt_name = os.getenv("UNIFAULT_TINY_FILENAME", "pretrain-epoch=1.ckpt")
    ckpt_path = cache_dir / ckpt_name

    if not ckpt_path.exists():
        print(f"Downloading UniFault Tiny checkpoint to: {ckpt_path}")
        _download_file(UNIFAULT_TINY_URL, ckpt_path)

    _validate_checkpoint(str(ckpt_path))
    return str(ckpt_path)


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super().__init__()
        self.scale = scale
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1.0 / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        return V.contiguous(), None


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None, lora_rank=None, lora_alpha=4):
        super().__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.lora_rank = lora_rank

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(queries, keys, values, attn_mask)
        out = out.view(B, L, -1)
        return self.out_projection(out), attn


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Linear(d_model, d_ff)
        self.conv2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.SiLU()

    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y)))
        y = self.dropout(self.conv2(y))
        return self.norm2(x + y), attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super().__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        attns = []
        for attn_layer in self.attn_layers:
            x, attn = attn_layer(x, attn_mask=attn_mask)
            attns.append(attn)
        x = self.norm(x)
        return x, attns


class PatchEmbed(nn.Module):
    def __init__(self, seq_len, patch_size=16, stride=16, embed_dim=768):
        super().__init__()
        self.num_patches = int((seq_len - patch_size) / stride + 1)
        self.kernel = patch_size
        self.stride = patch_size
        self.input_layer = nn.Linear(patch_size, embed_dim)

    def forward(self, x):
        x = x.unfold(dimension=-1, size=self.kernel, step=self.stride)  # [B, C, N, P]
        b, m, n, p = x.shape
        x = x.reshape(b * m, n, p)
        return self.input_layer(x)


class Transformer_bkbone(nn.Module):
    """
    This class mirrors the official UniFault key structure so that
    checkpoint keys like:
      model.pos_embed
      model.patch_embed.input_layer.weight
      model.encoder.attn_layers.0.attention.query_projection.weight
      ...
    can match almost one-to-one.
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.patch_embed = PatchEmbed(
            seq_len=args.seq_len, patch_size=args.patch_size, stride=args.patch_size, embed_dim=args.embed_dim
        )

        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, args.embed_dim), requires_grad=True)
        self.pos_drop = nn.Dropout(p=args.dropout)

        lora_rank = None
        lora_alpha = 16
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, 1, attention_dropout=args.dropout, output_attention=False),
                        args.embed_dim, args.heads, lora_rank=lora_rank, lora_alpha=lora_alpha
                    ),
                    args.embed_dim,
                    4 * args.embed_dim,
                    dropout=args.dropout,
                    activation='gelu'
                ) for _ in range(args.depth)
            ],
            norm_layer=torch.nn.LayerNorm(args.embed_dim)
        )

        # Keep these two modules to match the official checkpoint keys
        self.input_layer = nn.Linear(args.patch_size, args.embed_dim)
        self.pretrain_head = nn.Linear(args.embed_dim, args.patch_size)

        self.head = nn.Linear(args.embed_dim, args.num_classes)

    def forward_features(self, x):
        x_patch = self.patch_embed(x)
        x_patch = x_patch + self.pos_embed
        x_patch = self.pos_drop(x_patch)
        features, _ = self.encoder(x_patch)
        features = torch.reshape(features, (-1, self.args.num_channels * features.shape[-2], features.shape[-1]))
        return features

    def forward(self, x):
        x = _to_unifault_input(x)
        features = self.forward_features(x)
        predictions = self.head(features.mean(1)).squeeze()
        return predictions, features

    def load_pretrained_filtered(self, checkpoint_path: str) -> Tuple[int, int, List[str]]:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        current = self.state_dict()

        matched = {}
        unmatched = []
        for k, v in state_dict.items():
            k_clean = k
            if k_clean.startswith("model."):
                k_clean = k_clean[len("model."):]
            if k_clean in current and current[k_clean].shape == v.shape:
                matched[k_clean] = v
            else:
                unmatched.append(k)

        self.load_state_dict(matched, strict=False)
        return len(matched), len(current), unmatched


def _build_args(num_classes: int, seq_len: int, num_channels: int,
                model_type: str = "tiny", patch_size: int = 64,
                patch_stride: int = 64, dropout: float = 0.3) -> SimpleNamespace:
    config_map = {
        "tiny":  {"embed_dim": 128, "heads": 4,  "depth": 4},
        "small": {"embed_dim": 256, "heads": 8,  "depth": 8},
        "base":  {"embed_dim": 512, "heads": 12, "depth": 12},
    }
    if model_type not in config_map:
        raise ValueError(f"Unsupported model_type={model_type}. Choose from {list(config_map.keys())}.")
    cfg = config_map[model_type]
    return SimpleNamespace(
        num_classes=num_classes,
        seq_len=seq_len,
        num_channels=num_channels,
        patch_size=patch_size,
        patch_stride=patch_stride,
        dropout=dropout,
        **cfg,
    )


def UniFault(nb_classes: int, data_length: int, input_channels: int = 1,
             model_type: Optional[str] = None, patch_size: int = 64,
             patch_stride: int = 64, dropout: float = 0.3,
             pretrained_checkpoint: Optional[str] = None) -> Transformer_bkbone:
    model_type = model_type or os.getenv("UNIFAULT_MODEL_TYPE", "tiny")
    resolved_ckpt = _resolve_pretrained_checkpoint(model_type, pretrained_checkpoint)

    if resolved_ckpt is None and os.getenv("UNIFAULT_ALLOW_SCRATCH", "false").lower() not in {"1", "true", "yes"}:
        raise ValueError(
            "UniFault is running without a pretrained checkpoint. "
            "Set UNIFAULT_PRETRAINED_CKPT, enable UNIFAULT_AUTO_DOWNLOAD for tiny, "
            "or explicitly allow scratch with UNIFAULT_ALLOW_SCRATCH=true."
        )

    args = _build_args(nb_classes, data_length, input_channels, model_type, patch_size, patch_stride, dropout)
    model = Transformer_bkbone(args)

    if resolved_ckpt:
        matched, total, unmatched = model.load_pretrained_filtered(resolved_ckpt)
        print(f"UniFault pretrained weights loaded: matched {matched}/{total} tensors from {resolved_ckpt}")
        if unmatched:
            print("First unmatched keys:", unmatched[:10])

        min_ratio = float(os.getenv("UNIFAULT_MIN_MATCH_RATIO", "0.8"))
        if total > 0 and matched / total < min_ratio:
            raise ValueError(
                f"Too few UniFault tensors matched ({matched}/{total}). "
                "Checkpoint likely does not correspond to the selected model variant."
            )

    return model


def train_op(network, EPOCH, BATCH_SIZE, LR,
             train_x, train_y, val_x, val_y,
             output_directory_models, log_training_duration, test_split):
    drop_last_flag = bool(train_x.shape[0] % BATCH_SIZE == 1)
    dataset = Data.TensorDataset(torch.FloatTensor(train_x), torch.tensor(train_y).long())
    train_loader = Data.DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=drop_last_flag)

    lr_results, loss_train_results, accuracy_train_results = [], [], []
    loss_validation_results, accuracy_validation_results = [], []

    # optimizer = torch.optim.AdamW(network.parameters(), lr=LR, weight_decay=1e-4)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, T_max=max(EPOCH, 1), eta_min=max(LR * 0.01, 1e-6)
    # )
    # loss_function = nn.CrossEntropyLoss()

    # LR = 0.001
    # prepare optimizer&scheduler&loss_function
    optimizer = torch.optim.Adam(network.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5,
                                                           patience=10,
                                                           min_lr=LR/100)
    loss_function = nn.CrossEntropyLoss()

    torch.save(network.state_dict(), output_directory_models + "init_model.pkl")
    start_time = time.time()

    for epoch in range(EPOCH):
        network.train()
        for x, y in train_loader:
            batch_x = x.cuda()
            batch_y = y.cuda()
            output_bc, _ = network(batch_x)
            loss = loss_function(output_bc, batch_y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
            optimizer.step()

        network.eval()
        loss_train, accuracy_train = get_test_loss_acc(network, loss_function, train_x, train_y, test_split)
        loss_validation, accuracy_validation = get_test_loss_acc(network, loss_function, val_x, val_y, test_split)
        network.train()
        scheduler.step(loss_train)

        lr = optimizer.param_groups[0]['lr']
        lr_results.append(lr)
        loss_train_results.append(loss_train)
        accuracy_train_results.append(accuracy_train)
        loss_validation_results.append(loss_validation)
        accuracy_validation_results.append(accuracy_validation)

        # print training process
        if (epoch + 1) % 10 == 0:
            print('Epoch:', (epoch + 1), '|lr:', lr,
                  '| train_loss:', loss_train,
                  '| train_acc:', accuracy_train,
                  '| validation_loss:', loss_validation,
                  '| validation_acc:', accuracy_validation)

        save_models(network, output_directory_models,
                    loss_train, loss_train_results,
                    accuracy_validation, accuracy_validation_results)

    per_training_duration = time.time() - start_time
    log_training_duration.append(per_training_duration)

    torch.save(network.state_dict(), output_directory_models + "last_model.pkl")
    history = log_history(
        EPOCH, lr_results, loss_train_results, accuracy_train_results,
        loss_validation_results, accuracy_validation_results, output_directory_models
    )
    plot_learning_history(EPOCH, history, output_directory_models)
    return history, per_training_duration, log_training_duration


def predict_tr_val_test(network, nb_classes, LABELS,
                        train_x, val_x, test_x,
                        train_y, val_y, test_y,
                        scores, per_training_duration,
                        fold_id, valid_index,
                        output_directory_models,
                        test_split):
    network_obj = network
    best_validation_model = output_directory_models + "best_validation_model.pkl"
    network_obj.load_state_dict(torch.load(best_validation_model))
    network_obj.eval()

    pred_train = np.array(model_predict(network_obj, train_x, train_y, test_split))
    pred_valid = np.array(model_predict(network_obj, val_x, val_y, test_split))
    pred_test = np.array(model_predict(network_obj, test_x, test_y, test_split))

    score = {
        "logloss": {"train": [], "valid": [], "test": []},
        "accuracy": {"train": [], "valid": [], "test": []},
        "macro-precision": {"train": [], "valid": [], "test": []},
        "macro-recall": {"train": [], "valid": [], "test": []},
        "macro-f1": {"train": [], "valid": [], "test": []},
        "weighted-f1": {"train": [], "valid": [], "test": []},
        "micro-f1": {"train": [], "valid": [], "test": []},
        "per_class_f1": {"train": [], "valid": [], "test": []},
        "confusion_matrix": {"train": [], "valid": [], "test": []},
    }

    loss_function = nn.CrossEntropyLoss()
    for pred, X, y, mode in zip(
        [pred_train, pred_valid, pred_test],
        [train_x, val_x, test_x],
        [train_y, val_y, test_y],
        ["train", "valid", "test"]
    ):
        loss, acc = get_test_loss_acc(network_obj, loss_function, X, y, test_split)
        pred_label = pred.argmax(axis=1)

        scores["logloss"][mode].append(loss)
        scores["accuracy"][mode].append(acc)
        scores["macro-precision"][mode].append(precision_score(y, pred_label, average="macro"))
        scores["macro-recall"][mode].append(recall_score(y, pred_label, average="macro"))
        scores["macro-f1"][mode].append(f1_score(y, pred_label, average="macro"))
        scores["weighted-f1"][mode].append(f1_score(y, pred_label, average="weighted"))
        scores["micro-f1"][mode].append(f1_score(y, pred_label, average="micro"))
        scores["per_class_f1"][mode].append(f1_score(y, pred_label, average=None))
        scores["confusion_matrix"][mode].append(confusion_matrix(y, pred_label))

        score["logloss"][mode].append(loss)
        score["accuracy"][mode].append(acc)
        score["macro-precision"][mode].append(precision_score(y, pred_label, average="macro"))
        score["macro-recall"][mode].append(recall_score(y, pred_label, average="macro"))
        score["macro-f1"][mode].append(f1_score(y, pred_label, average="macro"))
        score["weighted-f1"][mode].append(f1_score(y, pred_label, average="weighted"))
        score["micro-f1"][mode].append(f1_score(y, pred_label, average="micro"))
        score["per_class_f1"][mode].append(f1_score(y, pred_label, average=None))
        score["confusion_matrix"][mode].append(confusion_matrix(y, pred_label))

    save_metrics_per_cv(score, per_training_duration, fold_id, nb_classes, LABELS, output_directory_models)
    return pred_train, pred_valid, pred_test, scores
