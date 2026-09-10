# -*- coding: utf-8 -*-
"""
Mantis classifier for AWD-Net-Under-Review.

Recommended location:
    AWD-Net-Under-Review/src/classifiers/compare/Mantis.py

What this file does:
    1. Integrates the official Mantis time-series foundation model into AWD-Net.
    2. Automatically downloads the pretrained Mantis checkpoint.
    3. Stores the Hugging Face cache INSIDE the AWD-Net project by default:
           AWD-Net-Under-Review/pretrained/mantis_hf_cache/
    4. Provides AWD-Net-compatible train_op(...) and predict_tr_val_test(...).

Required packages:
    pip install mantis-tsfm huggingface_hub einops

Required change in src/utils/constants.py::create_classifier:

    if classifier_name == 'Mantis':
        from classifiers.compare import Mantis
        return Mantis.MantisClassifier(INPUT_CHANNEL, data_length, nb_classes), Mantis

Recommended command:
    python main.py --PATTERN TRAIN --DATASETS CWRU_10 --CLASSIFIERS_all Mantis \
        --BATCH_SIZE 32 --EPOCH 100 --LR 0.0002 --CV_SPLITS 5 --test_split 20

The pretrained checkpoint is loaded from:
    paris-noah/Mantis-8M

The actual project-local cache path after download is usually:
    AWD-Net-Under-Review/pretrained/mantis_hf_cache/models--paris-noah--Mantis-8M/
"""

import os
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data

from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from utils.utils import (
    get_test_loss_acc,
    log_history,
    model_predict,
    plot_learning_history,
    save_metrics_per_cv,
    save_models,
)


MANTIS_REPO_ID = "paris-noah/Mantis-8M"


def _infer_awdn_project_root() -> Path:
    """
    Infer AWD-Net project root from this file location.

    Expected deployed path:
        <project_root>/src/classifiers/compare/Mantis.py

    Therefore:
        Path(__file__).resolve().parents[3] == <project_root>
    """
    current = Path(__file__).resolve()

    # Strong check: search upward for a directory that looks like AWD-Net project root.
    for parent in current.parents:
        if (parent / "src" / "classifiers").exists() and (parent / "src" / "utils").exists():
            return parent

    # Fallback for the expected fixed path.
    try:
        return current.parents[3]
    except Exception:
        return Path.cwd().resolve()


def _default_project_cache_dir() -> Path:
    """
    Return the project-local Hugging Face cache directory.

    Priority:
        1. AWDNET_MANTIS_CACHE_DIR
        2. MANTIS_CACHE_DIR
        3. <AWD-Net project>/pretrained/mantis_hf_cache
    """
    env_cache = os.environ.get("AWDNET_MANTIS_CACHE_DIR", None)
    if env_cache is None:
        env_cache = os.environ.get("MANTIS_CACHE_DIR", None)

    if env_cache:
        return Path(env_cache).expanduser().resolve()

    return (_infer_awdn_project_root() / "pretrained" / "mantis_hf_cache").resolve()


def _prepare_project_local_hf_cache(cache_dir: Path) -> Path:
    """
    Create the local cache directory and force Hugging Face Hub to use it.

    This is the key change compared with the previous version:
        pretrained weights are cached under the AWD-Net project instead of ~/.cache/huggingface/hub.
    """
    cache_dir = Path(cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Make the project-local path the active hub cache for this Python process.
    # This helps even when a third-party from_pretrained wrapper does not expose cache_dir cleanly.
    os.environ["HF_HUB_CACHE"] = str(cache_dir)

    # Keep token/config cache near the project as well, but do not override user's HF_HOME if already set.
    os.environ.setdefault("HF_HOME", str(cache_dir.parent / "hf_home"))

    return cache_dir


def _is_offline_mode() -> bool:
    return os.environ.get("HF_HUB_OFFLINE", "0").strip() in {"1", "ON", "TRUE", "true", "True"}


def _download_mantis_snapshot(repo_id: str, cache_dir: Path) -> Optional[str]:
    """
    Explicitly download the Mantis repository snapshot into the project-local cache.

    We still call MantisV1.from_pretrained(...) afterwards, because that method knows how to
    instantiate the official architecture. The snapshot_download call is used to guarantee that
    the cache location is the AWD-Net project directory.

    Returns
    -------
    local_snapshot_path:
        Local snapshot directory path if huggingface_hub is available and succeeds.
        None if snapshot_download is not available, in which case from_pretrained still attempts download.
    """
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        warnings.warn(
            "huggingface_hub is not available, so explicit snapshot_download is skipped. "
            "MantisV1.from_pretrained will still be attempted. "
            "Install with: pip install huggingface_hub. Original error: {}".format(repr(exc))
        )
        return None

    try:
        local_dir = snapshot_download(
            repo_id=repo_id,
            cache_dir=str(cache_dir),
            local_files_only=_is_offline_mode(),
        )
        return local_dir
    except TypeError:
        # Compatibility with older huggingface_hub versions.
        local_dir = snapshot_download(
            repo_id=repo_id,
            cache_dir=str(cache_dir),
        )
        return local_dir
    except Exception as online_exc:
        # If online download fails, try to use an already-cached local snapshot.
        try:
            local_dir = snapshot_download(
                repo_id=repo_id,
                cache_dir=str(cache_dir),
                local_files_only=True,
            )
            warnings.warn(
                "Online Mantis checkpoint download failed, but a local cached snapshot was found at: {}. "
                "Original online error: {}".format(local_dir, repr(online_exc))
            )
            return local_dir
        except Exception as local_exc:
            raise RuntimeError(
                "Failed to download or locate Mantis pretrained checkpoint.\n"
                "Repo id: {}\n"
                "Expected project-local cache dir: {}\n"
                "Online error: {}\n"
                "Local-cache error: {}\n"
                "Please check network/proxy access to Hugging Face, or pre-copy the Hugging Face cache "
                "into the above directory.".format(repo_id, cache_dir, repr(online_exc), repr(local_exc))
            )


def _load_mantis_v1_class():
    """Import Mantis lazily so AWD-Net can still import this file before installing the dependency."""
    try:
        from mantis.architecture import MantisV1
    except Exception as exc:
        raise ImportError(
            "Cannot import MantisV1. Please install the official Mantis package first:\n"
            "    pip install mantis-tsfm huggingface_hub einops\n"
            "Original import error: {}".format(repr(exc))
        )
    return MantisV1


def _safe_torch_load(path: str):
    """Load a PyTorch state dict in a version-compatible way."""
    map_location = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


class MantisClassifier(nn.Module):
    """
    AWD-Net-compatible classifier using the Mantis pretrained time-series foundation model.

    Parameters
    ----------
    input_channel:
        Number of input sensor channels. For common AWD-Net fault diagnosis datasets, this is usually 1.
    data_length:
        Original sequence length in AWD-Net. The forward pass resizes it to seq_len.
    numclass:
        Number of fault categories.
    seq_len:
        Mantis input length. 512 is recommended because Mantis-8M was pretrained with length 512.
    checkpoint:
        Hugging Face repo id or local model id. Default: paris-noah/Mantis-8M.
    cache_dir:
        Hugging Face cache root. If None, defaults to:
            <AWD-Net project>/pretrained/mantis_hf_cache
    freeze_backbone:
        True: train only the classification head.
        False: full fine-tuning of Mantis + head.
    aggregation:
        Multichannel aggregation method.
        "mean": encode each channel independently and average embeddings.
        "concat": encode each channel independently and concatenate embeddings.
    dropout:
        Dropout before final linear classifier.
    verbose:
        Print checkpoint cache path at initialization.
    """

    def __init__(
        self,
        input_channel: int = 1,
        data_length: Optional[int] = None,
        numclass: int = 10,
        seq_len: int = 512,
        checkpoint: str = MANTIS_REPO_ID,
        cache_dir: Optional[str] = None,
        freeze_backbone: bool = False,
        aggregation: str = "mean",
        dropout: float = 0.1,
        verbose: bool = True,
    ):
        super().__init__()

        if aggregation not in {"mean", "concat"}:
            raise ValueError("aggregation must be either 'mean' or 'concat'.")

        self.input_channel = int(input_channel)
        self.data_length = data_length
        self.numclass = int(numclass)
        self.seq_len = int(seq_len)
        self.checkpoint = checkpoint
        self.freeze_backbone = bool(freeze_backbone)
        self.aggregation = aggregation

        self.project_root = _infer_awdn_project_root()
        self.cache_dir = _prepare_project_local_hf_cache(
            Path(cache_dir).expanduser().resolve() if cache_dir is not None else _default_project_cache_dir()
        )

        if verbose:
            print("[Mantis] AWD-Net project root:", str(self.project_root))
            print("[Mantis] Hugging Face cache dir:", str(self.cache_dir))
            print("[Mantis] Checkpoint:", self.checkpoint)
            print(
                "[Mantis] Expected cached model folder:",
                str(self.cache_dir / "models--paris-noah--Mantis-8M"),
            )

        # Download/check local snapshot before model construction.
        _download_mantis_snapshot(self.checkpoint, self.cache_dir)

        MantisV1 = _load_mantis_v1_class()
        device = "cuda" if torch.cuda.is_available() else "cpu"

        backbone = MantisV1(seq_len=self.seq_len, device=device, pre_training=False)

        # Always pass cache_dir. If the installed hub mixin is older, fall back while HF_HUB_CACHE
        # has already been forced to the project-local directory.
        try:
            backbone = backbone.from_pretrained(
                self.checkpoint,
                cache_dir=str(self.cache_dir),
                local_files_only=_is_offline_mode(),
            )
        except TypeError:
            try:
                backbone = backbone.from_pretrained(
                    self.checkpoint,
                    cache_dir=str(self.cache_dir),
                )
            except TypeError:
                backbone = backbone.from_pretrained(self.checkpoint)

        backbone.pre_training = False
        self.backbone = backbone

        hidden_dim = int(getattr(self.backbone, "hidden_dim", 256))
        head_in_dim = hidden_dim if self.aggregation == "mean" else hidden_dim * self.input_channel

        self.head = nn.Sequential(
            nn.LayerNorm(head_in_dim),
            nn.Dropout(dropout),
            nn.Linear(head_in_dim, self.numclass),
        )

        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    @staticmethod
    def _to_bct(x: torch.Tensor) -> torch.Tensor:
        """
        Convert common PyTorch time-series tensor shapes to [B, C, T].

        Supported:
            [B, T]
            [B, C, T]
            [B, 1, C, T]
            [B, C, 1, T]
            [B, C, H, T] -> [B, C*H, T]
        """
        x = x.float()

        if x.dim() == 2:
            x = x.unsqueeze(1)

        elif x.dim() == 3:
            pass

        elif x.dim() == 4:
            if x.shape[1] == 1:
                x = x.squeeze(1)
            elif x.shape[2] == 1:
                x = x.squeeze(2)
            else:
                b, c, h, t = x.shape
                x = x.reshape(b, c * h, t)

        else:
            raise ValueError("Unsupported input shape for MantisClassifier: {}".format(tuple(x.shape)))

        if x.dim() != 3:
            raise ValueError("Input should be converted to [B, C, T], got {}".format(tuple(x.shape)))

        return x

    def _resize_to_mantis_length(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] == self.seq_len:
            return x
        return F.interpolate(x, size=self.seq_len, mode="linear", align_corners=False)

    def extract_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract Mantis embeddings.

        MantisV1 is used as a univariate encoder. For multichannel signals, each channel is
        sent independently through Mantis, then channel embeddings are aggregated.
        """
        x = self._to_bct(x)
        x = self._resize_to_mantis_length(x)

        b, c, t = x.shape
        x_flat = x.reshape(b * c, 1, t)

        z = self.backbone(x_flat)   # [B*C, D]
        z = z.reshape(b, c, -1)     # [B, C, D]

        if self.aggregation == "mean":
            z = z.mean(dim=1)
        else:
            # Keep fixed classifier input dimension when runtime channel count differs.
            if c < self.input_channel:
                pad = z.new_zeros(b, self.input_channel - c, z.shape[-1])
                z = torch.cat([z, pad], dim=1)
            elif c > self.input_channel:
                z = z[:, : self.input_channel, :]
            z = z.reshape(b, -1)

        return z

    def forward(self, x: torch.Tensor):
        embedding = self.extract_embedding(x)
        logits = self.head(embedding)
        return logits, embedding


# Alias for compatibility with possible AWD-Net naming conventions.
Mantis = MantisClassifier


def _effective_mantis_lr(lr: float) -> float:
    """
    AWD-Net's default LR can be too large for Transformer full fine-tuning.
    Cap large LR values to a stable default unless the user explicitly passes <= 1e-3.
    """
    lr = float(lr)
    if lr > 1e-3:
        warnings.warn(
            "Received LR={}. This is usually too large for Mantis full fine-tuning; "
            "using LR=2e-4 instead. Recommended grid: 1e-4, 2e-4, 1e-3.".format(lr)
        )
        return 2e-4
    return lr


def train_op(
    network,
    EPOCH,
    BATCH_SIZE,
    LR,
    train_x,
    train_y,
    val_x,
    val_y,
    output_directory_models,
    log_training_duration,
    test_split,
):
    """
    AWD-Net-compatible fine-tuning function.

    Returns
    -------
    history, per_training_duration, log_training_duration
    """
    os.makedirs(output_directory_models, exist_ok=True)

    if train_x.shape[0] % BATCH_SIZE == 1:
        drop_last_flag = True
    else:
        drop_last_flag = False

    torch_dataset = Data.TensorDataset(torch.FloatTensor(train_x), torch.tensor(train_y).long())
    train_loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=drop_last_flag,
        pin_memory=torch.cuda.is_available(),
    )

    lr_results: List[float] = []
    loss_train_results: List[float] = []
    accuracy_train_results: List[float] = []
    loss_validation_results: List[float] = []
    accuracy_validation_results: List[float] = []

    effective_lr = _effective_mantis_lr(float(LR))

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, network.parameters()),
        lr=effective_lr,
        weight_decay=0.05,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(int(EPOCH), 1),
        eta_min=max(effective_lr * 0.01, 1e-6),
    )
    loss_function = nn.CrossEntropyLoss()

    output_directory_init = os.path.join(output_directory_models, "init_model.pkl")
    torch.save(network.state_dict(), output_directory_init)

    start_time = time.time()

    for epoch in range(EPOCH):
        network.train()

        for _, (x, y) in enumerate(train_loader):
            if torch.cuda.is_available():
                batch_x = x.cuda(non_blocking=True)
                batch_y = y.cuda(non_blocking=True)
            else:
                batch_x = x
                batch_y = y

            output_bc, _ = network(batch_x)
            loss_total = loss_function(output_bc, batch_y)

            optimizer.zero_grad(set_to_none=True)
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()

        network.eval()
        with torch.no_grad():
            loss_train, accuracy_train = get_test_loss_acc(
                network, loss_function, train_x, train_y, test_split
            )
            loss_validation, accuracy_validation = get_test_loss_acc(
                network, loss_function, val_x, val_y, test_split
            )

        lr = optimizer.param_groups[0]["lr"]
        lr_results.append(lr)
        loss_train_results.append(loss_train)
        accuracy_train_results.append(accuracy_train)
        loss_validation_results.append(loss_validation)
        accuracy_validation_results.append(accuracy_validation)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                "Epoch:", epoch + 1,
                "| lr:", lr,
                "| train_loss:", loss_train,
                "| train_acc:", accuracy_train,
                "| validation_loss:", loss_validation,
                "| validation_acc:", accuracy_validation,
            )

        save_models(
            network,
            output_directory_models,
            loss_train,
            loss_train_results,
            accuracy_validation,
            accuracy_validation_results,
        )

    per_training_duration = time.time() - start_time
    log_training_duration.append(per_training_duration)

    output_directory_last = os.path.join(output_directory_models, "last_model.pkl")
    torch.save(network.state_dict(), output_directory_last)

    history = log_history(
        EPOCH,
        lr_results,
        loss_train_results,
        accuracy_train_results,
        loss_validation_results,
        accuracy_validation_results,
        output_directory_models,
    )
    plot_learning_history(EPOCH, history, output_directory_models)

    return history, per_training_duration, log_training_duration


def predict_tr_val_test(
    network,
    nb_classes,
    LABELS,
    train_x,
    val_x,
    test_x,
    train_y,
    val_y,
    test_y,
    scores,
    per_training_duration,
    fold_id,
    valid_index,
    output_directory_models,
    test_split,
):
    """
    AWD-Net-compatible prediction and metric logging function.
    """
    network_obj = network
    best_validation_model = os.path.join(output_directory_models, "best_validation_model.pkl")
    network_obj.load_state_dict(_safe_torch_load(best_validation_model))
    network_obj.eval()

    pred_train = np.array(model_predict(network_obj, train_x, train_y, test_split))
    pred_valid = np.array(model_predict(network_obj, val_x, val_y, test_split))
    pred_test = np.array(model_predict(network_obj, test_x, test_y, test_split))

    score: Dict[str, Dict[str, List]] = {
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
        ["train", "valid", "test"],
    ):
        loss, acc = get_test_loss_acc(network_obj, loss_function, X, y, test_split)
        pred_label = pred.argmax(axis=1)

        macro_precision = precision_score(y, pred_label, average="macro", zero_division=0)
        macro_recall = recall_score(y, pred_label, average="macro", zero_division=0)
        macro_f1 = f1_score(y, pred_label, average="macro", zero_division=0)
        weighted_f1 = f1_score(y, pred_label, average="weighted", zero_division=0)
        micro_f1 = f1_score(y, pred_label, average="micro", zero_division=0)
        per_class_f1 = f1_score(y, pred_label, average=None, zero_division=0)
        cm = confusion_matrix(y, pred_label)

        scores["logloss"][mode].append(loss)
        scores["accuracy"][mode].append(acc)
        scores["macro-precision"][mode].append(macro_precision)
        scores["macro-recall"][mode].append(macro_recall)
        scores["macro-f1"][mode].append(macro_f1)
        scores["weighted-f1"][mode].append(weighted_f1)
        scores["micro-f1"][mode].append(micro_f1)
        scores["per_class_f1"][mode].append(per_class_f1)
        scores["confusion_matrix"][mode].append(cm)

        score["logloss"][mode].append(loss)
        score["accuracy"][mode].append(acc)
        score["macro-precision"][mode].append(macro_precision)
        score["macro-recall"][mode].append(macro_recall)
        score["macro-f1"][mode].append(macro_f1)
        score["weighted-f1"][mode].append(weighted_f1)
        score["micro-f1"][mode].append(micro_f1)
        score["per_class_f1"][mode].append(per_class_f1)
        score["confusion_matrix"][mode].append(cm)

    save_metrics_per_cv(
        score,
        per_training_duration,
        fold_id,
        nb_classes,
        LABELS,
        output_directory_models,
    )

    return pred_train, pred_valid, pred_test, scores
