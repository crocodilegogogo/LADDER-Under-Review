
import os
import time
from pathlib import Path
from typing import Optional, Tuple

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

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
except Exception:
    AutoModelForCausalLM = None
    AutoTokenizer = None
    AutoConfig = None

try:
    from peft import LoraConfig, TaskType, get_peft_model
except Exception:
    LoraConfig = None
    TaskType = None
    get_peft_model = None

try:
    from huggingface_hub import snapshot_download
except Exception:
    snapshot_download = None

# DESCRIPTION_TEXT = [
#     "Fault-Free",
#     "Minor Inner Ring Fault",
#     "Moderate Inner Ring Fault",
#     "Severe Inner Ring Fault",
#     "Minor Ball Fault",
#     "Moderate Ball Fault",
#     "Severe Ball Fault",
#     "Minor Outer Ring Fault",
#     "Moderate Outer Ring Fault",
#     "Severe Outer Ring Fault",
# ]
DESCRIPTION_TEXT = [
    "Fault1",
    "Fault2",
    "Fault3",
    "Fault4",
    "Fault5",
    "Fault6",
    "Fault7",
    "Fault8",
    "Fault9",
    "Fault10",
    "Fault11",
    "Fault12",
    "Fault13",
    "Fault14",
    "Fault15",
    "Fault16",
    "Fault17",
    "Fault18",
    "Fault19",
    "Fault20",
]
SYS_PROMPT = (
    "As an expert in bearing fault diagnosis with extensive knowledge in mechanical engineering and failure "
    "analysis, you can assess the state of bearings. Typically, bearing states are categorized as ["
    "Fault-Free, Minor Inner Ring Fault, Moderate Inner Ring Fault, Severe Inner Ring Fault, Minor Ball "
    "Fault, Moderate Ball Fault, Severe Ball Fault, Minor Outer Ring Fault, Moderate Outer Ring Fault, "
    "Severe Outer Ring Fault]. Based on the description of the bearing state, answer my questions."
)
DEFAULT_USER_TEMPLATE = "Please identify the current bearing state according to the injected signal tokens."
SIGNAL_TOKEN_ID = int(os.getenv("BEARLLM_SIGNAL_TOKEN_ID", "151857"))
DESCRIPTION_LEN = int(os.getenv("BEARLLM_DESCRIPTION_LEN", "5"))
BEARLLM_NF = int(os.getenv("BEARLLM_NF", "24000"))
BEARLLM_BETA = float(os.getenv("BEARLLM_BETA", "0.01"))

# def _resolve_qwen_dir() -> str:
#     explicit = os.getenv("BEARLLM_QWEN_DIR")
#     if explicit and Path(explicit).exists():
#         return explicit

#     auto_download = os.getenv("BEARLLM_AUTO_DOWNLOAD", "false").lower() in {"1", "true", "yes"}
#     repo_id = os.getenv("BEARLLM_QWEN_REPO", "Qwen/Qwen2.5-1.5B-Instruct")
#     cache_dir = os.getenv("BEARLLM_CACHE_DIR", str(Path.home() / ".cache" / "bearllm_qwen2_5_1_5b"))

#     if auto_download:
#         if snapshot_download is None:
#             raise ImportError("huggingface_hub is required for BEARLLM_AUTO_DOWNLOAD")
#         return snapshot_download(repo_id=repo_id, local_dir=cache_dir, local_dir_use_symlinks=False)

#     return repo_id

def _resolve_qwen_dir() -> str:
    explicit = os.getenv("BEARLLM_QWEN_DIR")
    if explicit and Path(explicit).exists():
        return explicit

    auto_download = os.getenv("BEARLLM_AUTO_DOWNLOAD", "true").lower() in {"1", "true", "yes"}
    repo_id = os.getenv("BEARLLM_QWEN_REPO", "Qwen/Qwen2.5-1.5B-Instruct")
    # cache_dir = os.getenv("BEARLLM_CACHE_DIR", str(Path.home() / ".cache" / "bearllm_qwen2_5_1_5b"))
    cache_dir = os.getenv("BEARLLM_CACHE_DIR", os.path.join(os.getcwd(), "classifiers", "compare", "bearllm_qwen2_5_1_5b"))

    if auto_download:
        if snapshot_download is None:
            raise ImportError("huggingface_hub is required for BearLLM auto-download")
        local_dir = snapshot_download(repo_id=repo_id, local_dir=cache_dir, local_dir_use_symlinks=False)
        return local_dir

    return repo_id

def _squeeze_signal_batch(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 4:
        return x[:, 0, 0, :]
    if x.dim() == 3:
        if x.size(1) == 1:
            return x[:, 0, :]
        if x.size(-1) == 1:
            return x[:, :, 0]
    if x.dim() == 2:
        return x
    raise ValueError(f"Unsupported input shape for BearLLM adaptation: {tuple(x.shape)}")

def dct_torch(x: torch.Tensor) -> torch.Tensor:
    n = x.size(-1)
    v = torch.cat([x, x.flip(dims=[-1])], dim=-1)
    Vc = torch.fft.fft(v, dim=-1)
    k = torch.arange(n, device=x.device, dtype=x.dtype)
    Wr = torch.cos(-np.pi * k / (2.0 * n))
    Wi = torch.sin(-np.pi * k / (2.0 * n))
    V = Vc[..., :n]
    return 2.0 * (V.real * Wr - V.imag * Wi)

def dcn_transform(x: torch.Tensor, nf: int = BEARLLM_NF, beta: float = BEARLLM_BETA) -> torch.Tensor:
    f = dct_torch(x)
    if f.size(-1) >= nf:
        f = f[..., :nf]
    else:
        pad = torch.zeros(f.size(0), nf - f.size(-1), device=f.device, dtype=f.dtype)
        f = torch.cat([f, pad], dim=-1)
    norm = torch.linalg.norm(f, dim=-1, keepdim=True).clamp_min(1e-8)
    f = beta * (nf ** 0.5) * f / norm
    return f

class ChannelAttention(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(1, in_channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.se = nn.Sequential(
            nn.Conv1d(in_channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, in_channels, 1),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.se(self.avg_pool(x))
        max_out = self.se(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)

class ConvWide(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 8, stride: int = 8) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride)
        self.norm = nn.BatchNorm1d(out_channels)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.norm(self.conv(x)))

class ConvMultiScale(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        if out_channels % 4 != 0:
            raise ValueError("out_channels must be divisible by 4")
        branch = out_channels // 4
        self.conv1 = nn.Conv1d(in_channels, branch, 1, 4, padding=0)
        self.conv3 = nn.Conv1d(in_channels, branch, 3, 4, padding=1)
        self.conv5 = nn.Conv1d(in_channels, branch, 5, 4, padding=2)
        self.conv7 = nn.Conv1d(in_channels, branch, 7, 4, padding=3)
        self.norm = nn.BatchNorm1d(branch * 3)
        self.relu = nn.ReLU(inplace=True)
        self.ca = ChannelAttention(branch * 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv1(x)
        x3 = self.conv3(x)
        x5 = self.conv5(x)
        x7 = self.conv7(x)
        x_ms = torch.cat([x3, x5, x7], dim=1)
        x_ms = self.relu(self.norm(x_ms))
        x_ms = self.ca(x_ms) * x_ms
        return torch.cat([x1, x_ms], dim=1)

class FeatureEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv_query = ConvWide(1, 60, 8, 8)
        self.conv_ref = ConvWide(1, 8, 8, 8)
        self.conv_res = ConvWide(1, 60, 8, 8)
        self.conv = nn.Sequential(
            ConvMultiScale(128, 128),
            ConvMultiScale(128, 128),
            ConvMultiScale(128, 128),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        query = x[:, :1, :]
        ref = x[:, 1:, :]
        res = query - ref
        query = self.conv_query(query)
        ref = self.conv_ref(ref)
        res = self.conv_res(res)
        x = torch.cat([query, ref, res], dim=1)
        return self.conv(x)

class AlignmentLayer(nn.Module):
    def __init__(self, llm_hidden_size: int, num_classes: int, description_len: int = DESCRIPTION_LEN) -> None:
        super().__init__()
        self.linear1 = nn.Linear(128 * 47, 128)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(128, num_classes)
        self.softmax = nn.Softmax(dim=1)
        self.linear3 = nn.Linear(num_classes, description_len * llm_hidden_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.reshape(x.size(0), 47 * 128)
        x = self.linear1(x)
        x = self.relu(x)
        prior_logits = self.linear2(x)
        weights = self.softmax(prior_logits)
        x = self.linear3(weights)
        return x.reshape(x.size(0), DESCRIPTION_LEN, -1), prior_logits

class AlignmentAdapter(nn.Module):
    def __init__(self, llm_hidden_size: int, num_classes: int, tokenizer, embedding_weight: torch.Tensor) -> None:
        super().__init__()
        self.feature_encoder = FeatureEncoder()
        self.alignment_layer = AlignmentLayer(llm_hidden_size, num_classes)

        feature_ckpt = os.getenv("BEARLLM_FEATURE_ENCODER_CKPT")
        classifier_ckpt = os.getenv("BEARLLM_CLASSIFIER_CKPT")
        if feature_ckpt and Path(feature_ckpt).exists():
            try:
                self.feature_encoder.load_state_dict(torch.load(feature_ckpt, map_location="cpu"))
            except Exception as e:
                print(f"Warning: failed to load feature encoder checkpoint: {e}")
        if classifier_ckpt and Path(classifier_ckpt).exists():
            try:
                state = torch.load(classifier_ckpt, map_location="cpu")
                if "linear1.weight" in state:
                    self.alignment_layer.linear1.weight.data.copy_(state["linear1.weight"])
                    self.alignment_layer.linear1.bias.data.copy_(state["linear1.bias"])
                    self.alignment_layer.linear2.weight.data.copy_(state["linear2.weight"][:num_classes])
                    self.alignment_layer.linear2.bias.data.copy_(state["linear2.bias"][:num_classes])
            except Exception as e:
                print(f"Warning: failed to load classifier checkpoint: {e}")

        with torch.no_grad():
            rows = []
            for text in DESCRIPTION_TEXT[:num_classes]:
                token_ids = tokenizer(text, return_tensors='pt', add_special_tokens=False).input_ids[0]
                token_embeds = embedding_weight[token_ids]
                pooled = token_embeds.mean(dim=0)
                rows.append(pooled.repeat(DESCRIPTION_LEN))
            l3_weight = torch.stack(rows, dim=0)
            self.alignment_layer.linear3.weight.data.copy_(l3_weight.T)
            self.alignment_layer.linear3.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feats = self.feature_encoder(x)
        return self.alignment_layer(feats)

class ModifiedEmbedding(nn.Module):
    def __init__(self, base_embedding: nn.Module, adapter: AlignmentAdapter, llm_hidden_size: int) -> None:
        super().__init__()
        self.base_embedding = base_embedding
        self.adapter = adapter.to(base_embedding.weight.device)
        self.llm_hidden_size = llm_hidden_size
        self.current_signal_pair = None
        self.last_prior_logits = None

        if os.getenv("BEARLLM_TRAIN_TEXT_EMBED", "false").lower() not in {"1", "true", "yes"}:
            self.base_embedding.weight.requires_grad = False

    def set_signal_pair(self, signal_pair: torch.Tensor) -> None:
        self.current_signal_pair = signal_pair
        self.last_prior_logits = None

    def clear_cache(self) -> None:
        self.current_signal_pair = None
        self.last_prior_logits = None

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        device = input_ids.device
        vocab_max = self.base_embedding.weight.size(0) - 1
        output = torch.zeros(
            input_ids.size(0), input_ids.size(1), self.llm_hidden_size,
            device=device, dtype=self.base_embedding.weight.dtype
        )

        text_mask = input_ids < SIGNAL_TOKEN_ID
        if text_mask.any():
            text_ids = input_ids[text_mask].clamp(max=vocab_max)
            output[text_mask] = self.base_embedding(text_ids)

        signal_mask = input_ids >= SIGNAL_TOKEN_ID
        if signal_mask.any():
            if self.current_signal_pair is None:
                raise RuntimeError("Signal placeholders found but current_signal_pair is not set.")
            aligned, prior_logits = self.adapter(self.current_signal_pair.to(device))
            aligned = aligned.to(output.dtype)
            self.last_prior_logits = prior_logits
            for b in range(input_ids.size(0)):
                idx = torch.where(signal_mask[b])[0]
                if len(idx) > 0:
                    take = min(len(idx), aligned.size(1))
                    output[b, idx[:take], :] = aligned[b, :take, :]
        return output

class BearLLMLoRAClassifier(nn.Module):
    def __init__(self, num_classes: int, data_length: int, normal_label: int = 0) -> None:
        super().__init__()
        if AutoModelForCausalLM is None or AutoTokenizer is None or AutoConfig is None:
            raise ImportError("transformers is required for BearLLM")
        if LoraConfig is None or TaskType is None or get_peft_model is None:
            raise ImportError("peft is required for BearLLM")

        self.num_classes = int(num_classes)
        self.data_length = int(data_length)
        self.normal_label = int(normal_label)
        self.register_buffer("reference_signal", torch.zeros(1, self.data_length))

        qwen_dir = _resolve_qwen_dir()
        config = AutoConfig.from_pretrained(qwen_dir, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(qwen_dir, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        base_model = AutoModelForCausalLM.from_pretrained(
            qwen_dir, trust_remote_code=True, torch_dtype=dtype, config=config
        )
        self.llm_hidden_size = int(base_model.config.hidden_size)

        base_embedding = base_model.get_input_embeddings()
        self.adapter = AlignmentAdapter(self.llm_hidden_size, self.num_classes, self.tokenizer, base_embedding.weight.detach().cpu())
        self.modified_embedding = ModifiedEmbedding(base_embedding, self.adapter, self.llm_hidden_size)

        lora_config = LoraConfig(
            target_modules=os.getenv("BEARLLM_LORA_TARGET_MODULES", "all-linear"),
            task_type=TaskType.CAUSAL_LM,
            r=int(os.getenv("BEARLLM_LORA_R", "4")),
            lora_alpha=int(os.getenv("BEARLLM_LORA_ALPHA", "32")),
            lora_dropout=float(os.getenv("BEARLLM_LORA_DROPOUT", "0.1")),
        )
        self.qwen = get_peft_model(base_model, lora_config)

        self.classifier = nn.Sequential(
            nn.Linear(self.llm_hidden_size + self.num_classes, self.llm_hidden_size // 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.llm_hidden_size // 4, self.num_classes),
        )
        self.prompt_ids = self._build_prompt_template()

    def _build_prompt_template(self) -> torch.Tensor:
        user_part = os.getenv("BEARLLM_USER_PROMPT", DEFAULT_USER_TEMPLATE)
        system_ids = self.tokenizer(
            "<|im_start|>system\n" + SYS_PROMPT + "\n<|im_end|><|im_start|>user\n",
            return_tensors='pt', add_special_tokens=False
        ).input_ids[0]
        user_ids = self.tokenizer(
            user_part + "\n<|im_end|>\n<|im_start|>assistant\n",
            return_tensors='pt', add_special_tokens=False
        ).input_ids[0]
        placeholder = torch.arange(SIGNAL_TOKEN_ID, SIGNAL_TOKEN_ID + DESCRIPTION_LEN, dtype=torch.long)
        return torch.cat([system_ids, placeholder, user_ids], dim=0)

    def set_reference(self, x_data, y_data) -> None:
        signal = _squeeze_signal_batch(torch.as_tensor(x_data, dtype=torch.float32))
        labels = torch.as_tensor(y_data).view(-1)
        normal_mask = labels == self.normal_label
        ref = signal[normal_mask].mean(dim=0, keepdim=True) if normal_mask.any() else signal.mean(dim=0, keepdim=True)
        self.reference_signal.copy_(ref)

    def _build_signal_pair(self, x: torch.Tensor) -> torch.Tensor:
        signal = _squeeze_signal_batch(x)
        if signal.size(-1) != self.data_length:
            signal = torch.nn.functional.interpolate(signal.unsqueeze(1), size=self.data_length, mode="linear", align_corners=False).squeeze(1)
        ref = self.reference_signal.expand(signal.size(0), -1)
        query_freq = dcn_transform(signal)
        ref_freq = dcn_transform(ref)
        return torch.stack([query_freq, ref_freq], dim=1)

    def forward(self, x: torch.Tensor):
        signal_pair = self._build_signal_pair(x)
        device = signal_pair.device
        batch_size = signal_pair.size(0)

        input_ids = self.prompt_ids.to(device).unsqueeze(0).expand(batch_size, -1).clone()
        attention_mask = torch.ones_like(input_ids)

        self.modified_embedding.set_signal_pair(signal_pair)
        inputs_embeds = self.modified_embedding(input_ids)

        outputs = self.qwen(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden = outputs.hidden_states[-1]
        prior_logits = self.modified_embedding.last_prior_logits
        if prior_logits is None:
            raise RuntimeError("ModifiedEmbedding did not cache prior_logits.")
        prior_logits = prior_logits.to(hidden.dtype)

        placeholder_mask = input_ids >= SIGNAL_TOKEN_ID
        pooled_hidden = []
        for b in range(batch_size):
            idx = torch.where(placeholder_mask[b])[0]
            pooled_hidden.append(hidden[b, idx, :].mean(dim=0))
        pooled_hidden = torch.stack(pooled_hidden, dim=0)

        logits = self.classifier(torch.cat([pooled_hidden, prior_logits], dim=1).float())
        self.modified_embedding.clear_cache()
        return logits, pooled_hidden

def BearLLM(nb_classes: int, data_length: int, normal_label: int = 0) -> BearLLMLoRAClassifier:
    return BearLLMLoRAClassifier(nb_classes, data_length, normal_label)

def _create_optimizer_and_scheduler(network, lr, epochs, train_steps):
    params = [p for p in network.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    warmup_steps = max(10, train_steps)
    total_steps = max(epochs * train_steps, warmup_steps + 1)

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler

def train_op(network, EPOCH, BATCH_SIZE, LR,
             train_x, train_y, val_x, val_y,
             output_directory_models, log_training_duration, test_split):
    network.set_reference(train_x, train_y)
    drop_last_flag = bool(train_x.shape[0] % BATCH_SIZE == 1)
    dataset = Data.TensorDataset(torch.FloatTensor(train_x), torch.tensor(train_y).long())
    train_loader = Data.DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=drop_last_flag)

    lr_results, loss_train_results, accuracy_train_results = [], [], []
    loss_validation_results, accuracy_validation_results = [], []

    LR = 0.001
    optimizer, scheduler = _create_optimizer_and_scheduler(network, LR, EPOCH, max(1, len(train_loader)))
    loss_function = nn.CrossEntropyLoss()

    torch.save(network.state_dict(), output_directory_models + "init_model.pkl")
    start_time = time.time()

    for epoch in range(EPOCH):
        network.train()
        for x, y in train_loader:
            batch_x = x.cuda()
            batch_y = y.cuda()
            logits, _ = network(batch_x)
            loss = loss_function(logits, batch_y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

        network.eval()
        loss_train, accuracy_train = get_test_loss_acc(network, loss_function, train_x, train_y, test_split)
        loss_validation, accuracy_validation = get_test_loss_acc(network, loss_function, val_x, val_y, test_split)
        network.train()

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

        save_models(network, output_directory_models, loss_train, loss_train_results, accuracy_validation, accuracy_validation_results)

    per_training_duration = time.time() - start_time
    log_training_duration.append(per_training_duration)
    torch.save(network.state_dict(), output_directory_models + "last_model.pkl")

    history = log_history(EPOCH, lr_results, loss_train_results, accuracy_train_results,
                          loss_validation_results, accuracy_validation_results, output_directory_models)
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
    for pred, X, y, mode in zip([pred_train, pred_valid, pred_test],
                                [train_x, val_x, test_x],
                                [train_y, val_y, test_y],
                                ["train", "valid", "test"]):
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
