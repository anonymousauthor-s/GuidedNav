"""Stage 1/2 loaders with assistant-answer-only causal language targets.

Media loading, Qwen preprocessing, position IDs and collator contracts come
from the existing VQA/VLN datasets. These wrappers alter only label masking;
the Stage 3 training and final-model evaluation implementations are untouched.
"""

import torch

from .assistant_targets import assistant_target_spans
from .data_processor import IGNORE_INDEX, LazySupervisedDataset
from .vln_unified_dataset import VLNUnifiedDataset


class _AssistantTargets:
    def _answer_labels(self, input_ids):
        tokenizer = self.tokenizer
        start = tokenizer.convert_tokens_to_ids("<|im_start|>")
        end = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if start is None or end is None or start == end:
            raise ValueError("The tokenizer must expose Qwen chat boundary tokens")
        headers = {
            role: tokenizer.encode("<|im_start|>" + role + "\n", add_special_tokens=False)
            for role in ("system", "user", "assistant")
        }
        if any(not header or header[0] != start for header in headers.values()):
            raise ValueError("The tokenizer does not use the expected Qwen chat headers")
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("A dataset item must contain one tokenized conversation")
        spans = assistant_target_spans(
            input_ids[0].tolist(), im_start_id=start, im_end_id=end, role_headers=headers,
        )
        labels = torch.full_like(input_ids, IGNORE_INDEX)
        for begin, finish in spans:
            labels[0, begin:finish] = input_ids[0, begin:finish]
        return labels


class EarlyStageVQADataset(_AssistantTargets, LazySupervisedDataset):
    def _get_item(self, sources):
        item = super()._get_item(sources)
        item["labels"] = self._answer_labels(item["input_ids"])
        return item


class EarlyStageVLNDataset(_AssistantTargets, VLNUnifiedDataset):
    def __getitem__(self, index):
        item = super().__getitem__(index)
        item["labels"] = self._answer_labels(item["input_ids"])
        return item
