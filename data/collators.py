import torch


class BaseCollator(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def _pad_batch(self, batch, max_length):
        batch["input_ids"] = [torch.nn.functional.pad(ids, (max_length - len(ids), 0), value=self.tokenizer.pad_token_id) for ids in batch["input_ids"]]
        batch["labels"]    = [torch.nn.functional.pad(labels, (max_length - len(labels), 0), value=self.tokenizer.pad_token_id) for labels in batch["labels"]]
        batch["attention_mask"] = [torch.nn.functional.pad(attention_mask, (max_length - len(attention_mask), 0), value=0) for attention_mask in batch["attention_mask"]]
        if "doc_id" in batch:
            # -1 sentinel: never collides with a real doc id (real ids start at 0 per packed row).
            batch["doc_id"] = [torch.nn.functional.pad(doc_id, (max_length - len(doc_id), 0), value=-1) for doc_id in batch["doc_id"]]

    def prepare_batch(self, batch, max_length=None):
        # 1) Handle empty
        if not batch:
            return {"input_ids": [], "labels": [], "attention_mask": [], "images": [], "doc_id": []}

        # 2) Drop None rows
        batch = [s for s in batch if s is not None]
        if not batch:
            return {"input_ids": [], "labels": [], "attention_mask": [], "images": [], "doc_id": []}

        # batch is a list of dicts, each containing "input_ids", "attention_mask", "labels", "images"
        # let's convert it to a dict of lists of tensors
        batch = {k: [item[k] for item in batch] for k in batch[0]}

        if max_length is not None:
            batch = self._discard_samples_that_are_too_long(batch, max_length)

        if len(batch["input_ids"]) == 0:
            return batch

        # Pad samples to max length
        if max_length is not None:
            max_len = max_length
        else:
            max_len = max(map(len, batch["input_ids"]))
        self._pad_batch(batch, max_len) #  dictionaries in Python are mutable and passed by reference

        result = {
            "input_ids": torch.stack(batch["input_ids"]),
            "attention_mask": torch.stack(batch["attention_mask"]),
            "images": batch["images"],
            "labels": torch.stack(batch["labels"]),
        }
        if "doc_id" in batch:
            result["doc_id"] = torch.stack(batch["doc_id"])
        if "stream_state" in batch:
            # Per-row data position stamps (ConstantLengthDataset), consumed by train.py for exact resume
            result["stream_state"] = batch["stream_state"]
        return result

    def _discard_samples_that_are_too_long(self, batch, max_length):
        # Filters every per-row key together (input_ids, labels, attention_mask, images, and doc_id / stream_state when present)
        keep = [i for i, ids in enumerate(batch["input_ids"]) if len(ids) <= max_length]
        return {k: [v[i] for i in keep] for k, v in batch.items()}


class VQACollator(BaseCollator):  # Visual Question Answering Collator
    def __init__(self, tokenizer, max_length):
        self.max_length = max_length
        super().__init__(tokenizer)

    def _pad_batch(self, batch, max_length):  # Reimplementing to use -100 as the pad value for labels, so that it's ignored by the loss
        batch["input_ids"] = [torch.nn.functional.pad(ids, (max_length - len(ids), 0), value=self.tokenizer.pad_token_id) for ids in batch["input_ids"]]
        batch["labels"]    = [torch.nn.functional.pad(labels, (max_length - len(labels), 0), value=-100) for labels in batch["labels"]]
        batch["attention_mask"] = [torch.nn.functional.pad(attention_mask, (max_length - len(attention_mask), 0), value=0) for attention_mask in batch["attention_mask"]]
        if "doc_id" in batch:
            # -1 sentinel: never collides with a real doc id (real ids start at 0 per packed row).
            batch["doc_id"] = [torch.nn.functional.pad(doc_id, (max_length - len(doc_id), 0), value=-1) for doc_id in batch["doc_id"]]

    def __call__(self, batch):
        batch = self.prepare_batch(batch, max_length=self.max_length)
        return batch
