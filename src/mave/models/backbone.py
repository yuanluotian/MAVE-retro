from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from mave.core.registry import MODEL_REGISTRY


@dataclass(frozen=True, slots=True)
class CompletionScore:
    """
    Score assigned by a causal language model to one candidate
    completion.

    logprob_sum:
        Sum of token log-probabilities.

    logprob_mean:
        Mean token log-probability. This reduces token-length
        bias when choice labels tokenize into different numbers
        of tokens.
    """

    text: str
    logprob_sum: Tensor
    logprob_mean: Tensor
    num_tokens: int


@MODEL_REGISTRY.register("llama3_8b_instruct")
class LLMBackbone:
    """
    Causal-LM backbone used by one trainable policy.

    The paper specifies Llama-3-8B-Instruct as the backbone but
    does not specify the exact prompt templates, parameter-
    efficient fine-tuning scheme, or classification head.

    This reproduction therefore treats each policy as a
    constrained-choice language-model policies:
        prompt + candidate completion -> sequence log-probability.
    """

    def __init__(
        self,
        pretrained_name: str = (
            "meta-llama/Meta-Llama-3-8B-Instruct"
        ),
        *,
        dtype: torch.dtype | str = torch.bfloat16,
        device_map: str | Mapping[str, Any] | None = "auto",
        trust_remote_code: bool = False,
        use_flash_attention_2: bool = False,
        tokenizer_kwargs: Mapping[str, Any] | None = None,
        model_kwargs: Mapping[str, Any] | None = None,
    ) -> None:

        tokenizer_kwargs = dict(
            tokenizer_kwargs or {}
        )
        model_kwargs = dict(
            model_kwargs or {}
        )

        if isinstance(dtype, str):
            dtype = self._resolve_dtype(dtype)

        self.pretrained_name = pretrained_name

        self.tokenizer: PreTrainedTokenizerBase = (
            AutoTokenizer.from_pretrained(
                pretrained_name,
                trust_remote_code=trust_remote_code,
                **tokenizer_kwargs,
            )
        )

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = (
                self.tokenizer.eos_token
            )

        self.tokenizer.padding_side = "left"

        if use_flash_attention_2:
            model_kwargs.setdefault(
                "attn_implementation",
                "flash_attention_2",
            )

        self.model: PreTrainedModel = (
            AutoModelForCausalLM.from_pretrained(
                pretrained_name,
                torch_dtype=dtype,
                device_map=device_map,
                trust_remote_code=trust_remote_code,
                **model_kwargs,
            )
        )

        self.model.config.pad_token_id = (
            self.tokenizer.pad_token_id
        )

    # ========================================================
    # Chat formatting
    # ========================================================

    def render_chat(
        self,
        messages: Sequence[
            Mapping[str, str]
        ],
        *,
        add_generation_prompt: bool = True,
    ) -> str:
        """
        Convert structured chat messages to the model's native
        chat template.
        """

        if not messages:
            raise ValueError(
                "messages must not be empty."
            )

        return self.tokenizer.apply_chat_template(
            list(messages),
            tokenize=False,
            add_generation_prompt=(
                add_generation_prompt
            ),
        )

    # ========================================================
    # Candidate scoring
    # ========================================================

    def score_completions(
        self,
        prompt: str,
        completions: Sequence[str],
        *,
        normalize_by_length: bool = True,
    ) -> Tensor:
        """
        Compute one scalar score for every candidate completion.

        Parameters
        ----------
        prompt:
            Already-rendered model prompt.

        completions:
            Candidate action strings.

        normalize_by_length:
            If True, use mean token log-probability.
            If False, use total completion log-probability.

        Returns
        -------
        Tensor
            Shape [num_completions].

        Notes
        -----
        This operation intentionally preserves autograd so that
        the same scoring function can later be used while
        recomputing policy log-probabilities for GRPO.
        """

        if not completions:
            raise ValueError(
                "completions must not be empty."
            )

        scores = [
            self.score_completion(
                prompt,
                completion,
            )
            for completion in completions
        ]

        values = [
            (
                score.logprob_mean
                if normalize_by_length
                else score.logprob_sum
            )
            for score in scores
        ]

        return torch.stack(values)

    def score_completion(
        self,
        prompt: str,
        completion: str,
    ) -> CompletionScore:
        """
        Score one completion conditioned on prompt.
        """

        if not prompt:
            raise ValueError(
                "prompt must not be empty."
            )

        if not completion:
            raise ValueError(
                "completion must not be empty."
            )

        full_text = (
            prompt + completion
        )

        input_ids, completion_mask = (
            self._tokenize_with_completion_mask(
                prompt=prompt,
                full_text=full_text,
            )
        )

        input_ids = input_ids.to(
            self.input_device
        )

        completion_mask = (
            completion_mask.to(
                self.input_device
            )
        )

        outputs = self.model(
            input_ids=input_ids,
        )

        logits = outputs.logits

        # Causal shift:
        # token t is predicted from positions < t.
        shifted_logits = logits[:, :-1, :]
        shifted_targets = input_ids[:, 1:]

        shifted_mask = (
            completion_mask[:, 1:]
        )

        token_logprobs = (
            torch.log_softmax(
                shifted_logits,
                dim=-1,
            )
            .gather(
                dim=-1,
                index=shifted_targets.unsqueeze(-1),
            )
            .squeeze(-1)
        )

        selected = token_logprobs[
            shifted_mask
        ]

        if selected.numel() == 0:
            raise RuntimeError(
                "Completion produced no scoreable tokens."
            )

        return CompletionScore(
            text=completion,
            logprob_sum=selected.sum(),
            logprob_mean=selected.mean(),
            num_tokens=int(
                selected.numel()
            ),
        )

    # ========================================================
    # Tokenization
    # ========================================================

    def _tokenize_with_completion_mask(
        self,
        *,
        prompt: str,
        full_text: str,
    ) -> tuple[Tensor, Tensor]:
        """
        Identify completion tokens using character offsets.

        Fast tokenizers expose offset_mapping. A prefix-based
        fallback is provided for tokenizers without that feature.
        """

        prompt_char_length = len(prompt)

        try:
            encoded = self.tokenizer(
                full_text,
                add_special_tokens=False,
                return_tensors="pt",
                return_offsets_mapping=True,
            )

            offsets = encoded.pop(
                "offset_mapping"
            )[0]

            input_ids = encoded[
                "input_ids"
            ]

            completion_mask = (
                offsets[:, 1]
                > prompt_char_length
            )

            completion_mask = (
                completion_mask.unsqueeze(0)
            )

            return (
                input_ids,
                completion_mask,
            )

        except (
            TypeError,
            NotImplementedError,
        ):
            pass

        # ----------------------------------------------------
        # Fallback for slow tokenizers.
        # ----------------------------------------------------

        prompt_ids = self.tokenizer(
            prompt,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"]

        full_ids = self.tokenizer(
            full_text,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"]

        prompt_length = (
            prompt_ids.shape[1]
        )

        if (
            full_ids.shape[1]
            <= prompt_length
        ):
            raise RuntimeError(
                "Completion tokenization failed."
            )

        mask = torch.zeros_like(
            full_ids,
            dtype=torch.bool,
        )

        mask[:, prompt_length:] = True

        return full_ids, mask

    # ========================================================
    # Training helpers
    # ========================================================

    def train(self) -> None:
        self.model.train()

    def eval(self) -> None:
        self.model.eval()

    def parameters(self):
        return self.model.parameters()

    def named_parameters(self):
        return self.model.named_parameters()

    def freeze(self) -> None:
        for parameter in (
            self.model.parameters()
        ):
            parameter.requires_grad_(False)

    def unfreeze(self) -> None:
        for parameter in (
            self.model.parameters()
        ):
            parameter.requires_grad_(True)

    # ========================================================
    # Properties
    # ========================================================

    @property
    def input_device(
        self,
    ) -> torch.device:
        """
        Device on which input tensors should be placed.

        For ordinary and device_map='auto' loading, the embedding
        layer provides a reliable input device.
        """
        embeddings = (
            self.model
            .get_input_embeddings()
        )

        return next(
            embeddings.parameters()
        ).device

    # ========================================================
    # Internal
    # ========================================================

    @staticmethod
    def _resolve_dtype(
        dtype: str,
    ) -> torch.dtype:

        mapping = {
            "float32": torch.float32,
            "fp32": torch.float32,
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
        }

        key = dtype.lower()

        if key not in mapping:
            raise ValueError(
                f"Unsupported dtype: {dtype!r}"
            )

        return mapping[key]
