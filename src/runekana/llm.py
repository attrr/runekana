"""
llm.py — LLM-based reading verification for furigana annotations.

Exposes verify_candidates() as the single entry point.
Supports Gemini (Vertex AI) and OpenAI-compatible providers.
"""

from abc import ABC, abstractmethod
import textwrap
import logging
import random
import time
import re
import concurrent.futures
from typing import Optional

from pydantic import BaseModel
from openai import OpenAI as OpenAIClient
from google import genai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    wait_random,
    retry_if_exception,
    before_sleep_log,
)

log = logging.getLogger(__name__)


class Candidate(BaseModel):
    """Input structure for a reading candidate and its context."""

    word: str
    reading: str
    context: str


class Hint(BaseModel):
    """Output structure for a model-suggested reading."""

    word: str
    proposed: str
    is_correct: bool
    correction: Optional[str] = None


class Hints(BaseModel):
    """Wrapper for a collection of Hint objects."""

    hints: list[Hint]


class LLM(ABC):
    """Interface for handling LLM communication and usage tracking."""

    def __init__(self, model_name: str) -> None:
        """Initialize the LLM client and session state."""
        prompt = """\
            あなたは日本語言語学の専門家です。以下は日本語の小説から抽出した漢字語とその仮説読みです。
            各読みが文脈上正しいか検証してください。
            正しくない場合は正しい読みを提供してください。
            【重要ルールの厳守】
            1. 修正する際は、対象の単語「全体」の読み（送り仮名や平仮名部分を含む）を平仮名で提供してください。漢字部分のみの回答は厳禁です。
            2. 対象単語の活用形や送り仮名を「絶対に」勝手に変更・補完しないでください。（例：「学ん」という単語に対し「まなった」や「学んだ」と補完して答えるのは禁止です。必ず元の形のまま「まなん」と答えてください）
            3. 原文の平仮名部分と、読みの平仮名部分は完全に一致している必要があります。
            4. 小説のルビであるため、一般的に広く使われる慣用読みや訓読（例：「半年前」を「はんとしまえ」と読むなど）を、不必要に硬い音読（「はんとしぜん」など）に過剰修正しないでください。元の読みが日本語の口語として自然に通用するものであれば、そのまま正しいと判定してください。

            以下のJSON形式で回答してください:
            {"hints": [{"word": "対象語", "proposed": "提示読み", "is_correct": true/false, "correction": "修正読みまたはnull"}]}

            検証対象:
        """
        self.prompt = textwrap.dedent(prompt)
        self.input_tokens: int = 0
        self.output_tokens: int = 0
        self.provider = self.__class__.__name__.upper()
        self.model_name = model_name

    def increase_counter(self, input_tokens: int, output_tokens: int):
        """Record token consumption for the current session."""
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens

    def fetch_statistics(self) -> str:
        """Return token usage statistics in a multi-line formatted report.
        Generate a summary of token usage and costs."""
        total = self.input_tokens + self.output_tokens
        return (
            f"\n=== {self.provider} API Usage ===\n"
            f" Model: {self.model_name}\n"
            f" Input Tokens:      {self.input_tokens}\n"
            f" Output Tokens:     {self.output_tokens}\n"
            f" Total Tokens:      {total}\n"
            f"========================="
        )

    @abstractmethod
    def predict(self, prompt: str) -> str:
        """Execute a raw LLM request with retry and error handling."""

    @staticmethod
    def is_retryable(e: BaseException) -> bool:
        """Retry for known network/transport errors and server-side HTTP errors (429, 5xx).
        Fail fast for client errors (400, 401, 403, 404) and unknown exceptions."""

        status = getattr(e, "status_code", None) or getattr(e, "code", None)
        if isinstance(status, int):
            if 400 <= status < 500 and status != 429:
                return False
            if status == 429 or status >= 500:
                return True

        return any(
            cls.__name__
            in {
                "RemoteProtocolError",
                "ConnectError",
                "ReadTimeout",
                "WriteTimeout",
                "PoolTimeout",
                "ConnectTimeout",
                "TimeoutException",
                "NetworkError",
                "ReadError",
                "WriteError",
            }
            for cls in type(e).__mro__
        )

    def predict_with_retry(
        self, prompt: str, sleep_min: float = 0.5, sleep_max: float = 2.5
    ) -> str:
        """Execute a raw LLM request with retry and error handling."""
        time.sleep(random.uniform(sleep_min, sleep_max))

        @retry(
            wait=wait_exponential(multiplier=1.5, min=2, max=120)
            + wait_random(min=0, max=3),
            stop=stop_after_attempt(8),
            retry=retry_if_exception(self.is_retryable),
            before_sleep=before_sleep_log(log, logging.WARNING, exc_info=True),
            reraise=True,
        )
        def _execute():
            return self.predict(prompt)

        return _execute()

    def serialize_predict(self, s: str) -> list[Hint]:
        """Convert raw response string to a list of Hint objects."""
        try:
            return Hints.model_validate_json(s.strip()).hints
        except Exception as e:
            log.warning(f"Failed to serialize prediction: {e}")
            return []

    def infer(self, l: list[Candidate]) -> list[Hint]:
        """Infer readings for candidates through the complete pipeline."""
        if not l:
            return []

        # build prompt
        words = "\n".join(
            f"{i+1}. {w.word} → {w.reading}  (文脈: ...{w.context}...)"
            for i, w in enumerate(l)
        )
        prompt = self.prompt + words

        # predict
        response = self.predict_with_retry(prompt)
        log.debug("Raw response:\n%s", response[:500] if response else "(empty)")
        return self.serialize_predict(response)


class Vertex(LLM):
    """Google Cloud Vertex AI provider implementation via unified genai SDK."""

    def __init__(
        self,
        project: str,
        location: str = "global",
        model_name: str = "gemini-3.1-flash-lite-preview",
    ) -> None:
        super().__init__(model_name=model_name)
        self.client = genai.Client(vertexai=True, project=project, location=location)

    def predict(self, prompt: str) -> str:
        """Execute prediction using Vertex AI."""
        log.debug("Prompt:\n%s", prompt)
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": Hints,
                "temperature": 0.0,
            },
        )

        # response.text is a @property that raises ValueError when
        # safety-filtered or no valid candidates.
        try:
            text = response.text if response.text else ""
        except ValueError as e:
            log.warning("Gemini response has no usable text (safety-filtered?): %s", e)
            return ""

        if hasattr(response, "usage_metadata") and response.usage_metadata:
            input_tokens = getattr(response.usage_metadata, "prompt_token_count", 0)
            output_tokens = getattr(
                response.usage_metadata, "candidates_token_count", 0
            )
            self.increase_counter(input_tokens, output_tokens)

        return text


class Gemini(Vertex):
    """Google AI Studio (Gemini) provider implementation via unified genai SDK."""

    def __init__(
        self, api_key: str, model_name: str = "gemini-3.1-flash-lite-preview"
    ) -> None:
        LLM.__init__(self, model_name)
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name


class OpenAI(LLM):
    """OpenAI-compatible provider implementation."""

    def __init__(
        self, api_key: str, base_url: str | None = None, model_name: str = "gpt-4o-mini"
    ) -> None:
        super().__init__(model_name=model_name)
        self.client = OpenAIClient(
            api_key=api_key, base_url=base_url, max_retries=0, timeout=120.0
        )

    def predict(self, prompt: str) -> str:
        """Execute prediction using OpenAI-compatible API."""
        log.debug("Calling completions.create (model: %s)...", self.model_name)
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        log.debug("API returned successfully.")

        # Update token usage
        if hasattr(response, "usage") and response.usage:
            self.increase_counter(
                response.usage.prompt_tokens, response.usage.completion_tokens
            )

        if not response.choices:
            raise RuntimeError(f"Empty choices from {self.model_name}: {response}")

        return response.choices[0].message.content or ""

    def serialize_predict(self, s: str) -> list[Hint]:
        """Robustly parse JSON using regex from potential conversational responses."""
        match = re.search(r"\{.*\}", s, re.DOTALL)
        json_str = match.group() if match else s

        try:
            return Hints.model_validate_json(json_str).hints
        except Exception as e:
            log.warning(
                "OpenAI serialization failed: %s. Raw snippet: %s", e, json_str[:100]
            )
            return []


def verify_candidates(
    grouped_candidates: dict[tuple, list],
    local_dict: dict[str, str],
    dict_path: str,
    llm: LLM,
    save_fn=None,
    concurrency: int = 5,
    batch_size: int = 100,
) -> int:
    """
    Verify reading candidates via LLM in parallel batches.
    Returns number of corrections applied.
    Corrections are written into local_dict in-place AND applied to all Token instances in memory.
    """
    log.info(
        "Sending %d unique words to %s (%s)...",
        len(grouped_candidates),
        llm.provider,
        llm.model_name,
    )

    candidates_objs = []
    cand_to_key = {}
    for k, refs in grouped_candidates.items():
        if len(k) == 3:
            w, r, c = k
        else:
            w, r = k
            # Non-contextual mode: use context from first occurrence
            c = refs[0][1] if refs else ""
        cand = Candidate(word=w, reading=r, context=c)
        candidates_objs.append(cand)
        cand_to_key[id(cand)] = k

    corrections = 0
    batches = [
        candidates_objs[i : i + batch_size]
        for i in range(0, len(candidates_objs), batch_size)
    ]

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=concurrency)
    future_to_batch = {executor.submit(llm.infer, batch): batch for batch in batches}

    try:
        completed = 0
        for future in concurrent.futures.as_completed(future_to_batch):
            completed += 1
            log.info("-" * 60)
            log.info(">> [ BATCH %d / %d COMPLETED ] <<", completed, len(batches))
            hints = future.result()
            batch = future_to_batch[future]

            def apply_hint(cand: Candidate, hint: Hint):
                nonlocal corrections
                if (
                    not hint.is_correct
                    and hint.correction
                    and hint.correction != hint.proposed
                ):
                    ref_list = grouped_candidates[cand_to_key[id(cand)]]
                    first_context = ref_list[0][1] if ref_list else ""
                    ctx = cand.context or first_context

                    log.info(
                        "  Correction: %s (%s -> %s)",
                        hint.word,
                        hint.proposed,
                        hint.correction,
                    )
                    if ctx:
                        log.debug("    Context: %s", ctx)

                    local_dict[hint.word] = hint.correction
                    corrections += 1
                    for token, _ in ref_list:
                        token.reading = hint.correction

            if len(hints) == len(batch):
                for cand, hint in zip(batch, hints):
                    apply_hint(cand, hint)
            else:
                hint_map = {(h.word, h.proposed): h for h in hints}
                for cand in batch:
                    hint = hint_map.get((cand.word, cand.reading))
                    if hint:
                        apply_hint(cand, hint)

            # Save incrementally to prevent data loss if script is killed
            if corrections > 0 and dict_path and save_fn:
                save_fn(dict_path, local_dict)

    except KeyboardInterrupt:
        log.warning("\nVerification interrupted by user! Cancelling pending tasks...")
        for future in future_to_batch:
            future.cancel()
        executor.shutdown(wait=False)
        raise
    finally:
        executor.shutdown(wait=False)

    log.info(
        "Verification complete. Applied %d total corrections to local dictionary.",
        corrections,
    )
    if corrections > 0 and dict_path:
        log.info("Final dictionary saved to %s", dict_path)

    log.info(llm.fetch_statistics())
    return corrections
