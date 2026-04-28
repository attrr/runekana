"""
llm.py — LLM-based reading verification for furigana annotations.

Exposes Verifier class for parallel batch verification.
Supports Gemini (Vertex AI) and OpenAI-compatible providers.
"""

from pathlib import Path
import re
import json
import gzip
import asyncio
import textwrap
import logging
import random
from datetime import datetime
from abc import ABC, abstractmethod
from typing import Optional, Any, Callable
from dataclasses import dataclass, field

from google.genai.types import HttpOptions
import httpx
from pydantic import BaseModel
from openai import AsyncOpenAI
from google import genai
from rich.progress import Progress, SpinnerColumn, TextColumn
from tenacity import (
    AsyncRetrying,
    stop_after_attempt,
    wait_exponential,
    wait_random,
    retry_if_exception,
    before_sleep_log,
)

from runekana import console
from runekana.text import has_kanji, split_okurigana

log = logging.getLogger("runekana.llm")


class Connectivity:
    DEFAULT_URL: str = "http://connectivitycheck.gstatic.com/generate_204"

    def __init__(self, canary_url: str | None = None) -> None:
        if canary_url:
            self.canary_url = canary_url
        else:
            self.canary_url = self.DEFAULT_URL
        self.is_online = asyncio.Event()
        self.is_online.set()
        self._recovery_task: asyncio.Task | None = None

    async def wait_until_online(self):
        await self.is_online.wait()

    def disconnect_occurs(self):
        # query gen204 first before decide
        try:
            resp = httpx.get(self.canary_url)
            if resp.status_code == 204:
                return
        except Exception:
            pass

        if self.is_online.is_set():
            self.is_online.clear()
            log.warning("Internet connection lost.")

        if self._recovery_task is None or self._recovery_task.done():
            self._recovery_task = asyncio.create_task(self.probe_until_online())

    async def probe_until_online(self):
        log.info("Starting background connectivity check...")
        try:
            async for attempt in AsyncRetrying(
                wait=wait_exponential(multiplier=2, min=5, max=60),
                retry=retry_if_exception(lambda e: True),
                before_sleep=before_sleep_log(log, logging.DEBUG),
            ):
                with attempt:
                    async with httpx.AsyncClient(timeout=5.0) as client:
                        resp = await client.get(self.canary_url)
                        if resp.status_code == 204:
                            log.info("Internet connection restored.")
                            self.is_online.set()
                            return
                        raise RuntimeError(f"Unexpected status: {resp.status_code}")
        except Exception as e:
            log.error("Fatal error during connectivity check: %s", e)
            self.is_online.set()  # temp set online to avoid deadlock


class Candidate(BaseModel):
    """Input structure for a reading candidate and its context."""

    id: int
    word: str
    reading: str
    context: str


class Hint(BaseModel):
    """Output structure for a model-suggested reading."""

    id: int
    word: str
    proposed: str
    is_correct: bool
    correction: Optional[str] = None


class Hints(BaseModel):
    """Wrapper for a collection of Hint objects."""

    hints: list[Hint]


class LLM(ABC):
    """Interface for handling LLM communication and usage tracking."""

    def __init__(
        self,
        model_name: str,
        canary_url: str = Connectivity.DEFAULT_URL,
    ) -> None:
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
            5. 提供された各項目の「id」は、回答の際にも必ずそのまま保持してください。

            以下のJSON形式で回答してください:
            {"hints": [{"id": 0, "word": "対象語", "proposed": "提示読み", "is_correct": true/false, "correction": "修正読みまたはnull"}]}

            検証対象:
        """
        self.prompt = textwrap.dedent(prompt)
        self.input_tokens: int = 0
        self.output_tokens: int = 0
        self.provider = self.__class__.__name__.upper()
        self.model_name = model_name
        self.monitor = Connectivity(canary_url)

    def increase_counter(self, input_tokens: int, output_tokens: int):
        """Record token consumption for the current session."""
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens

    def fetch_statistics(
        self, price_input: float = 0.0, price_output: float = 0.0
    ) -> str:
        """Return token usage statistics in a multi-line formatted report.
        Generate a summary of token usage and costs."""
        total = self.input_tokens + self.output_tokens
        if total == 0:
            return ""

        report = (
            f"\n=== {self.provider} API Usage ===\n"
            f" Model: {self.model_name}\n"
            f" Input Tokens:      {self.input_tokens}\n"
            f" Output Tokens:     {self.output_tokens}\n"
            f" Total Tokens:      {total}\n"
        )

        if price_input > 0 or price_output > 0:
            cost_in = (self.input_tokens / 1_000_000) * price_input
            cost_out = (self.output_tokens / 1_000_000) * price_output
            total_cost = cost_in + cost_out
            report += f" Estimated Cost:    ${total_cost:.4f}\n"

        report += "========================="
        return report

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.aclose()

    async def aclose(self):
        """Release asynchronous resources. Override in subclasses."""
        pass

    @abstractmethod
    async def predict(self, prompt: str) -> str:
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

        # Check for common network error names across providers/libraries
        network_errors = {
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
            "ServerDisconnectedError",
            "ClientOSError",
            "ClientPayloadError",
            "ClientConnectorError",
        }

        return any(cls.__name__ in network_errors for cls in type(e).__mro__)

    async def predict_with_retry(
        self, prompt: str, sleep_min: float = 0.5, sleep_max: float = 2.5
    ) -> str:
        """Execute a raw LLM request with async retry and error handling."""
        await asyncio.sleep(random.uniform(sleep_min, sleep_max))

        async for attempt in AsyncRetrying(
            wait=wait_exponential(multiplier=1.5, min=2, max=120)
            + wait_random(min=0, max=3),
            stop=stop_after_attempt(12),
            retry=retry_if_exception(self.is_retryable),
            before_sleep=before_sleep_log(log, logging.WARNING, exc_info=True),
            reraise=True,
        ):
            with attempt:
                await self.monitor.wait_until_online()

                try:
                    return await self.predict(prompt)
                except Exception as e:
                    if self.is_retryable(e):
                        self.monitor.disconnect_occurs()
                    raise
        return ""

    def serialize_predict(self, s: str) -> list[Hint]:
        """Convert raw response string to a list of Hint objects."""
        try:
            return Hints.model_validate_json(s.strip()).hints
        except Exception as e:
            log.warning(f"Failed to serialize prediction: {e}")
            return []

    async def infer(self, candidates: list[Candidate]) -> list[Hint]:
        """Infer readings for candidates through the complete pipeline."""
        if not candidates:
            return []

        # build prompt
        words = "\n".join(
            f"{c.id}. {c.word} → {c.reading}  (文脈: ...{c.context}...)"
            for c in candidates
        )
        prompt = self.prompt + words

        response = await self.predict_with_retry(prompt)
        log.debug("Raw response:\n%s", response[:500] if response else "(empty)")
        return self.serialize_predict(response)


class Vertex(LLM):
    """Google Cloud Vertex AI provider implementation via unified genai SDK."""

    def __init__(
        self,
        project: str,
        location: str = "global",
        model_name: str = "gemini-3.1-flash-lite-preview",
        canary_url: str = Connectivity.DEFAULT_URL,
    ) -> None:
        super().__init__(model_name=model_name, canary_url=canary_url)
        self.client = genai.Client(vertexai=True, project=project, location=location)

    async def predict(self, prompt: str) -> str:
        """Execute async prediction using Vertex AI."""
        log.debug("Prompt:\n%s", prompt)
        response = await self.client.aio.models.generate_content(
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

    @staticmethod
    async def _log_http_request(req: httpx.Request):
        log.info(f"request_sent: {req.method} {req.url}")

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        model_name: str = "gemini-3.1-flash-lite-preview",
        canary_url: str = Connectivity.DEFAULT_URL,
    ) -> None:
        LLM.__init__(self, model_name, canary_url=canary_url)
        self._httpx_async_client = httpx.AsyncClient(
            event_hooks={"request": [self._log_http_request]}
        )
        http_options = HttpOptions(
            base_url=base_url,
            httpx_async_client=self._httpx_async_client,
        )
        self.client = genai.Client(
            vertexai=False, api_key=api_key, http_options=http_options
        )
        self.base_url = base_url
        self.model_name = model_name

    async def aclose(self):
        """Close the internal httpx client."""
        if hasattr(self, "_httpx_async_client"):
            await self._httpx_async_client.aclose()


class OpenAI(LLM):
    """OpenAI-compatible provider implementation using AsyncOpenAI."""

    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
        model_name: str = "gpt-4o-mini",
        canary_url: str = Connectivity.DEFAULT_URL,
    ) -> None:
        super().__init__(model_name=model_name, canary_url=canary_url)
        self.client = AsyncOpenAI(
            api_key=api_key, base_url=base_url, max_retries=0, timeout=120.0
        )
        self.base_url = base_url

    async def aclose(self):
        """Close the AsyncOpenAI client."""
        if hasattr(self, "client"):
            await self.client.close()

    async def predict(self, prompt: str) -> str:
        """Execute async prediction using OpenAI-compatible API."""
        log.debug("Calling completions.create (model: %s)...", self.model_name)
        response = await self.client.chat.completions.create(
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


@dataclass
class VerificationJob:
    """Domain object representing a unique word/reading verification task and its DOM token references."""

    word: str
    proposed_reading: str
    context: str
    token_refs: list[Any] = field(default_factory=list)

    def to_candidate(self, id: int) -> Candidate:
        """Create the payload for the LLM request."""
        return Candidate(
            id=id, word=self.word, reading=self.proposed_reading, context=self.context
        )

    def apply_hint(self, hint: Hint, local_dict: dict[str, str]) -> bool:
        """Apply LLM correction back to local dict and DOM tokens. Returns True if applied."""
        if not hint.is_correct and hint.correction and hint.correction != hint.proposed:
            # Validate okurigana sync to prevent LLM hallucination (e.g., 仰い: おっしゃい -> あおぎ)
            if has_kanji(hint.word):
                segments = split_okurigana(hint.word, hint.correction)
                if len(segments) == 1:
                    _text_part, ruby_part = segments[0]
                    if ruby_part is None:
                        log.warning(
                            "  Rejected LLM correction: %s (%s -> %s) due to okurigana mismatch.",
                            hint.word,
                            hint.proposed,
                            hint.correction,
                        )
                        return False

            log.info(
                "  Correction: %s (%s -> %s)",
                hint.word,
                hint.proposed,
                hint.correction,
            )
            if self.context:
                log.debug("    Context: %s", self.context)

            local_dict[hint.word] = hint.correction
            for token in self.token_refs:
                token.reading = hint.correction
            return True
        return False


class Verifier:
    """Orchestrator for parallel LLM verification batches. Supports context manager."""

    def __init__(
        self,
        llm: LLM,
        local_dict: dict[str, str],
        dict_path: str,
        save_fn: Optional[Callable[[str, dict], None]] = None,
        concurrency: int = 5,
        batch_size: int = 100,
        price_input: float = 0.0,
        price_output: float = 0.0,
        generated_dir: Optional[str] = None,
        book_name: Optional[str] = None,
    ):
        self.llm = llm
        self.local_dict = local_dict
        self.dict_path = dict_path
        self.save_fn = save_fn
        self.concurrency = concurrency
        self.batch_size = batch_size
        self.price_input = price_input
        self.price_output = price_output
        self.generated_dir = generated_dir
        self.book_name = book_name
        self.semaphore = asyncio.Semaphore(concurrency)
        self.corrections = 0
        self.completed_batches = 0

        # Persistent progress bar, manually started during verification
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        )
        self.current_task_id: Any = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Perform final cleanup and print statistics on exit."""
        self.progress.stop()

        stats = self.llm.fetch_statistics(self.price_input, self.price_output)
        if stats:
            console.print(
                f"\n[bold yellow]LLM Usage Statistics:[/bold yellow]\n{stats}"
            )

        if exc_type is None:
            console.print(
                f"[bold green]Verification complete.[/bold green] Applied {self.corrections} corrections."
            )

    def _save_llm_output(
        self,
        batch: list[VerificationJob],
        hints: list[Hint],
        batch_index: int,
        batch_corrections: int,
    ):
        """Save a successfully verified batch to the generated-dir for tracing/collection."""
        if not self.generated_dir or not self.book_name:
            return

        # Ensure directory exists: {generated_dir}/{book_name}/
        target_dir = Path(self.generated_dir, self.book_name)
        target_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = target_dir.joinpath(f"{timestamp}-batch-{batch_index:03d}.json.gz")
        try:
            base_url = getattr(self.llm, "base_url", None)
            data = []
            hint_map = {h.id: h for h in hints}
            for i, job in enumerate(batch):
                hint = hint_map.get(i)
                if hint:
                    data.append(
                        {
                            "candidate": job.to_candidate(i).model_dump(),
                            "hint": hint.model_dump(),
                        }
                    )

            payload = {
                "meta": {
                    "book": self.book_name,
                    "model": self.llm.model_name,
                    "base_url": base_url,
                    "timestamp": timestamp,
                    "batch_index": batch_index,
                    "is_official": base_url is None,
                    "corrections": batch_corrections,
                },
                "data": data,
            }

            with gzip.open(path, "wt", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)

            log.info("Saved batch %d to trace: %s", batch_index, path)
        except Exception as e:
            log.warning("Failed to save trace batch %d: %s", batch_index, e)

    async def _run_batch(
        self,
        batch: list[VerificationJob],
        batch_index: int,
        total_batches: int,
    ) -> int:
        """Execute a single verification batch with concurrency control."""
        async with self.semaphore:
            cands = [job.to_candidate(i) for i, job in enumerate(batch)]
            failed = False
            try:
                hints = await self.llm.infer(cands)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                log.error("Batch %d FAILED after retries: %s", batch_index, e)
                hints = []
                failed = True

            self.completed_batches += 1
            if self.current_task_id is not None:
                self.progress.update(
                    self.current_task_id,
                    advance=1,
                    description=f"Verifying batches ({self.completed_batches}/{total_batches})...",
                )

            log.info("-" * 60)
            status = "FAILED" if failed else "COMPLETED"
            log.info(">> [ BATCH %d / %d %s ] <<", batch_index, total_batches, status)

            batch_corrections = 0
            for hint in hints:
                if 0 <= hint.id < len(batch):
                    job = batch[hint.id]
                    if job.word == hint.word and job.apply_hint(hint, self.local_dict):
                        batch_corrections += 1
                else:
                    log.warning("LLM returned invalid ID %d", hint.id)

            if batch_corrections > 0:
                if self.dict_path and self.save_fn:
                    self.save_fn(self.dict_path, self.local_dict)

            if not failed and hints:
                self._save_llm_output(batch, hints, batch_index, batch_corrections)

            return batch_corrections

    async def _verify_async(self, jobs: list[VerificationJob]) -> int:
        """Internal async orchestrator for all jobs."""
        async with self.llm:
            batches = [
                jobs[i : i + self.batch_size]
                for i in range(0, len(jobs), self.batch_size)
            ]
            total_batches = len(batches)

            self.current_task_id = self.progress.add_task(
                f"Verifying batches ({self.completed_batches}/{total_batches})...",
                total=total_batches,
            )

            tasks = [
                self._run_batch(batch, i + 1, total_batches)
                for i, batch in enumerate(batches)
            ]

            results = await asyncio.gather(*tasks)
            self.corrections = sum(results)
            return self.corrections

    def verify(self, jobs: list[VerificationJob]) -> int:
        """
        Synchronous entry point that runs the async verification loop.
        """
        if not jobs or len(jobs) == 0:
            return 0

        # gathering info for logging
        batchs_count = (len(jobs) + self.batch_size - 1) // self.batch_size
        average_context = sum(len(j.context) for j in jobs) / len(jobs) if jobs else 0
        log.info(
            "Verifying %d unique words via %s (%s) in %d batches... (Avg context: %.1f chars)",
            len(jobs),
            self.llm.provider,
            self.llm.model_name,
            batchs_count,
            average_context,
        )

        self.progress.start()
        self.corrections = asyncio.run(self._verify_async(jobs))
        return self.corrections
