import asyncio
import base64
import json
import logging
import time
from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal, Optional

import openai
from google import genai
from json_repair import repair_json
from pydantic import BaseModel


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def extract_llm_message_text(content: str | list[dict[str, Any]] | None) -> str:
    """
    Extract plain text from ``LLMMessage.content``.
    OpenAI returns ``str``; Gemini (multimodal) can return list ``[{"type":"text","text":"..."}, ...]``.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(p.get("text", "")) for p in content if isinstance(p, dict)
        ).strip()
    return str(content)


class LLMToolCall(BaseModel):
    id: Optional[str]
    name: str
    args: dict[str, Any]


class LLMMessage(BaseModel):
    role: Optional[str]
    """Role of the message, used mainly in chat models"""

    content: Optional[str | list[dict[str, Any]]] = None
    """Content of LLMMessage should be either a string or a dictionary following OpenAI format"""

    refusal: Optional[str] = None
    """refusal message (for compatible with OpenAI API)"""

    tool_calls: Optional[list[Any]] = None
    """tool_calls field is only used to be compatible with OpenAI API"""

    tool_call_id: Optional[str] = None
    """id of the tool call (used in tool call messages for compatibility with OpenAI API)"""


class LLMUsage(BaseModel):
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    total_tokens: Optional[int]


class LLMResponse(BaseModel):
    finish_reason: Optional[Literal["stop", "length", "content_filter", "tool_calls", "other"]] = (
        None
    )
    """finish_reason follows OpenAI definitions"""

    message: Optional[LLMMessage] = None
    """message returned from the LLM"""

    tool_calls: Optional[list[LLMToolCall]] = None
    """tool calls requested by the LLM"""

    usage: Optional[LLMUsage] = None
    """usage metadata"""



class LLMService(ABC):
    @abstractmethod
    async def create(
        self, messages: list[LLMMessage], tools: Optional[list[Any]] = None
    ) -> LLMResponse:
        pass


class OpenAIService(LLMService):
    def __init__(self, api_key: Optional[str], base_url: Optional[str], model: str):
        self.client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    async def create(self, messages: list[LLMMessage], tools: Optional[list[Any]] = None):
        # logger.debug(f"Try to call OpenAIService(messages={messages})")
        openai_messages = self._convert_to_llm_messages(messages)
        # logger.debug(f"openai_messages={json.dumps(openai_messages, indent=2)}")
        # logger.debug(f"tools={json.dumps(tools, indent=2)}")

        start_ts = time.time()
        response = await self.client.chat.completions.create(
            model=self.model, messages=openai_messages, tools=tools
        )
        logger.info(f"Take {time.time() - start_ts}s to OpenAIService")
        # logger.debug(f"response={json.dumps(response.model_dump(), indent=2)}")

        r = self._parse_llm_response(response)
        # logger.debug(f"parsed response={r}")
        logger.debug(f"parsed response={r.usage}")
        return r

    def _convert_to_llm_messages(self, messages: list[LLMMessage]) -> list[dict[str, Any]]:
        openai_messages: list[dict[str, Any]] = [m.model_dump() for m in messages]

        return openai_messages

    def _convert_to_llm_tools(self, tools: list[dict[str, Any]]) -> Any:
        return tools

    def _parse_llm_response(self, llm_response: Any) -> LLMResponse:
        message = llm_response.choices[0].message

        llm_message = LLMMessage(
            role="assistant",
            content=llm_response.choices[0].message.content,
            refusal=llm_response.choices[0].message.refusal,
            tool_calls=(
                [tc.model_dump() for tc in llm_response.choices[0].message.tool_calls]
                if llm_response.choices[0].message.tool_calls
                else None
            ),
        )

        llm_tool_calls = []
        if message.tool_calls:
            for tc in message.tool_calls:
                llm_tool_calls.append(
                    LLMToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        args=json.loads(repair_json(tc.function.arguments)),
                    )
                )

        usage = LLMUsage(
            prompt_tokens=llm_response.usage.prompt_tokens,
            completion_tokens=llm_response.usage.completion_tokens,
            total_tokens=llm_response.usage.total_tokens,
        )

        return LLMResponse(
            finish_reason=llm_response.choices[0].finish_reason,
            message=llm_message,
            tool_calls=llm_tool_calls,
            usage=usage,
        )


def parse_b64_image(b64_image: str):
    b64_str = b64_image.split(",")[-1]

    image_bytes = base64.b64decode(b64_str)
    image_mime = b64_image.split(";")[0].split(":")[-1]

    return image_bytes, image_mime


class GeminiService(LLMService):
    FINISH_REASON_MAP = {
        genai.types.FinishReason.STOP: "stop",
        genai.types.FinishReason.MAX_TOKENS: "length",
    }

    def __init__(self, api_key: Optional[str], base_url: Optional[str], model: str):
        self.client = genai.Client(api_key=api_key)
        self.model = model

    @staticmethod
    def _split_leading_system(
        messages: list[LLMMessage],
    ) -> tuple[Optional[str], list[LLMMessage]]:
        """Split leading ``role=system`` messages → ``system_instruction`` (Gemini API)."""
        if not messages:
            return None, []
        chunks: list[str] = []
        i = 0
        while i < len(messages) and (messages[i].role or "").lower() == "system":
            m = messages[i]
            if isinstance(m.content, str):
                chunks.append(m.content)
            elif isinstance(m.content, list):
                for c in m.content:
                    if isinstance(c, dict) and c.get("type") == "text":
                        chunks.append(str(c.get("text", "")))
            i += 1
        if not chunks:
            return None, messages
        return "\n\n".join(chunks), messages[i:]

    async def create(self, messages: list[LLMMessage], tools: Optional[list[Any]] = None):
        system_instruction, rest = self._split_leading_system(messages)
        if not rest:
            raise ValueError("GeminiService.create requires at least one non-system message")

        gemini_messages = self._convert_to_llm_messages(rest)
        gemini_tools = self._convert_to_llm_tools(tools)

        # Don't log raw ``contents`` — very long and leaks prompt (only log summary when DEBUG is needed).
        logger.debug(
            "GeminiService.create model=%s n_content=%s has_tools=%s has_system=%s",
            self.model,
            len(gemini_messages),
            bool(gemini_tools),
            bool(system_instruction),
        )
        cfg_kwargs: dict[str, Any] = {"tools": gemini_tools}
        if system_instruction:
            cfg_kwargs["system_instruction"] = system_instruction
        config = genai.types.GenerateContentConfig(**cfg_kwargs)

        start_ts = time.time()
        response = await asyncio.to_thread(
            self.client.models.generate_content,
            model=self.model,
            contents=gemini_messages,
            config=config,
        )
        logger.info(f"Take {time.time() - start_ts}s to call GeminiService")

        return self._parse_llm_response(response)

    def _convert_to_llm_messages(self, messages: list[LLMMessage]) -> list[genai.types.Content]:
        gemini_messages = []
        for m in messages:
            if m.role == "assistant":
                role = "model"
            else:
                role = "user"

            parts = []
            if isinstance(m.content, str):
                parts.append(genai.types.Part(text=m.content))

            elif isinstance(m.content, list):
                for c in m.content:
                    if c["type"] == "text":
                        parts.append(genai.types.Part(text=c["text"]))
                    elif c["type"] == "image_url":
                        image_bytes, image_mime = parse_b64_image(c["image_url"]["url"])
                        parts.append(
                            genai.types.Part.from_bytes(data=image_bytes, mime_type=image_mime)
                        )

            gemini_messages.append(genai.types.Content(role=role, parts=parts))

        return gemini_messages

    def _convert_to_llm_tools(self, tools: Optional[list[Any]] = None):
        gemini_tools = []
        if tools:
            for t in tools:
                fn = t["function"]
                gemini_t = {
                    "name": fn["name"],
                    "description": fn["description"],
                    "parameters": fn["parameters"],
                    "response": fn["output_schema"],
                }
                gemini_tools.append(genai.types.Tool(function_declarations=[gemini_t]))

        logger.debug("GeminiService tools n_decl=%s", len(gemini_tools))
        return gemini_tools

    def _parse_llm_response(self, llm_response: genai.types.GenerateContentResponse) -> LLMResponse:
        message = None
        if llm_response.candidates:
            candidate = llm_response.candidates[0]
            if candidate.content:
                message = candidate.content
        else:
            candidate = None

        finish_reason = "other"
        if candidate:
            if candidate.finish_reason in GeminiService.FINISH_REASON_MAP:
                finish_reason = GeminiService.FINISH_REASON_MAP[candidate.finish_reason]

        llm_tool_calls = []
        message_contents: list[dict[str, Any]] = []
        text_chunks: list[str] = []
        if message and message.parts:
            for p in message.parts:
                if p.text:
                    message_contents.append({"type": "text", "text": p.text})
                    text_chunks.append(p.text)
                elif p.function_call:
                    if p.function_call.name is not None:
                        llm_tool_calls.append(
                            LLMToolCall(
                                id=p.function_call.id,
                                name=p.function_call.name,
                                args=p.function_call.args or {},
                            )
                        )

        # TODO: This only for compatible with the current BaseSession.
        if finish_reason == "other" and len(llm_tool_calls):
            finish_reason = "tool_calls"

        # Match OpenAI: response only text → ``content`` is ``str``; has tool/multimodal → ``list``.
        if llm_tool_calls:
            llm_message = LLMMessage(
                role="assistant",
                content=message_contents if message_contents else None,
            )
        elif text_chunks:
            llm_message = LLMMessage(
                role="assistant",
                content="\n".join(text_chunks).strip(),
            )
        else:
            llm_message = LLMMessage(role="assistant", content="")

        if llm_response.usage_metadata:
            usage = LLMUsage(
                prompt_tokens=llm_response.usage_metadata.prompt_token_count,
                completion_tokens=llm_response.usage_metadata.candidates_token_count,
                total_tokens=llm_response.usage_metadata.total_token_count,
            )
        else:
            usage = None

        return LLMResponse(
            finish_reason=finish_reason,
            message=llm_message,
            tool_calls=llm_tool_calls,
            usage=usage,
        )


class LocalGemmaService(LLMService):
    """
    Run a Gemma instruct model locally via Hugging Face transformers.

    Loads weights from ``local_dir`` if present (existence of ``config.json`` is the marker).
    Otherwise downloads ``model_name`` from the Hub into ``local_dir`` first, then loads from disk.

    Drop-in replacement for ``OpenAIService`` / ``GeminiService`` — exposes the same
    ``async create(messages, tools=None) -> LLMResponse`` contract, so it plugs into
    ``gen_multihop.generate_conversation`` without changes.
    """

    DEFAULT_MODEL_NAME = "google/gemma-4-26B-A4B-it"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        local_dir: Optional[str] = None,
        hf_token: Optional[str] = None,
        device_map: str = "auto",
        torch_dtype: str = "bfloat16",
        max_new_tokens: int = 4096,
        temperature: float = 0.7,
        top_p: float = 0.95,
        do_sample: bool = True,
    ):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from huggingface_hub import snapshot_download
        except ImportError as e:
            raise RuntimeError(
                "LocalGemmaService requires `torch`, `transformers`, "
                "`huggingface_hub`, `accelerate`. Install them first."
            ) from e

        self._torch = torch
        self.model_name = model_name
        # Mirror ``model`` attribute used by the API services so logging
        # (``getattr(llm, "model", "?")``) keeps working.
        self.model = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.do_sample = do_sample

        path = self._ensure_local_weights(model_name, local_dir, hf_token, snapshot_download)
        self._model_path = path

        dtype = self._resolve_dtype(torch, torch_dtype)
        logger.info(
            "LocalGemmaService loading: name=%s path=%s dtype=%s device_map=%s",
            model_name,
            path,
            torch_dtype,
            device_map,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(str(path))
        load_kwargs: dict[str, Any] = {
            "device_map": device_map,
            # `torch_dtype` is deprecated on newer transformers; prefer `dtype`.
            "dtype": dtype,
        }
        try:
            self.hf_model = AutoModelForCausalLM.from_pretrained(
                str(path),
                **load_kwargs,
            )
        except TypeError:
            # Backward compatibility for older transformers versions.
            load_kwargs.pop("dtype", None)
            load_kwargs["torch_dtype"] = dtype
            self.hf_model = AutoModelForCausalLM.from_pretrained(
                str(path),
                **load_kwargs,
            )
        self.hf_model.eval()
        # Single-resident model — serialize requests even if the caller fires async tasks concurrently.
        self._gen_lock = asyncio.Lock()

    @staticmethod
    def _resolve_dtype(torch, name: str):
        if name == "auto":
            return "auto"
        return getattr(torch, name)

    @staticmethod
    def _ensure_local_weights(
        model_name: str,
        local_dir: Optional[str],
        hf_token: Optional[str],
        snapshot_download,
    ) -> Path:
        if local_dir is None:
            target = Path(__file__).parent / "models" / model_name.replace("/", "__")
        else:
            target = Path(local_dir)

        marker = target / "config.json"
        if marker.is_file():
            logger.info("LocalGemmaService: cached weights found at %s", target)
            return target

        target.mkdir(parents=True, exist_ok=True)
        logger.info("LocalGemmaService: downloading %s -> %s", model_name, target)
        snapshot_download(
            repo_id=model_name,
            local_dir=str(target),
            local_dir_use_symlinks=False,
            token=hf_token,
        )
        if not marker.is_file():
            raise RuntimeError(
                f"Downloaded {model_name} but config.json not found at {target}"
            )
        return target

    async def create(
        self, messages: list[LLMMessage], tools: Optional[list[Any]] = None
    ) -> LLMResponse:
        if tools:
            logger.warning(
                "LocalGemmaService ignores tools=%s — only chat completion is supported.",
                len(tools),
            )
        logger.info(
            "LocalGemmaService.create start | model=%s | n_messages=%s",
            self.model,
            len(messages),
        )
        async with self._gen_lock:
            logger.info("LocalGemmaService.create acquired generation lock")
            try:
                resp = await asyncio.to_thread(self._sync_create, messages)
                logger.info(
                    "LocalGemmaService.create done | finish_reason=%s | prompt_tokens=%s | completion_tokens=%s",
                    resp.finish_reason,
                    resp.usage.prompt_tokens if resp.usage else None,
                    resp.usage.completion_tokens if resp.usage else None,
                )
                return resp
            except Exception as e:
                logger.exception("LocalGemmaService.create failed")
                # Ensure upper layers always get a non-empty error string.
                raise RuntimeError(f"{type(e).__name__}: {e!r}") from e

    def _resolve_input_device(self):
        """
        Resolve a concrete device for input_ids.
        With `device_map=auto`, model.device can be `meta`; use input embedding
        weight device (or first non-meta parameter) instead.
        """
        emb = self.hf_model.get_input_embeddings()
        if emb is not None and hasattr(emb, "weight"):
            dev = emb.weight.device
            if str(dev) != "meta":
                return dev
        for p in self.hf_model.parameters():
            dev = p.device
            if str(dev) != "meta":
                return dev
        raise RuntimeError("Could not resolve a non-meta device for LocalGemmaService.")

    def _sync_create(self, messages: list[LLMMessage]) -> LLMResponse:
        logger.info("LocalGemmaService._sync_create step=1 messages_to_chat")
        chat = self._messages_to_chat(messages)
        logger.info("LocalGemmaService._sync_create step=1 done | n_chat_turns=%s", len(chat))
        try:
            logger.info("LocalGemmaService._sync_create step=2 apply_chat_template")
            tokenized = self.tokenizer.apply_chat_template(
                chat,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            logger.info(
                "LocalGemmaService._sync_create step=2 done | tokenized_type=%s",
                type(tokenized).__name__,
            )
        except Exception as e:
            # Some Gemma chat templates reject role="system" — fall back to merging it into the first user turn.
            logger.debug(
                "apply_chat_template failed (%s) — merging system into first user message.",
                e,
            )
            logger.info("LocalGemmaService._sync_create step=2 fallback merge_system_into_user")
            chat = self._merge_system_into_user(chat)
            tokenized = self.tokenizer.apply_chat_template(
                chat,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            logger.info(
                "LocalGemmaService._sync_create step=2 fallback done | tokenized_type=%s",
                type(tokenized).__name__,
            )

        logger.info("LocalGemmaService._sync_create step=3 resolve_input_device")
        input_device = self._resolve_input_device()
        logger.info("LocalGemmaService._sync_create step=3 done | input_device=%s", input_device)
        # transformers versions differ:
        # - some return a Tensor
        # - some return a BatchEncoding with input_ids/attention_mask
        if isinstance(tokenized, self._torch.Tensor):
            input_ids = tokenized.to(input_device)
            attention_mask = None
            logger.info("LocalGemmaService._sync_create step=4 tokenized=tensor")
        else:
            input_ids = tokenized["input_ids"].to(input_device)
            attention_mask = (
                tokenized["attention_mask"].to(input_device)
                if "attention_mask" in tokenized
                else None
            )
            logger.info(
                "LocalGemmaService._sync_create step=4 tokenized=batch | has_attention_mask=%s",
                attention_mask is not None,
            )
        prompt_len = int(input_ids.shape[-1])
        logger.info("LocalGemmaService._sync_create step=4 done | prompt_len=%s", prompt_len)

        gen_kwargs: dict[str, Any] = dict(
            max_new_tokens=self.max_new_tokens,
            do_sample=self.do_sample,
            temperature=self.temperature,
            top_p=self.top_p,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        if attention_mask is not None:
            gen_kwargs["attention_mask"] = attention_mask

        logger.info(
            "LocalGemmaService._sync_create step=5 generate start | max_new_tokens=%s | do_sample=%s | temperature=%s | top_p=%s",
            self.max_new_tokens,
            self.do_sample,
            self.temperature,
            self.top_p,
        )
        start_ts = time.time()
        with self._torch.inference_mode():
            output = self.hf_model.generate(input_ids, **gen_kwargs)
        logger.info("LocalGemmaService._sync_create step=5 generate done")
        gen_ids = output[0, prompt_len:]
        completion_len = int(gen_ids.shape[-1])
        logger.info(
            "LocalGemmaService generated %d tokens in %.1fs",
            completion_len,
            time.time() - start_ts,
        )

        logger.info("LocalGemmaService._sync_create step=6 decode")
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        finish_reason = "length" if completion_len >= self.max_new_tokens else "stop"
        logger.info(
            "LocalGemmaService._sync_create step=6 done | finish_reason=%s | text_len=%s",
            finish_reason,
            len(text),
        )

        return LLMResponse(
            finish_reason=finish_reason,
            message=LLMMessage(role="assistant", content=text),
            tool_calls=[],
            usage=LLMUsage(
                prompt_tokens=prompt_len,
                completion_tokens=completion_len,
                total_tokens=prompt_len + completion_len,
            ),
        )

    @staticmethod
    def _messages_to_chat(messages: list[LLMMessage]) -> list[dict[str, str]]:
        out: list[dict[str, str]] = []
        for m in messages:
            role = (m.role or "user").lower()
            if role not in ("user", "assistant", "system"):
                role = "user"
            text = (
                m.content
                if isinstance(m.content, str)
                else extract_llm_message_text(m.content)
            )
            out.append({"role": role, "content": text or ""})
        return out

    @staticmethod
    def _merge_system_into_user(chat: list[dict[str, str]]) -> list[dict[str, str]]:
        sys_chunks = [c["content"] for c in chat if c["role"] == "system" and c["content"]]
        rest = [c for c in chat if c["role"] != "system"]
        if not sys_chunks:
            return rest or chat
        merged_sys = "\n\n".join(sys_chunks)
        if not rest:
            return [{"role": "user", "content": merged_sys}]
        head = rest[0]
        if head["role"] == "user":
            head = {"role": "user", "content": f"{merged_sys}\n\n{head['content']}"}
            return [head, *rest[1:]]
        return [{"role": "user", "content": merged_sys}, *rest]