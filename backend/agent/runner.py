import asyncio
import json
import os
from typing import Any

from loguru import logger

from backend.agent.hook import AgentHook, AgentHookContext
from backend.agent.providers.openai_provider import LLMProvider
from backend.agent.schemas import AgentRunSpec, AgentRunResult, LLMResponse, ToolCallRequest
from backend.agent.message_dump import log_agent_messages_block
from backend.agent.tools.base import Tool
from backend.utils.helpers import (
    build_assistant_message,
    estimate_message_tokens,
    estimate_prompt_tokens_chain,
    find_legal_message_start,
)
from backend.utils.runtime import (
    EMPTY_FINAL_RESPONSE_MESSAGE,
    build_finalization_retry_message,
    build_length_recovery_message,
    ensure_nonempty_tool_result,
    is_blank_text,
    repeated_external_lookup_error,
)

_BACKFILL_CONTENT = "[Tool result unavailable — call was interrupted or lost]"
_SNIP_SAFETY_BUFFER = 1024
_MAX_EMPTY_RETRIES = 2
_MAX_LENGTH_RECOVERIES = 3
_PERSISTED_MODEL_ERROR_PLACEHOLDER = "[助手回复因模型错误不可用]"

class AgentRunner:
    """ Agent 核心执行引擎 """
    def __init__(self, provider: LLMProvider) -> None:
        """ 根据 LLM  Provider 进行初始化 """
        self.provider = provider

    @staticmethod
    def _log_tool_calls_batch(iteration: int, tool_calls: list[ToolCallRequest]) -> None:
        """每次模型返回待执行工具时，打印完整 tool_calls（避免把 JSON 放进 loguru 的 format 占位符）。"""
        if not tool_calls:
            return
        payload = [
            {"id": tc.id, "name": tc.name, "arguments": dict(tc.arguments)}
            for tc in tool_calls
        ]
        try:
            body = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
        except TypeError:
            body = str(payload)
        logger.info(
            f"[agent tool_calls] iteration={iteration} count={len(tool_calls)}\n" + body
        )


    async def run(self, spec: AgentRunSpec) -> AgentRunResult:
        """ 核心入口
        Args:
            spec: agent 一次运行的配置
        Returns:
            AgentRunResult: agent 运行一次结束的配置
        """
        # 钩子
        hook = spec.hook or AgentHook()
        # 消息列表
        messages = list(spec.initial_messages)
        # 返回给用户的内容
        final_content : str | None = None
        # 实际调用的工具列表
        tools_used: list[str] = []
        # token 累积用量
        usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0}
        # 错误信息
        error: str | None = None
        # 停止原因
        stop_reason: str = "completed"
        # 工具执行的事件
        tool_events: list[dict[str, str]] = []
        external_lookup_counts: dict[str, int] = {}
        empty_content_retries = 0
        length_recovery_count = 0
        had_injections = False

        for iteration in range(spec.max_iterations):
            try:
                """
                上下文治理
                """
                # 删除没有对应 assistant.tool_calls 的 tool 消息
                messages_for_model = self._drop_orphan_tool_results(messages)
                # 对于有 tool_calls 但缺少 tool 结果的，插入一个占位 tool 消息
                messages_for_model = self._backfill_missing_tool_results(messages_for_model)
                # 将每个 tool 消息的 content 根据限制截断
                messages_for_model = self._apply_tool_result_budget(spec, messages_for_model)
                # 根据估算的 token，如果超出预算，从后往前保留新的消息
                messages_for_model = self._snip_history(spec, messages_for_model)
                messages_for_model = self._drop_orphan_tool_results(messages_for_model)
                messages_for_model = self._backfill_missing_tool_results(messages_for_model)

            except Exception as exc:
                logger.warning(
                    "Context governance failed on turn {}: {}; applying minimal repair",
                    iteration,
                    exc,
                )
                try:
                    messages_for_model = self._drop_orphan_tool_results(messages)
                    messages_for_model = self._backfill_missing_tool_results(messages_for_model)
                except Exception:
                    messages_for_model = list(messages)

            # 创建上下文并触发钩子
            context = AgentHookContext(iteration=iteration, messages=messages)
            await hook.before_iteration(context)

            log_agent_messages_block(
                "[runner raw messages chain — before governance→snip]",
                messages,
                iteration=context.iteration,
            )
            log_agent_messages_block(
                "[runner messages_for_model — actual payload to LLM]",
                messages_for_model,
                iteration=context.iteration,
            )

            # 请求模型
            response = await self._request_model(spec, messages_for_model, hook, context)

            # 将用量、响应、工具调用列表存入 context 提供 hook 的调用
            raw_usage = self._usage_dict(response.usage)
            context.response = response
            context.usage = dict(raw_usage)
            context.tool_calls = list(response.tool_calls)
            self._accumulate_usage(usage, raw_usage)

            # 根据相应判断是否需要执行工具
            if response.should_execute_tools:
                print(f"需要执行工具\n")
            else:
                print(f"不需要执行工具\n")
            if response.should_execute_tools:
                # 获取 tool_calls
                tool_calls = list(response.tool_calls)
                context.tool_calls = list(tool_calls)

                # 构建 assistant 消息并将此消息追加到 messages 历史中以供迭代
                assistant_message = build_assistant_message(
                    response.content or "",
                    tool_calls=[tc.to_openai_tool_call() for tc in tool_calls],
                    reasoning_content=response.reasoning_content,
                    thinking_blocks=response.thinking_blocks,
                )
                messages.append(assistant_message)
                tools_used.extend(tc.name for tc in tool_calls)

                await self._emit_checkpoint(
                    spec,
                    {
                        "phase": "awaiting_tools",
                        "iteration": iteration,
                        "model": spec.model,
                        "assistant_message": assistant_message,
                        "completed_tool_results": [],
                        "pending_tool_calls": [tc.to_openai_tool_call() for tc in tool_calls],
                    },
                )

                self._log_tool_calls_batch(iteration, tool_calls)

                # 执行 执行工具前 的时机钩子
                await hook.before_execute_tools(context)

                # 开始执行工具，返回：工具执行结果、每个工具执行的事件、异常
                results, new_events, fatal_error = await self._execute_tools(
                    spec,
                    tool_calls,
                    external_lookup_counts,
                )
                tool_events.extend(new_events)
                context.tool_results = list(results)
                context.tool_events = list(new_events)

                completed_tool_results: list[dict[str, Any]] = []
                for tool_call, result in zip(tool_calls, results):
                    tool_message = {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_call.name,
                        "content": self._normalize_tool_result(
                            spec,
                            tool_call.id,
                            tool_call.name,
                            result,
                        ),
                    }
                    messages.append(tool_message)
                    completed_tool_results.append(tool_message)

                if fatal_error is not None:
                    err_text = f"{type(fatal_error).__name__}: {fatal_error}"
                    error = err_text
                    final_content = err_text
                    stop_reason = "tool_error"
                    self._append_final_message(messages, final_content)
                    context.final_content = final_content
                    context.error = error
                    context.stop_reason = stop_reason
                    await hook.after_iteration(context)
                    break

                await self._emit_checkpoint(
                    spec,
                    {
                        "phase": "tools_completed",
                        "iteration": iteration,
                        "model": spec.model,
                        "assistant_message": assistant_message,
                        "completed_tool_results": completed_tool_results,
                        "pending_tool_calls": [],
                    },
                )
                empty_content_retries = 0
                length_recovery_count = 0
                # 执行迭代后钩子
                await hook.after_iteration(context)
                # 进行下一次循环
                continue

            # 没有工具调用
            # 如果存在 tool_call 说明不该执行
            if response.has_tool_calls:
                logger.warning(
                    "Ignoring tool calls under finish_reason={!r}",
                    response.finish_reason,
                )

            # 允许钩子对模型返回的内容进行最终处理
            clean = hook.finalize_content(context, response.content)

            if response.finish_reason != "error" and is_blank_text(clean):
                empty_content_retries += 1
                if empty_content_retries < _MAX_EMPTY_RETRIES:
                    logger.warning(
                        "Empty model response on turn {} ({}/{}); retrying",
                        iteration,
                        empty_content_retries,
                        _MAX_EMPTY_RETRIES,
                    )
                    await hook.after_iteration(context)
                    continue
                logger.warning(
                    "Empty model response on turn {} after {} retries; finalization retry",
                    iteration,
                    empty_content_retries,
                )
                response = await self._request_finalization_retry(
                    spec,
                    messages_for_model,
                    iteration=iteration,
                )
                retry_usage = self._usage_dict(response.usage)
                self._accumulate_usage(usage, retry_usage)
                raw_usage = self._merge_usage(raw_usage, retry_usage)
                context.response = response
                context.usage = dict(raw_usage)
                context.tool_calls = list(response.tool_calls)
                clean = hook.finalize_content(context, response.content)

            # 长度截断恢复
            if response.finish_reason == "length" and clean and not is_blank_text(clean):
                length_recovery_count += 1
                if length_recovery_count <= _MAX_LENGTH_RECOVERIES:
                    logger.info(
                        "Output truncated on turn {} ({}/{}); continuing",
                        iteration,
                        length_recovery_count,
                        _MAX_LENGTH_RECOVERIES,
                    )
                    messages.append(
                        build_assistant_message(
                            clean,
                            reasoning_content=response.reasoning_content,
                            thinking_blocks=response.thinking_blocks,
                        )
                    )
                    messages.append(build_length_recovery_message())
                    await hook.after_iteration(context)
                    continue

            assistant_message: dict[str, Any] | None = None
            if response.finish_reason != "error" and clean and not is_blank_text(clean):
                assistant_message = build_assistant_message(
                    clean,
                    reasoning_content=response.reasoning_content,
                    thinking_blocks=response.thinking_blocks,
                )

            # 如果停止原因是错误 error
            if response.finish_reason == "error":
                final_content = clean or spec.error_message
                stop_reason = "error"
                error = final_content
                self._append_model_error_placeholder(messages)
                context.final_content = final_content
                context.error = error
                context.stop_reason = stop_reason
                await hook.after_iteration(context)
                break

            if is_blank_text(clean):
                final_content = EMPTY_FINAL_RESPONSE_MESSAGE
                stop_reason = "empty_final_response"
                error = final_content
                self._append_final_message(messages, final_content)
                context.final_content = final_content
                context.error = error
                context.stop_reason = stop_reason
                await hook.after_iteration(context)
                break

            messages.append(
                assistant_message
                or build_assistant_message(
                    clean,
                    reasoning_content=response.reasoning_content,
                    thinking_blocks=response.thinking_blocks,
                )
            )
            await self._emit_checkpoint(
                spec,
                {
                    "phase": "final_response",
                    "iteration": iteration,
                    "model": spec.model,
                    "assistant_message": messages[-1],
                    "completed_tool_results": [],
                    "pending_tool_calls": [],
                },
            )
            final_content = clean
            context.final_content = final_content
            context.stop_reason = stop_reason
            await hook.after_iteration(context)
            break
        else:
            # 达到最大循环次数
            stop_reason = "max_iterations"
            try:
                final_content = spec.max_iterations_message.format(
                    max_iterations=spec.max_iterations,
                )
            except Exception:
                final_content = spec.max_iterations_message
            self._append_final_message(messages, final_content)

        return AgentRunResult(
            final_content=final_content,
            messages=messages,
            tools_used=tools_used,
            usage=usage,
            stop_reason=stop_reason,
            error=error,
            tool_events=tool_events,
            had_injections=had_injections,
        )



    @staticmethod
    def _usage_dict(usage: dict[str, Any] | None) -> dict[str, int]:
        if not usage:
            return {}
        result: dict[str, int] = {}
        for key, value in usage.items():
            try:
                result[key] = int(value or 0)
            except (TypeError, ValueError):
                continue
        return result

    @staticmethod
    def _accumulate_usage(target: dict[str, int], addition: dict[str, int]) -> None:
        for key, value in addition.items():
            target[key] = target.get(key, 0) + value

    @staticmethod
    def _merge_usage(left: dict[str, int], right: dict[str, int]) -> dict[str, int]:
        merged = dict(left)
        for key, value in right.items():
            merged[key] = merged.get(key, 0) + value
        return merged

    async def _emit_checkpoint(self, spec: AgentRunSpec, payload: dict[str, Any]) -> None:
        callback = getattr(spec, "checkpoint_callback", None)
        if callback is not None:
            await callback(payload)

    @staticmethod
    def _append_final_message(messages: list[dict[str, Any]], content: str | None) -> None:
        if not content:
            return
        if (
            messages
            and messages[-1].get("role") == "assistant"
            and not messages[-1].get("tool_calls")
        ):
            if messages[-1].get("content") == content:
                return
            messages[-1] = build_assistant_message(content)
            return
        messages.append(build_assistant_message(content))

    @staticmethod
    def _append_model_error_placeholder(messages: list[dict[str, Any]]) -> None:
        if messages and messages[-1].get("role") == "assistant" and not messages[-1].get(
            "tool_calls",
        ):
            return
        messages.append(build_assistant_message(_PERSISTED_MODEL_ERROR_PLACEHOLDER))

    async def _request_finalization_retry(
            self,
            spec: AgentRunSpec,
            messages: list[dict[str, Any]],
            *,
            iteration: int | None = None,
    ) -> LLMResponse:
        retry_messages = list(messages)
        retry_messages.append(build_finalization_retry_message())
        log_agent_messages_block(
            "[runner finalization_retry — payload to LLM]",
            retry_messages,
            iteration=iteration,
        )
        kwargs = self._build_request_kwargs(spec, retry_messages, tools=[])
        kwargs["tool_choice"] = None
        coro = self.provider.chat_with_retry(**kwargs)
        timeout_s = spec.llm_timeout_s
        if timeout_s is None:
            raw = os.environ.get("LLM_RUN_TIMEOUT_S", "300").strip()
            try:
                timeout_s = float(raw)
            except (TypeError, ValueError):
                timeout_s = 300.0
        if timeout_s is not None and timeout_s <= 0:
            timeout_s = None
        if timeout_s is None:
            return await coro
        try:
            return await asyncio.wait_for(coro, timeout=timeout_s)
        except asyncio.TimeoutError:
            return LLMResponse(
                content=f"调用模型超时（{timeout_s:g}s）",
                finish_reason="error",
                error_kind="timeout",
            )

    def _partition_tool_batches(
            self,
            spec: AgentRunSpec,
            tool_calls: list[ToolCallRequest],
    ) -> list[list[ToolCallRequest]]:
        if not getattr(spec, "concurrent_tools", False):
            return [[tc] for tc in tool_calls]
        batches: list[list[ToolCallRequest]] = []
        current: list[ToolCallRequest] = []
        for tool_call in tool_calls:
            tool_obj = spec.tools.get(tool_call.name) if hasattr(spec.tools, "get") else None
            can_batch = bool(
                tool_obj is not None and getattr(tool_obj, "concurrency_safe", False),
            )
            if can_batch:
                current.append(tool_call)
                continue
            if current:
                batches.append(current)
                current = []
            batches.append([tool_call])
        if current:
            batches.append(current)
        return batches

    async def _execute_tools(
            self,
            spec: AgentRunSpec,
            tool_calls: list[ToolCallRequest],
            external_lookup_counts: dict[str, int],
    ) -> tuple[list[Any], list[dict[str, str]], BaseException | None]:
        batches = self._partition_tool_batches(spec, tool_calls)
        tool_results: list[tuple[Any, dict[str, str], BaseException | None]] = []
        for batch in batches:
            if getattr(spec, "concurrent_tools", False) and len(batch) > 1:
                batch_results = await asyncio.gather(*(
                    self._run_tool(spec, tc, external_lookup_counts)
                    for tc in batch
                ))
                tool_results.extend(batch_results)
            else:
                for tc in batch:
                    tool_results.append(await self._run_tool(spec, tc, external_lookup_counts))

        results: list[Any] = []
        events: list[dict[str, str]] = []
        fatal_error: BaseException | None = None
        for result, event, err in tool_results:
            results.append(result)
            events.append(event)
            if err is not None and fatal_error is None:
                fatal_error = err
        return results, events, fatal_error

    _WORKSPACE_BLOCK_MARKERS: tuple[str, ...] = (
        "outside the configured workspace",
        "outside allowed directory",
        "working_dir is outside",
        "working_dir could not be resolved",
        "path traversal detected",
        "path outside working dir",
        "internal/private url detected",
    )

    @classmethod
    def _is_workspace_violation(cls, text: str) -> bool:
        if not text:
            return False
        lowered = text.lower()
        return any(marker in lowered for marker in cls._WORKSPACE_BLOCK_MARKERS)

    async def _run_tool(
            self,
            spec: AgentRunSpec,
            tool_call: ToolCallRequest,
            external_lookup_counts: dict[str, int],
    ) -> tuple[Any, dict[str, str], BaseException | None]:
        hint = "\n\n[分析上方错误并换一种做法。]"
        lookup_error = repeated_external_lookup_error(
            tool_call.name,
            tool_call.arguments,
            external_lookup_counts,
        )
        if lookup_error:
            event = {
                "name": tool_call.name,
                "status": "error",
                "detail": "repeated external lookup blocked",
            }
            if spec.fail_on_tool_error:
                return lookup_error + hint, event, RuntimeError(lookup_error)
            return lookup_error + hint, event, None

        prepared = spec.tools.prepare_call(tool_call.name, tool_call.arguments)
        tool: Tool | None
        params: dict[str, Any]
        prep_error: str | None
        tool, params, prep_error = prepared[0], prepared[1], prepared[2]

        if prep_error:
            event = {
                "name": tool_call.name,
                "status": "error",
                "detail": prep_error.split(": ", 1)[-1][:120],
            }
            if self._is_workspace_violation(prep_error):
                logger.warning(
                    "Tool {} workspace guard (prepare): {}",
                    tool_call.name,
                    prep_error.replace("\n", " ").strip()[:200],
                )
                event["detail"] = (
                    "workspace_violation: " + prep_error.replace("\n", " ").strip()
                )[:160]
                return prep_error, event, RuntimeError(prep_error)
            return prep_error + hint, event, RuntimeError(prep_error) if spec.fail_on_tool_error else None

        try:
            assert tool is not None
            result = await tool.execute(**params)
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            event = {
                "name": tool_call.name,
                "status": "error",
                "detail": str(exc),
            }
            if self._is_workspace_violation(str(exc)):
                logger.warning(
                    "Tool {} workspace guard: {}",
                    tool_call.name,
                    str(exc).replace("\n", " ").strip()[:200],
                )
                event["detail"] = (
                    "workspace_violation: " + str(exc).replace("\n", " ").strip()
                )[:160]
                return f"Error: {type(exc).__name__}: {exc}", event, exc
            if spec.fail_on_tool_error:
                return f"Error: {type(exc).__name__}: {exc}", event, exc
            return f"Error: {type(exc).__name__}: {exc}", event, None

        if isinstance(result, str) and result.startswith("Error"):
            event = {
                "name": tool_call.name,
                "status": "error",
                "detail": result.replace("\n", " ").strip()[:120],
            }
            if self._is_workspace_violation(result):
                event["detail"] = ("workspace_violation: " + result.replace("\n", " ").strip())[
                    :160
                ]
                return result, event, RuntimeError(result)
            if spec.fail_on_tool_error:
                return result + hint, event, RuntimeError(result)
            return result + hint, event, None

        detail = "" if result is None else str(result)
        detail = detail.replace("\n", " ").strip()
        if not detail:
            detail = "(empty)"
        elif len(detail) > 120:
            detail = detail[:120] + "..."
        return result, {"name": tool_call.name, "status": "ok", "detail": detail}, None

    @staticmethod
    def _drop_orphan_tool_results(
            messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """ 从 messages 列表里删除孤儿 "role": "tool"
        在 assistant 的 tool_calls 中没有出现过的 tool_call_id 删除
        """
        declared: set[str] = set()
        updated: list[dict[str, Any]] | None = None

        for idx, msg in enumerate(messages):
            role = msg.get("role")
            if role == "assistant":
                for tool_call in msg.get("tool_calls") or []:
                    if isinstance(tool_call, dict) and tool_call.get("id"):
                        declared.add(str(tool_call["id"]))
            if role == "tool":
                tid = msg.get("tool_call_id")
                # 如果该 ID 不存在，说明没有对应的 tool_call
                if tid and str(tid) not in declared:
                    if updated is None:
                        updated = [dict(m) for m in messages[:idx]]
                    continue
            if updated is not None:
                updated.append(dict(msg))
        if updated is None:
            return messages
        return updated

    @staticmethod
    def _backfill_missing_tool_results(
        messages: list[dict[str, Any]],
    )-> list[dict[str, Any]]:
        declared: list[tuple[int, str, str]] = []
        fulfilled: set[str] = set()
        for idx, msg in enumerate(messages):
            role = msg.get("role")
            if role == "assistant":
                for tool_call in msg.get("tool_calls") or []:
                    if isinstance(tool_call, dict) and tool_call.get("id"):
                        name = ""
                        func = tool_call.get("function")
                        if isinstance(func, dict):
                            name = func.get("name")
                        declared.append((idx, str(tool_call["id"]), name))
            elif role == "tool":
                tid = msg.get("tool_call_id")
                if tid:
                    fulfilled.add(str(tid))
        missing = [(ai, cid, name) for ai, cid, name in declared if cid not in fulfilled]
        if not missing:
            return messages

        updated = list(messages)
        offset = 0
        for assistant_idx, call_id, name in missing:
            insert_at = assistant_idx + 1 + offset
            while insert_at < len(updated) and updated[insert_at].get("role") == "tool":
                insert_at += 1
            updated.insert(insert_at, {
                "role": "tool",
                "tool_call_id": call_id,
                "name": name,
                "content": _BACKFILL_CONTENT,
            })
            offset += 1
        return updated

    def _apply_tool_result_budget(
            self,
            spec: AgentRunSpec,
            messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        updated = messages
        for idx, message in enumerate(messages):
            if message.get("role") != "tool":
                continue
            normalized = self._normalize_tool_result(
                spec,
                str(message.get("tool_call_id") or f"tool_{idx}"),
                str(message.get("name") or "tool"),
                message.get("content"),
            )
            if normalized != message.get("content"):
                if updated is messages:
                    updated = [dict(m) for m in messages]
                updated[idx]["content"] = normalized
        return updated

    def _normalize_tool_result(
            self,
            spec: AgentRunSpec,
            tool_call_id: str,
            tool_name: str,
            result: Any,
    ) -> str:
        """不落盘：空结果占位 → 统一成 str → 按 max_tool_result_chars 截断。"""
        del tool_call_id  # 与 nanobot 签名对齐，持久化时才需要 id
        normalized = ensure_nonempty_tool_result(tool_name, result)
        if isinstance(normalized, str):
            text = normalized
        else:
            try:
                text = json.dumps(normalized, ensure_ascii=False, default=str)
            except (TypeError, ValueError):
                text = str(normalized)
        max_chars = spec.max_tool_result_chars
        if max_chars is not None and max_chars > 0 and len(text) > max_chars:
            return text[:max_chars] + "\n...[truncated]"
        return text

    def _snip_history(
            self,
            spec: AgentRunSpec,
            messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        当前对话历史的估计 token 数超过上下文窗口的预算时，智能截断历史
        """
        # 没有消息或者没有显示窗口大小
        if not messages or not spec.context_window_tokens:
            return messages

        # 从 provider 的 generation 配置中获取模型允许的最大 token
        provider_max_tokens = getattr(getattr(self.provider, "generation", None), "max_tokens", 4096)

        # 优先使用本轮输出长度限制，本轮最多生成多少 Token
        max_output = spec.max_tokens if isinstance(spec.max_tokens, int) else (
            provider_max_tokens if isinstance(provider_max_tokens, int) else 4096
        )

        # 本次可用于输入发给模型的 token 预算，上限
        budget = spec.context_block_limit or (
                spec.context_window_tokens - max_output - _SNIP_SAFETY_BUFFER      # 最大窗口token - 本轮最大输出 - 安全预留
        )
        if budget <= 0:
            return messages

        # 计算大约消耗的 Token
        estimate, _ = estimate_prompt_tokens_chain(
            self.provider,
            spec.model,
            messages,
            spec.tools.get_definitions(),
        )
        # messages 的消耗token小于预算，不需要裁剪
        if estimate <= budget:
            return messages

        # 所有 system 的 message 留下
        system_messages = [dict(msg) for msg in messages if msg.get("role") == "system"]
        non_system = [dict(msg) for msg in messages if msg.get("role") != "system"]
        if not non_system:
            return messages

        # 计算 system 消息的 token
        system_tokens = sum(estimate_message_tokens(msg) for msg in system_messages)
        # 预算 - system_token 表示给非 system 消息的 token 预留，确保至少有 128
        remaining_budget = max(128, budget - system_tokens)
        # 保留的 message 对象
        kept: list[dict[str, Any]] = []
        kept_tokens = 0

        # 贪心策略，从最新的消息开始向上装
        for message in reversed(non_system):
            msg_tokens = estimate_message_tokens(message)
            # 当前的 token + 已经有的 token >= 非sys的token上限，不能继续装了
            if kept and kept_tokens + msg_tokens > remaining_budget:
                break
            kept.append(message)
            kept_tokens += msg_tokens
        # 再反转回来，确保消息的连贯
        kept.reverse()


        if kept:
            for i, message in enumerate(kept):
                # system 后必须是 "user"
                if message.get("role") == "user":
                    kept = kept[i:]
                    break
            else:
                # Recover nearest user message from outside the kept window;
                # GLM rejects system→assistant (error 1214).  Budget is
                # intentionally exceeded — oversized beats invalid.
                for idx in range(len(non_system) - 1, -1, -1):
                    if non_system[idx].get("role") == "user":
                        kept = non_system[idx:]
                        break
                # If no user exists at all, _enforce_role_alternation
                # will insert a synthetic one as a safety net.
            start = find_legal_message_start(kept)
            if start:
                kept = kept[start:]
        if not kept:
            kept = non_system[-min(len(non_system), 4):]
            start = find_legal_message_start(kept)
            if start:
                kept = kept[start:]
        return system_messages + kept

    def _build_request_kwargs(
            self,
            spec: AgentRunSpec,
            messages: list[dict[str, Any]],
            *,
            tools: list[dict[str, Any]] | None,
            tool_choice: str | dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """ 构建传递给 LLM 的参数 """
        return {
            # 消息
            "messages": messages,
            # 已注册的工具
            "tools": tools or [],
            # 模型
            "model": spec.model or self.provider.get_default_model(),
            "max_tokens": spec.max_tokens,
            "temperature": spec.temperature,
            "reasoning_effort": spec.reasoning_effort,
            # 选择的工具
            "tool_choice": tool_choice,
        }

    async def _request_model(
            self,
            spec: AgentRunSpec,
            messages: list[dict[str, Any]],
            hook: AgentHook,
            context: AgentHookContext,
    ) -> LLMResponse:
        """调用 Provider.chat_with_retry；可选 asyncio 层超时（与 SDK HTTP timeout 独立）。"""
        _ = hook, context  # 预留流式 / Hook 扩展；本条请求的 payload 已在 ``run()`` 内打印

        # 确定超时时间
        timeout_s: float | None = spec.llm_timeout_s
        if timeout_s is None:
            raw = os.environ.get("LLM_RUN_TIMEOUT_S", "300").strip()
            try:
                timeout_s = float(raw)
            except (TypeError, ValueError):
                timeout_s = 300.0
        if timeout_s is not None and timeout_s <= 0:
            timeout_s = None

        # 构建请求参数，准备发送给 LLM Provider
        kwargs = self._build_request_kwargs(
            spec,
            messages,
            tools=spec.tools.get_definitions(),
        )
        # 只创建协程对象，不使用 await
        coro = self.provider.chat_with_retry(**kwargs)

        if timeout_s is None:
            return await coro
        try:
            # 执行协程
            return await asyncio.wait_for(coro, timeout=timeout_s)
        except asyncio.TimeoutError:
            # 超时
            return LLMResponse(
                content=f"调用模型超时（{timeout_s:g}s）",
                finish_reason="error",
                error_kind="timeout",
            )
