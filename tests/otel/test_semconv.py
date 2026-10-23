from pitstop.otel.semconv import MAX_TEXT, read_span


def test_semconv_reads_current_genai_names():
    chat = read_span("chat gpt-4o", {
        "gen_ai.operation.name": "chat", "gen_ai.request.model": "gpt-4o", "gen_ai.response.model": "gpt-4o-2024",
        "server.address": "api.openai.com", "gen_ai.usage.input_tokens": 120, "gen_ai.usage.output_tokens": 30,
        "gen_ai.response.finish_reasons": ("tool_calls",)}, False)
    assert chat.kind == "llm" and chat.model == "gpt-4o-2024" and chat.server == "api.openai.com"
    assert (chat.input_tokens, chat.output_tokens, chat.finish_reasons) == (120, 30, ("tool_calls",))
    assert chat.error is None

    tool = read_span("execute_tool search", {"gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": "search",
                                             "gen_ai.tool.call.arguments": '{"q": "x"}',
                                             "error.type": "TimeoutError"}, True)
    assert (tool.kind, tool.tool_name, tool.tool_args, tool.error) == ("tool", "search", '{"q": "x"}',
                                                                      "TimeoutError")
    agent = read_span("invoke_agent planner", {"gen_ai.operation.name": "invoke_agent",
                                               "gen_ai.agent.name": "planner"}, True)
    assert (agent.kind, agent.agent_name, agent.error) == ("agent", "planner", "error")
    for op in ("text_completion", "generate_content"):
        assert read_span("x", {"gen_ai.operation.name": op}, False).kind == "llm"
    assert read_span("x", {"gen_ai.operation.name": "embeddings"}, False).kind == "other"


def test_semconv_reads_old_genai_names():
    v = read_span("openai.chat", {"llm.request.type": "chat", "gen_ai.request.model": "gpt-4",
                                  "gen_ai.usage.prompt_tokens": 7, "gen_ai.usage.completion_tokens": 3}, False)
    assert (v.kind, v.model, v.input_tokens, v.output_tokens) == ("llm", "gpt-4", 7, 3)
    assert read_span("c", {"llm.request.type": "completion"}, False).kind == "llm"
    # the current name wins over the old one when both are present
    both = read_span("c", {"gen_ai.operation.name": "chat", "gen_ai.usage.input_tokens": 5,
                           "gen_ai.usage.prompt_tokens": 9}, False)
    assert both.input_tokens == 5


def test_semconv_reads_openinference():
    llm = read_span("ChatOpenAI", {"openinference.span.kind": "LLM", "llm.model_name": "gpt-4o-mini",
                                   "llm.token_count.prompt": 11, "llm.token_count.completion": 4}, False)
    assert (llm.kind, llm.model, llm.input_tokens, llm.output_tokens) == ("llm", "gpt-4o-mini", 11, 4)
    tool = read_span("lookup", {"openinference.span.kind": "TOOL", "tool.name": "lookup",
                                "input.value": '{"id": 3}'}, False)
    assert (tool.kind, tool.tool_name, tool.tool_args) == ("tool", "lookup", '{"id": 3}')
    assert read_span("a", {"openinference.span.kind": "AGENT"}, False).kind == "agent"
    chain = read_span("c", {"openinference.span.kind": "CHAIN", "input.value": "hello"}, False)
    assert chain.kind == "other" and chain.tool_args is None  # input.value is a tool's args only on TOOL spans


def test_semconv_reads_openllmetry():
    llm = read_span("openai.chat", {"llm.request.type": "chat", "gen_ai.request.model": "gpt-4o",
                                    "llm.response.finish_reason": "content_filter",
                                    "llm.response.model_name": "ignored-later-name"}, False)
    assert (llm.kind, llm.model, llm.finish_reasons) == ("llm", "gpt-4o", ("content_filter",))
    assert read_span("search.tool", {"traceloop.span.kind": "tool"}, False).kind == "tool"
    assert read_span("planner.agent", {"traceloop.span.kind": "agent"}, False).kind == "agent"
    assert read_span("w.workflow", {"traceloop.span.kind": "workflow"}, False).kind == "other"


def test_semconv_ignores_wrong_types_and_truncates():
    v = read_span("x", {"gen_ai.operation.name": "chat", "gen_ai.usage.input_tokens": "12",
                        "gen_ai.usage.output_tokens": True, "gen_ai.response.model": 42,
                        "gen_ai.request.model": "m" * 5000, "gen_ai.response.finish_reasons": [1, "stop"]}, False)
    assert v.input_tokens is None and v.output_tokens is None
    assert v.model == "m" * MAX_TEXT and MAX_TEXT == 2000
    assert v.finish_reasons == ("stop",)
    t = read_span("x", {"gen_ai.operation.name": "execute_tool", "gen_ai.tool.call.arguments": "a" * 10**6}, False)
    assert len(t.tool_args) == MAX_TEXT
    odd = read_span("x", {"gen_ai.operation.name": 7}, False)
    assert odd.kind == "other" and odd.error is None
