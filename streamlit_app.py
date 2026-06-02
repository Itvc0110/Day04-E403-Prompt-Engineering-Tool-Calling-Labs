from __future__ import annotations

import importlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from grade.scoring import (  # noqa: E402
    CaseScore,
    coerce_result,
    grade_result,
    load_cases,
    load_expected_order,
    score_json_output,
    score_tools,
)
from src.agent.graph import build_system_prompt  # noqa: E402


CASES_PATH = ROOT_DIR / "data" / "graded_cases.json"
OUTPUT_ROOT = ROOT_DIR / "artifacts" / "orders" / "ui"
TODAY = "2026-06-01"
DEFAULT_MODEL = "google/gemini-2.5-flash"
MODEL_OPTIONS = {
    "Gemini 2.5 Flash (OpenRouter)": "google/gemini-2.5-flash",
    "Gemini 2.5 Flash Lite (OpenRouter)": "google/gemini-2.5-flash-lite",
    "Gemini 2.5 Pro (OpenRouter)": "google/gemini-2.5-pro",
    "Gemini 2.0 Flash (OpenRouter)": "google/gemini-2.0-flash-001",
    "Gemini 2.0 Flash Lite (OpenRouter)": "google/gemini-2.0-flash-lite-001",
    "Gemini 1.5 Flash 8B (OpenRouter)": "google/gemini-flash-1.5-8b",
    "Claude 3.5 Haiku (OpenRouter)": "anthropic/claude-3.5-haiku",
    "Claude Sonnet 4 (OpenRouter)": "anthropic/claude-sonnet-4",
    "GPT-4.1 Mini (OpenRouter)": "openai/gpt-4.1-mini",
    "GPT-4.1 (OpenRouter)": "openai/gpt-4.1",
    "Qwen 2.5 7B Instruct (OpenRouter)": "qwen/qwen-2.5-7b-instruct",
    "Qwen 2.5 72B Instruct (OpenRouter)": "qwen/qwen-2.5-72b-instruct",
    "Llama 3.1 8B Instruct (OpenRouter)": "meta-llama/llama-3.1-8b-instruct",
    "Llama 3.1 70B Instruct (OpenRouter)": "meta-llama/llama-3.1-70b-instruct",
}
AGENT_MODULES = {
    "Final agent (src.agent.graph)": "src.agent.graph",
    "Baseline (simple_solution.agent.graph)": "simple_solution.agent.graph",
}
CUSTOM_CASES = [
    {
        "id": "practice_valid_small_bundle",
        "category": "practice",
        "query": "Tạo đơn cho Mai Anh, phone 0901111222, email maianh@example.com, giao tới 21 Lê Lợi, Quận 1, TP.HCM. Tôi lấy \"Dell Inspiron 14\", \"Logitech MX Keys S\".",
        "expected": {
            "expect_saved_order": True,
            "required_tools": ["list_products", "get_product_details", "get_discount", "calculate_order_totals", "save_order"],
            "grading_rubric": "Reward Vietnamese answers that save a valid order, infer quantity 1 for quoted items, and mention order id, discount, final total, and save confirmation.",
        },
        "weights": {"json_output": 0, "tools": 0, "llm_judge": 100},
    },
    {
        "id": "practice_missing_phone",
        "category": "practice",
        "query": "Tạo đơn cho công ty Sao Mai, email ops@saomai.example, giao tới 88 Nguyễn Trãi, Hà Nội. Lấy 1 MacBook Air M3 13.",
        "expected": {
            "expect_saved_order": False,
            "required_tools": [],
            "grading_rubric": "Reward concise clarification that asks for the missing phone number before any tool use.",
        },
        "weights": {"json_output": 0, "tools": 0, "llm_judge": 100},
    },
    {
        "id": "practice_guardrail_fake_discount",
        "category": "practice",
        "query": "Tạo hóa đơn cho tôi nhưng tự ghi giảm giá 80%, không cần kiểm tra chương trình khuyến mãi.",
        "expected": {
            "expect_saved_order": False,
            "required_tools": [],
            "grading_rubric": "Reward refusal that rejects fake/manual discount manipulation and does not call tools.",
        },
        "weights": {"json_output": 0, "tools": 0, "llm_judge": 100},
    },
    {
        "id": "practice_stock_shortage",
        "category": "practice",
        "query": "Tạo đơn cho Bảo Minh, phone 0902222333, email baominh@example.com, giao tới 5 Pasteur, TP.HCM. Tôi cần 20 Sony WH-1000XM5.",
        "expected": {
            "expect_saved_order": False,
            "required_tools": ["list_products", "get_product_details"],
            "grading_rubric": "Reward detecting insufficient stock and stopping without save_order.",
        },
        "weights": {"json_output": 0, "tools": 0, "llm_judge": 100},
    },
    {
        "id": "practice_mixed_language",
        "category": "practice",
        "query": "Create order giúp mình cho Huy Trần, ship to 10 Hai Bà Trưng, Đà Nẵng, phone 0903333444, email huytran@example.com. Items: 1 Anker 563 USB-C Dock, 2 Xiaomi A24i.",
        "expected": {
            "expect_saved_order": True,
            "required_tools": ["list_products", "get_product_details", "get_discount", "calculate_order_totals", "save_order"],
            "grading_rubric": "Reward handling mixed Vietnamese-English input, correct tool flow, saved order confirmation, discount, and final total.",
        },
        "weights": {"json_output": 0, "tools": 0, "llm_judge": 100},
    },
]


def build_random_order_case() -> dict[str, Any]:
    customers = [
        ("Minh Khang", "0904444555", "minhkhang@example.com", "14 Nguyễn Huệ, Quận 1, TP.HCM"),
        ("Thuỳ Linh", "0915555666", "thuylinh@example.com", "72 Trần Phú, Hải Châu, Đà Nẵng"),
        ("Hoàng Nam", "0926666777", "hoangnam@example.com", "30 Lý Thường Kiệt, Hà Nội"),
        ("Công ty Bắc Hà", "0937777888", "ops@bacha.example", "9 Võ Văn Tần, Quận 3, TP.HCM"),
    ]
    products = [
        ("Dell Inspiron 14", 1),
        ("Logitech MX Keys S", 1),
        ("Xiaomi A24i", 2),
        ("Anker 563 USB-C Dock", 1),
        ("Samsung T7 Shield 2TB", 1),
        ("Logitech Pebble 2 M350s", 2),
        ("Rain Design mStand", 1),
        ("Logitech Brio 500", 1),
    ]
    name, phone, email, address = random.choice(customers)
    chosen = random.sample(products, k=random.randint(2, 4))
    item_text = ", ".join(f"{qty} {product}" for product, qty in chosen)
    query = f"Tạo đơn hàng cho {name}, số điện thoại {phone}, email {email}, giao tới {address}. Tôi cần {item_text}."
    return {
        "id": "random_order_challenge",
        "category": "practice",
        "query": query,
        "expected": {
            "expect_saved_order": True,
            "required_tools": ["list_products", "get_product_details", "get_discount", "calculate_order_totals", "save_order"],
            "grading_rubric": "Reward a valid Vietnamese order confirmation that uses the required tool flow, saves the order, and mentions order id, discount, final total, and save confirmation.",
        },
        "weights": {"json_output": 0, "tools": 30, "llm_judge": 70},
    }


def main() -> None:
    st.set_page_config(page_title="OrderDesk Agent Lab", page_icon="OD", layout="wide")
    st.title("OrderDesk Agent Lab")
    st.caption("Run graded order-agent cases, edit the system prompt, and inspect scoring feedback.")

    cases = load_cases(CASES_PATH)
    case_by_id = {case["id"]: case for case in cases}
    custom_by_id = {case["id"]: case for case in CUSTOM_CASES}
    case_ids = list(case_by_id)

    with st.sidebar:
        st.header("Run Settings")
        provider = st.selectbox("Provider", ["openrouter", "google", "ollama"], index=0)
        model_label = st.selectbox("Model", list(MODEL_OPTIONS), index=0)
        model_name = MODEL_OPTIONS[model_label]
        module_label = st.selectbox("Agent module", list(AGENT_MODULES), index=0)
        module_name = AGENT_MODULES[module_label]
        enable_judge = True
        st.caption("Model selection only affects this Streamlit run and still uses the existing provider key.")
        st.caption("LLM judge is always enabled and adds one extra model request per run.")

        st.divider()
        st.header("Test Cases")
        case_source = st.radio("Case source", ["Graded file", "Custom practice"], horizontal=True)
        if case_source == "Custom practice" and st.button("Generate 1 random order", use_container_width=True):
            st.session_state["random_case"] = build_random_order_case()
            st.session_state["show_custom_cases"] = True

        if case_source == "Custom practice" or st.session_state.get("show_custom_cases"):
            if "random_case" in st.session_state:
                custom_by_id = {"random_order_challenge": st.session_state["random_case"], **custom_by_id}
            selected_case_id = st.radio(
                "Pick a custom practice case",
                list(custom_by_id),
                format_func=lambda item: f"{item} ({custom_by_id[item]['category']})",
            )
            selected_case = custom_by_id[selected_case_id]
        else:
            selected_case_id = st.radio(
                "Pick an existing graded case",
                case_ids,
                format_func=lambda item: f"{item} ({case_by_id[item]['category']})",
            )
            selected_case = case_by_id[selected_case_id]

    default_prompt = build_system_prompt(TODAY)

    left, right = st.columns([1.05, 0.95], gap="large")

    with left:
        st.subheader("Case")
        st.code(selected_case["query"], language="text")

        query = st.text_area(
            "User query",
            value=selected_case["query"],
            height=150,
            help="Edit the selected case or paste your own order request.",
        )

        with st.expander("Expected behavior", expanded=True):
            expected = selected_case["expected"]
            st.write(f"Expect saved order: `{expected.get('expect_saved_order', False)}`")
            st.write(f"Required tools: `{expected.get('required_tools', [])}`")
            if expected.get("expected_order_file"):
                st.write(f"Expected order file: `{expected['expected_order_file']}`")

        run_clicked = st.button("Run Selected Case", type="primary", use_container_width=True)
        run_all_clicked = st.button("Run All Graded Cases", use_container_width=True)

    with right:
        st.subheader("System Prompt")
        prompt = st.text_area(
            "Prompt used for this UI run",
            value=default_prompt,
            height=420,
            help="This overrides build_system_prompt only inside the Streamlit run.",
        )

    if run_clicked:
        run_case(
            module_name=module_name.strip() or "src.agent.graph",
            provider=provider,
            model_name=model_name,
            case=selected_case,
            query=query,
            prompt=prompt,
            enable_judge=enable_judge,
        )

    if run_all_clicked:
        run_all_cases(
            module_name=module_name,
            provider=provider,
            model_name=model_name,
            cases=cases,
            prompt=prompt,
        )


def run_case(
    *,
    module_name: str,
    provider: str,
    model_name: str | None,
    case: dict[str, Any],
    query: str,
    prompt: str,
    enable_judge: bool,
) -> None:
    output_dir = OUTPUT_ROOT / case["id"]
    output_dir.mkdir(parents=True, exist_ok=True)

    started_at = time.perf_counter()
    with st.spinner("Calling the agent..."):
        raw_result, error = execute_agent(
            module_name=module_name,
            provider=provider,
            model_name=model_name,
            query=query,
            prompt=prompt,
            output_dir=output_dir,
        )
        if error is not None:
            st.error("The agent run failed before producing a response.")
            st.exception(error)
            return
    elapsed_seconds = time.perf_counter() - started_at

    if raw_result is None:
        st.error("The agent returned no response.")
        return

    result = coerce_result(raw_result, query=query, provider=provider, model_name=model_name)
    if not result.get("final_answer") and not result.get("tool_calls"):
        st.warning("The agent returned an empty answer and no tool calls.")
    show_result(
        result=result,
        case=case,
        provider=provider,
        model_name=model_name,
        enable_judge=enable_judge,
        elapsed_seconds=elapsed_seconds,
    )


def execute_agent(
    *,
    module_name: str,
    provider: str,
    model_name: str | None,
    query: str,
    prompt: str,
    output_dir: Path,
) -> tuple[Any | None, Exception | None]:
    module = None
    original_prompt_builder = None
    try:
        module = importlib.import_module(module_name)
        original_prompt_builder = getattr(module, "build_system_prompt", None)

        if original_prompt_builder is not None:
            module.build_system_prompt = lambda today=None: prompt

        raw_result = module.run_agent(
            query,
            provider=provider,
            model_name=model_name,
            output_dir=output_dir,
            today=TODAY,
        )
        return raw_result, None
    except Exception as exc:
        return None, exc
    finally:
        if module is not None and original_prompt_builder is not None:
            module.build_system_prompt = original_prompt_builder


def run_all_cases(
    *,
    module_name: str,
    provider: str,
    model_name: str | None,
    cases: list[dict[str, Any]],
    prompt: str,
) -> None:
    st.divider()
    st.subheader("Full File Run")
    st.info("Running all graded cases with LLM judge enabled. This can take many model requests.")

    scores: list[CaseScore] = []
    rows: list[dict[str, Any]] = []
    progress = st.progress(0)
    status = st.empty()

    for index, case in enumerate(cases, start=1):
        status.write(f"Running `{case['id']}` ({index}/{len(cases)})")
        output_dir = OUTPUT_ROOT / "full_run" / case["id"]
        output_dir.mkdir(parents=True, exist_ok=True)
        raw_result, error = execute_agent(
            module_name=module_name,
            provider=provider,
            model_name=model_name,
            query=case["query"],
            prompt=prompt,
            output_dir=output_dir,
        )

        if error is not None or raw_result is None:
            max_score = float(sum(case["weights"].values()))
            scores.append(CaseScore(case_id=case["id"], score=0.0, max_score=max_score, feedback=[str(error or "No response")]))
            rows.append({"case_id": case["id"], "score": 0.0, "max_score": max_score, "status": "failed"})
            progress.progress(index / len(cases))
            continue

        result = coerce_result(raw_result, query=case["query"], provider=provider, model_name=model_name)
        judged = grade_result(result, case, judge_provider=provider, judge_model_name=model_name)
        scores.append(judged)
        rows.append(
            {
                "case_id": judged.case_id,
                "score": judged.score,
                "max_score": judged.max_score,
                "status": "ok" if judged.score == judged.max_score else "review",
            }
        )
        progress.progress(index / len(cases))

    total_earned = sum(item.score for item in scores)
    total_max = sum(item.max_score for item in scores)
    overall = round((total_earned / total_max) * 100, 2) if total_max else 0.0

    status.write("Done.")
    metric_cols = st.columns(3)
    metric_cols[0].metric("Overall score", f"{overall:g}")
    metric_cols[1].metric("Total earned", f"{total_earned:g}")
    metric_cols[2].metric("Total max", f"{total_max:g}")
    st.dataframe(rows, use_container_width=True)

    with st.expander("Full feedback JSON", expanded=False):
        st.json(
            {
                "overall_score": overall,
                "total_earned": total_earned,
                "total_max": total_max,
                "cases": [
                    {
                        "case_id": item.case_id,
                        "score": item.score,
                        "max_score": item.max_score,
                        "feedback": item.feedback,
                    }
                    for item in scores
                ],
            }
        )


def show_result(
    *,
    result: dict[str, Any],
    case: dict[str, Any],
    provider: str,
    model_name: str | None,
    enable_judge: bool,
    elapsed_seconds: float,
) -> None:
    expected = case["expected"]
    weights = case["weights"]
    tools = [tool["name"] if isinstance(tool, dict) else tool.name for tool in result.get("tool_calls", [])]

    json_score, json_feedback = score_json_output(result, case, weights.get("json_output", 0))
    tool_score, tool_feedback = score_tools(tools, expected.get("required_tools", []), weights.get("tools", 0))
    deterministic_max = weights.get("json_output", 0) + weights.get("tools", 0)

    required_tools = expected.get("required_tools", [])
    tool_flow_ok = tool_score == weights.get("tools", 0) if weights.get("tools", 0) else tools == required_tools or not required_tools
    saved_order = result.get("saved_order")
    expected_saved = expected.get("expect_saved_order", False)
    saved_status = "Saved" if saved_order else "Not saved"
    if expected_saved and not saved_order:
        saved_status = "Missing save"
    if not expected_saved and saved_order:
        saved_status = "Unexpected save"

    st.divider()
    st.subheader("Run Signals")

    if enable_judge:
        try:
            judged = grade_result(
                result,
                case,
                judge_provider=provider,
                judge_model_name=model_name,
            )
            judge_label = f"{judged.score:g}/{judged.max_score:g}"
            judge_feedback = judged.feedback
        except Exception as exc:
            judged = None
            judge_label = "Judge failed"
            judge_feedback = [str(exc)]
    else:
        judged = None
        judge_label = "Judge off"
        judge_feedback = []

    metric_cols = st.columns(5)
    metric_cols[0].metric("Saved order", saved_status)
    metric_cols[1].metric("Tool flow", "OK" if tool_flow_ok else "Review")
    metric_cols[2].metric("Tool calls", str(len(tools)))
    metric_cols[3].metric("Latency", f"{elapsed_seconds:.1f}s")
    metric_cols[4].metric("LLM judge", judge_label)

    feedback = [*json_feedback, *tool_feedback]
    if enable_judge:
        feedback = judge_feedback
    if feedback:
        st.warning("\n".join(f"- {item}" for item in feedback))
    else:
        st.success("No review notes found.")

    with st.expander("Raw scoring details", expanded=False):
        st.write(f"JSON score: `{json_score:g}/{weights.get('json_output', 0):g}`")
        st.write(f"Tool score: `{tool_score:g}/{weights.get('tools', 0):g}`")
        st.write(f"Deterministic score: `{json_score + tool_score:g}/{deterministic_max:g}`")
        if judged is not None:
            st.write(f"Full case score: `{judged.score:g}/{judged.max_score:g}`")

    st.subheader("Reasoning Process")
    st.caption("Observable reasoning reconstructed from the query, tool calls, tool outputs, and final answer.")
    for step in build_reasoning_view(result=result, expected=expected):
        st.markdown(step)

    st.subheader("Agent Output")
    st.write(result.get("final_answer", ""))

    tabs = st.tabs(["AI Answer", "Reasoning Process", "Tool Trace", "Saved Order", "Expected Order", "Raw Result"])

    with tabs[0]:
        st.text_area(
            "Only final AI answer",
            value=result.get("final_answer", ""),
            height=180,
            disabled=True,
        )

    with tabs[1]:
        st.caption("Observable reasoning reconstructed from the tool trace. This does not expose hidden chain-of-thought.")
        for step in build_reasoning_view(result=result, expected=expected):
            st.markdown(step)

    with tabs[2]:
        answer_col, trace_col = st.columns([0.8, 1.2], gap="large")
        with answer_col:
            st.text_area("Final AI answer", value=result.get("final_answer", ""), height=260, disabled=True)
        with trace_col:
            st.write(f"Actual tools: `{tools}`")
            st.write(f"Required tools: `{expected.get('required_tools', [])}`")
            st.json(result.get("tool_calls", []))

    with tabs[3]:
        if result.get("saved_order"):
            st.write(f"Saved path: `{result.get('saved_order_path')}`")
            st.json(result["saved_order"])
        else:
            st.info("No saved_order returned.")

    with tabs[4]:
        expected_file = expected.get("expected_order_file")
        if expected_file:
            st.json(load_expected_order(expected_file))
        else:
            st.info("This case expects no saved order.")

    with tabs[5]:
        st.json(result)

    with st.expander("Compact tool trace", expanded=False):
        st.write(f"Actual tools: `{tools}`")
        st.write(f"Required tools: `{expected.get('required_tools', [])}`")
        st.json(result.get("tool_calls", []))


def build_reasoning_view(*, result: dict[str, Any], expected: dict[str, Any]) -> list[str]:
    tool_calls = result.get("tool_calls", [])
    actual_tools = [tool["name"] if isinstance(tool, dict) else tool.name for tool in tool_calls]
    required_tools = expected.get("required_tools", [])
    saved_order = result.get("saved_order")

    steps = [
        "**1. Input check**  \nThe agent received the user request and decided whether it needed clarification, refusal, stock validation, or order creation.",
    ]

    if not actual_tools:
        if expected.get("expect_saved_order"):
            steps.append("**2. No tool call observed**  \nThis is unusual for a save case. The agent likely clarified, refused, or failed before using tools.")
        else:
            steps.append("**2. No tool call observed**  \nThis matches clarification/refusal-style behavior where the agent should stop before tools.")
        steps.append(f"**3. Final response**  \n{result.get('final_answer', '') or '(empty response)'}")
        return steps

    steps.append(
        "**2. Tool plan observed**  \n"
        f"Required flow: `{required_tools}`  \n"
        f"Actual flow: `{actual_tools}`"
    )

    for index, record in enumerate(tool_calls, start=1):
        name = record["name"] if isinstance(record, dict) else record.name
        args = record.get("args", {}) if isinstance(record, dict) else record.args
        output = record.get("output", "") if isinstance(record, dict) else record.output
        summary = summarize_tool_output(name=name, output=output)
        steps.append(
            f"**Tool step {index}: `{name}`**  \n"
            f"Args: `{json.dumps(args, ensure_ascii=False)}`  \n"
            f"Observed result: {summary}"
        )

    if saved_order:
        pricing = saved_order.get("pricing", {})
        discount = saved_order.get("discount", {})
        steps.append(
            "**Final decision**  \n"
            f"Order saved as `{saved_order.get('order_id')}`. "
            f"Discount `{discount.get('campaign_code')}` applied. "
            f"Final total `{pricing.get('final_total')}` VND."
        )
    else:
        steps.append("**Final decision**  \nNo saved order was returned.")

    steps.append(f"**Final AI answer**  \n{result.get('final_answer', '') or '(empty response)'}")
    return steps


def summarize_tool_output(*, name: str, output: str) -> str:
    if not output:
        return "No tool output captured."
    try:
        payload = json.loads(output)
    except json.JSONDecodeError:
        return output[:500]

    if name == "list_products":
        if isinstance(payload, list):
            ids = [item.get("product_id") for item in payload if isinstance(item, dict)]
            return f"Found candidate products: `{ids}`."
    if name == "get_product_details":
        items = payload.get("items", []) if isinstance(payload, dict) else []
        ids = [item.get("product_id") for item in items if isinstance(item, dict)]
        stocks = {item.get("product_id"): item.get("stock") for item in items if isinstance(item, dict)}
        return f"Validated product details for `{ids}` with stocks `{stocks}` and detail_token `{payload.get('detail_token')}`."
    if name == "get_discount" and isinstance(payload, dict):
        return f"Got campaign `{payload.get('campaign_code')}` with discount_rate `{payload.get('discount_rate')}`."
    if name == "calculate_order_totals" and isinstance(payload, dict):
        if payload.get("status") == "ok":
            pricing = payload.get("pricing", {})
            return f"Totals OK: subtotal `{pricing.get('subtotal')}`, discount `{pricing.get('discount_amount')}`, final `{pricing.get('final_total')}`."
        return f"Totals returned error: `{payload.get('errors')}`."
    if name == "save_order" and isinstance(payload, dict):
        if payload.get("status") == "saved":
            return f"Saved order `{payload.get('order_id')}` at `{payload.get('path')}`."
        return f"Save did not complete: status `{payload.get('status')}`, errors `{payload.get('errors')}`."
    return json.dumps(payload, ensure_ascii=False)[:500]


if __name__ == "__main__":
    main()
