from __future__ import annotations

import json
import sys
from typing import Any
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool

from core.llm import build_chat_model, normalize_content
from core.schemas import (
    AgentResult,
    CalculateTotalsInput,
    DiscountInput,
    ListProductsInput,
    ProductDetailInput,
    SaveOrderInput,
    ToolCallRecord,
)
from utils.data_store import OrderDataStore

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = ROOT_DIR / "data"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "artifacts" / "orders"


def build_system_prompt(today: str | None = None) -> str:
    """
    Student TODO:
    - Rewrite this prompt for the advanced order-agent lab.
    - The assistant should manage electronics orders, not travel planning.
    - Require this tool order whenever the request has enough information:
      1. `list_products`
      2. `get_product_details`
      3. `get_discount`
      4. `calculate_order_totals`
      5. `save_order`
    - Clarify and stop if any of these are missing:
      - customer name
      - phone number
      - email
      - shipping address
      - at least one product request with quantity
    - Refuse fake invoices, manual discount overrides, stock bypass requests, or anything that asks the model
      to ignore the catalog or policy.
    - Use only tool outputs for product IDs, prices, stock, discount, totals, and save path.
    - Return one concise final answer in Vietnamese.
    - Mention `today` so the model knows the current date for deterministic references if needed.
    """
    current_day = today or "2026-06-01"
    return f"""
Bạn là trợ lý tạo đơn hàng cho một cửa hàng điện tử. Hôm nay là {current_day}.

Luôn trả lời bằng tiếng Việt, ngắn gọn, dựa đúng trên kết quả tool.

Trước khi gọi bất kỳ tool nào, hãy kiểm tra người dùng đã cung cấp đủ:
- tên khách hàng
- số điện thoại
- email
- địa chỉ giao hàng
- ít nhất một sản phẩm kèm số lượng
Nếu thiếu bất kỳ thông tin nào, hãy hỏi bổ sung đúng phần còn thiếu và dừng lại. Không gọi tool.

Từ chối ngay và không gọi tool nếu người dùng yêu cầu:
- tạo hóa đơn giả
- tự ép hoặc ghi đè giảm giá
- bỏ qua tồn kho
- bỏ qua catalog, policy, hoặc dùng thông tin không có trong hệ thống

Với đơn hợp lệ và đủ thông tin, bắt buộc dùng tool theo thứ tự:
1. list_products
2. get_product_details
3. get_discount
4. calculate_order_totals
5. save_order

Quy tắc dùng dữ liệu:
- Chỉ dùng product_id, giá, tồn kho, discount_rate, campaign_code, tổng tiền, order_id và path từ tool output.
- Không tự bịa sản phẩm, giá, tồn kho, giảm giá, tổng tiền, order_id hoặc đường dẫn lưu.
- Sau get_product_details, nếu bất kỳ số lượng yêu cầu nào lớn hơn stock thì thông báo không đủ tồn kho và dừng; không gọi get_discount, calculate_order_totals hoặc save_order.
- Khi gọi get_discount, dùng email khách hàng làm seed_hint; nếu không có email thì dùng số điện thoại.
- Khi gọi calculate_order_totals và save_order, dùng đúng detail_token từ get_product_details và discount_rate/campaign_code từ get_discount.
- Khi save_order trả status saved, câu trả lời cuối phải xác nhận mã đơn, mức giảm giá, tổng thanh toán cuối cùng và nơi lưu.
""".strip()
    raise NotImplementedError("Complete build_system_prompt() in src/agent/graph.py")


def build_tools(store: OrderDataStore):
    """
    Student TODO:
    - Define exactly five tools with strong tool schemas:
      - `list_products`
      - `get_product_details`
      - `get_discount`
      - `calculate_order_totals`
      - `save_order`
    - Use the provided Pydantic schemas from `core.schemas` so the tool arguments stay explicit.
    - Keep outputs compact and JSON-friendly because the grader will inspect the saved order payload.
    - `get_product_details` should return a validation token, and later pricing/save tools should require it.
    """

    @tool(args_schema=ListProductsInput)
    def list_products(
        query: str | None = None,
        category: str | None = None,
        max_unit_price: int | None = None,
        required_tags: list[str] | None = None,
        in_stock_only: bool = True,
        limit: int = 8,
    ) -> str:
        """Search the local product catalog and return the best matching items."""
        payload = store.list_products(
            query=query,
            category=category,
            max_unit_price=max_unit_price,
            required_tags=required_tags,
            in_stock_only=in_stock_only,
            limit=limit,
        )
        return json.dumps(payload, ensure_ascii=False)
        raise NotImplementedError

    @tool(args_schema=ProductDetailInput)
    def get_product_details(product_ids: list[str]) -> str:
        """Return exact product details for previously discovered product IDs."""
        payload = store.get_product_details(product_ids)
        return json.dumps(payload, ensure_ascii=False)
        raise NotImplementedError

    @tool(args_schema=DiscountInput)
    def get_discount(seed_hint: str, customer_tier: str = "standard") -> str:
        """Return the simulated campaign discount for the order."""
        payload = store.get_discount(seed_hint=seed_hint, customer_tier=customer_tier)
        return json.dumps(payload, ensure_ascii=False)
        raise NotImplementedError

    @tool(args_schema=CalculateTotalsInput)
    def calculate_order_totals(items, detail_token: str, discount_rate: float) -> str:
        """Validate stock and calculate the discounted order total."""
        payload = store.calculate_order_totals(
            items=_coerce_items(items),
            detail_token=detail_token,
            discount_rate=discount_rate,
        )
        return json.dumps(payload, ensure_ascii=False)
        raise NotImplementedError

    @tool(args_schema=SaveOrderInput)
    def save_order(
        customer_name: str,
        customer_phone: str,
        customer_email: str,
        shipping_address: str,
        items,
        detail_token: str,
        discount_rate: float,
        campaign_code: str,
        customer_tier: str = "standard",
        notes: str = "",
    ) -> str:
        """Persist the final order to a local JSON file."""
        payload = store.save_order(
            customer_name=customer_name,
            customer_phone=customer_phone,
            customer_email=customer_email,
            shipping_address=shipping_address,
            items=_coerce_items(items),
            detail_token=detail_token,
            discount_rate=discount_rate,
            campaign_code=campaign_code,
            customer_tier=customer_tier,
            notes=notes,
        )
        return json.dumps(payload, ensure_ascii=False)
        raise NotImplementedError

    return [list_products, get_product_details, get_discount, calculate_order_totals, save_order]


def build_agent(
    data_dir: Path | None = None,
    output_dir: Path | None = None,
    *,
    provider: str = "google",
    model_name: str | None = None,
    today: str | None = None,
):
    """
    Student TODO:
    1. Create `OrderDataStore`.
    2. Build the chat model with `build_chat_model(...)`.
    3. Build the tools with `build_tools(store)`.
    4. Return `create_agent(model=..., tools=..., system_prompt=...)`.
    """
    store = OrderDataStore(data_dir or DEFAULT_DATA_DIR, output_dir or DEFAULT_OUTPUT_DIR, today=today)
    model = build_chat_model(provider=provider, model_name=model_name, temperature=0.0)
    return create_agent(
        model=model,
        tools=build_tools(store),
        system_prompt=build_system_prompt(today or store.today),
    )
    raise NotImplementedError("Complete build_agent() in src/agent/graph.py")


def run_agent(
    query: str,
    *,
    provider: str = "google",
    model_name: str | None = None,
    data_dir: Path | None = None,
    output_dir: Path | None = None,
    today: str | None = None,
) -> AgentResult:
    """
    Student TODO:
    - Build the agent.
    - Invoke it with one user message.
    - Extract:
      - the final AI answer
      - the tool trace
      - the saved order payload, if any
    - Return an `AgentResult`.
    """
    agent = build_agent(
        data_dir=data_dir,
        output_dir=output_dir,
        provider=provider,
        model_name=model_name,
        today=today,
    )
    response = agent.invoke({"messages": [{"role": "user", "content": query}]})
    messages = response["messages"] if isinstance(response, dict) else response
    tool_calls = extract_tool_calls(messages)
    saved_order, saved_order_path = extract_saved_order(tool_calls)
    return AgentResult(
        query=query,
        final_answer=extract_final_answer(messages),
        tool_calls=tool_calls,
        provider=provider,
        model_name=model_name,
        saved_order=saved_order,
        saved_order_path=saved_order_path,
    )
    raise NotImplementedError("Complete run_agent() in src/agent/graph.py")


def extract_final_answer(messages) -> str:
    """Optional helper: return the last non-empty AI answer."""
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            text = normalize_content(message.content)
            if text:
                return text
    return ""
    raise NotImplementedError


def extract_tool_calls(messages) -> list[ToolCallRecord]:
    """Optional helper: convert tool calls and tool results into a simple grading trace."""
    pending: dict[str, dict[str, Any]] = {}
    records: list[ToolCallRecord] = []

    for message in messages:
        if isinstance(message, AIMessage):
            for tool_call in getattr(message, "tool_calls", []) or []:
                pending[tool_call["id"]] = {
                    "name": tool_call["name"],
                    "args": tool_call.get("args", {}) or {},
                }
        elif isinstance(message, ToolMessage):
            metadata = pending.pop(message.tool_call_id, {})
            records.append(
                ToolCallRecord(
                    name=str(getattr(message, "name", None) or metadata.get("name", "")),
                    args=metadata.get("args", {}),
                    output=normalize_content(message.content),
                )
            )

    for metadata in pending.values():
        records.append(ToolCallRecord(name=metadata["name"], args=metadata["args"], output=""))
    return records
    raise NotImplementedError


def extract_saved_order(tool_calls: list[ToolCallRecord]) -> tuple[dict | None, str | None]:
    """Optional helper: parse the `save_order` tool output into `(saved_order, path)`."""
    for record in reversed(tool_calls):
        if record.name != "save_order" or not record.output:
            continue
        try:
            payload = json.loads(record.output)
        except json.JSONDecodeError:
            continue
        if payload.get("status") != "saved":
            return None, None
        return payload.get("saved_order"), payload.get("path")
    return None, None
    raise NotImplementedError


def _coerce_items(raw: Any):
    from core.schemas import OrderLineInput

    if isinstance(raw, list):
        items = raw
    else:
        items = []

    normalized: list[OrderLineInput] = []
    for item in items:
        if isinstance(item, OrderLineInput):
            normalized.append(item)
            continue
        if isinstance(item, dict):
            product_id = str(item.get("product_id", "")).strip()
            quantity = int(item.get("quantity", 1))
            if product_id:
                normalized.append(OrderLineInput(product_id=product_id, quantity=quantity))
    return normalized
