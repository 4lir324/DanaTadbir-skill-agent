"""Calculator skill: the LLM only *extracts* the expression; a real evaluator computes it."""

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

from skill_agent.safe_math import CalculatorError, evaluate, format_number
from skill_agent.skills.base import Skill, SkillCard, SkillRequest

SYSTEM_PROMPT = """\
You convert the user's math request into a single expression and MUST call the `calculate` tool \
with it. Never compute or guess the result yourself.

Rules:
- Use only numbers, + - * / ** // %, parentheses, and the functions sqrt, sin, cos, tan, log, \
log10, log2, exp, abs, round, floor, ceil, factorial, plus the constants pi and e.
- Percentages: "15% of 240" -> 0.15 * 240. Powers: "2 to the 10th" -> 2 ** 10.
- Persian wording: "ضرب" is multiplication, "تقسیم" division, "جمع" addition, "منها" \
subtraction, "به توان" power, "جذر" square root. Persian digits are fine.
- If the request has several separate calculations, call the tool once for each.
"""


@tool
def calculate(expression: str) -> str:
    """Evaluate a math expression exactly and return the numeric result."""
    return format_number(evaluate(expression))


class Calculator(Skill):
    card = SkillCard(
        name="calculator",
        description="Computes math: arithmetic, percentages, powers, roots and simple word problems.",
        examples=[
            "What is 15% of 240?",
            "Calculate (3 + 4) * 12 / 5",
            "حاصل ۱۲ ضرب در ۳۴۵ چقدر می‌شود؟",
            "جذر ۱۴۴ به علاوه ۲ به توان ۵ چنده؟",
        ],
    )

    def run(self, request: SkillRequest, llm: BaseChatModel) -> str:
        message = request.instruction if request.previous_output is None else request.render()
        reply = llm.bind_tools([calculate]).invoke(
            [SystemMessage(SYSTEM_PROMPT), HumanMessage(message)]
        )
        if not reply.tool_calls:
            return "I could not find a math expression to compute in this request."
        return "\n".join(self._compute(call["args"].get("expression", "")) for call in reply.tool_calls)

    @staticmethod
    def _compute(expression: str) -> str:
        try:
            return f"{expression} = {format_number(evaluate(expression))}"
        except CalculatorError as error:
            return f"{expression} -> error: {error}"
