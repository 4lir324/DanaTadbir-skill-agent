"""General chat skill: normal conversation and the graph's fallback."""

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from skill_agent.skills.base import Skill, SkillCard, SkillRequest

SYSTEM_PROMPT = """\
You are a friendly, concise assistant. Answer greetings, small talk and general-knowledge \
questions.

Rules:
- Reply in the language the user wrote in (Persian -> natural, conversational Persian).
- Keep answers short and helpful: a sentence or two for small talk, a short paragraph at most \
for questions.
- If you are not sure of a fact, say so instead of guessing.
- You can also mention, when relevant, that the user can ask you to summarize a text, \
translate a text, or calculate math.
"""


class GeneralChat(Skill):
    card = SkillCard(
        name="general_chat",
        description="Greetings, small talk and general questions that no other skill covers.",
        examples=[
            "Hi, how are you?",
            "What is the capital of France?",
            "سلام، حالت چطوره؟",
            "پایتخت ایتالیا کجاست؟",
        ],
    )

    def run(self, request: SkillRequest, llm: BaseChatModel) -> str:
        reply = llm.invoke([SystemMessage(SYSTEM_PROMPT), HumanMessage(request.original_prompt)])
        return str(reply.content).strip()
