"""Shortlist stage: keep the router prompt O(k) no matter how many skills exist.

With N <= k skills the stage is a pass-through. With more skills, a retriever
picks the k most relevant cards for the prompt, and only those reach the router.
"""

import math
from collections import Counter
from typing import Protocol

from skill_agent.skills.base import SkillCard
from skill_agent.text import tokenize

DEFAULT_K = 4
_STEM_LENGTH = 5  # crude prefix stemming: "summarize"/"summary" share "summ"


class SkillRetriever(Protocol):
    """Swap-in point: e.g. an embedding retriever can replace the lexical one."""

    def select(self, prompt: str, cards: list[SkillCard], k: int) -> list[SkillCard]: ...


_STOPWORDS = frozenset(
    # Persian function words
    "این آن را رو که با به از در برای لطفا می هم و یا من تو ما شما است هست یک ببین بده"
    # English function words
    " the a an this that these those to of in on and or please me my it is are for you can"
    " with do does what how".split()
)


def _stem(token: str) -> str:
    # Tokens with digits are identifiers ("model2", "gpt4"): truncating would merge them.
    return token if any(ch.isdigit() for ch in token) else token[:_STEM_LENGTH]


def _stems(text: str) -> set[str]:
    return {_stem(token) for token in tokenize(text) if token not in _STOPWORDS}


def _card_text(card: SkillCard) -> str:
    return " ".join([card.name.replace("_", " "), card.description, *card.examples])


class LexicalRetriever:
    """IDF-weighted stem overlap between the prompt and each card. No downloads, works offline."""

    def select(self, prompt: str, cards: list[SkillCard], k: int) -> list[SkillCard]:
        card_stems = [_stems(_card_text(card)) for card in cards]
        document_frequency = Counter(stem for stems in card_stems for stem in stems)
        prompt_stems = _stems(prompt)

        def score(stems: set[str]) -> float:
            return sum(
                math.log(1 + len(cards) / document_frequency[stem])
                for stem in prompt_stems & stems
            )

        scored = sorted(zip(cards, map(score, card_stems)), key=lambda pair: -pair[1])
        return [card for card, value in scored[:k] if value > 0]


def select_candidates(
    prompt: str,
    cards: list[SkillCard],
    retriever: SkillRetriever | None = None,
    k: int = DEFAULT_K,
) -> list[SkillCard]:
    """Return the cards the router may choose from."""
    if len(cards) <= k:
        return cards
    return (retriever or LexicalRetriever()).select(prompt, cards, k)
