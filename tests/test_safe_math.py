import pytest

from skill_agent.safe_math import CalculatorError, evaluate, format_number


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("12*345", 4140),
        ("2 + 3 * 4", 14),
        ("(2 + 3) * 4", 20),
        ("2^10", 1024),
        ("2**3**2", 512),
        ("10 / 4", 2.5),
        ("7 // 2", 3),
        ("7 % 3", 1),
        ("-5 + 2", -3),
        ("sqrt(144) + 2", 14),
        ("0.15 * 240", 36),
        ("factorial(5)", 120),
        ("pi", 3.141592653589793),
        # Persian / Arabic digits and symbols
        ("۱۲ × ۳۴۵", 4140),
        ("٣ + ٤", 7),
        ("۱۰ ÷ ۴", 2.5),
        ("۲٫۵ * ۲", 5),
    ],
)
def test_valid_expressions(expression, expected):
    assert evaluate(expression) == pytest.approx(expected)


@pytest.mark.parametrize(
    "expression",
    [
        "",
        "2 +",
        "1/0",
        "__import__('os').system('ls')",
        "open('/etc/passwd')",
        "().__class__.__bases__",
        "[1, 2, 3]",
        "'a' * 3",
        "lambda: 1",
        "x + 1",
        "sqrt(-1)",
        "2 ** 10000",
        "(9**999)**999",
        "factorial(100000)",
        "1" * 300,
    ],
)
def test_rejects_invalid_or_unsafe_expressions(expression):
    with pytest.raises(CalculatorError):
        evaluate(expression)


def test_float_noise_is_hidden():
    assert format_number(evaluate("0.1 + 0.2")) == "0.3"
    assert format_number(4140.0) == "4140"
    assert format_number(float("inf")) == "inf"
