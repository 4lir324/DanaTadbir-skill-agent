import httpx
import openai

from evals.run import render_report, run_case, summarize
from skill_agent import check_llm


class FailingAgent:
    def invoke(self, _prompt):
        raise openai.APITimeoutError(request=httpx.Request("POST", "http://example.test"))


CASE = {"id": 5, "category": "clear_fa", "prompt": "p", "expected": ["summarizer"]}


def test_llm_failure_is_recorded_as_a_failed_case_not_an_abort():
    result = run_case(FailingAgent(), CASE)
    assert result["routing_correct"] is False
    assert result["outcome"].startswith("error")
    assert result["answer_ok"] is None  # no answer check on this case


def test_failed_case_with_answer_check_counts_as_wrong_answer():
    result = run_case(FailingAgent(), {**CASE, "answer_contains": "746"})
    assert result["answer_ok"] is False


def test_report_and_summary_handle_failed_cases():
    results = [run_case(FailingAgent(), CASE)]
    summary = summarize(results)
    assert summary["routing_correct"] == 0 and "error" in render_report(results, summary, "m")


def test_check_llm_reports_missing_key_cleanly(monkeypatch, capsys):
    def no_key():
        raise check_llm.ConfigError("Missing API key")

    monkeypatch.setattr(check_llm, "load_settings", no_key)
    assert check_llm.main() == 2
    assert "Missing API key" in capsys.readouterr().err
