"""
LLM back edge: explain the final search result in plain engineering
language, grounded strictly in the logged numbers -- never invented
reasoning. This is the second (and last) place the LLM touches this
project.
"""
from __future__ import annotations

from llm.client import get_client

SYSTEM_PROMPT = """You are an engineering assistant explaining the result of \
an automated bend-radius optimization to a stamping process engineer. \
You will be given the final search result as structured data. Write a short \
(3-5 sentence) plain-English explanation of the result: what radius was \
found, which constraint governs it, and why. Use ONLY the numbers given to \
you -- never invent or estimate values not present in the data."""


def explain_result(search_result: dict, final_evaluation: dict) -> str:
    """
    search_result: output of search.search_loop.bisection_search()
    final_evaluation: EvaluationResult.as_log_dict() for the final radius
    """
    client, model = get_client()

    context = (
        f"Search result: {search_result}\n"
        f"Final design evaluation: {final_evaluation}"
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": context},
        ],
        temperature=0.2,
        max_tokens=250,
    )

    return response.choices[0].message.content.strip()


if __name__ == "__main__":
    example_search_result = {
        "status": "converged",
        "r_min_feasible": 5.63,
        "bracket_width_mm": 0.04,
        "iterations": 9,
    }
    example_evaluation = {
        "p99_springback_mm": 1.987,
        "max_springback_mm": 2.015,
        "feasible": True,
        "crack_constraint_ok": True,
        "springback_constraint_ok": True,
        "source": "physics",
    }
    print(explain_result(example_search_result, example_evaluation))
