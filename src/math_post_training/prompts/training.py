"""The single prompt protocol used by the post-trained math policy."""

MATH_INSTRUCTION = (
    "Solve the following math problem step by step. "
    "Put your final answer within \\boxed{}."
)


def build_math_prompt(problem):
    """Render the plain-text prompt shared by SFT, GRPO, eval, and inference."""

    return f"{MATH_INSTRUCTION}\n\nProblem: {problem}\n\nSolution:\n"
