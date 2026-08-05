"""GenerationBackend adapter for vLLM.

The optional vLLM dependency must be imported lazily so importing ``math_post_training`` remains
possible in development and training environments where the inference group is absent.
"""

