"""Claude client on Amazon Bedrock (Messages API through the Bedrock Mantle endpoint)."""

from __future__ import annotations

from .config import Settings


def make_client(settings: Settings):
    """Build the Bedrock client.

    Bedrock does not support the server-side ``fallbacks`` parameter, so refusals are retried
    client-side with ``BetaRefusalFallbackMiddleware`` (guardrail G5). The middleware applies to
    ``client.beta.messages`` calls, which is what the agent uses.
    """
    from anthropic import AnthropicBedrockMantle, BetaRefusalFallbackMiddleware

    middleware = []
    if settings.fallback_model:
        middleware.append(BetaRefusalFallbackMiddleware([{"model": settings.fallback_model}]))
    return AnthropicBedrockMantle(aws_region=settings.aws_region, middleware=middleware or None)


def new_fallback_state():
    """One state per conversation: pins follow-up turns to the model that accepted."""
    from anthropic import BetaFallbackState

    return BetaFallbackState()
