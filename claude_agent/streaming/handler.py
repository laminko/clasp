from __future__ import annotations

from collections.abc import AsyncIterator

from ..types.responses import Response, Usage
from ..utils.errors import ParseError
from ..utils.helpers import get_logger
from .events import Event, ResultEvent, TextChunkEvent, UsageEvent
from .parser import parse_line

logger = get_logger(__name__)


class StreamHandler:
    """Converts a raw line stream into typed Events and aggregates a final Response."""

    async def process_stream(
        self, lines: AsyncIterator[str]
    ) -> AsyncIterator[Event]:
        """Yield typed Events from an async line iterator."""
        async for line in lines:
            if not line.strip():
                continue
            try:
                event = parse_line(line)
                if event is not None:
                    yield event
            except ParseError as exc:
                logger.warning("ParseError on line: %s | %s", exc.raw[:80], exc)

    async def collect_result(self, events: AsyncIterator[Event]) -> Response:
        """Consume all events and return the aggregated Response."""
        text_parts: list[str] = []
        session_id = ""
        duration_ms = 0
        is_error = False
        usage = Usage()

        async for event in events:
            if isinstance(event, TextChunkEvent):
                text_parts.append(event.text)
            elif isinstance(event, ResultEvent):
                # The result event contains the authoritative final text
                if event.result:
                    return Response(
                        result=event.result,
                        session_id=event.session_id,
                        duration_ms=event.duration_ms,
                        usage=usage,
                        is_error=event.is_error,
                    )
                session_id = event.session_id
                duration_ms = event.duration_ms
                is_error = event.is_error
            elif isinstance(event, UsageEvent):
                usage = Usage(
                    input_tokens=event.input_tokens,
                    output_tokens=event.output_tokens,
                    cache_read_tokens=event.cache_read_tokens,
                    cache_write_tokens=event.cache_write_tokens,
                )

        return Response(
            result="".join(text_parts),
            session_id=session_id,
            duration_ms=duration_ms,
            usage=usage,
            is_error=is_error,
        )
