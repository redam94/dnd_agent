"""
Chat memory tools.

These functions provide a lightweight in‑memory mechanism for
recording and retrieving chat messages for a D&D campaign.  They
store messages in a module‑level dictionary keyed by campaign ID so
that both players and the dungeon master (DM) can recall recent
conversation without requiring any external database.  Each message
record includes a timestamp, the session identifier, the role
(e.g. ``"player"`` or ``"dm"``) and the message content.

Use these tools through the orchestrator's ``dm_chat`` entry point
to build a simple DM chat bot, or call them directly to inspect
history.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic_ai import RunContext

try:
    from dnd_agent.models.agent_deps import CampaignDeps  # type: ignore
except Exception:
    # Fallback type when CampaignDeps is unavailable.  The generic
    # ``Any`` avoids type check errors but has no runtime impact.
    CampaignDeps = Any  # type: ignore


# Module‑level store for chat history.  This is a mapping from
# campaign ID to a list of message dictionaries.  Each entry
# contains ``timestamp``, ``session_id``, ``role`` and ``content``.
_CHAT_HISTORY: Dict[str, List[Dict[str, Any]]] = {}


async def save_chat_message(
    ctx: RunContext[CampaignDeps],
    campaign_id: str,
    session_id: str,
    role: str,
    content: str,
) -> str:
    """Persist a single chat message to the in‑memory store.

    Messages are grouped by campaign ID and appended to a list of
    previous messages.  The timestamp is recorded using the current
    UTC time.  This function never raises an exception; any errors
    will result in an error message being returned instead of
    propagating.

    :param ctx: run context with campaign dependencies (unused)
    :param campaign_id: identifier for the campaign
    :param session_id: identifier for the current session or scene
    :param role: the speaker's role (e.g. ``"player"``, ``"dm"``)
    :param content: the message text
    :returns: a human‑friendly confirmation string
    """
    try:
        hist = _CHAT_HISTORY.setdefault(campaign_id, [])
        hist.append(
            {
                "timestamp": datetime.utcnow(),
                "session_id": session_id,
                "role": role,
                "content": content,
            }
        )
        return "✅ Message saved to chat history."
    except Exception as exc:  # noqa: BLE001
        return f"❌ Error saving chat message: {exc}"


async def get_chat_history(
    ctx: RunContext[CampaignDeps],
    campaign_id: str,
    session_id: Optional[str] = None,
    limit: int = 20,
) -> str:
    """Retrieve the most recent chat messages for a campaign.

    If a ``session_id`` is provided, only messages matching that
    session are returned.  Messages are ordered from oldest to
    newest, and the result is truncated to the specified ``limit``.

    :param ctx: run context with campaign dependencies (unused)
    :param campaign_id: identifier for the campaign
    :param session_id: optionally filter by session ID
    :param limit: maximum number of messages to include
    :returns: a formatted string containing the chat history
    """
    try:
        hist = _CHAT_HISTORY.get(campaign_id, [])
        # Optionally filter by session ID
        if session_id:
            hist = [m for m in hist if m["session_id"] == session_id]
        # Only show the most recent ``limit`` messages
        messages = hist[-limit:]
        if not messages:
            return "No chat history found."
        lines: List[str] = []
        for msg in messages:
            ts = msg["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
            role = msg["role"].title()
            lines.append(f"{role} ({ts}): {msg['content']}")
        return "\n".join(lines)
    except Exception as exc:  # noqa: BLE001
        return f"❌ Error retrieving chat history: {exc}"


async def search_chat_history(
    ctx: RunContext[CampaignDeps],
    campaign_id: str,
    query: str,
    session_id: Optional[str] = None,
    limit: int = 20,
) -> str:
    """Search the chat history for messages containing a substring.

    This performs a case‑insensitive substring match against the
    message content.  If a ``session_id`` is provided, only
    messages from that session are searched.  Results are
    returned in the order they were recorded and truncated to
    ``limit`` items.

    :param ctx: run context with campaign dependencies (unused)
    :param campaign_id: identifier for the campaign
    :param query: substring to search for
    :param session_id: optionally filter by session ID
    :param limit: maximum number of matches to return
    :returns: a formatted string containing the matching messages
    """
    try:
        hist = _CHAT_HISTORY.get(campaign_id, [])
        if session_id:
            hist = [m for m in hist if m["session_id"] == session_id]
        query_lower = query.lower()
        matches = [m for m in hist if query_lower in m["content"].lower()]
        if not matches:
            return f"No chat messages found containing '{query}'."
        # Truncate to the requested limit
        matches = matches[:limit]
        lines: List[str] = []
        for msg in matches:
            ts = msg["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
            role = msg["role"].title()
            lines.append(f"{role} ({ts}): {msg['content']}")
        return "\n".join(lines)
    except Exception as exc:  # noqa: BLE001
        return f"❌ Error searching chat history: {exc}"