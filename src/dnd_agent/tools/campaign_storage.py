"""
Campaign storage tools.

These functions store and retrieve narrative information from a
PostgreSQL vector database.  They support saving rich campaign notes,
performing semantic search across stored content and recalling chat
history.  If the PostgreSQL connection is not configured on the
``RunContext.deps`` then operations return a warning message rather
than raising.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic_ai import RunContext

from dnd_agent.database.vector_db import PostgresVectorManager
from dnd_agent.models.agent_deps import CampaignDeps


async def save_campaign_info(
    ctx: RunContext[CampaignDeps],
    campaign_id: str,
    info_type: str,
    title: str,
    content: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Persist narrative information in the vector database."""
    if not getattr(ctx.deps, "postgres_conn", None):
        return "⚠️  PostgreSQL not configured. Campaign info storage disabled."
    try:
        pg_manager = PostgresVectorManager(
            host=ctx.deps.postgres_conn.host,
            port=ctx.deps.postgres_conn.port,
            database=ctx.deps.postgres_conn.database,
            user=ctx.deps.postgres_conn.user,
            password=ctx.deps.postgres_conn.password,
        )
        info_id = pg_manager.store_campaign_info(
            campaign_id=campaign_id,
            info_type=info_type,
            title=title,
            content=content,
            metadata=metadata,
        )
        pg_manager.close()
        return f"✅ Saved campaign info (ID: {info_id}) - Type: {info_type}, Title: '{title}'"
    except Exception as exc:  # noqa: BLE001
        print(f"Error saving campaign info: {exc}")
        return f"❌ Error saving campaign info: {exc}"


async def search_campaign_info(
    ctx: RunContext[CampaignDeps],
    campaign_id: str,
    query: str,
    info_type: Optional[str] = None,
    limit: int = 5,
) -> str:
    """Perform a semantic search over campaign info."""
    if not getattr(ctx.deps, "postgres_conn", None):
        return "⚠️  PostgreSQL not configured. Campaign info search disabled."
    try:
        pg_manager = PostgresVectorManager(
            host=ctx.deps.postgres_conn.host,
            port=ctx.deps.postgres_conn.port,
            database=ctx.deps.postgres_conn.database,
            user=ctx.deps.postgres_conn.user,
            password=ctx.deps.postgres_conn.password,
        )
        results = pg_manager.search_campaign_info(
            campaign_id=campaign_id,
            query=query,
            info_type=info_type,
            limit=limit,
        )
        pg_manager.close()
        
        if not results:
            return f"No campaign info found matching: '{query}'"
        output_lines = [f"**Campaign Info Search Results** (Query: '{query}')"]
        for i, result in enumerate(results, 1):
            output_lines.append(f"**{i}. {result['title']}** ({result['info_type']})")
            output_lines.append(result['content'])
            if result.get('metadata'):
                output_lines.append(f"_Metadata: {result['metadata']}_")
            output_lines.append("")
        return "\n".join(output_lines)
    except Exception as exc:  # noqa: BLE001
        print(f"Error searching campaign info: {exc}")
        return f"❌ Error searching campaign info: {exc}"


async def recall_chat_history(
    ctx: RunContext[CampaignDeps],
    campaign_id: str,
    session_id: Optional[str] = None,
    query: Optional[str] = None,
    limit: int = 20,
) -> str:
    """Retrieve or search conversation history."""
    if not getattr(ctx.deps, "postgres_conn", None):
        return "⚠️  PostgreSQL not configured. Chat history disabled."
    try:
        pg_manager = PostgresVectorManager(
            host=ctx.deps.postgres_conn.host,
            port=ctx.deps.postgres_conn.port,
            database=ctx.deps.postgres_conn.database,
            user=ctx.deps.postgres_conn.user,
            password=ctx.deps.postgres_conn.password,
        )
        if query:
            results = pg_manager.search_chat_history(
                campaign_id=campaign_id,
                query=query,
                session_id=session_id,
                limit=limit,
            )
        else:
            results = pg_manager.get_chat_history(
                campaign_id=campaign_id,
                session_id=session_id,
                limit=limit,
            )
        pg_manager.close()
        if not results:
            return "No chat history found."
        lines = [f"**Chat History** ({len(results)} messages)\n"]
        for msg in results:
            ts = msg['timestamp'].strftime("%Y-%m-%d %H:%M:%S")
            role = msg['role'].title()
            lines.append(f"{role} ({ts}):")
            lines.append(msg['content'])
            lines.append("")
        return "\n".join(lines)
    except Exception as exc:  # noqa: BLE001
        print(f"Error retrieving chat history: {exc}")
        return f"❌ Error retrieving chat history: {exc}"