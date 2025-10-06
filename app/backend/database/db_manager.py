"""
Database Manager for D&D Campaign System
=========================================

Handles PostgreSQL connections and CRUD operations for campaigns,
sessions, and messages.
"""

import os
from datetime import datetime
from typing import List, Dict, Any, Optional
import psycopg2
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager


class DatabaseManager:
    """Manages PostgreSQL database connections and operations."""
    
    def __init__(self):
        """Initialize database connection parameters."""
        self.connection_params = {
            "host": os.environ.get("POSTGRES_HOST", "localhost"),
            "port": int(os.environ.get("POSTGRES_PORT", "5432")),
            "database": os.environ.get("POSTGRES_DB", "dnd_campaign"),
            "user": os.environ.get("POSTGRES_USER", "postgres"),
            "password": os.environ.get("POSTGRES_PASSWORD", "password")
        }
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections."""
        conn = psycopg2.connect(**self.connection_params)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def init_database(self):
        """Initialize database tables if they don't exist."""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                # Create campaigns table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS campaigns (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        name VARCHAR(255) NOT NULL,
                        description TEXT,
                        setting VARCHAR(255),
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create sessions table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS sessions (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        campaign_id UUID NOT NULL REFERENCES campaigns(id) ON DELETE CASCADE,
                        name VARCHAR(255),
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        ended_at TIMESTAMP WITH TIME ZONE,
                        active BOOLEAN DEFAULT TRUE
                    )
                """)
                
                # Create messages table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS messages (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        campaign_id UUID NOT NULL REFERENCES campaigns(id) ON DELETE CASCADE,
                        session_id UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                        role VARCHAR(50) NOT NULL CHECK (role IN ('player', 'dm', 'system')),
                        player_name VARCHAR(100),
                        content TEXT NOT NULL,
                        timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        metadata JSONB
                    )
                """)
                
                # Create indexes
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_sessions_campaign_id 
                    ON sessions(campaign_id)
                """)
                
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_messages_campaign_id 
                    ON messages(campaign_id)
                """)
                
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_messages_session_id 
                    ON messages(session_id)
                """)
                
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_messages_timestamp 
                    ON messages(timestamp)
                """)
                
                # Create characters table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS characters (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        campaign_id UUID NOT NULL REFERENCES campaigns(id) ON DELETE CASCADE,
                        player_name VARCHAR(100) NOT NULL,
                        character_data JSONB NOT NULL,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_characters_campaign_id 
                    ON characters(campaign_id)
                """)
                
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_characters_player_name 
                    ON characters(campaign_id, player_name)
                """)
                
                # Create campaign elements table (NPCs, quests, locations, monsters, narrative)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS campaign_elements (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        campaign_id UUID NOT NULL REFERENCES campaigns(id) ON DELETE CASCADE,
                        element_type VARCHAR(50) NOT NULL,
                        element_data JSONB NOT NULL,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_campaign_elements_campaign_id 
                    ON campaign_elements(campaign_id)
                """)
                
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_campaign_elements_type 
                    ON campaign_elements(campaign_id, element_type)
                """)
    
    # ========================================================================
    # Campaign Operations
    # ========================================================================
    
    def create_campaign(
        self,
        name: str,
        description: Optional[str] = None,
        setting: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a new campaign."""
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    INSERT INTO campaigns (name, description, setting)
                    VALUES (%s, %s, %s)
                    RETURNING id, name, description, setting, created_at, updated_at
                """, (name, description, setting))
                
                result = cur.fetchone()
                return dict(result)
    
    def get_campaign(self, campaign_id: str) -> Optional[Dict[str, Any]]:
        """Get a campaign by ID."""
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT id, name, description, setting, created_at, updated_at
                    FROM campaigns
                    WHERE id = %s
                """, (campaign_id,))
                
                result = cur.fetchone()
                return dict(result) if result else None
    
    def list_campaigns(self) -> List[Dict[str, Any]]:
        """List all campaigns."""
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT id, name, description, setting, created_at, updated_at
                    FROM campaigns
                    ORDER BY created_at DESC
                """)
                
                results = cur.fetchall()
                return [dict(row) for row in results]
    
    def update_campaign(
        self,
        campaign_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        setting: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Update a campaign."""
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Build dynamic update query
                updates = []
                params = []
                
                if name is not None:
                    updates.append("name = %s")
                    params.append(name)
                if description is not None:
                    updates.append("description = %s")
                    params.append(description)
                if setting is not None:
                    updates.append("setting = %s")
                    params.append(setting)
                
                if not updates:
                    return self.get_campaign(campaign_id)
                
                updates.append("updated_at = CURRENT_TIMESTAMP")
                params.append(campaign_id)
                
                query = f"""
                    UPDATE campaigns
                    SET {', '.join(updates)}
                    WHERE id = %s
                    RETURNING id, name, description, setting, created_at, updated_at
                """
                
                cur.execute(query, params)
                result = cur.fetchone()
                return dict(result) if result else None
    
    def delete_campaign(self, campaign_id: str) -> bool:
        """Delete a campaign and all related data."""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM campaigns WHERE id = %s", (campaign_id,))
                return cur.rowcount > 0
    
    # ========================================================================
    # Session Operations
    # ========================================================================
    
    def create_session(
        self,
        campaign_id: str,
        name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a new session."""
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    INSERT INTO sessions (campaign_id, name)
                    VALUES (%s, %s)
                    RETURNING id, campaign_id, name, created_at, ended_at, active
                """, (campaign_id, name))
                
                result = cur.fetchone()
                return dict(result)
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get a session by ID."""
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT id, campaign_id, name, created_at, ended_at, active
                    FROM sessions
                    WHERE id = %s
                """, (session_id,))
                
                result = cur.fetchone()
                return dict(result) if result else None
    
    def list_sessions(self, campaign_id: str) -> List[Dict[str, Any]]:
        """List all sessions for a campaign."""
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT id, campaign_id, name, created_at, ended_at, active
                    FROM sessions
                    WHERE campaign_id = %s
                    ORDER BY created_at DESC
                """, (campaign_id,))
                
                results = cur.fetchall()
                return [dict(row) for row in results]
    
    def end_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Mark a session as ended."""
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    UPDATE sessions
                    SET active = FALSE, ended_at = CURRENT_TIMESTAMP
                    WHERE id = %s
                    RETURNING id, campaign_id, name, created_at, ended_at, active
                """, (session_id,))
                
                result = cur.fetchone()
                return dict(result) if result else None
    
    # ========================================================================
    # Message Operations
    # ========================================================================
    
    def save_message(
        self,
        campaign_id: str,
        session_id: str,
        role: str,
        content: str,
        player_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Save a message."""
        import json
        
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    INSERT INTO messages (campaign_id, session_id, role, player_name, content, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    RETURNING id, campaign_id, session_id, role, player_name, content, timestamp, metadata
                """, (campaign_id, session_id, role, player_name, content, json.dumps(metadata) if metadata else None))
                
                result = cur.fetchone()
                return dict(result)
    
    def get_messages(
        self,
        campaign_id: str,
        session_id: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get messages for a campaign or session."""
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                if session_id:
                    cur.execute("""
                        SELECT id, campaign_id, session_id, role, player_name, content, timestamp, metadata
                        FROM messages
                        WHERE campaign_id = %s AND session_id = %s
                        ORDER BY timestamp ASC
                        LIMIT %s
                    """, (campaign_id, session_id, limit))
                else:
                    cur.execute("""
                        SELECT id, campaign_id, session_id, role, player_name, content, timestamp, metadata
                        FROM messages
                        WHERE campaign_id = %s
                        ORDER BY timestamp ASC
                        LIMIT %s
                    """, (campaign_id, limit))
                
                results = cur.fetchall()
                return [dict(row) for row in results]
    
    def delete_message(self, message_id: str) -> bool:
        """Delete a message."""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM messages WHERE id = %s", (message_id,))
                return cur.rowcount > 0
    
    # ========================================================================
    # Character Operations
    # ========================================================================
    
    def save_character(
        self,
        campaign_id: str,
        player_name: str,
        character_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Save or update a character."""
        import json
        
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Check if character exists
                cur.execute("""
                    SELECT id FROM characters
                    WHERE campaign_id = %s AND player_name = %s
                """, (campaign_id, player_name))
                
                existing = cur.fetchone()
                
                if existing:
                    # Update existing character
                    cur.execute("""
                        UPDATE characters
                        SET character_data = %s, updated_at = CURRENT_TIMESTAMP
                        WHERE campaign_id = %s AND player_name = %s
                        RETURNING id, campaign_id, player_name, character_data, created_at, updated_at
                    """, (json.dumps(character_data), campaign_id, player_name))
                else:
                    # Insert new character
                    cur.execute("""
                        INSERT INTO characters (campaign_id, player_name, character_data)
                        VALUES (%s, %s, %s)
                        RETURNING id, campaign_id, player_name, character_data, created_at, updated_at
                    """, (campaign_id, player_name, json.dumps(character_data)))
                
                result = cur.fetchone()
                return dict(result)
    
    def get_character(
        self,
        campaign_id: str,
        player_name: str
    ) -> Optional[Dict[str, Any]]:
        """Get a character by campaign and player name."""
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT id, campaign_id, player_name, character_data, created_at, updated_at
                    FROM characters
                    WHERE campaign_id = %s AND player_name = %s
                """, (campaign_id, player_name))
                
                result = cur.fetchone()
                return dict(result) if result else None
    
    def list_characters(self, campaign_id: str) -> List[Dict[str, Any]]:
        """List all characters in a campaign."""
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT id, campaign_id, player_name, character_data, created_at, updated_at
                    FROM characters
                    WHERE campaign_id = %s
                    ORDER BY updated_at DESC
                """, (campaign_id,))
                
                results = cur.fetchall()
                return [dict(row) for row in results]
    
    def delete_character(self, character_id: str) -> bool:
        """Delete a character."""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM characters WHERE id = %s", (character_id,))
                return cur.rowcount > 0
    
    # ========================================================================
    # Campaign Elements Operations (NPCs, Locations, Quests, Monsters)
    # ========================================================================
    
    def save_campaign_element(
        self,
        campaign_id: str,
        element_type: str,
        element_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Save a campaign element (NPC, location, quest, monster, narrative point)."""
        import json
        
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    INSERT INTO campaign_elements (campaign_id, element_type, element_data)
                    VALUES (%s, %s, %s)
                    RETURNING id, campaign_id, element_type, element_data, created_at, updated_at
                """, (campaign_id, element_type, json.dumps(element_data)))
                
                result = cur.fetchone()
                return dict(result)
    
    def get_campaign_element(self, element_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific campaign element."""
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT id, campaign_id, element_type, element_data, created_at, updated_at
                    FROM campaign_elements
                    WHERE id = %s
                """, (element_id,))
                
                result = cur.fetchone()
                return dict(result) if result else None
    
    def list_campaign_elements(
        self,
        campaign_id: str,
        element_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List campaign elements, optionally filtered by type."""
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                if element_type:
                    cur.execute("""
                        SELECT id, campaign_id, element_type, element_data, created_at, updated_at
                        FROM campaign_elements
                        WHERE campaign_id = %s AND element_type = %s
                        ORDER BY created_at DESC
                    """, (campaign_id, element_type))
                else:
                    cur.execute("""
                        SELECT id, campaign_id, element_type, element_data, created_at, updated_at
                        FROM campaign_elements
                        WHERE campaign_id = %s
                        ORDER BY element_type, created_at DESC
                    """, (campaign_id,))
                
                results = cur.fetchall()
                return [dict(row) for row in results]
    
    def update_campaign_element(
        self,
        element_id: str,
        element_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Update a campaign element."""
        import json
        
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    UPDATE campaign_elements
                    SET element_data = %s, updated_at = CURRENT_TIMESTAMP
                    WHERE id = %s
                    RETURNING id, campaign_id, element_type, element_data, created_at, updated_at
                """, (json.dumps(element_data), element_id))
                
                result = cur.fetchone()
                return dict(result) if result else None
    
    def delete_campaign_element(self, element_id: str) -> bool:
        """Delete a campaign element."""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM campaign_elements WHERE id = %s", (element_id,))
                return cur.rowcount > 0
    
    # ========================================================================
    # Health Check
    # ========================================================================
    
    def health_check(self) -> bool:
        """Check if database connection is healthy."""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    return True
        except Exception:
            return False