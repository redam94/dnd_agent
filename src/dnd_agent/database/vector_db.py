"""PostgreSQL Vector Database Manager"""
from typing import Optional, Any, Dict, List
import os

# PostgreSQL with vector support
try:
    import psycopg2
    from psycopg2.extras import Json, RealDictCursor
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False
    print("⚠️  psycopg2 not installed. PostgreSQL features disabled.")
    print("   Install with: pip install psycopg2-binary")

# For embeddings
try:
    from openai import OpenAI
    OPENAI_EMBEDDINGS_AVAILABLE = True
except ImportError:
    OPENAI_EMBEDDINGS_AVAILABLE = False
    print("⚠️  OpenAI client not installed for embeddings.")
    print("   Install with: pip install openai")


class PostgresVectorManager:
    """Manages PostgreSQL database with pgvector for campaign info and chat history"""
    
    def __init__(self, host: str, port: int, database: str, user: str, password: str):
        if not POSTGRES_AVAILABLE:
            raise ImportError("psycopg2 not installed. Install with: pip install psycopg2-binary")
        
        self.conn = psycopg2.connect(
            host=host,
            port=port,
            database=database,
            user=user,
            password=password
        )
        self.conn.autocommit = False
        self._setup_database()
    
    def _setup_database(self):
        """Create tables and enable pgvector extension"""
        with self.conn.cursor() as cur:
            # Enable pgvector extension
            try:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                self.conn.commit()
            except Exception as e:
                self.conn.rollback()
                print(f"Note: pgvector extension may not be available: {e}")
                print("      Install with: https://github.com/pgvector/pgvector")
            
            # Create campaign_info table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS campaign_info (
                    id SERIAL PRIMARY KEY,
                    campaign_id VARCHAR(255) NOT NULL,
                    info_type VARCHAR(100) NOT NULL,
                    title VARCHAR(500) NOT NULL,
                    content TEXT NOT NULL,
                    metadata JSONB,
                    embedding vector(1536),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create chat_history table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS chat_history (
                    id SERIAL PRIMARY KEY,
                    campaign_id VARCHAR(255) NOT NULL,
                    session_id VARCHAR(255) NOT NULL,
                    role VARCHAR(50) NOT NULL,
                    content TEXT NOT NULL,
                    metadata JSONB,
                    embedding vector(1536),
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_campaign_info_campaign_id 
                ON campaign_info(campaign_id)
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_campaign_info_type 
                ON campaign_info(info_type)
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_chat_history_campaign_id 
                ON chat_history(campaign_id)
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_chat_history_session_id 
                ON chat_history(session_id)
            """)
            
            self.conn.commit()
    
    def close(self):
        """Close database connection"""
        self.conn.close()
    
    def get_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding for text using OpenAI"""
        if not OPENAI_EMBEDDINGS_AVAILABLE:
            return None
        
        try:
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Warning: Could not generate embedding: {e}")
            return None
    
    def store_campaign_info(
        self,
        campaign_id: str,
        info_type: str,
        title: str,
        content: str,
        metadata: Dict[str, Any] = None
    ) -> int:
        """Store campaign information with vector embedding"""
        embedding = self.get_embedding(f"{title}. {content}")
        
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO campaign_info 
                (campaign_id, info_type, title, content, metadata, embedding)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (
                campaign_id,
                info_type,
                title,
                content,
                Json(metadata or {}),
                embedding
            ))
            result = cur.fetchone()
            self.conn.commit()
            return result[0] if result else None
    
    def store_chat_message(
        self,
        campaign_id: str,
        session_id: str,
        role: str,
        content: str,
        metadata: Dict[str, Any] = None
    ) -> int:
        """Store chat message with vector embedding"""
        embedding = self.get_embedding(content)
        
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO chat_history 
                (campaign_id, session_id, role, content, metadata, embedding)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (
                campaign_id,
                session_id,
                role,
                content,
                Json(metadata or {}),
                embedding
            ))
            result = cur.fetchone()
            self.conn.commit()
            return result[0] if result else None
    
    def search_campaign_info(
        self,
        campaign_id: str,
        query: str,
        info_type: Optional[str] = None,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Search campaign info using vector similarity"""
        query_embedding = self.get_embedding(query)
        if not query_embedding:
            # Fallback to text search
            return self.search_campaign_info_text(campaign_id, query, info_type, limit)
        
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            type_filter = "AND info_type = %s" if info_type else ""
            params = [campaign_id, query_embedding, limit]
            if info_type:
                params.insert(2, info_type)
            
            cur.execute(f"""
                SELECT id, info_type, title, content, metadata, created_at,
                       embedding <=> %s::vector as distance
                FROM campaign_info
                WHERE campaign_id = %s
                {type_filter}
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """.replace('%s::vector', '%s').replace('<=> %s', '<=> %s::vector'), 
            [campaign_id] + ([info_type] if info_type else []) + [query_embedding, limit])
            
            return [dict(row) for row in cur.fetchall()]
    
    def search_campaign_info_text(
        self,
        campaign_id: str,
        query: str,
        info_type: Optional[str] = None,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Search campaign info using text search (fallback)"""
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            type_filter = "AND info_type = %s" if info_type else ""
            params = [campaign_id, f"%{query}%", f"%{query}%", limit]
            if info_type:
                params.insert(1, info_type)
            
            cur.execute(f"""
                SELECT id, info_type, title, content, metadata, created_at
                FROM campaign_info
                WHERE campaign_id = %s
                {type_filter}
                AND (title ILIKE %s OR content ILIKE %s)
                ORDER BY created_at DESC
                LIMIT %s
            """, params)
            
            return [dict(row) for row in cur.fetchall()]
    
    def get_chat_history(
        self,
        campaign_id: str,
        session_id: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Retrieve chat history"""
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            session_filter = "AND session_id = %s" if session_id else ""
            params = [campaign_id, limit]
            if session_id:
                params.insert(1, session_id)
            
            cur.execute(f"""
                SELECT id, session_id, role, content, metadata, timestamp
                FROM chat_history
                WHERE campaign_id = %s
                {session_filter}
                ORDER BY timestamp DESC
                LIMIT %s
            """, params)
            
            results = [dict(row) for row in cur.fetchall()]
            return list(reversed(results))  # Return in chronological order
    
    def search_chat_history(
        self,
        campaign_id: str,
        query: str,
        session_id: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Search chat history using vector similarity"""
        query_embedding = self.get_embedding(query)
        if not query_embedding:
            # Fallback to text search
            return self.search_chat_history_text(campaign_id, query, session_id, limit)
        
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            session_filter = "AND session_id = %s" if session_id else ""
            params = [campaign_id, query_embedding, limit]
            if session_id:
                params.insert(2, session_id)
            
            cur.execute(f"""
                SELECT id, session_id, role, content, metadata, timestamp,
                       embedding <=> %s::vector as distance
                FROM chat_history
                WHERE campaign_id = %s
                {session_filter}
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """.replace('%s::vector', '%s').replace('<=> %s', '<=> %s::vector'),
            [campaign_id] + ([session_id] if session_id else []) + [query_embedding, limit])
            
            return [dict(row) for row in cur.fetchall()]
    
    def search_chat_history_text(
        self,
        campaign_id: str,
        query: str,
        session_id: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Search chat history using text search (fallback)"""
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            session_filter = "AND session_id = %s" if session_id else ""
            params = [campaign_id, f"%{query}%", limit]
            if session_id:
                params.insert(1, session_id)
            
            cur.execute(f"""
                SELECT id, session_id, role, content, metadata, timestamp
                FROM chat_history
                WHERE campaign_id = %s
                {session_filter}
                AND content ILIKE %s
                ORDER BY timestamp DESC
                LIMIT %s
            """, params)
            
            return [dict(row) for row in cur.fetchall()]

