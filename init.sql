-- PostgreSQL initialization script for D&D Campaign Database
-- This script sets up the basic schema for campaigns, sessions, and vector embeddings

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "pgcrypto";  -- For gen_random_uuid()

-- Campaigns table
CREATE TABLE IF NOT EXISTS campaigns (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    setting VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_campaigns_created_at ON campaigns(created_at);

-- Sessions table
CREATE TABLE IF NOT EXISTS sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    campaign_id UUID NOT NULL REFERENCES campaigns(id) ON DELETE CASCADE,
    name VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    ended_at TIMESTAMP WITH TIME ZONE,
    active BOOLEAN DEFAULT TRUE
);

CREATE INDEX IF NOT EXISTS idx_sessions_campaign_id ON sessions(campaign_id);
CREATE INDEX IF NOT EXISTS idx_sessions_active ON sessions(active);

-- Messages/Chat history table
CREATE TABLE IF NOT EXISTS messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    campaign_id UUID NOT NULL REFERENCES campaigns(id) ON DELETE CASCADE,
    session_id UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role VARCHAR(50) NOT NULL CHECK (role IN ('player', 'dm', 'system')),
    player_name VARCHAR(100),
    content TEXT NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB
);

CREATE INDEX IF NOT EXISTS idx_messages_campaign_id ON messages(campaign_id);
CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages(session_id);
CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp);

-- Campaign lore/notes with vector embeddings for semantic search
CREATE TABLE IF NOT EXISTS campaign_lore (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    campaign_id UUID NOT NULL REFERENCES campaigns(id) ON DELETE CASCADE,
    title VARCHAR(255) NOT NULL,
    content TEXT NOT NULL,
    category VARCHAR(100),  -- e.g., 'location', 'npc', 'quest', 'lore', 'item'
    tags TEXT[],
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    embedding vector(1536)  -- OpenAI ada-002 embedding dimension
);

CREATE INDEX idx_lore_campaign_id ON campaign_lore(campaign_id);
CREATE INDEX idx_lore_category ON campaign_lore(category);
CREATE INDEX idx_lore_tags ON campaign_lore USING GIN(tags);
-- For vector similarity search (requires pgvector extension)
-- CREATE INDEX idx_lore_embedding ON campaign_lore USING ivfflat (embedding vector_cosine_ops);

-- Game state snapshots
CREATE TABLE IF NOT EXISTS game_states (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    campaign_id UUID NOT NULL REFERENCES campaigns(id) ON DELETE CASCADE,
    session_id UUID REFERENCES sessions(id) ON DELETE CASCADE,
    state_type VARCHAR(50) NOT NULL,  -- e.g., 'combat', 'exploration', 'social'
    state_data JSONB NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_game_states_campaign_id ON game_states(campaign_id);
CREATE INDEX idx_game_states_session_id ON game_states(session_id);
CREATE INDEX idx_game_states_timestamp ON game_states(timestamp);

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggers for updated_at
CREATE TRIGGER update_campaigns_updated_at
    BEFORE UPDATE ON campaigns
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_campaign_lore_updated_at
    BEFORE UPDATE ON campaign_lore
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Views for convenient querying

-- Active sessions view
CREATE OR REPLACE VIEW active_sessions AS
SELECT 
    s.id,
    s.campaign_id,
    s.name,
    s.created_at,
    c.name as campaign_name,
    COUNT(m.id) as message_count,
    MAX(m.timestamp) as last_message_at
FROM sessions s
JOIN campaigns c ON s.campaign_id = c.id
LEFT JOIN messages m ON s.id = m.session_id
WHERE s.active = TRUE
GROUP BY s.id, s.campaign_id, s.name, s.created_at, c.name;

-- Recent campaign activity view
CREATE OR REPLACE VIEW recent_campaign_activity AS
SELECT 
    c.id as campaign_id,
    c.name as campaign_name,
    COUNT(DISTINCT s.id) as session_count,
    COUNT(m.id) as total_messages,
    MAX(m.timestamp) as last_activity
FROM campaigns c
LEFT JOIN sessions s ON c.id = s.campaign_id
LEFT JOIN messages m ON s.id = m.session_id
GROUP BY c.id, c.name
ORDER BY last_activity DESC;

-- Insert sample data (optional - comment out if not needed)
INSERT INTO campaigns (name, description, setting) VALUES
    ('Tutorial Campaign', 'A beginner-friendly adventure', 'Generic Fantasy');

-- Grant permissions (adjust as needed)
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO your_api_user;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO your_api_user;

-- Success message
DO $$
BEGIN
    RAISE NOTICE 'D&D Campaign Database initialized successfully!';
    RAISE NOTICE 'Tables created: campaigns, sessions, messages, campaign_lore, game_states';
    RAISE NOTICE 'Views created: active_sessions, recent_campaign_activity';
END $$;