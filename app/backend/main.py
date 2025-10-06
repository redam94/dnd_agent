"""
FastAPI Backend for D&D Campaign Chatbot
=========================================

This backend provides a REST API for a D&D campaign chatbot powered by
the multi-agent system. The DM orchestrator delegates to specialized
agents for rules, combat, narrative, entities, spatial, and memory.

Features:
- Campaign management (create, list, get)
- Session management with chat history
- Message streaming and regular responses
- Game state retrieval
- Health checks for dependencies

Environment Variables Required:
- OPENAI_API_KEY: OpenAI API key for LLM
- NEO4J_URI: Neo4j database URI
- NEO4J_USER: Neo4j username
- NEO4J_PASSWORD: Neo4j password
- POSTGRES_HOST: PostgreSQL host (optional)
- POSTGRES_PORT: PostgreSQL port (optional)
- POSTGRES_DB: PostgreSQL database name (optional)
- POSTGRES_USER: PostgreSQL username (optional)
- POSTGRES_PASSWORD: PostgreSQL password (optional)
"""

import os
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from dnd_agent import create_campaign_system, AgentType
from dnd_agent.agents.base import AgentRequest, AgentResponse
from dnd_agent.models.agent_deps import CampaignDeps


# ============================================================================
# Pydantic Models for API
# ============================================================================

class CampaignCreate(BaseModel):
    """Request to create a new campaign."""
    name: str = Field(..., description="Campaign name")
    description: Optional[str] = Field(None, description="Campaign description")
    setting: Optional[str] = Field(None, description="Campaign setting/world")


class Campaign(BaseModel):
    """Campaign information."""
    id: str
    name: str
    description: Optional[str]
    setting: Optional[str]
    created_at: datetime
    updated_at: datetime


class SessionCreate(BaseModel):
    """Request to create a new session."""
    campaign_id: str
    name: Optional[str] = Field(None, description="Session name/title")


class Session(BaseModel):
    """Session information."""
    id: str
    campaign_id: str
    name: Optional[str]
    created_at: datetime
    active: bool


class MessageRequest(BaseModel):
    """Request to send a message to the DM."""
    campaign_id: str
    session_id: str
    message: str
    player_name: Optional[str] = Field(None, description="Name of the player sending message")


class MessageResponse(BaseModel):
    """Response from the DM."""
    id: str
    campaign_id: str
    session_id: str
    role: str  # "player" or "dm"
    content: str
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None


class GameStateRequest(BaseModel):
    """Request to get current game state."""
    campaign_id: str
    session_id: Optional[str] = None


class RuleLookupRequest(BaseModel):
    """Request to lookup a rule, spell, monster, etc."""
    query: str = Field(..., description="Search query")
    resource_type: str = Field(..., description="Type of resource (spell, monster, equipment, etc.)")


class DiceRollRequest(BaseModel):
    """Request to roll dice."""
    campaign_id: str
    session_id: str
    dice_notation: str = Field(..., description="Dice notation (e.g., '2d6+3', '1d20')")


class CombatActionRequest(BaseModel):
    """Request to perform a combat action."""
    campaign_id: str
    session_id: str
    action: str = Field(..., description="Combat action type (attack, cast_spell, etc.)")
    parameters: Dict[str, Any] = Field(..., description="Action parameters")


class CharacterSheet(BaseModel):
    """Character sheet data."""
    campaign_id: str
    player_name: str
    character_data: Dict[str, Any]


class CharacterResponse(BaseModel):
    """Character response."""
    id: str
    campaign_id: str
    player_name: str
    character_data: Dict[str, Any]
    created_at: datetime
    updated_at: datetime


class CampaignElement(BaseModel):
    """Campaign element (NPC, quest, location, monster, narrative)."""
    campaign_id: str
    element_type: str = Field(..., description="Type: npc, quest, location, monster, narrative")
    element_data: Dict[str, Any]


class CampaignElementResponse(BaseModel):
    """Campaign element response."""
    id: str
    campaign_id: str
    element_type: str
    element_data: Dict[str, Any]
    created_at: datetime
    updated_at: datetime


class AIGenerateRequest(BaseModel):
    """Request to generate content with AI."""
    campaign_id: str
    element_type: str
    prompt: str
    context: Optional[Dict[str, Any]] = None


class HealthStatus(BaseModel):
    """System health status."""
    status: str
    neo4j_connected: bool
    postgres_connected: bool
    agents_initialized: bool


# ============================================================================
# Database Manager Import
# ============================================================================

# Import the database manager (create this file as shown in previous artifact)
try:
    from database.db_manager import DatabaseManager
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False
    print("⚠️  DatabaseManager not found. Using in-memory storage (data won't persist).")


# ============================================================================
# In-Memory Storage Fallback
# ============================================================================

campaigns_db: Dict[str, Campaign] = {}
sessions_db: Dict[str, Session] = {}
messages_db: Dict[str, List[MessageResponse]] = {}


# ============================================================================
# Dependency Setup
# ============================================================================

class AppState:
    """Application state container."""
    def __init__(self):
        self.orchestrator = None
        self.deps = None
        self.db_manager = None
        self.initialized = False


app_state = AppState()


def setup_dependencies() -> CampaignDeps:
    """Initialize campaign dependencies."""
    from dnd_agent.database.neo4j_manager import Neo4jSpatialManager
    
    # Initialize Neo4j
    neo4j_manager = Neo4jSpatialManager(
        uri=os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
        user=os.environ.get("NEO4J_USER", "neo4j"),
        password=os.environ.get("NEO4J_PASSWORD", "password")
    )
    
    # Initialize PostgreSQL connection params (optional)
    postgres_conn = None
    if all([
        os.environ.get("POSTGRES_HOST"),
        os.environ.get("POSTGRES_DB"),
        os.environ.get("POSTGRES_USER"),
        os.environ.get("POSTGRES_PASSWORD")
    ]):
        class PostgresParams:
            def __init__(self, host: str, port: str, database: str, user: str, password: str):
                self.host = host
                self.port = port
                self.database = database
                self.user = user
                self.password = password
        
        postgres_conn = PostgresParams(
            host=os.environ["POSTGRES_HOST"],
            port=os.environ.get("POSTGRES_PORT", "5432"),
            database=os.environ["POSTGRES_DB"],
            user=os.environ["POSTGRES_USER"],
            password=os.environ["POSTGRES_PASSWORD"]
        )
    
    deps = CampaignDeps(
        neo4j_driver=neo4j_manager,
        postgres_conn=postgres_conn,
        dnd_api_base="https://www.dnd5eapi.co"
    )
    
    return deps


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the campaign system on startup."""
    try:
        # Initialize database manager
        if DB_AVAILABLE:
            app_state.db_manager = DatabaseManager()
            app_state.db_manager.init_database()
            print("✅ Database initialized successfully")
        else:
            print("⚠️  Using in-memory storage - data will be lost on restart")
        
        # Setup dependencies
        app_state.deps = setup_dependencies()
        
        # Initialize the orchestrator with all agents
        model_config = {
            "model_name": os.environ.get("OPENAI_MODEL", "gpt-4o")
        }
        app_state.orchestrator = create_campaign_system(app_state.deps, model_config)
        app_state.initialized = True
        
        print("✅ Campaign system initialized successfully")
        
    except Exception as e:
        print(f"❌ Failed to initialize campaign system: {e}")
        app_state.initialized = False
    
    yield
    
    # Cleanup on shutdown
    if app_state.deps and hasattr(app_state.deps.neo4j_driver, 'close'):
        app_state.deps.neo4j_driver.close()
    print("🛑 Campaign system shutdown")


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="D&D Campaign Chatbot API",
    description="Multi-agent D&D campaign management system with DM chatbot",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_orchestrator():
    """Dependency to get the initialized orchestrator."""
    if not app_state.initialized or app_state.orchestrator is None:
        raise HTTPException(status_code=503, detail="Campaign system not initialized")
    return app_state.orchestrator


# ============================================================================
# Health & Status Endpoints
# ============================================================================

@app.get("/health", response_model=HealthStatus)
async def health_check():
    """Check system health and dependencies."""
    neo4j_ok = False
    postgres_ok = False
    
    if app_state.deps:
        # Check Neo4j
        try:
            if hasattr(app_state.deps.neo4j_driver, 'check_database_status'):
                neo4j_ok = True
        except:
            pass
        
        # Check Postgres with DatabaseManager
        if DB_AVAILABLE and app_state.db_manager:
            postgres_ok = app_state.db_manager.health_check()
        else:
            postgres_ok = app_state.deps.postgres_conn is not None
    
    return HealthStatus(
        status="healthy" if app_state.initialized else "unhealthy",
        neo4j_connected=neo4j_ok,
        postgres_connected=postgres_ok,
        agents_initialized=app_state.initialized
    )


# ============================================================================
# Campaign Management Endpoints
# ============================================================================

@app.post("/campaigns", response_model=Campaign)
async def create_campaign(
    campaign: CampaignCreate,
    orchestrator=Depends(get_orchestrator)
):
    """Create a new D&D campaign."""
    try:
        if DB_AVAILABLE and app_state.db_manager:
            # Use database
            db_campaign = app_state.db_manager.create_campaign(
                name=campaign.name,
                description=campaign.description,
                setting=campaign.setting
            )
            
            # Convert to Campaign model
            new_campaign = Campaign(
                id=str(db_campaign['id']),
                name=db_campaign['name'],
                description=db_campaign['description'],
                setting=db_campaign['setting'],
                created_at=db_campaign['created_at'],
                updated_at=db_campaign['updated_at']
            )
        else:
            # Fallback to in-memory
            campaign_id = str(uuid.uuid4())
            now = datetime.utcnow()
            
            new_campaign = Campaign(
                id=campaign_id,
                name=campaign.name,
                description=campaign.description,
                setting=campaign.setting,
                created_at=now,
                updated_at=now
            )
            
            campaigns_db[campaign_id] = new_campaign
            messages_db[campaign_id] = []
        
        # Initialize campaign in the system via entity agent
        request = AgentRequest(
            agent_type=AgentType.ORCHESTRATOR,
            action="initialize_campaign",
            parameters={
                "campaign_id": new_campaign.id,
                "name": campaign.name,
                "description": campaign.description,
                "setting": campaign.setting
            }
        )
        
        await orchestrator.process(request)
        
        return new_campaign
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create campaign: {str(e)}")


@app.get("/campaigns", response_model=List[Campaign])
async def list_campaigns():
    """List all campaigns."""
    try:
        if DB_AVAILABLE and app_state.db_manager:
            # Use database
            db_campaigns = app_state.db_manager.list_campaigns()
            return [
                Campaign(
                    id=str(c['id']),
                    name=c['name'],
                    description=c['description'],
                    setting=c['setting'],
                    created_at=c['created_at'],
                    updated_at=c['updated_at']
                )
                for c in db_campaigns
            ]
        else:
            # Fallback to in-memory
            return list(campaigns_db.values())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list campaigns: {str(e)}")


@app.get("/campaigns/{campaign_id}", response_model=Campaign)
async def get_campaign(campaign_id: str):
    """Get a specific campaign."""
    try:
        if DB_AVAILABLE and app_state.db_manager:
            # Use database
            db_campaign = app_state.db_manager.get_campaign(campaign_id)
            if not db_campaign:
                raise HTTPException(status_code=404, detail="Campaign not found")
            
            return Campaign(
                id=str(db_campaign['id']),
                name=db_campaign['name'],
                description=db_campaign['description'],
                setting=db_campaign['setting'],
                created_at=db_campaign['created_at'],
                updated_at=db_campaign['updated_at']
            )
        else:
            # Fallback to in-memory
            if campaign_id not in campaigns_db:
                raise HTTPException(status_code=404, detail="Campaign not found")
            return campaigns_db[campaign_id]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get campaign: {str(e)}")


# ============================================================================
# Session Management Endpoints
# ============================================================================

@app.post("/sessions", response_model=Session)
async def create_session(session: SessionCreate):
    """Create a new game session within a campaign."""
    try:
        if DB_AVAILABLE and app_state.db_manager:
            # Check if campaign exists
            campaign = app_state.db_manager.get_campaign(session.campaign_id)
            if not campaign:
                raise HTTPException(status_code=404, detail="Campaign not found")
            
            # Use database
            db_session = app_state.db_manager.create_session(
                campaign_id=session.campaign_id,
                name=session.name
            )
            
            return Session(
                id=str(db_session['id']),
                campaign_id=str(db_session['campaign_id']),
                name=db_session['name'],
                created_at=db_session['created_at'],
                active=db_session['active']
            )
        else:
            # Fallback to in-memory
            if session.campaign_id not in campaigns_db:
                raise HTTPException(status_code=404, detail="Campaign not found")
            
            session_id = str(uuid.uuid4())
            
            new_session = Session(
                id=session_id,
                campaign_id=session.campaign_id,
                name=session.name,
                created_at=datetime.utcnow(),
                active=True
            )
            
            sessions_db[session_id] = new_session
            
            return new_session
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create session: {str(e)}")


@app.get("/campaigns/{campaign_id}/sessions", response_model=List[Session])
async def list_sessions(campaign_id: str):
    """List all sessions for a campaign."""
    try:
        if DB_AVAILABLE and app_state.db_manager:
            # Check if campaign exists
            campaign = app_state.db_manager.get_campaign(campaign_id)
            if not campaign:
                raise HTTPException(status_code=404, detail="Campaign not found")
            
            # Use database
            db_sessions = app_state.db_manager.list_sessions(campaign_id)
            return [
                Session(
                    id=str(s['id']),
                    campaign_id=str(s['campaign_id']),
                    name=s['name'],
                    created_at=s['created_at'],
                    active=s['active']
                )
                for s in db_sessions
            ]
        else:
            # Fallback to in-memory
            if campaign_id not in campaigns_db:
                raise HTTPException(status_code=404, detail="Campaign not found")
            
            return [s for s in sessions_db.values() if s.campaign_id == campaign_id]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list sessions: {str(e)}")


# ============================================================================
# Chat/Message Endpoints
# ============================================================================

@app.post("/messages", response_model=MessageResponse)
async def send_message(
    msg: MessageRequest,
    orchestrator=Depends(get_orchestrator)
):
    """Send a message to the DM and get a response."""
    try:
        if DB_AVAILABLE and app_state.db_manager:
            # Check if campaign and session exist
            campaign = app_state.db_manager.get_campaign(msg.campaign_id)
            if not campaign:
                raise HTTPException(status_code=404, detail="Campaign not found")
            
            session = app_state.db_manager.get_session(msg.session_id)
            if not session:
                raise HTTPException(status_code=404, detail="Session not found")
            
            # Save player message
            player_msg_data = app_state.db_manager.save_message(
                campaign_id=msg.campaign_id,
                session_id=msg.session_id,
                role="player",
                content=msg.message,
                player_name=msg.player_name
            )
            
            # Get recent chat history for context
            recent_messages = app_state.db_manager.get_messages(msg.campaign_id, msg.session_id, limit=10)
            context = "\n".join([
                f"{m['role'].upper()}: {m['content']}" 
                for m in recent_messages
            ])
        else:
            # Fallback to in-memory
            if msg.campaign_id not in campaigns_db:
                raise HTTPException(status_code=404, detail="Campaign not found")
            
            if msg.session_id not in sessions_db:
                raise HTTPException(status_code=404, detail="Session not found")
            
            # Save player message
            player_msg = MessageResponse(
                id=str(uuid.uuid4()),
                campaign_id=msg.campaign_id,
                session_id=msg.session_id,
                role="player",
                content=msg.message,
                timestamp=datetime.utcnow()
            )
            
            messages_db.setdefault(msg.campaign_id, []).append(player_msg)
            
            # Get recent chat history for context
            recent_messages = messages_db.get(msg.campaign_id, [])[-10:]
            context = "\n".join([
                f"{m.role.upper()}: {m.content}" 
                for m in recent_messages
            ])
        
        # Process message through orchestrator
        request = AgentRequest(
            agent_type=AgentType.ORCHESTRATOR,
            action="process_player_message",
            parameters={
                "campaign_id": msg.campaign_id,
                "session_id": msg.session_id,
                "player_name": msg.player_name or "Player",
                "message": msg.message,
                "context": context
            }
        )
        
        response = await orchestrator.process(request)
        
        # Save DM response
        if DB_AVAILABLE and app_state.db_manager:
            dm_msg_data = app_state.db_manager.save_message(
                campaign_id=msg.campaign_id,
                session_id=msg.session_id,
                role="dm",
                content=response.message,
                metadata=response.metadata
            )
            
            dm_msg = MessageResponse(
                id=str(dm_msg_data['id']),
                campaign_id=msg.campaign_id,
                session_id=msg.session_id,
                role="dm",
                content=response.message,
                timestamp=dm_msg_data['timestamp'],
                metadata=response.metadata
            )
        else:
            # Fallback to in-memory
            dm_msg = MessageResponse(
                id=str(uuid.uuid4()),
                campaign_id=msg.campaign_id,
                session_id=msg.session_id,
                role="dm",
                content=response.message,
                timestamp=datetime.utcnow(),
                metadata=response.metadata
            )
            
            messages_db[msg.campaign_id].append(dm_msg)
        
        return dm_msg
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to send message: {str(e)}")


@app.get("/campaigns/{campaign_id}/messages", response_model=List[MessageResponse])
async def get_messages(
    campaign_id: str,
    session_id: Optional[str] = None,
    limit: int = 50
):
    """Get chat history for a campaign or session."""
    try:
        if DB_AVAILABLE and app_state.db_manager:
            # Check if campaign exists
            campaign = app_state.db_manager.get_campaign(campaign_id)
            if not campaign:
                raise HTTPException(status_code=404, detail="Campaign not found")
            
            # Get messages from database
            db_messages = app_state.db_manager.get_messages(campaign_id, session_id, limit)
            
            return [
                MessageResponse(
                    id=str(m['id']),
                    campaign_id=str(m['campaign_id']),
                    session_id=str(m['session_id']),
                    role=m['role'],
                    content=m['content'],
                    timestamp=m['timestamp'],
                    metadata=m.get('metadata')
                )
                for m in db_messages
            ]
        else:
            # Fallback to in-memory
            if campaign_id not in campaigns_db:
                raise HTTPException(status_code=404, detail="Campaign not found")
            
            messages = messages_db.get(campaign_id, [])
            
            if session_id:
                messages = [m for m in messages if m.session_id == session_id]
            
            return messages[-limit:]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get messages: {str(e)}")


# ============================================================================
# Game State Endpoints
# ============================================================================

@app.post("/game-state")
async def get_game_state(
    request: GameStateRequest,
    orchestrator=Depends(get_orchestrator)
):
    """Get current game state including entities, locations, and combat status."""
    if request.campaign_id not in campaigns_db:
        raise HTTPException(status_code=404, detail="Campaign not found")
    
    agent_request = AgentRequest(
        agent_type=AgentType.ORCHESTRATOR,
        action="get_game_state",
        parameters={
            "campaign_id": request.campaign_id,
            "session_id": request.session_id
        }
    )
    
    response = await orchestrator.process(agent_request)
    
    return {
        "campaign_id": request.campaign_id,
        "session_id": request.session_id,
        "state": response.data,
        "message": response.message,
        "success": response.success
    }


# ============================================================================
# Special Action Endpoints
# ============================================================================

@app.post("/actions/roll-dice")
async def roll_dice(
    roll_request: DiceRollRequest,
    orchestrator=Depends(get_orchestrator)
):
    """Roll dice using standard notation (e.g., '2d6+3', '1d20')."""
    request = AgentRequest(
        agent_type=AgentType.ORCHESTRATOR,
        action="roll_dice",
        parameters={
            "campaign_id": roll_request.campaign_id,
            "session_id": roll_request.session_id,
            "dice_notation": roll_request.dice_notation
        }
    )
    
    response = await orchestrator.process(request)
    
    return {
        "dice_notation": roll_request.dice_notation,
        "result": response.message,
        "success": response.success
    }


@app.post("/actions/combat")
async def handle_combat_action(
    combat_request: CombatActionRequest,
    orchestrator=Depends(get_orchestrator)
):
    """Handle combat actions (attack, cast spell, etc.)."""
    request = AgentRequest(
        agent_type=AgentType.ORCHESTRATOR,
        action=f"combat_{combat_request.action}",
        parameters={
            "campaign_id": combat_request.campaign_id,
            "session_id": combat_request.session_id,
            **combat_request.parameters
        }
    )
    
    response = await orchestrator.process(request)
    
    return {
        "action": combat_request.action,
        "result": response.message,
        "success": response.success,
        "metadata": response.metadata
    }


class RuleLookupRequest(BaseModel):
    """Request to lookup a rule, spell, monster, etc."""
    query: str = Field(..., description="Search query")
    resource_type: str = Field(..., description="Type of resource (spell, monster, equipment, etc.)")


@app.post("/actions/lookup")
async def lookup_rule(
    lookup: RuleLookupRequest,
    orchestrator=Depends(get_orchestrator)
):
    """Look up D&D 5e rules, spells, monsters, etc."""
    request = AgentRequest(
        agent_type=AgentType.RULES,
        action="lookup_resource",
        parameters={
            "query": lookup.query,
            "resource_type": lookup.resource_type
        }
    )
    
    response = await orchestrator.process(request)
    
    return {
        "query": lookup.query,
        "resource_type": lookup.resource_type,
        "result": response.message,
        "success": response.success
    }


# ============================================================================
# Character Sheet Endpoints
# ============================================================================

@app.post("/characters", response_model=CharacterResponse)
async def save_character(character: CharacterSheet):
    """Save or update a character sheet."""
    try:
        if DB_AVAILABLE and app_state.db_manager:
            # Use database
            db_char = app_state.db_manager.save_character(
                campaign_id=character.campaign_id,
                player_name=character.player_name,
                character_data=character.character_data
            )
            
            return CharacterResponse(
                id=str(db_char['id']),
                campaign_id=str(db_char['campaign_id']),
                player_name=db_char['player_name'],
                character_data=db_char['character_data'],
                created_at=db_char['created_at'],
                updated_at=db_char['updated_at']
            )
        else:
            # Fallback - return error since we need persistence for characters
            raise HTTPException(
                status_code=503,
                detail="Database not available. Character persistence requires PostgreSQL."
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save character: {str(e)}")


@app.get("/characters/{campaign_id}/{player_name}", response_model=CharacterResponse)
async def get_character(campaign_id: str, player_name: str):
    """Get a character sheet."""
    try:
        if DB_AVAILABLE and app_state.db_manager:
            db_char = app_state.db_manager.get_character(campaign_id, player_name)
            
            if not db_char:
                raise HTTPException(status_code=404, detail="Character not found")
            
            return CharacterResponse(
                id=str(db_char['id']),
                campaign_id=str(db_char['campaign_id']),
                player_name=db_char['player_name'],
                character_data=db_char['character_data'],
                created_at=db_char['created_at'],
                updated_at=db_char['updated_at']
            )
        else:
            raise HTTPException(
                status_code=503,
                detail="Database not available"
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get character: {str(e)}")


@app.get("/campaigns/{campaign_id}/characters", response_model=List[CharacterResponse])
async def list_campaign_characters(campaign_id: str):
    """List all characters in a campaign."""
    try:
        if DB_AVAILABLE and app_state.db_manager:
            db_chars = app_state.db_manager.list_characters(campaign_id)
            
            return [
                CharacterResponse(
                    id=str(c['id']),
                    campaign_id=str(c['campaign_id']),
                    player_name=c['player_name'],
                    character_data=c['character_data'],
                    created_at=c['created_at'],
                    updated_at=c['updated_at']
                )
                for c in db_chars
            ]
        else:
            raise HTTPException(
                status_code=503,
                detail="Database not available"
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list characters: {str(e)}")


# ============================================================================
# Campaign Elements Endpoints (DM Dashboard)
# ============================================================================

@app.post("/campaign-elements", response_model=CampaignElementResponse)
async def create_campaign_element(element: CampaignElement):
    """Create a campaign element (NPC, quest, location, monster, narrative point)."""
    try:
        if DB_AVAILABLE and app_state.db_manager:
            db_element = app_state.db_manager.save_campaign_element(
                campaign_id=element.campaign_id,
                element_type=element.element_type,
                element_data=element.element_data
            )
            
            return CampaignElementResponse(
                id=str(db_element['id']),
                campaign_id=str(db_element['campaign_id']),
                element_type=db_element['element_type'],
                element_data=db_element['element_data'],
                created_at=db_element['created_at'],
                updated_at=db_element['updated_at']
            )
        else:
            raise HTTPException(status_code=503, detail="Database not available")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create element: {str(e)}")


@app.get("/campaigns/{campaign_id}/elements", response_model=List[CampaignElementResponse])
async def list_campaign_elements(campaign_id: str, element_type: Optional[str] = None):
    """List campaign elements, optionally filtered by type."""
    try:
        if DB_AVAILABLE and app_state.db_manager:
            db_elements = app_state.db_manager.list_campaign_elements(campaign_id, element_type)
            
            return [
                CampaignElementResponse(
                    id=str(e['id']),
                    campaign_id=str(e['campaign_id']),
                    element_type=e['element_type'],
                    element_data=e['element_data'],
                    created_at=e['created_at'],
                    updated_at=e['updated_at']
                )
                for e in db_elements
            ]
        else:
            raise HTTPException(status_code=503, detail="Database not available")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list elements: {str(e)}")


@app.put("/campaign-elements/{element_id}", response_model=CampaignElementResponse)
async def update_campaign_element(element_id: str, element_data: Dict[str, Any]):
    """Update a campaign element."""
    try:
        if DB_AVAILABLE and app_state.db_manager:
            db_element = app_state.db_manager.update_campaign_element(element_id, element_data)
            
            if not db_element:
                raise HTTPException(status_code=404, detail="Element not found")
            
            return CampaignElementResponse(
                id=str(db_element['id']),
                campaign_id=str(db_element['campaign_id']),
                element_type=db_element['element_type'],
                element_data=db_element['element_data'],
                created_at=db_element['created_at'],
                updated_at=db_element['updated_at']
            )
        else:
            raise HTTPException(status_code=503, detail="Database not available")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update element: {str(e)}")


@app.delete("/campaign-elements/{element_id}")
async def delete_campaign_element(element_id: str):
    """Delete a campaign element."""
    try:
        if DB_AVAILABLE and app_state.db_manager:
            success = app_state.db_manager.delete_campaign_element(element_id)
            
            if not success:
                raise HTTPException(status_code=404, detail="Element not found")
            
            return {"success": True, "message": "Element deleted"}
        else:
            raise HTTPException(status_code=503, detail="Database not available")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete element: {str(e)}")


@app.post("/ai-generate")
async def ai_generate_content(
    request: AIGenerateRequest,
    orchestrator=Depends(get_orchestrator)
):
    """Use AI agents to generate campaign content."""
    try:
        # Route to appropriate agent based on element type
        if request.element_type in ["npc", "monster", "location"]:
            agent_type = AgentType.ENTITY
        elif request.element_type == "narrative":
            agent_type = AgentType.NARRATIVE
        else:
            agent_type = AgentType.ORCHESTRATOR
        
        agent_request = AgentRequest(
            agent_type=agent_type,
            action="generate_campaign_content",
            parameters={
                "campaign_id": request.campaign_id,
                "element_type": request.element_type,
                "prompt": request.prompt,
                "context": request.context or {}
            }
        )
        
        response = await orchestrator.process(agent_request)
        
        return {
            "success": response.success,
            "content": response.message,
            "data": response.data,
            "metadata": response.metadata
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate content: {str(e)}")


# ============================================================================
# Run the application
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )