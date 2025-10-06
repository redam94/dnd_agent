"""
D&D Campaign Chatbot - Streamlit Frontend
==========================================

A beautiful, interactive frontend for the D&D multi-agent campaign system.

Features:
- Campaign and session management
- Chat interface with the DM
- Dice roller with animations
- Rule lookup
- Character management
- Game state visualization
- Combat tracker

Usage:
    streamlit run streamlit_app.py
"""

import streamlit as st
import requests
from datetime import datetime
from typing import Dict, List, Any, Optional
import time
import random

# ============================================================================
# Configuration
# ============================================================================

# Set page config (must be first Streamlit command)
st.set_page_config(
    page_title="D&D Campaign Manager",
    page_icon="🎲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_BASE_URL = st.secrets.get("API_BASE_URL", "http://localhost:8000")

# ============================================================================
# Custom CSS Styling
# ============================================================================

st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --dm-color: #8b0000;
        --player-color: #1e3a8a;
        --success-color: #22c55e;
        --warning-color: #f59e0b;
    }
    
    /* Chat message styling */
    .dm-message {
        background: linear-gradient(135deg, #1a0000 0%, #3a0000 100%);
        border-left: 4px solid #8b0000;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 8px;
        color: #f0f0f0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    }
    
    .player-message {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 8px;
        color: #e0e0e0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    }
    
    .system-message {
        background: linear-gradient(135deg, #1a1a1a 0%, #2a2a2a 100%);
        border-left: 4px solid #6b7280;
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-radius: 8px;
        color: #d0d0d0;
        font-style: italic;
        text-align: center;
    }
    
    /* Message headers */
    .message-header {
        font-weight: bold;
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
        opacity: 0.9;
    }
    
    .dm-header { color: #ff6b6b; }
    .player-header { color: #4dabf7; }
    
    /* Dice roller styling */
    .dice-result {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        padding: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 16px rgba(0,0,0,0.3);
        animation: pulse 0.5s ease-in-out;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    
    /* Campaign card styling */
    .campaign-card {
        background: linear-gradient(135deg, #2d3748 0%, #1a202c 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border: 2px solid #4a5568;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .campaign-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.4);
        border-color: #667eea;
    }
    
    /* Character sheet styling */
    .stat-box {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-radius: 8px;
        border: 2px solid #4a5568;
        margin: 0.5rem 0;
    }
    
    .stat-modifier {
        font-size: 1.5rem;
        color: #667eea;
        font-weight: bold;
    }
    
    .inventory-item {
        padding: 0.75rem;
        margin: 0.5rem 0;
        background: linear-gradient(135deg, #1a202c 0%, #2d3748 100%);
        border-radius: 8px;
        border-left: 4px solid #667eea;
    }
    
    .hp-bar {
        width: 100%;
        height: 24px;
        background: #1a202c;
        border-radius: 12px;
        overflow: hidden;
        border: 2px solid #4a5568;
    }
    
    .hp-bar-fill {
        height: 100%;
        background: linear-gradient(90deg, #22c55e 0%, #16a34a 100%);
        transition: width 0.3s ease;
    }
    
    .hp-bar-fill.warning {
        background: linear-gradient(90deg, #f59e0b 0%, #d97706 100%);
    }
    
    .hp-bar-fill.critical {
        background: linear-gradient(90deg, #ef4444 0%, #dc2626 100%);
    }
    
    /* Status badge styling */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.25rem;
    }
    
    .badge-active { background: #22c55e; color: white; }
    .badge-inactive { background: #6b7280; color: white; }
    .badge-combat { background: #ef4444; color: white; }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1a1a1a;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #4a5568;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #667eea;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# API Client Functions
# ============================================================================

class APIClient:
    """Client for interacting with the D&D Campaign API."""
    
    @staticmethod
    def check_health() -> Dict[str, Any]:
        """Check API health status."""
        try:
            response = requests.get(f"{API_BASE_URL}/health", timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    @staticmethod
    def create_campaign(name: str, description: str = None, setting: str = None) -> Dict[str, Any]:
        """Create a new campaign."""
        data = {"name": name}
        if description:
            data["description"] = description
        if setting:
            data["setting"] = setting
        
        response = requests.post(f"{API_BASE_URL}/campaigns", json=data)
        response.raise_for_status()
        return response.json()
    
    @staticmethod
    def list_campaigns() -> List[Dict[str, Any]]:
        """List all campaigns."""
        response = requests.get(f"{API_BASE_URL}/campaigns")
        response.raise_for_status()
        return response.json()
    
    @staticmethod
    def get_campaign(campaign_id: str) -> Dict[str, Any]:
        """Get campaign details."""
        response = requests.get(f"{API_BASE_URL}/campaigns/{campaign_id}")
        response.raise_for_status()
        return response.json()
    
    @staticmethod
    def create_session(campaign_id: str, name: str = None) -> Dict[str, Any]:
        """Create a new session."""
        data = {"campaign_id": campaign_id}
        if name:
            data["name"] = name
        
        response = requests.post(f"{API_BASE_URL}/sessions", json=data)
        response.raise_for_status()
        return response.json()
    
    @staticmethod
    def list_sessions(campaign_id: str) -> List[Dict[str, Any]]:
        """List sessions for a campaign."""
        response = requests.get(f"{API_BASE_URL}/campaigns/{campaign_id}/sessions")
        response.raise_for_status()
        return response.json()
    
    @staticmethod
    def send_message(campaign_id: str, session_id: str, message: str, player_name: str = None) -> Dict[str, Any]:
        """Send a message to the DM."""
        data = {
            "campaign_id": campaign_id,
            "session_id": session_id,
            "message": message
        }
        if player_name:
            data["player_name"] = player_name
        
        response = requests.post(f"{API_BASE_URL}/messages", json=data)
        response.raise_for_status()
        return response.json()
    
    @staticmethod
    def get_messages(campaign_id: str, session_id: str = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Get chat history."""
        params = {"limit": limit}
        if session_id:
            params["session_id"] = session_id
        
        response = requests.get(f"{API_BASE_URL}/campaigns/{campaign_id}/messages", params=params)
        response.raise_for_status()
        return response.json()
    
    @staticmethod
    def roll_dice(campaign_id: str, session_id: str, dice_notation: str) -> Dict[str, Any]:
        """Roll dice."""
        data = {
            "campaign_id": campaign_id,
            "session_id": session_id,
            "dice_notation": dice_notation
        }
        response = requests.post(f"{API_BASE_URL}/actions/roll-dice", json=data)
        response.raise_for_status()
        return response.json()
    
    @staticmethod
    def lookup_rule(query: str, resource_type: str) -> Dict[str, Any]:
        """Look up a rule, spell, monster, etc."""
        data = {"query": query, "resource_type": resource_type}
        response = requests.post(f"{API_BASE_URL}/actions/lookup", json=data)
        response.raise_for_status()
        return response.json()
    
    @staticmethod
    def get_game_state(campaign_id: str, session_id: str = None) -> Dict[str, Any]:
        """Get current game state."""
        data = {"campaign_id": campaign_id}
        if session_id:
            data["session_id"] = session_id
        
        response = requests.post(f"{API_BASE_URL}/game-state", json=data)
        response.raise_for_status()
        return response.json()
    
    @staticmethod
    def save_character(campaign_id: str, player_name: str, character_data: Dict[str, Any]) -> Dict[str, Any]:
        """Save a character sheet."""
        data = {
            "campaign_id": campaign_id,
            "player_name": player_name,
            "character_data": character_data
        }
        response = requests.post(f"{API_BASE_URL}/characters", json=data)
        response.raise_for_status()
        return response.json()
    
    @staticmethod
    def get_character(campaign_id: str, player_name: str) -> Dict[str, Any]:
        """Get a character sheet."""
        response = requests.get(f"{API_BASE_URL}/characters/{campaign_id}/{player_name}")
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json()
    
    @staticmethod
    def list_campaign_characters(campaign_id: str) -> List[Dict[str, Any]]:
        """List all characters in a campaign."""
        response = requests.get(f"{API_BASE_URL}/campaigns/{campaign_id}/characters")
        response.raise_for_status()
        return response.json()
    
    @staticmethod
    def create_campaign_element(campaign_id: str, element_type: str, element_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a campaign element (NPC, quest, location, monster, narrative)."""
        data = {
            "campaign_id": campaign_id,
            "element_type": element_type,
            "element_data": element_data
        }
        response = requests.post(f"{API_BASE_URL}/campaign-elements", json=data)
        response.raise_for_status()
        return response.json()
    
    @staticmethod
    def list_campaign_elements(campaign_id: str, element_type: str = None) -> List[Dict[str, Any]]:
        """List campaign elements."""
        params = {"element_type": element_type} if element_type else {}
        response = requests.get(f"{API_BASE_URL}/campaigns/{campaign_id}/elements", params=params)
        response.raise_for_status()
        return response.json()
    
    @staticmethod
    def update_campaign_element(element_id: str, element_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update a campaign element."""
        response = requests.put(f"{API_BASE_URL}/campaign-elements/{element_id}", json=element_data)
        response.raise_for_status()
        return response.json()
    
    @staticmethod
    def delete_campaign_element(element_id: str) -> Dict[str, Any]:
        """Delete a campaign element."""
        response = requests.delete(f"{API_BASE_URL}/campaign-elements/{element_id}")
        response.raise_for_status()
        return response.json()
    
    @staticmethod
    def ai_generate_content(campaign_id: str, element_type: str, prompt: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Use AI to generate campaign content."""
        data = {
            "campaign_id": campaign_id,
            "element_type": element_type,
            "prompt": prompt,
            "context": context or {}
        }
        response = requests.post(f"{API_BASE_URL}/ai-generate", json=data)
        response.raise_for_status()
        return response.json()

# ============================================================================
# Session State Initialization
# ============================================================================

def init_session_state():
    """Initialize session state variables."""
    if "current_campaign" not in st.session_state:
        st.session_state.current_campaign = None
    if "current_session" not in st.session_state:
        st.session_state.current_session = None
    if "player_name" not in st.session_state:
        st.session_state.player_name = "Adventurer"
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_roll" not in st.session_state:
        st.session_state.last_roll = None
    if "show_new_session" not in st.session_state:
        st.session_state.show_new_session = False
    if "show_new_session_form" not in st.session_state:
        st.session_state.show_new_session_form = False
    if "character" not in st.session_state:
        st.session_state.character = None

init_session_state()

# ============================================================================
# Helper Functions
# ============================================================================

def display_message(msg: Dict[str, Any]):
    """Display a chat message with proper styling."""
    role = msg.get("role", "system")
    content = msg.get("content", "")
    timestamp = msg.get("timestamp", "")
    
    if isinstance(timestamp, str):
        try:
            ts = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            time_str = ts.strftime("%I:%M %p")
        except:
            time_str = ""
    else:
        time_str = ""
    
    if role == "dm":
        st.markdown(f"""
        <div class="dm-message">
            <div class="message-header dm-header">🎭 Dungeon Master {time_str}</div>
            <div>{content}</div>
        </div>
        """, unsafe_allow_html=True)
    elif role == "player":
        player_name = msg.get("player_name", "Player")
        st.markdown(f"""
        <div class="player-message">
            <div class="message-header player-header">⚔️ {player_name} {time_str}</div>
            <div>{content}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="system-message">
            <div>{content}</div>
        </div>
        """, unsafe_allow_html=True)

def display_dice_roll(notation: str, result: str):
    """Display dice roll result with animation."""
    st.markdown(f"""
    <div class="dice-result">
        🎲 {notation}<br>
        <span style="font-size: 4rem; color: #ffd700;">{result}</span>
    </div>
    """, unsafe_allow_html=True)

def parse_dice_notation(notation: str) -> tuple:
    """Parse dice notation like '2d6+3' into components."""
    try:
        parts = notation.lower().replace(" ", "").split("d")
        num_dice = int(parts[0]) if parts[0] else 1
        
        if "+" in parts[1]:
            dice_size, modifier = parts[1].split("+")
            modifier = int(modifier)
        elif "-" in parts[1]:
            dice_size, modifier = parts[1].split("-")
            modifier = -int(modifier)
        else:
            dice_size = parts[1]
            modifier = 0
        
        return num_dice, int(dice_size), modifier
    except:
        return None

def roll_dice_local(notation: str) -> str:
    """Roll dice locally for immediate feedback."""
    parsed = parse_dice_notation(notation)
    if not parsed:
        return "Invalid notation"
    
    num_dice, dice_size, modifier = parsed
    rolls = [random.randint(1, dice_size) for _ in range(num_dice)]
    total = sum(rolls) + modifier
    
    roll_details = " + ".join(map(str, rolls))
    if modifier != 0:
        roll_details += f" {'+' if modifier > 0 else ''}{modifier}"
    
    return f"{total} ({roll_details})"

# ============================================================================
# Sidebar - Navigation and Settings
# ============================================================================

with st.sidebar:
    st.title("🎲 D&D Campaign Manager")
    
    # API Health Check
    with st.expander("🔧 API Status", expanded=False):
        if st.button("Check Connection", use_container_width=True):
            with st.spinner("Checking..."):
                health = APIClient.check_health()
                if health.get("status") == "healthy":
                    st.success("✅ API Connected")
                    st.json({
                        "Neo4j": "✅" if health.get("neo4j_connected") else "❌",
                        "PostgreSQL": "✅" if health.get("postgres_connected") else "❌",
                        "Agents": "✅" if health.get("agents_initialized") else "❌"
                    })
                else:
                    st.error(f"❌ {health.get('message', 'Connection failed')}")
    
    st.divider()
    
    # Navigation
    page = st.radio(
        "Navigation",
        ["🏰 Campaign", "🎭 DM Dashboard", "💬 Chat", "🎲 Dice Roller", "📖 Lookup", "👤 Character", "⚔️ Combat"],
        label_visibility="collapsed"
    )
    
    st.divider()
    
    # Campaign Selection
    st.subheader("Current Campaign")
    
    try:
        campaigns = APIClient.list_campaigns()
        
        if campaigns:
            campaign_names = {c["name"]: c["id"] for c in campaigns}
            selected_campaign = st.selectbox(
                "Select Campaign",
                options=list(campaign_names.keys()),
                index=0 if not st.session_state.current_campaign else 
                      list(campaign_names.keys()).index(
                          next((c["name"] for c in campaigns if c["id"] == st.session_state.current_campaign), 
                               list(campaign_names.keys())[0])
                      ),
                label_visibility="collapsed"
            )
            
            if selected_campaign:
                st.session_state.current_campaign = campaign_names[selected_campaign]
                
                # Session Selection
                sessions = APIClient.list_sessions(st.session_state.current_campaign)
                if sessions:
                    session_names = {s.get("name", f"Session {i+1}"): s["id"] 
                                   for i, s in enumerate(sessions)}
                    selected_session = st.selectbox(
                        "Select Session",
                        options=list(session_names.keys()),
                        label_visibility="collapsed"
                    )
                    st.session_state.current_session = session_names[selected_session]
                else:
                    st.info("No active sessions")
                    if st.button("➕ New Session", use_container_width=True):
                        st.session_state.show_new_session = True
                        st.rerun()
        else:
            st.info("No campaigns found")
            
    except Exception as e:
        st.error(f"Error loading campaigns: {str(e)}")
    
    st.divider()
    
    # Player Settings
    st.subheader("Player Settings")
    st.session_state.player_name = st.text_input(
        "Character Name",
        value=st.session_state.player_name,
        placeholder="Enter your character name"
    )

# ============================================================================
# Main Content Area
# ============================================================================

if page == "🏰 Campaign":
    st.title("🏰 Campaign Management")
    
    col1, col2 = st.columns(2)
    
    # Create new campaign
    with col1:
        with st.expander("➕ Create New Campaign", expanded=False):
            with st.form("new_campaign_form"):
                camp_name = st.text_input("Campaign Name*", placeholder="e.g., Curse of Strahd")
                camp_desc = st.text_area("Description", placeholder="A dark tale of horror and intrigue...")
                camp_setting = st.text_input("Setting", placeholder="e.g., Ravenloft, Forgotten Realms")
                
                if st.form_submit_button("Create Campaign", use_container_width=True):
                    if camp_name:
                        try:
                            with st.spinner("Creating campaign..."):
                                campaign = APIClient.create_campaign(camp_name, camp_desc, camp_setting)
                                st.success(f"✅ Created campaign: {campaign['name']}")
                                st.session_state.current_campaign = campaign['id']
                                time.sleep(1)
                                st.rerun()
                        except Exception as e:
                            st.error(f"Error creating campaign: {str(e)}")
                    else:
                        st.warning("Campaign name is required")
    
    # Create new session
    with col2:
        with st.expander("➕ Create New Session", expanded=False):
            if not st.session_state.current_campaign:
                st.warning("⚠️ Select a campaign first")
            else:
                with st.form("new_session_form"):
                    sess_name = st.text_input(
                        "Session Name (Optional)", 
                        placeholder="e.g., Session 1: The Beginning"
                    )
                    
                    if st.form_submit_button("Create Session", use_container_width=True):
                        try:
                            with st.spinner("Creating session..."):
                                session = APIClient.create_session(
                                    st.session_state.current_campaign,
                                    sess_name if sess_name else None
                                )
                                st.success(f"✅ Created session!")
                                st.session_state.current_session = session['id']
                                time.sleep(1)
                                st.rerun()
                        except Exception as e:
                            st.error(f"Error creating session: {str(e)}")
    
    # List existing campaigns
    st.subheader("Your Campaigns")
    
    try:
        campaigns = APIClient.list_campaigns()
        
        if campaigns:
            for campaign in campaigns:
                with st.container():
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.markdown(f"""
                        <div class="campaign-card">
                            <h3>🏰 {campaign['name']}</h3>
                            <p><strong>Setting:</strong> {campaign.get('setting', 'N/A')}</p>
                            <p>{campaign.get('description', 'No description')}</p>
                            <small>Created: {campaign['created_at'][:10]}</small>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        if st.button("Select", key=f"select_{campaign['id']}", use_container_width=True):
                            st.session_state.current_campaign = campaign['id']
                            st.success(f"Selected: {campaign['name']}")
                            st.rerun()
                        
                        # Show sessions count
                        try:
                            sessions = APIClient.list_sessions(campaign['id'])
                            st.metric("Sessions", len(sessions))
                        except:
                            pass
        else:
            st.info("No campaigns yet. Create your first campaign above!")
            
    except Exception as e:
                    st.error(f"Error loading campaigns: {str(e)}")

elif page == "🎭 DM Dashboard":
    st.title("🎭 Dungeon Master Dashboard")
    
    if not st.session_state.current_campaign:
        st.warning("⚠️ Please select a campaign from the sidebar first.")
    else:
        st.markdown("**Welcome, Dungeon Master!** Build your campaign world with AI assistance.")
        
        # Quick Stats
        col1, col2, col3, col4, col5 = st.columns(5)
        
        try:
            npcs = APIClient.list_campaign_elements(st.session_state.current_campaign, "npc")
            locations = APIClient.list_campaign_elements(st.session_state.current_campaign, "location")
            quests = APIClient.list_campaign_elements(st.session_state.current_campaign, "quest")
            monsters = APIClient.list_campaign_elements(st.session_state.current_campaign, "monster")
            narratives = APIClient.list_campaign_elements(st.session_state.current_campaign, "narrative")
            
            with col1:
                st.metric("👥 NPCs", len(npcs))
            with col2:
                st.metric("🗺️ Locations", len(locations))
            with col3:
                st.metric("📜 Quests", len(quests))
            with col4:
                st.metric("👹 Monsters", len(monsters))
            with col5:
                st.metric("📖 Story Points", len(narratives))
        except:
            st.info("Loading campaign data...")
        
        st.divider()
        
        # Main tabs for different aspects
        tabs = st.tabs(["📖 Narrative", "👥 NPCs", "🗺️ Locations", "📜 Quests", "👹 Monsters"])
        
        # ===================================================================
        # NARRATIVE TAB
        # ===================================================================
        with tabs[0]:
            st.subheader("📖 Campaign Narrative")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("**Create narrative points, story arcs, and plot hooks for your campaign.**")
            
            with col2:
                if st.button("🤖 AI Generate Story", use_container_width=True, type="primary"):
                    st.session_state.show_ai_narrative = True
            
            # AI Generation Modal
            if st.session_state.get("show_ai_narrative", False):
                with st.form("ai_narrative_form"):
                    st.write("**AI Story Generator**")
                    prompt = st.text_area(
                        "Describe what you want to generate",
                        placeholder="e.g., Create an opening scene where the party arrives at a mysterious village...",
                        height=100
                    )
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.form_submit_button("Generate", use_container_width=True, type="primary"):
                            if prompt:
                                with st.spinner("🤖 AI is crafting your narrative..."):
                                    try:
                                        result = APIClient.ai_generate_content(
                                            st.session_state.current_campaign,
                                            "narrative",
                                            prompt
                                        )
                                        
                                        if result.get("success"):
                                            # Auto-fill the form below with AI content
                                            st.session_state.ai_narrative_content = result.get("content", "")
                                            st.success("✅ Narrative generated!")
                                            st.session_state.show_ai_narrative = False
                                            st.rerun()
                                        else:
                                            st.error("Failed to generate narrative")
                                    except Exception as e:
                                        st.error(f"Error: {str(e)}")
                    
                    with col2:
                        if st.form_submit_button("Cancel", use_container_width=True):
                            st.session_state.show_ai_narrative = False
                            st.rerun()
            
            # Create new narrative point
            with st.expander("➕ Add Narrative Point", expanded=False):
                with st.form("new_narrative"):
                    title = st.text_input("Title", placeholder="e.g., The Dark Prophecy")
                    
                    content = st.text_area(
                        "Narrative Content",
                        value=st.session_state.get("ai_narrative_content", ""),
                        placeholder="Describe the story point, plot hook, or narrative element...",
                        height=150
                    )
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        arc = st.text_input("Story Arc", placeholder="e.g., Main Quest")
                    with col2:
                        importance = st.selectbox("Importance", ["Low", "Medium", "High", "Critical"])
                    with col3:
                        status = st.selectbox("Status", ["Planned", "Active", "Completed", "Abandoned"])
                    
                    if st.form_submit_button("Add Narrative Point", use_container_width=True, type="primary"):
                        if title and content:
                            try:
                                APIClient.create_campaign_element(
                                    st.session_state.current_campaign,
                                    "narrative",
                                    {
                                        "title": title,
                                        "content": content,
                                        "arc": arc,
                                        "importance": importance,
                                        "status": status
                                    }
                                )
                                st.success("✅ Narrative point added!")
                                st.session_state.ai_narrative_content = ""
                                time.sleep(1)
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error: {str(e)}")
                        else:
                            st.warning("Title and content are required")
            
            # Display narrative points
            st.divider()
            try:
                narratives = APIClient.list_campaign_elements(st.session_state.current_campaign, "narrative")
                
                if narratives:
                    for narrative in narratives:
                        data = narrative['element_data']
                        importance_colors = {
                            "Low": "🟢",
                            "Medium": "🟡",
                            "High": "🟠",
                            "Critical": "🔴"
                        }
                        
                        with st.container():
                            col1, col2 = st.columns([5, 1])
                            
                            with col1:
                                st.markdown(f"""
                                <div class="campaign-card">
                                    <h4>{importance_colors.get(data.get('importance', 'Medium'), '⚪')} {data.get('title', 'Untitled')}</h4>
                                    <p><strong>Arc:</strong> {data.get('arc', 'N/A')} | <strong>Status:</strong> {data.get('status', 'Planned')}</p>
                                    <p>{data.get('content', '')}</p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with col2:
                                st.write("")
                                st.write("")
                                if st.button("🗑️", key=f"del_narr_{narrative['id']}"):
                                    try:
                                        APIClient.delete_campaign_element(narrative['id'])
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Error: {str(e)}")
                else:
                    st.info("No narrative points yet. Create your first story element above!")
            except Exception as e:
                st.error(f"Error loading narratives: {str(e)}")
        
        # ===================================================================
        # NPCS TAB  
        # ===================================================================
        with tabs[1]:
            st.subheader("👥 Non-Player Characters")
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown("**Create and manage NPCs for your campaign.**")
            with col2:
                if st.button("🤖 AI Generate NPC", use_container_width=True, type="primary"):
                    with st.spinner("🤖 Creating NPC..."):
                        try:
                            result = APIClient.ai_generate_content(
                                st.session_state.current_campaign,
                                "npc",
                                "Generate a detailed NPC with name, appearance, personality, and motivation"
                            )
                            st.session_state.ai_npc_result = result.get("content", "")
                            st.success("✅ NPC generated! Check below.")
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
            
            # Show AI result if available
            if st.session_state.get("ai_npc_result"):
                st.info(st.session_state.ai_npc_result)
                if st.button("Clear AI Result"):
                    st.session_state.ai_npc_result = None
                    st.rerun()
            
            # Create NPC form
            with st.expander("➕ Add NPC", expanded=False):
                with st.form("new_npc"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        npc_name = st.text_input("Name*", placeholder="e.g., Eldrin the Wise")
                        npc_race = st.text_input("Race", placeholder="e.g., Elf")
                        npc_class = st.text_input("Class/Occupation", placeholder="e.g., Wizard, Merchant")
                    
                    with col2:
                        npc_alignment = st.selectbox("Alignment", 
                            ["Lawful Good", "Neutral Good", "Chaotic Good",
                             "Lawful Neutral", "True Neutral", "Chaotic Neutral",
                             "Lawful Evil", "Neutral Evil", "Chaotic Evil"])
                        npc_role = st.selectbox("Role", ["Ally", "Neutral", "Antagonist", "Quest Giver", "Merchant", "Other"])
                        npc_location = st.text_input("Location", placeholder="Where can they be found?")
                    
                    npc_appearance = st.text_area("Appearance", placeholder="Physical description...", height=80)
                    npc_personality = st.text_area("Personality", placeholder="Traits, mannerisms, speech patterns...", height=80)
                    npc_motivation = st.text_area("Motivation", placeholder="What do they want? What drives them?", height=80)
                    npc_secrets = st.text_area("Secrets/Notes", placeholder="DM notes, plot hooks, secrets...", height=80)
                    
                    if st.form_submit_button("Add NPC", use_container_width=True, type="primary"):
                        if npc_name:
                            try:
                                APIClient.create_campaign_element(
                                    st.session_state.current_campaign,
                                    "npc",
                                    {
                                        "name": npc_name,
                                        "race": npc_race,
                                        "class": npc_class,
                                        "alignment": npc_alignment,
                                        "role": npc_role,
                                        "location": npc_location,
                                        "appearance": npc_appearance,
                                        "personality": npc_personality,
                                        "motivation": npc_motivation,
                                        "secrets": npc_secrets
                                    }
                                )
                                st.success("✅ NPC added!")
                                st.session_state.ai_npc_result = None
                                time.sleep(1)
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error: {str(e)}")
                        else:
                            st.warning("Name is required")
            
            # Display NPCs
            st.divider()
            try:
                npcs = APIClient.list_campaign_elements(st.session_state.current_campaign, "npc")
                
                if npcs:
                    for npc in npcs:
                        data = npc['element_data']
                        
                        with st.expander(f"👤 {data.get('name', 'Unnamed')} - {data.get('role', 'NPC')}"):
                            col1, col2 = st.columns([4, 1])
                            
                            with col1:
                                st.markdown(f"**Race:** {data.get('race', 'Unknown')} | **Class:** {data.get('class', 'N/A')}")
                                st.markdown(f"**Alignment:** {data.get('alignment', 'Unknown')} | **Location:** {data.get('location', 'Unknown')}")
                                
                                if data.get('appearance'):
                                    st.markdown(f"**Appearance:** {data.get('appearance')}")
                                if data.get('personality'):
                                    st.markdown(f"**Personality:** {data.get('personality')}")
                                if data.get('motivation'):
                                    st.markdown(f"**Motivation:** {data.get('motivation')}")
                                if data.get('secrets'):
                                    st.markdown(f"**DM Notes:** {data.get('secrets')}")
                            
                            with col2:
                                if st.button("🗑️ Delete", key=f"del_npc_{npc['id']}"):
                                    try:
                                        APIClient.delete_campaign_element(npc['id'])
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Error: {str(e)}")
                else:
                    st.info("No NPCs yet. Create your first character above!")
            except Exception as e:
                st.error(f"Error loading NPCs: {str(e)}")
        
        # ===================================================================
        # LOCATIONS TAB
        # ===================================================================
        with tabs[2]:
            st.subheader("🗺️ Locations")
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown("**Build your campaign world with memorable locations.**")
            with col2:
                if st.button("🤖 AI Generate Location", use_container_width=True, type="primary"):
                    with st.spinner("🤖 Creating location..."):
                        try:
                            result = APIClient.ai_generate_content(
                                st.session_state.current_campaign,
                                "location",
                                "Generate a detailed location with description, inhabitants, and points of interest"
                            )
                            st.info(result.get("content", ""))
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
            
            with st.expander("➕ Add Location", expanded=False):
                with st.form("new_location"):
                    loc_name = st.text_input("Name*", placeholder="e.g., The Whispering Woods")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        loc_type = st.selectbox("Type", ["City", "Town", "Village", "Dungeon", "Wilderness", "Landmark", "Building", "Other"])
                    with col2:
                        loc_danger = st.selectbox("Danger Level", ["Safe", "Low", "Medium", "High", "Deadly"])
                    
                    loc_description = st.text_area("Description", placeholder="What does this place look like? What's the atmosphere?", height=100)
                    loc_inhabitants = st.text_area("Inhabitants", placeholder="Who or what lives here?", height=80)
                    loc_poi = st.text_area("Points of Interest", placeholder="Notable features, buildings, areas...", height=80)
                    loc_secrets = st.text_area("DM Notes", placeholder="Hidden areas, traps, treasure, plot hooks...", height=80)
                    
                    if st.form_submit_button("Add Location", use_container_width=True, type="primary"):
                        if loc_name:
                            try:
                                APIClient.create_campaign_element(
                                    st.session_state.current_campaign,
                                    "location",
                                    {
                                        "name": loc_name,
                                        "type": loc_type,
                                        "danger_level": loc_danger,
                                        "description": loc_description,
                                        "inhabitants": loc_inhabitants,
                                        "points_of_interest": loc_poi,
                                        "secrets": loc_secrets
                                    }
                                )
                                st.success("✅ Location added!")
                                time.sleep(1)
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error: {str(e)}")
                        else:
                            st.warning("Name is required")
            
            # Display locations
            st.divider()
            try:
                locations = APIClient.list_campaign_elements(st.session_state.current_campaign, "location")
                
                if locations:
                    for loc in locations:
                        data = loc['element_data']
                        danger_icons = {"Safe": "🟢", "Low": "🟡", "Medium": "🟠", "High": "🔴", "Deadly": "💀"}
                        
                        with st.expander(f"🗺️ {data.get('name', 'Unnamed')} - {data.get('type', 'Location')} {danger_icons.get(data.get('danger_level', 'Low'), '⚪')}"):
                            col1, col2 = st.columns([4, 1])
                            
                            with col1:
                                st.markdown(f"**Type:** {data.get('type', 'Unknown')} | **Danger:** {data.get('danger_level', 'Unknown')}")
                                
                                if data.get('description'):
                                    st.markdown(f"**Description:** {data.get('description')}")
                                if data.get('inhabitants'):
                                    st.markdown(f"**Inhabitants:** {data.get('inhabitants')}")
                                if data.get('points_of_interest'):
                                    st.markdown(f"**Points of Interest:** {data.get('points_of_interest')}")
                                if data.get('secrets'):
                                    st.markdown(f"**DM Notes:** {data.get('secrets')}")
                            
                            with col2:
                                if st.button("🗑️ Delete", key=f"del_loc_{loc['id']}"):
                                    try:
                                        APIClient.delete_campaign_element(loc['id'])
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Error: {str(e)}")
                else:
                    st.info("No locations yet. Start building your world!")
            except Exception as e:
                st.error(f"Error loading locations: {str(e)}")
        
        # ===================================================================
        # QUESTS TAB
        # ===================================================================
        with tabs[3]:
            st.subheader("📜 Quests & Adventures")
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown("**Manage quests, missions, and adventures.**")
            with col2:
                if st.button("🤖 AI Generate Quest", use_container_width=True, type="primary"):
                    with st.spinner("🤖 Creating quest..."):
                        try:
                            result = APIClient.ai_generate_content(
                                st.session_state.current_campaign,
                                "quest",
                                "Generate an interesting quest with objectives, rewards, and potential complications"
                            )
                            st.info(result.get("content", ""))
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
            
            with st.expander("➕ Add Quest", expanded=False):
                with st.form("new_quest"):
                    quest_name = st.text_input("Quest Name*", placeholder="e.g., The Missing Merchant")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        quest_type = st.selectbox("Type", ["Main Quest", "Side Quest", "Personal Quest", "Random Encounter"])
                    with col2:
                        quest_difficulty = st.selectbox("Difficulty", ["Easy", "Medium", "Hard", "Deadly"])
                    with col3:
                        quest_status = st.selectbox("Status", ["Not Started", "Active", "Completed", "Failed", "Abandoned"])
                    
                    quest_giver = st.text_input("Quest Giver", placeholder="Who gives this quest?")
                    quest_description = st.text_area("Description", placeholder="What is the quest about?", height=100)
                    quest_objectives = st.text_area("Objectives", placeholder="What needs to be done?", height=80)
                    quest_rewards = st.text_area("Rewards", placeholder="Gold, items, favor, information...", height=80)
                    quest_notes = st.text_area("DM Notes", placeholder="Complications, alternate solutions, secrets...", height=80)
                    
                    if st.form_submit_button("Add Quest", use_container_width=True, type="primary"):
                        if quest_name:
                            try:
                                APIClient.create_campaign_element(
                                    st.session_state.current_campaign,
                                    "quest",
                                    {
                                        "name": quest_name,
                                        "type": quest_type,
                                        "difficulty": quest_difficulty,
                                        "status": quest_status,
                                        "quest_giver": quest_giver,
                                        "description": quest_description,
                                        "objectives": quest_objectives,
                                        "rewards": quest_rewards,
                                        "notes": quest_notes
                                    }
                                )
                                st.success("✅ Quest added!")
                                time.sleep(1)
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error: {str(e)}")
                        else:
                            st.warning("Quest name is required")
            
            # Display quests
            st.divider()
            try:
                quests = APIClient.list_campaign_elements(st.session_state.current_campaign, "quest")
                
                if quests:
                    for quest in quests:
                        data = quest['element_data']
                        status_icons = {
                            "Not Started": "⭕",
                            "Active": "🟢",
                            "Completed": "✅",
                            "Failed": "❌",
                            "Abandoned": "⚪"
                        }
                        
                        with st.expander(f"📜 {data.get('name', 'Unnamed Quest')} {status_icons.get(data.get('status', 'Active'), '❓')}"):
                            col1, col2 = st.columns([4, 1])
                            
                            with col1:
                                st.markdown(f"**Type:** {data.get('type', 'Quest')} | **Difficulty:** {data.get('difficulty', 'Unknown')} | **Status:** {data.get('status', 'Active')}")
                                if data.get('quest_giver'):
                                    st.markdown(f"**Quest Giver:** {data.get('quest_giver')}")
                                if data.get('description'):
                                    st.markdown(f"**Description:** {data.get('description')}")
                                if data.get('objectives'):
                                    st.markdown(f"**Objectives:** {data.get('objectives')}")
                                if data.get('rewards'):
                                    st.markdown(f"**Rewards:** {data.get('rewards')}")
                                if data.get('notes'):
                                    st.markdown(f"**DM Notes:** {data.get('notes')}")
                            
                            with col2:
                                # Update status button
                                new_status = st.selectbox(
                                    "Update Status",
                                    ["Not Started", "Active", "Completed", "Failed", "Abandoned"],
                                    index=["Not Started", "Active", "Completed", "Failed", "Abandoned"].index(data.get('status', 'Active')),
                                    key=f"status_{quest['id']}"
                                )
                                
                                if new_status != data.get('status'):
                                    data['status'] = new_status
                                    try:
                                        APIClient.update_campaign_element(quest['id'], data)
                                        st.success("Updated!")
                                        time.sleep(0.5)
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Error: {str(e)}")
                                
                                if st.button("🗑️ Delete", key=f"del_quest_{quest['id']}"):
                                    try:
                                        APIClient.delete_campaign_element(quest['id'])
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Error: {str(e)}")
                else:
                    st.info("No quests yet. Create your first adventure!")
            except Exception as e:
                st.error(f"Error loading quests: {str(e)}")
        
        # ===================================================================
        # MONSTERS TAB
        # ===================================================================
        with tabs[4]:
            st.subheader("👹 Monsters & Enemies")
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown("**Populate your world with fearsome creatures.**")
            with col2:
                if st.button("🤖 AI Generate Monster", use_container_width=True, type="primary"):
                    with st.spinner("🤖 Creating monster..."):
                        try:
                            result = APIClient.ai_generate_content(
                                st.session_state.current_campaign,
                                "monster",
                                "Generate a unique monster with stats, abilities, and tactical notes"
                            )
                            st.info(result.get("content", ""))
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
            
            with st.expander("➕ Add Monster", expanded=False):
                with st.form("new_monster"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        monster_name = st.text_input("Name*", placeholder="e.g., Shadow Drake")
                        monster_type = st.text_input("Type", placeholder="e.g., Dragon, Undead, Aberration")
                        monster_cr = st.text_input("CR", placeholder="Challenge Rating")
                    
                    with col2:
                        monster_size = st.selectbox("Size", ["Tiny", "Small", "Medium", "Large", "Huge", "Gargantuan"])
                        monster_alignment = st.text_input("Alignment", placeholder="e.g., Chaotic Evil")
                        monster_habitat = st.text_input("Habitat", placeholder="Where is it found?")
                    
                    monster_description = st.text_area("Description", placeholder="Appearance and behavior...", height=80)
                    monster_abilities = st.text_area("Abilities", placeholder="Special attacks, resistances, abilities...", height=80)
                    monster_tactics = st.text_area("Tactics", placeholder="How does it fight? What's its strategy?", height=80)
                    monster_loot = st.text_area("Loot/Treasure", placeholder="What does it carry or guard?", height=60)
                    
                    if st.form_submit_button("Add Monster", use_container_width=True, type="primary"):
                        if monster_name:
                            try:
                                APIClient.create_campaign_element(
                                    st.session_state.current_campaign,
                                    "monster",
                                    {
                                        "name": monster_name,
                                        "type": monster_type,
                                        "cr": monster_cr,
                                        "size": monster_size,
                                        "alignment": monster_alignment,
                                        "habitat": monster_habitat,
                                        "description": monster_description,
                                        "abilities": monster_abilities,
                                        "tactics": monster_tactics,
                                        "loot": monster_loot
                                    }
                                )
                                st.success("✅ Monster added!")
                                time.sleep(1)
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error: {str(e)}")
                        else:
                            st.warning("Name is required")
            
            # Display monsters
            st.divider()
            try:
                monsters = APIClient.list_campaign_elements(st.session_state.current_campaign, "monster")
                
                if monsters:
                    for monster in monsters:
                        data = monster['element_data']
                        
                        with st.expander(f"👹 {data.get('name', 'Unnamed')} - CR {data.get('cr', '?')}"):
                            col1, col2 = st.columns([4, 1])
                            
                            with col1:
                                st.markdown(f"**Type:** {data.get('type', 'Unknown')} | **Size:** {data.get('size', 'Medium')} | **CR:** {data.get('cr', 'Unknown')}")
                                st.markdown(f"**Alignment:** {data.get('alignment', 'Unknown')} | **Habitat:** {data.get('habitat', 'Unknown')}")
                                
                                if data.get('description'):
                                    st.markdown(f"**Description:** {data.get('description')}")
                                if data.get('abilities'):
                                    st.markdown(f"**Abilities:** {data.get('abilities')}")
                                if data.get('tactics'):
                                    st.markdown(f"**Tactics:** {data.get('tactics')}")
                                if data.get('loot'):
                                    st.markdown(f"**Loot:** {data.get('loot')}")
                            
                            with col2:
                                if st.button("🗑️ Delete", key=f"del_monster_{monster['id']}"):
                                    try:
                                        APIClient.delete_campaign_element(monster['id'])
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Error: {str(e)}")
                else:
                    st.info("No monsters yet. Add some creatures to your world!")
            except Exception as e:
                st.error(f"Error loading monsters: {str(e)}")

elif page == "💬 Chat":
    st.title("💬 Chat with the DM")
    
    if not st.session_state.current_campaign or not st.session_state.current_session:
        st.warning("⚠️ Please select a campaign and session from the sidebar first.")
    else:
        # Load messages
        try:
            messages = APIClient.get_messages(
                st.session_state.current_campaign,
                st.session_state.current_session,
                limit=100
            )
            
            # Display chat history
            chat_container = st.container()
            with chat_container:
                if messages:
                    for msg in messages:
                        display_message(msg)
                else:
                    st.info("🎭 The adventure begins... Send a message to start!")
            
            # Chat input
            st.divider()
            
            col1, col2 = st.columns([5, 1])
            
            with col1:
                user_message = st.text_input(
                    "Your message",
                    placeholder="What do you do?",
                    key="chat_input",
                    label_visibility="collapsed"
                )
            
            with col2:
                send_button = st.button("Send", use_container_width=True, type="primary")
            
            if send_button and user_message:
                try:
                    with st.spinner("🎭 DM is thinking..."):
                        # Send message
                        response = APIClient.send_message(
                            st.session_state.current_campaign,
                            st.session_state.current_session,
                            user_message,
                            st.session_state.player_name
                        )
                        
                        # Rerun to show new messages
                        time.sleep(0.5)
                        st.rerun()
                        
                except Exception as e:
                    st.error(f"Error sending message: {str(e)}")
            
            # Quick actions
            st.divider()
            st.subheader("Quick Actions")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if st.button("🔍 Perception Check", use_container_width=True):
                    st.session_state.quick_action = "I make a perception check"
                    st.rerun()
            
            with col2:
                if st.button("🗡️ Attack", use_container_width=True):
                    st.session_state.quick_action = "I attack with my weapon"
                    st.rerun()
            
            with col3:
                if st.button("💬 Talk", use_container_width=True):
                    st.session_state.quick_action = "I want to talk to them"
                    st.rerun()
            
            with col4:
                if st.button("🏃 Run", use_container_width=True):
                    st.session_state.quick_action = "I try to escape"
                    st.rerun()
                    
        except Exception as e:
            st.error(f"Error loading chat: {str(e)}")

elif page == "🎲 Dice Roller":
    st.title("🎲 Dice Roller")
    
    # Common dice presets
    st.subheader("Quick Roll")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🎲 d20", use_container_width=True, type="primary"):
            result = roll_dice_local("1d20")
            st.session_state.last_roll = ("1d20", result)
    
    with col2:
        if st.button("🎲 d12", use_container_width=True):
            result = roll_dice_local("1d12")
            st.session_state.last_roll = ("1d12", result)
    
    with col3:
        if st.button("🎲 d10", use_container_width=True):
            result = roll_dice_local("1d10")
            st.session_state.last_roll = ("1d10", result)
    
    with col4:
        if st.button("🎲 d8", use_container_width=True):
            result = roll_dice_local("1d8")
            st.session_state.last_roll = ("1d8", result)
    
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        if st.button("🎲 d6", use_container_width=True):
            result = roll_dice_local("1d6")
            st.session_state.last_roll = ("1d6", result)
    
    with col6:
        if st.button("🎲 d4", use_container_width=True):
            result = roll_dice_local("1d4")
            st.session_state.last_roll = ("1d4", result)
    
    with col7:
        if st.button("🎲 2d6", use_container_width=True):
            result = roll_dice_local("2d6")
            st.session_state.last_roll = ("2d6", result)
    
    with col8:
        if st.button("🎲 d100", use_container_width=True):
            result = roll_dice_local("1d100")
            st.session_state.last_roll = ("1d100", result)
    
    st.divider()
    
    # Custom roll
    st.subheader("Custom Roll")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        dice_notation = st.text_input(
            "Dice Notation",
            placeholder="e.g., 2d6+3, 1d20, 3d8-2",
            help="Format: NdN+M where N is number of dice, N is dice size, M is modifier"
        )
    
    with col2:
        st.write("")  # Spacing
        st.write("")  # Spacing
        if st.button("Roll!", use_container_width=True, type="primary"):
            if dice_notation:
                result = roll_dice_local(dice_notation)
                st.session_state.last_roll = (dice_notation, result)
    
    # Display last roll
    if st.session_state.last_roll:
        notation, result = st.session_state.last_roll
        display_dice_roll(notation, result)
        
        # Send to DM if in a session
        if st.session_state.current_campaign and st.session_state.current_session:
            if st.button("📤 Send to DM", use_container_width=True):
                try:
                    APIClient.send_message(
                        st.session_state.current_campaign,
                        st.session_state.current_session,
                        f"I rolled {notation}: {result}",
                        st.session_state.player_name
                    )
                    st.success("Sent to DM!")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    st.divider()
    
    # Common skill checks with modifiers
    st.subheader("Skill Checks")
    
    col1, col2 = st.columns(2)
    
    with col1:
        skill = st.selectbox(
            "Skill",
            ["Athletics", "Acrobatics", "Stealth", "Perception", "Investigation",
             "Arcana", "History", "Religion", "Nature", "Medicine",
             "Insight", "Survival", "Persuasion", "Deception", "Intimidation", "Performance"]
        )
    
    with col2:
        modifier = st.number_input("Modifier", min_value=-10, max_value=20, value=0)
    
    if st.button(f"Roll {skill} Check", use_container_width=True, type="primary"):
        notation = f"1d20{'+' if modifier >= 0 else ''}{modifier}"
        result = roll_dice_local(notation)
        st.session_state.last_roll = (f"{skill} Check ({notation})", result)
        st.rerun()

elif page == "📖 Lookup":
    st.title("📖 Rule Lookup")
    
    st.markdown("""
    Search for D&D 5e rules, spells, monsters, items, and more!
    """)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query = st.text_input(
            "Search",
            placeholder="e.g., fireball, beholder, longsword"
        )
    
    with col2:
        resource_type = st.selectbox(
            "Type",
            ["spell", "monster", "equipment", "class", "race", "feature", "condition"]
        )
    
    if st.button("🔍 Search", use_container_width=True, type="primary"):
        if query:
            try:
                with st.spinner("Searching..."):
                    result = APIClient.lookup_rule(query, resource_type)
                    
                    st.success("Found!")
                    
                    # Display result in a nice format
                    st.markdown(f"""
                    <div class="campaign-card">
                        <h3>📖 {query.title()}</h3>
                        <p><strong>Type:</strong> {resource_type.title()}</p>
                        <hr>
                        {result.get('result', 'No information found')}
                    </div>
                    """, unsafe_allow_html=True)
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")
        else:
            st.warning("Please enter a search query")
    
    st.divider()
    
    # Quick reference
    st.subheader("Quick Reference")
    
    with st.expander("🎯 Common Actions"):
        st.markdown("""
        - **Attack:** Roll 1d20 + Attack Modifier vs. AC
        - **Spell Attack:** Roll 1d20 + Spell Attack Modifier vs. AC
        - **Saving Throw:** Roll 1d20 + Ability Modifier vs. DC
        - **Skill Check:** Roll 1d20 + Skill Modifier vs. DC
        """)
    
    with st.expander("⚔️ Combat"):
        st.markdown("""
        - **Initiative:** Roll 1d20 + Dexterity Modifier
        - **Critical Hit:** Natural 20 on attack roll (double damage dice)
        - **Critical Fail:** Natural 1 on attack roll (automatic miss)
        - **Advantage:** Roll 2d20, take higher result
        - **Disadvantage:** Roll 2d20, take lower result
        """)
    
    with st.expander("💫 Conditions"):
        st.markdown("""
        - **Blinded:** Disadvantage on attacks, attacks against have advantage
        - **Prone:** Disadvantage on attacks, melee attacks against have advantage
        - **Restrained:** Speed 0, disadvantage on attacks and Dex saves
        - **Stunned:** Incapacitated, auto-fail Str/Dex saves, attacks have advantage
        """)

elif page == "👤 Character":
    st.title("👤 Character Sheet")
    
    if not st.session_state.current_campaign:
        st.warning("⚠️ Please select a campaign from the sidebar first.")
    else:
        # Load character from backend on first visit
        if st.session_state.character is None:
            try:
                # Try to load existing character
                loaded_char = APIClient.get_character(
                    st.session_state.current_campaign,
                    st.session_state.player_name
                )
                
                if loaded_char:
                    st.session_state.character = loaded_char['character_data']
                    st.success(f"✅ Loaded character: {st.session_state.character['name']}")
                else:
                    # Create new character
                    st.session_state.character = {
                        "name": st.session_state.player_name,
                        "class": "Fighter",
                        "level": 1,
                        "race": "Human",
                        "max_hp": 10,
                        "current_hp": 10,
                        "ac": 10,
                        "stats": {
                            "STR": 10, "DEX": 10, "CON": 10,
                            "INT": 10, "WIS": 10, "CHA": 10
                        },
                        "inventory": [],
                        "spells": [],
                        "features": [],
                        "background": "",
                        "notes": ""
                    }
                    st.info("📝 New character sheet created")
            except Exception as e:
                st.error(f"Error loading character: {str(e)}")
        # Character Header
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.session_state.character["name"] = st.text_input(
                "Character Name",
                value=st.session_state.character["name"],
                key="char_name_input"
            )
        
        with col2:
            st.session_state.character["class"] = st.selectbox(
                "Class",
                ["Barbarian", "Bard", "Cleric", "Druid", "Fighter", "Monk", 
                 "Paladin", "Ranger", "Rogue", "Sorcerer", "Warlock", "Wizard"],
                index=["Barbarian", "Bard", "Cleric", "Druid", "Fighter", "Monk", 
                       "Paladin", "Ranger", "Rogue", "Sorcerer", "Warlock", "Wizard"].index(
                    st.session_state.character["class"]
                ) if st.session_state.character["class"] in ["Barbarian", "Bard", "Cleric", "Druid", "Fighter", "Monk", "Paladin", "Ranger", "Rogue", "Sorcerer", "Warlock", "Wizard"] else 0
            )
        
        with col3:
            st.session_state.character["level"] = st.number_input(
                "Level",
                min_value=1,
                max_value=20,
                value=st.session_state.character["level"]
            )
        
        st.divider()
        
        # Core Stats
        st.subheader("⚔️ Combat Stats")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Armor Class", st.session_state.character["ac"])
            new_ac = st.number_input("Update AC", min_value=1, max_value=30, 
                                     value=st.session_state.character["ac"], 
                                     key="ac_input", label_visibility="collapsed")
            if new_ac != st.session_state.character["ac"]:
                st.session_state.character["ac"] = new_ac
        
        with col2:
            hp_percent = (st.session_state.character["current_hp"] / st.session_state.character["max_hp"]) * 100
            hp_color = "🟢" if hp_percent > 50 else "🟡" if hp_percent > 25 else "🔴"
            st.metric("Hit Points", f"{hp_color} {st.session_state.character['current_hp']}/{st.session_state.character['max_hp']}")
            
            # HP adjustment buttons
            subcol1, subcol2 = st.columns(2)
            with subcol1:
                heal = st.number_input("Heal", min_value=0, value=0, key="heal_input")
                if st.button("+ Heal", use_container_width=True):
                    st.session_state.character["current_hp"] = min(
                        st.session_state.character["current_hp"] + heal,
                        st.session_state.character["max_hp"]
                    )
                    st.rerun()
            
            with subcol2:
                damage = st.number_input("Damage", min_value=0, value=0, key="damage_input")
                if st.button("- Damage", use_container_width=True, type="secondary"):
                    st.session_state.character["current_hp"] = max(
                        st.session_state.character["current_hp"] - damage,
                        0
                    )
                    st.rerun()
        
        with col3:
            st.metric("Max HP", st.session_state.character["max_hp"])
            new_max_hp = st.number_input("Update Max HP", min_value=1, max_value=500,
                                         value=st.session_state.character["max_hp"],
                                         key="max_hp_input", label_visibility="collapsed")
            if new_max_hp != st.session_state.character["max_hp"]:
                st.session_state.character["max_hp"] = new_max_hp
                st.session_state.character["current_hp"] = min(
                    st.session_state.character["current_hp"],
                    new_max_hp
                )
        
        with col4:
            if st.button("🔄 Long Rest", use_container_width=True):
                st.session_state.character["current_hp"] = st.session_state.character["max_hp"]
                st.success("Fully healed!")
                st.rerun()
        
        st.divider()
        
        # Ability Scores
        st.subheader("📊 Ability Scores")
        
        cols = st.columns(6)
        abilities = ["STR", "DEX", "CON", "INT", "WIS", "CHA"]
        
        for i, ability in enumerate(abilities):
            with cols[i]:
                score = st.session_state.character["stats"][ability]
                modifier = (score - 10) // 2
                mod_str = f"+{modifier}" if modifier >= 0 else str(modifier)
                
                st.markdown(f"""
                <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); border-radius: 8px; border: 2px solid #4a5568;">
                    <div style="font-size: 0.9rem; color: #a0aec0; margin-bottom: 0.5rem;">{ability}</div>
                    <div style="font-size: 2rem; font-weight: bold; color: #f0f0f0;">{score}</div>
                    <div style="font-size: 1.2rem; color: #667eea; margin-top: 0.5rem;">{mod_str}</div>
                </div>
                """, unsafe_allow_html=True)
                
                new_score = st.number_input(
                    f"{ability}",
                    min_value=1,
                    max_value=30,
                    value=score,
                    key=f"stat_{ability}",
                    label_visibility="collapsed"
                )
                if new_score != score:
                    st.session_state.character["stats"][ability] = new_score
        
        st.divider()
        
        # Inventory
        st.subheader("🎒 Inventory")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            new_item = st.text_input("Add Item", placeholder="e.g., Longsword, Health Potion")
        
        with col2:
            st.write("")
            st.write("")
            if st.button("➕ Add", use_container_width=True):
                if new_item:
                    if "inventory" not in st.session_state.character:
                        st.session_state.character["inventory"] = []
                    st.session_state.character["inventory"].append({
                        "name": new_item,
                        "quantity": 1,
                        "equipped": False
                    })
                    st.rerun()
        
        # Display inventory
        if st.session_state.character.get("inventory"):
            for i, item in enumerate(st.session_state.character["inventory"]):
                col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                
                with col1:
                    equipped_icon = "✅" if item.get("equipped", False) else "⬜"
                    st.write(f"{equipped_icon} **{item['name']}**")
                
                with col2:
                    st.write(f"Qty: {item.get('quantity', 1)}")
                
                with col3:
                    if st.button("Equip" if not item.get("equipped") else "Unequip", key=f"equip_{i}"):
                        st.session_state.character["inventory"][i]["equipped"] = not item.get("equipped", False)
                        st.rerun()
                
                with col4:
                    if st.button("🗑️", key=f"delete_item_{i}"):
                        st.session_state.character["inventory"].pop(i)
                        st.rerun()
        else:
            st.info("No items in inventory. Add items above!")
        
        st.divider()
        
        # Character Notes & Background
        st.subheader("📝 Character Notes")
        
        tab1, tab2, tab3 = st.tabs(["Background", "Notes", "AI Summary"])
        
        with tab1:
            st.session_state.character["background"] = st.text_area(
                "Character Background",
                value=st.session_state.character.get("background", ""),
                placeholder="Tell the story of your character...",
                height=200,
                key="background_input"
            )
        
        with tab2:
            st.session_state.character["notes"] = st.text_area(
                "Campaign Notes",
                value=st.session_state.character.get("notes", ""),
                placeholder="Keep track of important information, NPCs, quests, etc.",
                height=200,
                key="notes_input"
            )
        
        with tab3:
            st.write("Get an AI-generated summary of your character's journey!")
            
            if st.button("🤖 Generate Summary", use_container_width=True, type="primary"):
                if st.session_state.current_campaign and st.session_state.current_session:
                    try:
                        with st.spinner("Generating character summary..."):
                            # Get recent messages for context
                            messages = APIClient.get_messages(
                                st.session_state.current_campaign,
                                st.session_state.current_session,
                                limit=50
                            )
                            
                            # Create summary request
                            summary_request = f"""Generate a brief character summary for {st.session_state.character['name']}, a level {st.session_state.character['level']} {st.session_state.character['class']}. 
                            
Current Status:
- HP: {st.session_state.character['current_hp']}/{st.session_state.character['max_hp']}
- Inventory: {', '.join([item['name'] for item in st.session_state.character.get('inventory', [])])}
- Background: {st.session_state.character.get('background', 'Unknown')}

Based on recent campaign events, provide a 2-3 paragraph summary of their journey, current state, and character development."""
                            
                            response = APIClient.send_message(
                                st.session_state.current_campaign,
                                st.session_state.current_session,
                                summary_request,
                                "System"
                            )
                            
                            st.markdown(f"""
                            <div class="dm-message">
                                <div class="message-header dm-header">🤖 Character Summary</div>
                                <div>{response['content']}</div>
                            </div>
                            """, unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Error generating summary: {str(e)}")
                else:
                    st.warning("Select a campaign and session first")
        
        st.divider()
        
        # Save/Load Character
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("💾 Save Character", use_container_width=True, type="primary"):
                try:
                    with st.spinner("Saving character..."):
                        APIClient.save_character(
                            st.session_state.current_campaign,
                            st.session_state.character['name'],
                            st.session_state.character
                        )
                        st.success("✅ Character saved to database!")
                except Exception as e:
                    st.error(f"Error saving character: {str(e)}")
        
        with col2:
            if st.button("📥 Export JSON", use_container_width=True):
                import json
                char_json = json.dumps(st.session_state.character, indent=2)
                st.download_button(
                    "Download Character",
                    char_json,
                    file_name=f"{st.session_state.character['name']}_character.json",
                    mime="application/json",
                    use_container_width=True
                )
        
        with col3:
            if st.button("🔄 Reset HP", use_container_width=True):
                st.session_state.character["current_hp"] = st.session_state.character["max_hp"]
                st.success("HP restored!")
                st.rerun()
        
        # Load Other Characters
        st.divider()
        with st.expander("📚 Load Other Characters"):
            try:
                other_chars = APIClient.list_campaign_characters(st.session_state.current_campaign)
                
                if other_chars:
                    st.write("**Characters in this campaign:**")
                    for char in other_chars:
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            char_data = char['character_data']
                            st.write(f"**{char_data['name']}** - Level {char_data['level']} {char_data['class']}")
                        with col2:
                            if st.button("Load", key=f"load_{char['id']}", use_container_width=True):
                                st.session_state.character = char_data
                                st.session_state.player_name = char['player_name']
                                st.success(f"Loaded {char_data['name']}!")
                                st.rerun()
                else:
                    st.info("No other characters in this campaign")
            except Exception as e:
                st.error(f"Error loading characters: {str(e)}")

elif page == "⚔️ Combat":
    st.title("⚔️ Combat Tracker")
    
    if not st.session_state.current_campaign or not st.session_state.current_session:
        st.warning("⚠️ Please select a campaign and session first.")
    else:
        st.markdown("""
        Track combat encounters, initiative, and actions.
        """)
        
        # Initiative tracker
        st.subheader("Initiative Order")
        
        if "initiative_order" not in st.session_state:
            st.session_state.initiative_order = []
        
        # Add combatant
        with st.expander("➕ Add Combatant"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                combatant_name = st.text_input("Name", placeholder="Character/Monster")
            
            with col2:
                initiative = st.number_input("Initiative", min_value=0, max_value=30, value=10)
            
            with col3:
                hp = st.number_input("HP", min_value=1, max_value=500, value=20)
            
            if st.button("Add", use_container_width=True):
                if combatant_name:
                    st.session_state.initiative_order.append({
                        "name": combatant_name,
                        "initiative": initiative,
                        "hp": hp,
                        "max_hp": hp
                    })
                    st.session_state.initiative_order.sort(key=lambda x: x["initiative"], reverse=True)
                    st.success(f"Added {combatant_name}!")
                    st.rerun()
        
        # Display initiative order
        if st.session_state.initiative_order:
            for i, combatant in enumerate(st.session_state.initiative_order):
                col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
                
                with col1:
                    st.markdown(f"**{i+1}. {combatant['name']}**")
                
                with col2:
                    st.markdown(f"Initiative: **{combatant['initiative']}**")
                
                with col3:
                    hp_percent = (combatant['hp'] / combatant['max_hp']) * 100
                    color = "🟢" if hp_percent > 50 else "🟡" if hp_percent > 25 else "🔴"
                    st.markdown(f"{color} HP: **{combatant['hp']}/{combatant['max_hp']}**")
                
                with col4:
                    if st.button("❌", key=f"remove_{i}"):
                        st.session_state.initiative_order.pop(i)
                        st.rerun()
            
            # Combat actions
            st.divider()
            st.subheader("Combat Actions")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🔄 Next Turn", use_container_width=True, type="primary"):
                    # Rotate initiative
                    if st.session_state.initiative_order:
                        first = st.session_state.initiative_order.pop(0)
                        st.session_state.initiative_order.append(first)
                        st.rerun()
            
            with col2:
                if st.button("🎲 Roll Initiative", use_container_width=True):
                    for combatant in st.session_state.initiative_order:
                        combatant['initiative'] = random.randint(1, 20)
                    st.session_state.initiative_order.sort(key=lambda x: x["initiative"], reverse=True)
                    st.rerun()
            
            with col3:
                if st.button("🗑️ Clear All", use_container_width=True):
                    st.session_state.initiative_order = []
                    st.rerun()
        else:
            st.info("No combatants in initiative order. Add some above!")

# ============================================================================
# Footer
# ============================================================================

st.divider()
st.markdown("""
<div style="text-align: center; opacity: 0.6; padding: 2rem;">
    🎲 Powered by Multi-Agent D&D System | Made with ❤️ and Streamlit
</div>
""", unsafe_allow_html=True)