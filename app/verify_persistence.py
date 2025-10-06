#!/usr/bin/env python3
"""
Verify Data Persistence
=======================

This script tests if PostgreSQL persistence is working correctly.
Run this to verify your setup before going live.

Usage:
    python verify_persistence.py
"""

import sys
import os

# Colors for output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'

def print_success(msg):
    print(f"{Colors.GREEN}✓ {msg}{Colors.END}")

def print_error(msg):
    print(f"{Colors.RED}✗ {msg}{Colors.END}")

def print_info(msg):
    print(f"{Colors.BLUE}ℹ {msg}{Colors.END}")

def print_warning(msg):
    print(f"{Colors.YELLOW}⚠ {msg}{Colors.END}")

def check_database_module():
    """Check if database module can be imported."""
    print_info("Checking database module...")
    try:
        from backend.database.db_manager import DatabaseManager
        print_success("DatabaseManager imported successfully")
        return True
    except ImportError as e:
        print_error(f"Cannot import DatabaseManager: {e}")
        print_warning("Make sure backend/database/db_manager.py exists")
        print_warning("Make sure backend/database/__init__.py exists")
        return False

def check_database_connection():
    """Check if can connect to PostgreSQL."""
    print_info("Checking database connection...")
    try:
        from backend.database.db_manager import DatabaseManager
        db = DatabaseManager()
        
        if db.health_check():
            print_success("Database connection successful")
            return True, db
        else:
            print_error("Database health check failed")
            return False, None
    except Exception as e:
        print_error(f"Cannot connect to database: {e}")
        print_warning("Make sure PostgreSQL is running: docker ps | grep postgres")
        print_warning("Check .env for correct connection parameters")
        return False, None

def check_tables_exist(db):
    """Check if required tables exist."""
    print_info("Checking if tables exist...")
    try:
        with db.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name IN ('campaigns', 'sessions', 'messages')
                """)
                tables = [row[0] for row in cur.fetchall()]
                
                required_tables = ['campaigns', 'sessions', 'messages']
                missing_tables = set(required_tables) - set(tables)
                
                if not missing_tables:
                    print_success(f"All required tables exist: {', '.join(required_tables)}")
                    return True
                else:
                    print_error(f"Missing tables: {', '.join(missing_tables)}")
                    print_warning("Tables will be created when backend starts")
                    return False
    except Exception as e:
        print_error(f"Error checking tables: {e}")
        return False

def test_crud_operations(db):
    """Test create, read operations."""
    print_info("Testing database operations...")
    
    try:
        # Test Create
        print_info("  Creating test campaign...")
        campaign = db.create_campaign(
            name="Persistence Test Campaign",
            description="Testing if data persists",
            setting="Test World"
        )
        campaign_id = campaign['id']
        print_success(f"  Campaign created with ID: {campaign_id}")
        
        # Test Read
        print_info("  Reading campaign back...")
        retrieved = db.get_campaign(campaign_id)
        if retrieved and retrieved['name'] == "Persistence Test Campaign":
            print_success("  Campaign retrieved successfully")
        else:
            print_error("  Failed to retrieve campaign")
            return False
        
        # Test Session Creation
        print_info("  Creating test session...")
        session = db.create_session(
            campaign_id=campaign_id,
            name="Test Session"
        )
        session_id = session['id']
        print_success(f"  Session created with ID: {session_id}")
        
        # Test Message Creation
        print_info("  Creating test message...")
        message = db.save_message(
            campaign_id=campaign_id,
            session_id=session_id,
            role="player",
            content="This is a test message",
            player_name="TestPlayer"
        )
        print_success(f"  Message created with ID: {message['id']}")
        
        # Test List Operations
        print_info("  Testing list operations...")
        campaigns = db.list_campaigns()
        sessions = db.list_sessions(campaign_id)
        messages = db.get_messages(campaign_id, session_id)
        
        print_success(f"  Found {len(campaigns)} campaign(s)")
        print_success(f"  Found {len(sessions)} session(s)")
        print_success(f"  Found {len(messages)} message(s)")
        
        # Cleanup
        print_info("  Cleaning up test data...")
        db.delete_campaign(campaign_id)
        print_success("  Test data cleaned up")
        
        return True
        
    except Exception as e:
        print_error(f"Error during CRUD test: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_api_endpoint():
    """Check if API is using database."""
    print_info("Checking API endpoint...")
    try:
        import requests
        
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            health = response.json()
            
            if health.get('postgres_connected'):
                print_success("API is using PostgreSQL")
                return True
            else:
                print_warning("API is not connected to PostgreSQL")
                print_warning("Backend might be using in-memory storage")
                return False
        else:
            print_error(f"API returned status {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print_warning("Cannot connect to API at http://localhost:8000")
        print_warning("Make sure backend is running: python main.py")
        return False
    except Exception as e:
        print_error(f"Error checking API: {e}")
        return False

def main():
    """Run all verification checks."""
    print("\n" + "="*60)
    print("D&D Campaign Manager - Persistence Verification")
    print("="*60 + "\n")
    
    all_passed = True
    
    # Check 1: Module import
    if not check_database_module():
        print_error("\nSetup incomplete: Cannot import database module")
        print_info("See PERSISTENCE_SETUP.md for instructions")
        return 1
    
    print()
    
    # Check 2: Database connection
    connected, db = check_database_connection()
    if not connected:
        print_error("\nSetup incomplete: Cannot connect to database")
        print_info("Make sure PostgreSQL is running:")
        print_info("  docker-compose up -d postgres")
        return 1
    
    print()
    
    # Check 3: Tables exist
    tables_ok = check_tables_exist(db)
    if not tables_ok:
        print_info("\nInitializing database...")
        try:
            db.init_database()
            print_success("Database initialized")
        except Exception as e:
            print_error(f"Failed to initialize database: {e}")
            all_passed = False
    
    print()
    
    # Check 4: CRUD operations
    crud_ok = test_crud_operations(db)
    if not crud_ok:
        all_passed = False
    
    print()
    
    # Check 5: API endpoint
    api_ok = check_api_endpoint()
    if not api_ok:
        all_passed = False
    
    # Summary
    print("\n" + "="*60)
    if all_passed:
        print_success("All checks passed! ✨")
        print_success("Data persistence is working correctly")
        print_info("\nYour campaigns will now survive server restarts!")
        print_info("Try it: Create a campaign, restart backend, check it's still there")
        print("="*60 + "\n")
        return 0
    else:
        print_warning("Some checks failed")
        print_info("See errors above for details")
        print_info("Check PERSISTENCE_SETUP.md for troubleshooting")
        print("="*60 + "\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())