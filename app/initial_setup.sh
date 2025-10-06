#!/bin/bash

# Enable Data Persistence for D&D Campaign Manager
# This script sets up PostgreSQL persistence

set -e

echo "🎲 Enabling Data Persistence"
echo "=============================="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Step 1: Create database directory structure
echo "Step 1: Creating database directory..."
if [ ! -d "backend/database" ]; then
    mkdir -p backend/database
    touch backend/database/__init__.py
    print_success "Created backend/database/ directory"
else
    print_info "backend/database/ already exists"
fi

# Step 2: Check if db_manager.py exists
echo ""
echo "Step 2: Checking for db_manager.py..."
if [ ! -f "backend/database/db_manager.py" ]; then
    print_error "db_manager.py not found!"
    echo ""
    echo "Please create backend/database/db_manager.py with the DatabaseManager code"
    echo "See the 'database/db_manager.py' artifact for the code"
    exit 1
else
    print_success "db_manager.py found"
fi

# Step 3: Check PostgreSQL
echo ""
echo "Step 3: Checking PostgreSQL..."
if docker ps | grep -q dnd_postgres; then
    print_success "PostgreSQL is running"
else
    print_info "Starting PostgreSQL..."
    docker-compose up -d postgres
    print_info "Waiting for PostgreSQL to initialize..."
    sleep 5
    print_success "PostgreSQL started"
fi

# Step 4: Install dependencies
echo ""
echo "Step 4: Checking Python dependencies..."
if python -c "import psycopg2" 2>/dev/null; then
    print_success "psycopg2 is installed"
else
    print_info "Installing psycopg2..."
    pip install psycopg2-binary
    print_success "psycopg2 installed"
fi

# Step 5: Test database connection
echo ""
echo "Step 5: Testing database connection..."
if docker exec dnd_postgres psql -U postgres -d dnd_campaign -c "SELECT 1;" > /dev/null 2>&1; then
    print_success "Database connection successful"
else
    print_error "Cannot connect to database"
    print_info "Try restarting PostgreSQL: docker-compose restart postgres"
    exit 1
fi

# Step 6: Test Python import
echo ""
echo "Step 6: Testing DatabaseManager import..."
cd backend
if python -c "from database.db_manager import DatabaseManager; print('OK')" > /dev/null 2>&1; then
    print_success "DatabaseManager imports successfully"
else
    print_error "Cannot import DatabaseManager"
    print_info "Check that backend/database/__init__.py exists"
    print_info "Check that backend/database/db_manager.py is valid Python code"
    exit 1
fi
cd ..

# Step 7: Run verification script
echo ""
echo "Step 7: Running full verification..."
if [ -f "verify_persistence.py" ]; then
    python verify_persistence.py
else
    print_info "Skipping verification (verify_persistence.py not found)"
fi

echo ""
echo "=============================="
print_success "Setup complete!"
echo "=============================="
echo ""
echo "Next steps:"
echo "1. Restart your backend: python backend/main.py"
echo "2. Look for: '✅ Database initialized successfully'"
echo "3. Create a campaign in Streamlit"
echo "4. Restart backend again"
echo "5. Verify campaign still exists!"
echo ""
echo "Your data will now persist across restarts! 🎉"
echo ""