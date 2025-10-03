#!/bin/bash
# PostgreSQL Database Restore Script for Truvo Intent Database

if [ $# -eq 0 ]; then
    echo "Usage: $0 <backup_file.sql>"
    echo "Example: $0 truvo_intent_backup_20251003_104000.sql"
    echo ""
    echo "Available backups:"
    ls -la backups/truvo_intent_backup_*.sql 2>/dev/null || echo "No backups found in backups/ folder"
    exit 1
fi

BACKUP_FILE="$1"

# Check if backup file exists
if [ ! -f "$BACKUP_FILE" ]; then
    # Try in backups folder
    if [ -f "backups/$BACKUP_FILE" ]; then
        BACKUP_FILE="backups/$BACKUP_FILE"
    else
        echo "❌ Backup file not found: $BACKUP_FILE"
        exit 1
    fi
fi

echo "🔄 Restoring database from: $BACKUP_FILE"
echo "⚠️  This will OVERWRITE the current database!"
read -p "Are you sure? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Restore cancelled"
    exit 1
fi

echo "🗑️  Dropping existing database..."
sudo -u postgres dropdb truvo_intent

echo "🆕 Creating new database..."
sudo -u postgres createdb truvo_intent

echo "👤 Recreating user and permissions..."
sudo -u postgres psql -c "CREATE USER truvo WITH PASSWORD 'truv0';" 2>/dev/null || true
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE truvo_intent TO truvo;"
sudo -u postgres psql -d truvo_intent -c "GRANT ALL ON SCHEMA public TO truvo;"
sudo -u postgres psql -d truvo_intent -c "GRANT CREATE ON SCHEMA public TO truvo;"

echo "📥 Restoring data from backup..."
sudo -u postgres psql truvo_intent < "$BACKUP_FILE"

if [ $? -eq 0 ]; then
    echo "✅ Database restored successfully!"
    echo "🔍 Verifying data..."
    
    # Test the connection and count records
    python3 -c "
from ai.intent_database import IntentTrainingDatabase
try:
    db = IntentTrainingDatabase()
    data = db.get_training_data()
    print(f'✅ Verification successful: {len(data)} training examples restored')
except Exception as e:
    print(f'❌ Verification failed: {e}')
"
else
    echo "❌ Restore failed!"
    exit 1
fi