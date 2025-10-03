#!/bin/bash
# PostgreSQL Database Backup Script for Truvo Intent Database

echo "Creating backup of truvo_intent database..."

# Create backup filename with timestamp
BACKUP_FILE="truvo_intent_backup_$(date +%Y%m%d_%H%M%S).sql"
BACKUP_DIR="/mnt/c/Users/deb0p/truvo/backups"

# Create backup directory if it doesn't exist
mkdir -p "$BACKUP_DIR"

# Create the backup
sudo -u postgres pg_dump truvo_intent > "$BACKUP_DIR/$BACKUP_FILE"

if [ $? -eq 0 ]; then
    echo "✅ Backup created successfully: $BACKUP_DIR/$BACKUP_FILE"
    echo "📁 Backup size: $(ls -lh "$BACKUP_DIR/$BACKUP_FILE" | awk '{print $5}')"
else
    echo "❌ Backup failed!"
    exit 1
fi

# Keep only last 5 backups (cleanup old ones)
echo "🧹 Cleaning up old backups (keeping last 5)..."
cd "$BACKUP_DIR"
ls -t truvo_intent_backup_*.sql | tail -n +6 | xargs -r rm --
echo "✅ Backup completed!"