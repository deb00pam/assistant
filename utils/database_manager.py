#!/usr/bin/env python3
"""
Database Backup and Restore Utility for Truvo Intent Database
"""

import os
import sys
import subprocess
import datetime
from pathlib import Path
import getpass

def run_command(cmd, shell=True, need_password=False):
    """Run a command and return success status"""
    try:
        if need_password:
            # For commands that need sudo password, run interactively
            result = subprocess.run(cmd, shell=shell, check=True, text=True)
            return True, "Success"
        else:
            result = subprocess.run(cmd, shell=shell, check=True, capture_output=True, text=True)
            return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, str(e)

def backup_database():
    """Create a backup of the database"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = Path("backups")
    backup_dir.mkdir(exist_ok=True)
    
    backup_file = backup_dir / f"truvo_intent_backup_{timestamp}.sql"
    
    print("🔄 Creating database backup...")
    print("📝 Note: You may be prompted for your WSL password")
    
    # Create backup using WSL with interactive sudo
    cmd = f'wsl sudo -u postgres pg_dump truvo_intent > "{backup_file}"'
    success, output = run_command(cmd, need_password=True)
    
    if success and backup_file.exists() and backup_file.stat().st_size > 0:
        size = backup_file.stat().st_size
        print(f"✅ Backup created: {backup_file}")
        print(f"📁 Size: {size:,} bytes")
        
        # Cleanup old backups (keep last 5)
        cleanup_old_backups(backup_dir)
        return str(backup_file)
    else:
        print(f"❌ Backup failed: {output}")
        # Try to remove empty backup file
        if backup_file.exists():
            backup_file.unlink()
        return None

def cleanup_old_backups(backup_dir, keep=5):
    """Keep only the last N backups"""
    backups = sorted(backup_dir.glob("truvo_intent_backup_*.sql"), 
                    key=lambda x: x.stat().st_mtime, reverse=True)
    
    if len(backups) > keep:
        for old_backup in backups[keep:]:
            old_backup.unlink()
            print(f"🗑️ Removed old backup: {old_backup.name}")

def list_backups():
    """List available backups"""
    backup_dir = Path("backups")
    if not backup_dir.exists():
        print("❌ No backups directory found")
        return []
    
    backups = sorted(backup_dir.glob("truvo_intent_backup_*.sql"), 
                    key=lambda x: x.stat().st_mtime, reverse=True)
    
    if not backups:
        print("❌ No backups found")
        return []
    
    print("📋 Available backups:")
    for i, backup in enumerate(backups, 1):
        mtime = datetime.datetime.fromtimestamp(backup.stat().st_mtime)
        size = backup.stat().st_size
        print(f"  {i}. {backup.name} ({mtime.strftime('%Y-%m-%d %H:%M')} - {size:,} bytes)")
    
    return backups

def restore_database(backup_file):
    """Restore database from backup"""
    if not Path(backup_file).exists():
        print(f"❌ Backup file not found: {backup_file}")
        return False
    
    print(f"🔄 Restoring from: {backup_file}")
    print("⚠️  This will OVERWRITE the current database!")
    
    confirm = input("Are you sure? (y/N): ").strip().lower()
    if confirm != 'y':
        print("❌ Restore cancelled")
        return False
    
    print("� Note: You may be prompted for your WSL password")
    
    print("�🗑️ Dropping existing database...")
    run_command('wsl sudo -u postgres dropdb truvo_intent', need_password=True)
    
    print("🆕 Creating new database...")
    success, _ = run_command('wsl sudo -u postgres createdb truvo_intent', need_password=True)
    if not success:
        print("❌ Failed to create database")
        return False
    
    print("👤 Setting up user permissions...")
    commands = [
        "wsl sudo -u postgres psql -c \"GRANT ALL PRIVILEGES ON DATABASE truvo_intent TO truvo;\"",
        "wsl sudo -u postgres psql -d truvo_intent -c \"GRANT ALL ON SCHEMA public TO truvo;\"",
        "wsl sudo -u postgres psql -d truvo_intent -c \"GRANT CREATE ON SCHEMA public TO truvo;\""
    ]
    
    for cmd in commands:
        run_command(cmd, need_password=True)
    
    print("📥 Restoring data...")
    cmd = f'wsl sudo -u postgres psql truvo_intent < "{backup_file}"'
    success, output = run_command(cmd, need_password=True)
    
    if success:
        print("✅ Database restored successfully!")
        
        # Verify the restore
        try:
            from ai.intent_database import IntentTrainingDatabase
            db = IntentTrainingDatabase()
            data = db.get_training_data()
            print(f"🔍 Verification: {len(data)} training examples restored")
            return True
        except Exception as e:
            print(f"⚠️ Verification failed: {e}")
            return False
    else:
        print(f"❌ Restore failed: {output}")
        return False

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python database_manager.py backup          # Create a backup")
        print("  python database_manager.py list            # List available backups")
        print("  python database_manager.py restore <file>  # Restore from backup")
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    if command == "backup":
        backup_file = backup_database()
        if backup_file:
            print(f"\n💾 Backup saved to: {backup_file}")
    
    elif command == "list":
        list_backups()
    
    elif command == "restore":
        if len(sys.argv) < 3:
            print("❌ Please specify a backup file")
            list_backups()
            sys.exit(1)
        
        backup_file = sys.argv[2]
        # If no path, look in backups folder
        if not os.path.dirname(backup_file):
            backup_file = f"backups/{backup_file}"
        
        restore_database(backup_file)
    
    else:
        print(f"❌ Unknown command: {command}")
        sys.exit(1)

if __name__ == "__main__":
    main()