"""
Quick script to check database tables
"""
import sqlite3

conn = sqlite3.connect('ingestion.db')
cursor = conn.cursor()

# Get all tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = cursor.fetchall()

print("\n=== DATABASE TABLES ===")
for table in tables:
    table_name = table[0]
    print(f"\n{table_name}:")
    
    # Get row count
    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    count = cursor.fetchone()[0]
    print(f"  Rows: {count}")
    
    # Show column names
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = cursor.fetchall()
    print(f"  Columns: {', '.join([col[1] for col in columns])}")

conn.close()
print("\n=== DONE ===\n")
