import sqlite3
import os

DATABASE_PATH = "data/face_embeddings.db"

def init_db():
    """Initialize the SQLite database and create the embeddings table."""
    os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS face_embeddings (
        key TEXT PRIMARY_KEY,
        filename TEXT,
        embedding BLOB,
        image BLOB
    )
""")
    conn.commit()
    conn.close()

# Call this when the application starts
if __name__ == "__main__":
    init_db()