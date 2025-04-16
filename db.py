# db.py

import sqlite3
import os

DB_TYPE = "sqlite"
DB_PATH = os.path.join(os.path.dirname(__file__), "history.db")

def get_connection():
    if DB_TYPE == "sqlite":
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        return conn
    else:
        raise NotImplementedError("Only SQLite is supported in this version.")

def init_db():
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt TEXT NOT NULL,
                response TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
