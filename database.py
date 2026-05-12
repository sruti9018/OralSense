import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'patients.db')

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    cursor = conn.cursor()
    cursor.executescript('''
        CREATE TABLE IF NOT EXISTS patients (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id    TEXT NOT NULL UNIQUE,
            name          TEXT,
            age           INTEGER,
            sex           TEXT,
            phone         TEXT,
            address       TEXT,
            created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS scans (
            id                      INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id              TEXT NOT NULL,
            scan_date               TEXT,
            doctor                  TEXT,
            department              TEXT,
            clinical_notes          TEXT,
            smoking                 INTEGER DEFAULT 0,
            chewing                 INTEGER DEFAULT 0,
            arecanut                INTEGER DEFAULT 0,
            alcohol                 INTEGER DEFAULT 0,
            img_class               TEXT,
            img_confidence          REAL,
            img_uncertainty         REAL,
            img_uncertainty_label   TEXT,
            multi_class             TEXT,
            multi_confidence        REAL,
            multi_uncertainty       REAL,
            multi_uncertainty_label TEXT,
            risk_key                TEXT,
            gradcam                 TEXT,
            created_at              TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    ''')
    conn.commit()
    conn.close()
    print("Database initialized at:", DB_PATH)

init_db()