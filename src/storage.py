import sqlite3 
from datetime import datetime 
DB_PATH = "outputs/events.db" 
def init_db(): 
    conn = sqlite3.connect(DB_PATH) 
    conn.execute(""" 
        CREATE TABLE IF NOT EXISTS events ( 
            id INTEGER PRIMARY KEY AUTOINCREMENT, 
            timestamp TEXT, 
            location TEXT, 
            detected_condition TEXT, 
            confidence_score REAL, 
            alert_triggered INTEGER 
        ) 
    """) 
    conn.commit() 
    conn.close() 
def save_event(location, detected_condition, confidence_score, alert_triggered): 
    conn = sqlite3.connect(DB_PATH) 
    conn.execute( 
        "INSERT INTO events (timestamp, location, detected_condition, confidence_score, alert_triggered) VALUES (?, ?, ?, ?, ?)", 
        (datetime.now().isoformat(), location, detected_condition, confidence_score, int(alert_triggered)) 
    )
    conn.commit() 
    conn.close() 
def get_all_events(): 
    conn = sqlite3.connect(DB_PATH) 
    rows = conn.execute("SELECT * FROM events").fetchall() 
    conn.close() 
    return rows 