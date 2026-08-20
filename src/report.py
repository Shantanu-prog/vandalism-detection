import sqlite3 
import pandas as pd 
DB_PATH = "outputs/events.db" 
def export_csv(output_path="outputs/report.csv"): 
    conn = sqlite3.connect(DB_PATH) 
    df = pd.read_sql_query( 
        "SELECT timestamp, location, detected_condition, confidence_score FROM events", 
        conn 
    )
    conn.close() 
    df.to_csv(output_path, index=False) 
    return output_path