import sqlite3
import os
from datetime import datetime
from contextlib import contextmanager

DB_PATH = os.getenv('DB_PATH', 'traffic_watcher.db')

def init_db():
    """Initialize the database with required tables"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS observations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            observation_number INTEGER NOT NULL,
            vehicle_type TEXT NOT NULL,
            timestamp DATETIME NOT NULL,
            direction TEXT NOT NULL,
            duration REAL NOT NULL,
            distance REAL NOT NULL,
            speed REAL NOT NULL,
            image_filename TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create index for faster queries
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_timestamp ON observations(timestamp)
    ''')
    
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_observation_number ON observations(observation_number)
    ''')
    
    conn.commit()
    conn.close()
    print(f"Database initialized at {DB_PATH}")

@contextmanager
def get_db_connection():
    """Context manager for database connections"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

def add_observation(observation_number, vehicle_type, timestamp, direction, duration, distance, speed, image_filename):
    """Add a new observation to the database"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO observations 
            (observation_number, vehicle_type, timestamp, direction, duration, distance, speed, image_filename)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (observation_number, vehicle_type, timestamp, direction, duration, distance, speed, image_filename))
        conn.commit()
        return cursor.lastrowid

def get_all_observations(limit=100, offset=0):
    """Get all observations with pagination"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM observations 
            ORDER BY timestamp DESC 
            LIMIT ? OFFSET ?
        ''', (limit, offset))
        return [dict(row) for row in cursor.fetchall()]

def get_observation_by_id(obs_id):
    """Get a single observation by ID"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM observations WHERE id = ?', (obs_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

def get_total_count():
    """Get total count of observations"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) as count FROM observations')
        return cursor.fetchone()['count']

def get_hourly_averages():
    """Get average vehicles per hour of day"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT 
                strftime('%H', timestamp) as hour,
                COUNT(*) as count,
                COUNT(DISTINCT DATE(timestamp)) as days
            FROM observations
            GROUP BY hour
            ORDER BY hour
        ''')
        results = cursor.fetchall()
        
        # Calculate averages
        hourly_data = []
        for row in results:
            hour = int(row['hour'])
            count = row['count']
            days = row['days']
            avg = count / days if days > 0 else 0
            hourly_data.append({
                'hour': hour,
                'average': round(avg, 2),
                'total': count
            })
        
        return hourly_data

def get_stats():
    """Get summary statistics"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Total count
        cursor.execute('SELECT COUNT(*) as total FROM observations')
        total = cursor.fetchone()['total']
        
        # Average speed
        cursor.execute('SELECT AVG(speed) as avg_speed FROM observations')
        avg_speed = cursor.fetchone()['avg_speed'] or 0
        
        # Count by direction
        cursor.execute('''
            SELECT direction, COUNT(*) as count 
            FROM observations 
            GROUP BY direction
        ''')
        by_direction = {row['direction']: row['count'] for row in cursor.fetchall()}
        
        # Count by vehicle type
        cursor.execute('''
            SELECT vehicle_type, COUNT(*) as count 
            FROM observations 
            GROUP BY vehicle_type
        ''')
        by_type = {row['vehicle_type']: row['count'] for row in cursor.fetchall()}
        
        # Recent observations (last 24 hours)
        cursor.execute('''
            SELECT COUNT(*) as count 
            FROM observations 
            WHERE timestamp >= datetime('now', '-1 day')
        ''')
        last_24h = cursor.fetchone()['count']
        
        return {
            'total': total,
            'average_speed': round(avg_speed, 2),
            'by_direction': by_direction,
            'by_type': by_type,
            'last_24h': last_24h
        }

if __name__ == '__main__':
    init_db()
