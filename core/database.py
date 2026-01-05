"""
Database Manager for storing violation records
Supports both SQLite and CSV storage
"""

import sqlite3
import pandas as pd
import os
from datetime import datetime
import config


class ViolationDatabase:
    """Manage violation records in database"""
    
    def __init__(self, use_sqlite=True):
        """
        Initialize database
        
        Args:
            use_sqlite: If True, use SQLite; otherwise use CSV
        """
        self.use_sqlite = use_sqlite
        
        # Create data directory if not exists
        os.makedirs('data', exist_ok=True)
        
        if self.use_sqlite:
            self._init_sqlite()
        else:
            self._init_csv()
    
    def _init_sqlite(self):
        """Initialize SQLite database"""
        self.conn = sqlite3.connect(config.DB_NAME, check_same_thread=False)
        self.cursor = self.conn.cursor()
        
        # Create table if not exists
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS violations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                violation_type TEXT NOT NULL,
                violation_name TEXT,
                number_plate TEXT,
                confidence REAL,
                severity TEXT,
                fine INTEGER,
                image_path TEXT
            )
        ''')
        self.conn.commit()
    
    def _init_csv(self):
        """Initialize CSV file"""
        self.csv_path = config.CSV_NAME
        
        # Create CSV with headers if not exists
        if not os.path.exists(self.csv_path):
            df = pd.DataFrame(columns=[
                'timestamp', 'violation_type', 'violation_name', 
                'number_plate', 'confidence', 'severity', 'fine', 'image_path'
            ])
            df.to_csv(self.csv_path, index=False)
    
    def add_violation(self, violation_type, number_plate='N/A', confidence=0.0, 
                     image_path='', violation_name='', severity='', fine=0):
        """
        Add a violation record
        
        Args:
            violation_type: Type code (e.g., 'HELMETLESS')
            number_plate: Extracted plate number
            confidence: Detection confidence
            image_path: Path to image file
            violation_name: Display name
            severity: Severity level
            fine: Fine amount
        
        Returns:
            Record ID
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        record = {
            'timestamp': timestamp,
            'violation_type': violation_type,
            'violation_name': violation_name,
            'number_plate': number_plate,
            'confidence': round(confidence, 2),
            'severity': severity,
            'fine': fine,
            'image_path': image_path
        }
        
        if self.use_sqlite:
            self.cursor.execute('''
                INSERT INTO violations 
                (timestamp, violation_type, violation_name, number_plate, 
                 confidence, severity, fine, image_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (timestamp, violation_type, violation_name, number_plate, 
                  confidence, severity, fine, image_path))
            self.conn.commit()
            return self.cursor.lastrowid
        else:
            # Append to CSV
            df = pd.DataFrame([record])
            df.to_csv(self.csv_path, mode='a', header=False, index=False)
            return None
    
    def get_all_violations(self, limit=100):
        """
        Get all violation records
        
        Args:
            limit: Maximum number of records to return
        
        Returns:
            DataFrame of violations
        """
        if self.use_sqlite:
            query = f'''
                SELECT * FROM violations 
                ORDER BY timestamp DESC 
                LIMIT {limit}
            '''
            df = pd.read_sql_query(query, self.conn)
        else:
            df = pd.read_csv(self.csv_path)
            df = df.tail(limit).sort_values('timestamp', ascending=False)
        
        return df
    
    def get_violations_by_type(self, violation_type):
        """Get violations filtered by type"""
        if self.use_sqlite:
            query = '''
                SELECT * FROM violations 
                WHERE violation_type = ? 
                ORDER BY timestamp DESC
            '''
            df = pd.read_sql_query(query, self.conn, params=(violation_type,))
        else:
            df = pd.read_csv(self.csv_path)
            df = df[df['violation_type'] == violation_type].sort_values('timestamp', ascending=False)
        
        return df
    
    def get_violations_by_plate(self, plate_number):
        """Get violations for a specific vehicle"""
        if self.use_sqlite:
            query = '''
                SELECT * FROM violations 
                WHERE number_plate = ? 
                ORDER BY timestamp DESC
            '''
            df = pd.read_sql_query(query, self.conn, params=(plate_number,))
        else:
            df = pd.read_csv(self.csv_path)
            df = df[df['number_plate'] == plate_number].sort_values('timestamp', ascending=False)
        
        return df
    
    def get_statistics(self):
        """
        Get violation statistics
        
        Returns:
            Dictionary with stats
        """
        df = self.get_all_violations(limit=10000)
        
        stats = {
            'total_violations': len(df),
            'by_type': df['violation_type'].value_counts().to_dict() if len(df) > 0 else {},
            'by_severity': df['severity'].value_counts().to_dict() if len(df) > 0 else {},
            'total_fines': df['fine'].sum() if len(df) > 0 else 0,
            'avg_confidence': df['confidence'].mean() if len(df) > 0 else 0
        }
        
        return stats
    
    def export_to_csv(self, filepath):
        """Export database to CSV"""
        df = self.get_all_violations(limit=100000)
        df.to_csv(filepath, index=False)
        return filepath
    
    def close(self):
        """Close database connection"""
        if self.use_sqlite and hasattr(self, 'conn'):
            self.conn.close()
