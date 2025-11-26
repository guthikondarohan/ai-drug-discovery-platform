"""
Prediction history database using SQLite.

Stores predictions, user queries, and molecular data for analytics.
"""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
import json


class PredictionDatabase:
    """
    SQLite database for storing prediction history.
    """
    
    def __init__(self, db_path: str = "data/predictions.db"):
        """Initialize database connection."""
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._create_tables()
    
    def _create_tables(self):
        """Create database tables if they don't exist."""
        cursor = self.conn.cursor()
        
        # Predictions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                smiles TEXT NOT NULL,
                chemical_name TEXT,
                prediction TEXT NOT NULL,
                probability REAL NOT NULL,
                confidence REAL NOT NULL,
                model_type TEXT NOT NULL,
                features TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # User queries table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS queries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_text TEXT NOT NULL,
                query_type TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Favorites table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS favorites (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                smiles TEXT NOT NULL UNIQUE,
                name TEXT,
                notes TEXT,
                added_date DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        self.conn.commit()
    
    def add_prediction(
        self,
        smiles: str,
        prediction: str,
        probability: float,
        confidence: float,
        model_type: str = "mlp",
        chemical_name: Optional[str] = None,
        features: Optional[Dict] = None
    ) -> int:
        """
        Add a prediction to the database.
        
        Returns:
            Prediction ID
        """
        cursor = self.conn.cursor()
        
        features_json = json.dumps(features) if features else None
        
        cursor.execute('''
            INSERT INTO predictions 
            (smiles, chemical_name, prediction, probability, confidence, model_type, features)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (smiles, chemical_name, prediction, probability, confidence, model_type, features_json))
        
        self.conn.commit()
        return cursor.lastrowid
    
    def get_recent_predictions(self, limit: int = 50) -> List[Dict]:
        """Get recent predictions."""
        cursor = self.conn.cursor()
        
        cursor.execute('''
            SELECT id, smiles, chemical_name, prediction, probability, 
                   confidence, model_type, timestamp
            FROM predictions
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (limit,))
        
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    def get_prediction_stats(self) -> Dict:
        """Get statistics about predictions."""
        cursor = self.conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM predictions')
        total = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM predictions WHERE prediction = "Active"')
        active = cursor.fetchone()[0]
        
        cursor.execute('SELECT AVG(probability) FROM predictions')
        avg_prob = cursor.fetchone()[0] or 0
        
        cursor.execute('SELECT AVG(confidence) FROM predictions')
        avg_conf = cursor.fetchone()[0] or 0
        
        return {
            'total_predictions': total,
            'active_count': active,
            'inactive_count': total - active,
            'average_probability': avg_prob,
            'average_confidence': avg_conf
        }
    
    def search_predictions(self, query: str) -> List[Dict]:
        """Search predictions by SMILES or chemical name."""
        cursor = self.conn.cursor()
        
        cursor.execute('''
            SELECT id, smiles, chemical_name, prediction, probability, 
                   confidence, timestamp
            FROM predictions
            WHERE smiles LIKE ? OR chemical_name LIKE ?
            ORDER BY timestamp DESC
            LIMIT 100
        ''', (f'%{query}%', f'%{query}%'))
        
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    def add_favorite(self, smiles: str, name: Optional[str] = None, notes: Optional[str] = None):
        """Add molecule to favorites."""
        cursor = self.conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO favorites (smiles, name, notes)
                VALUES (?, ?, ?)
            ''', (smiles, name, notes))
            self.conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False  # Already exists
    
    def get_favorites(self) -> List[Dict]:
        """Get all favorite molecules."""
        cursor = self.conn.cursor()
        
        cursor.execute('''
            SELECT id, smiles, name, notes, added_date
            FROM favorites
            ORDER BY added_date DESC
        ''')
        
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    def close(self):
        """Close database connection."""
        self.conn.close()
