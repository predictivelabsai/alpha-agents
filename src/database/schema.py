"""
Database schema for Alpha Agents equity portfolio construction system.
"""

import sqlite3
import os
from datetime import datetime
from typing import Optional, List, Dict

# Database path in db directory
DATABASE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'db', 'alpha_agents.db')

class DatabaseManager:
    def __init__(self, db_path: str = None):
        self.db_path = db_path or DATABASE_PATH
        # Ensure db directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.init_database()
    
    def init_database(self):
        """Initialize the database with required tables for equity analysis."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Stock analyses table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS stock_analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                stock_symbol TEXT NOT NULL,
                company_name TEXT,
                sector TEXT,
                current_price REAL,
                market_cap REAL,
                analysis_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Agent analyses table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS agent_analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                analysis_id INTEGER,
                agent_type TEXT NOT NULL,
                recommendation TEXT,
                confidence_score REAL,
                target_price REAL,
                risk_assessment TEXT,
                reasoning TEXT,
                key_factors TEXT,
                concerns TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (analysis_id) REFERENCES stock_analyses (id)
            )
        ''')
        
        # Multi-agent debates table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS agent_debates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                analysis_id INTEGER,
                debate_round INTEGER,
                agent_type TEXT NOT NULL,
                message TEXT,
                consensus_reached BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (analysis_id) REFERENCES stock_analyses (id)
            )
        ''')
        
        # Portfolio recommendations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS portfolio_recommendations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                portfolio_name TEXT,
                risk_tolerance TEXT,
                expected_return REAL,
                risk_level TEXT,
                diversification_score REAL,
                reasoning TEXT,
                confidence_score REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Portfolio stocks table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS portfolio_stocks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                portfolio_id INTEGER,
                stock_symbol TEXT NOT NULL,
                weight REAL,
                target_price REAL,
                recommendation TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (portfolio_id) REFERENCES portfolio_recommendations (id)
            )
        ''')
        
        # Performance metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE,
                total_analyses INTEGER,
                buy_recommendations INTEGER,
                hold_recommendations INTEGER,
                sell_recommendations INTEGER,
                average_confidence_score REAL,
                agent_performance TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def get_connection(self):
        """Get database connection."""
        return sqlite3.connect(self.db_path)
    
    def insert_stock_analysis(self, stock_data: dict) -> int:
        """Insert a new stock analysis and return its ID."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO stock_analyses 
            (stock_symbol, company_name, sector, current_price, market_cap)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            stock_data.get('symbol'),
            stock_data.get('company_name'),
            stock_data.get('sector'),
            stock_data.get('current_price'),
            stock_data.get('market_cap')
        ))
        
        analysis_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return analysis_id
    
    def insert_agent_analysis(self, analysis_data: dict):
        """Insert agent analysis result."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO agent_analyses 
            (analysis_id, agent_type, recommendation, confidence_score, 
             target_price, risk_assessment, reasoning, key_factors, concerns)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            analysis_data.get('analysis_id'),
            analysis_data.get('agent_type'),
            analysis_data.get('recommendation'),
            analysis_data.get('confidence_score'),
            analysis_data.get('target_price'),
            analysis_data.get('risk_assessment'),
            analysis_data.get('reasoning'),
            analysis_data.get('key_factors'),
            analysis_data.get('concerns')
        ))
        
        conn.commit()
        conn.close()
    
    def insert_portfolio_recommendation(self, portfolio_data: dict) -> int:
        """Insert portfolio recommendation and return its ID."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO portfolio_recommendations 
            (portfolio_name, risk_tolerance, expected_return, risk_level, 
             diversification_score, reasoning, confidence_score)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            portfolio_data.get('portfolio_name'),
            portfolio_data.get('risk_tolerance'),
            portfolio_data.get('expected_return'),
            portfolio_data.get('risk_level'),
            portfolio_data.get('diversification_score'),
            portfolio_data.get('reasoning'),
            portfolio_data.get('confidence_score')
        ))
        
        portfolio_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return portfolio_id
    
    def insert_portfolio_stock(self, stock_data: dict):
        """Insert stock in portfolio."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO portfolio_stocks 
            (portfolio_id, stock_symbol, weight, target_price, recommendation)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            stock_data.get('portfolio_id'),
            stock_data.get('stock_symbol'),
            stock_data.get('weight'),
            stock_data.get('target_price'),
            stock_data.get('recommendation')
        ))
        
        conn.commit()
        conn.close()
    
    def get_stock_analysis_by_id(self, analysis_id: int) -> Optional[dict]:
        """Get stock analysis by ID."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM stock_analyses WHERE id = ?', (analysis_id,))
        row = cursor.fetchone()
        
        if row:
            columns = [description[0] for description in cursor.description]
            result = dict(zip(columns, row))
            conn.close()
            return result
        
        conn.close()
        return None
    
    def get_agent_analyses(self, analysis_id: int) -> List[dict]:
        """Get all agent analyses for a stock analysis."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM agent_analyses 
            WHERE analysis_id = ? 
            ORDER BY created_at
        ''', (analysis_id,))
        
        rows = cursor.fetchall()
        columns = [description[0] for description in cursor.description]
        conn.close()
        
        return [dict(zip(columns, row)) for row in rows]
    
    def get_recent_analyses(self, limit: int = 10) -> List[dict]:
        """Get recent stock analyses."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT sa.*, COUNT(aa.id) as agent_count
            FROM stock_analyses sa
            LEFT JOIN agent_analyses aa ON sa.id = aa.analysis_id
            GROUP BY sa.id
            ORDER BY sa.created_at DESC
            LIMIT ?
        ''', (limit,))
        
        rows = cursor.fetchall()
        columns = [description[0] for description in cursor.description]
        conn.close()
        
        return [dict(zip(columns, row)) for row in rows]
    
    def get_performance_summary(self) -> dict:
        """Get overall performance summary."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Get total analyses
        cursor.execute('SELECT COUNT(*) FROM stock_analyses')
        total_analyses = cursor.fetchone()[0]
        
        # Get recommendation distribution
        cursor.execute('''
            SELECT recommendation, COUNT(*) 
            FROM agent_analyses 
            GROUP BY recommendation
        ''')
        recommendations = dict(cursor.fetchall())
        
        # Get average confidence
        cursor.execute('SELECT AVG(confidence_score) FROM agent_analyses')
        avg_confidence = cursor.fetchone()[0] or 0.0
        
        conn.close()
        
        return {
            'total_analyses': total_analyses,
            'recommendations': recommendations,
            'average_confidence': avg_confidence
        }

if __name__ == "__main__":
    # Initialize database
    db = DatabaseManager()
    print(f"Database initialized successfully at: {db.db_path}")

