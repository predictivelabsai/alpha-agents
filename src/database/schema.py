"""
Database schema for Alpha Agents consumer lending application.
"""

import sqlite3
import os
from datetime import datetime
from typing import Optional

class DatabaseManager:
    def __init__(self, db_path: str = "alpha_agents.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the database with required tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Loan applications table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS loan_applications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                applicant_name TEXT NOT NULL,
                email TEXT,
                phone TEXT,
                loan_amount REAL NOT NULL,
                loan_purpose TEXT,
                annual_income REAL,
                credit_score INTEGER,
                employment_status TEXT,
                debt_to_income_ratio REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Agent analyses table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS agent_analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                application_id INTEGER,
                agent_type TEXT NOT NULL,
                analysis_result TEXT,
                recommendation TEXT,
                confidence_score REAL,
                risk_assessment TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (application_id) REFERENCES loan_applications (id)
            )
        ''')
        
        # Agent debates table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS agent_debates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                application_id INTEGER,
                debate_round INTEGER,
                agent_type TEXT NOT NULL,
                message TEXT,
                consensus_reached BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (application_id) REFERENCES loan_applications (id)
            )
        ''')
        
        # Final decisions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS final_decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                application_id INTEGER,
                decision TEXT NOT NULL,
                confidence_score REAL,
                risk_level TEXT,
                loan_terms TEXT,
                interest_rate REAL,
                reasoning TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (application_id) REFERENCES loan_applications (id)
            )
        ''')
        
        # Performance metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE,
                total_applications INTEGER,
                approved_applications INTEGER,
                rejected_applications INTEGER,
                average_processing_time REAL,
                average_confidence_score REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def get_connection(self):
        """Get database connection."""
        return sqlite3.connect(self.db_path)
    
    def insert_loan_application(self, application_data: dict) -> int:
        """Insert a new loan application and return its ID."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO loan_applications 
            (applicant_name, email, phone, loan_amount, loan_purpose, 
             annual_income, credit_score, employment_status, debt_to_income_ratio)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            application_data.get('applicant_name'),
            application_data.get('email'),
            application_data.get('phone'),
            application_data.get('loan_amount'),
            application_data.get('loan_purpose'),
            application_data.get('annual_income'),
            application_data.get('credit_score'),
            application_data.get('employment_status'),
            application_data.get('debt_to_income_ratio')
        ))
        
        application_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return application_id
    
    def insert_agent_analysis(self, analysis_data: dict):
        """Insert agent analysis result."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO agent_analyses 
            (application_id, agent_type, analysis_result, recommendation, 
             confidence_score, risk_assessment)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            analysis_data.get('application_id'),
            analysis_data.get('agent_type'),
            analysis_data.get('analysis_result'),
            analysis_data.get('recommendation'),
            analysis_data.get('confidence_score'),
            analysis_data.get('risk_assessment')
        ))
        
        conn.commit()
        conn.close()
    
    def insert_debate_message(self, debate_data: dict):
        """Insert agent debate message."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO agent_debates 
            (application_id, debate_round, agent_type, message, consensus_reached)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            debate_data.get('application_id'),
            debate_data.get('debate_round'),
            debate_data.get('agent_type'),
            debate_data.get('message'),
            debate_data.get('consensus_reached', False)
        ))
        
        conn.commit()
        conn.close()
    
    def insert_final_decision(self, decision_data: dict):
        """Insert final lending decision."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO final_decisions 
            (application_id, decision, confidence_score, risk_level, 
             loan_terms, interest_rate, reasoning)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            decision_data.get('application_id'),
            decision_data.get('decision'),
            decision_data.get('confidence_score'),
            decision_data.get('risk_level'),
            decision_data.get('loan_terms'),
            decision_data.get('interest_rate'),
            decision_data.get('reasoning')
        ))
        
        conn.commit()
        conn.close()
    
    def get_application_by_id(self, application_id: int) -> Optional[dict]:
        """Get loan application by ID."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM loan_applications WHERE id = ?', (application_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            columns = [description[0] for description in cursor.description]
            return dict(zip(columns, row))
        return None
    
    def get_agent_analyses(self, application_id: int) -> list:
        """Get all agent analyses for an application."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM agent_analyses 
            WHERE application_id = ? 
            ORDER BY created_at
        ''', (application_id,))
        
        rows = cursor.fetchall()
        columns = [description[0] for description in cursor.description]
        conn.close()
        
        return [dict(zip(columns, row)) for row in rows]
    
    def get_debate_history(self, application_id: int) -> list:
        """Get debate history for an application."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM agent_debates 
            WHERE application_id = ? 
            ORDER BY debate_round, created_at
        ''', (application_id,))
        
        rows = cursor.fetchall()
        columns = [description[0] for description in cursor.description]
        conn.close()
        
        return [dict(zip(columns, row)) for row in rows]
    
    def get_performance_metrics(self, days: int = 30) -> list:
        """Get performance metrics for the last N days."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM performance_metrics 
            WHERE date >= date('now', '-{} days')
            ORDER BY date DESC
        '''.format(days))
        
        rows = cursor.fetchall()
        columns = [description[0] for description in cursor.description]
        conn.close()
        
        return [dict(zip(columns, row)) for row in rows]

if __name__ == "__main__":
    # Initialize database
    db = DatabaseManager()
    print("Database initialized successfully!")

