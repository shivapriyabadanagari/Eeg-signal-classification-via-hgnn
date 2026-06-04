import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import UserMixin
import os

DATABASE = 'users.db'

def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            mobile TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            address TEXT NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'user'
        )
    ''')
    conn.commit()
    conn.close()

class User(UserMixin):
    def __init__(self, id, name, mobile, email, address, password_hash, role):
        self.id = id
        self.name = name
        self.mobile = mobile
        self.email = email
        self.address = address
        self.password_hash = password_hash
        self.role = role
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    
    @staticmethod
    def get(user_id):
        conn = get_db_connection()
        user_data = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
        conn.close()
        if user_data:
            return User(
                id=user_data['id'],
                name=user_data['name'],
                mobile=user_data['mobile'],
                email=user_data['email'],
                address=user_data['address'],
                password_hash=user_data['password_hash'],
                role=user_data['role']
            )
        return None
    
    @staticmethod
    def get_by_email(email):
        conn = get_db_connection()
        user_data = conn.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()
        conn.close()
        if user_data:
            return User(
                id=user_data['id'],
                name=user_data['name'],
                mobile=user_data['mobile'],
                email=user_data['email'],
                address=user_data['address'],
                password_hash=user_data['password_hash'],
                role=user_data['role']
            )
        return None
    
    @staticmethod
    def create(name, mobile, email, address, password, role='user'):
        conn = get_db_connection()
        password_hash = generate_password_hash(password)
        try:
            conn.execute(
                'INSERT INTO users (name, mobile, email, address, password_hash, role) VALUES (?, ?, ?, ?, ?, ?)',
                (name, mobile, email, address, password_hash, role)
            )
            conn.commit()
            conn.close()
            return True
        except sqlite3.IntegrityError:
            conn.close()
            return False
