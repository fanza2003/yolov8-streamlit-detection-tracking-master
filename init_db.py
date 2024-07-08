import sqlite3
import hashlib

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Connect to the SQLite database
conn = sqlite3.connect('users.db')
c = conn.cursor()

# Create users table
c.execute('''CREATE TABLE IF NOT EXISTS users
             (username TEXT PRIMARY KEY, name TEXT, password TEXT)''')

# Add users
users = [
    ('admin', 'Admin', hash_password('your_admin_password')),
    ('rmiller', 'Rebecca Miller', hash_password('your_rebecca_password')),
    ('fanza', 'eldanendra', hash_password('123')),
    ('admin', 'Admin', hash_password('123')),
    ('Admin', 'Admin', hash_password('123'))
]

c.executemany('INSERT OR REPLACE INTO users VALUES (?,?,?)', users)

# Save (commit) the changes and close the connection
conn.commit()
conn.close()
