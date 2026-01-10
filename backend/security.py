# In backend/security.py

from passlib.context import CryptContext

# This tells passlib to use the "bcrypt" hashing algorithm
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(plain_password, hashed_password):
    """Checks if a plain password matches a hashed one."""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    """Hashes a plain-text password."""
    return pwd_context.hash(password)