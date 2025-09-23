import os
from dotenv import load_dotenv
from datetime import datetime
from sqlalchemy import (
    create_engine, Column, String, Integer, Float, Text,
    ForeignKey, DateTime, JSON
)
from sqlalchemy.orm import sessionmaker, declarative_base, relationship

# -------------------- Load Environment Variables --------------------
load_dotenv()

DATABASE_URI = os.getenv("DATABASE_URI")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MAX_CHATS_PER_USER = os.getenv("MAX_CHATS_PER_USER")

# -------------------- Database Setup --------------------
engine = create_engine(DATABASE_URI, pool_size=20, max_overflow=0)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    """Dependency for getting DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# -------------------- ORM Models --------------------
class User(Base):
    __tablename__ = "users"
    id = Column(String, primary_key=True, index=True)  # User-provided ID
    created_at = Column(DateTime, default=datetime.utcnow)

    chats = relationship("Chat", back_populates="user", cascade="all, delete-orphan")
    profile = relationship("StudentProfile", back_populates="user", uselist=False, cascade="all, delete-orphan")

class Chat(Base):
    __tablename__ = "chats"
    id = Column(String, primary_key=True, index=True)  # System-generated ID
    user_id = Column(String, ForeignKey("users.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    status = Column(String, default="active")

    user = relationship("User", back_populates="chats")
    records = relationship("ChatRecord", back_populates="chat", cascade="all, delete-orphan")
    title = relationship("Title", back_populates="chat", uselist=False, cascade="all, delete-orphan")  # ✅ One-to-one

class Title(Base):
    __tablename__ = "titles"
    id = Column(Integer, primary_key=True, index=True)
    chat_id = Column(String, ForeignKey("chats.id"), unique=True)  # ✅ One title per chat
    title = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    chat = relationship("Chat", back_populates="title")

class APILog(Base):
    __tablename__ = "api_logs"
    id = Column(Integer, primary_key=True, index=True)
    endpoint = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)
    request_data = Column(JSON)
    response_data = Column(JSON)
    status_code = Column(Integer)
    error_message = Column(Text, nullable=True)

class StudentProfile(Base):
    __tablename__ = "student_profiles"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, ForeignKey("users.id"), unique=True)

    location_preference = Column(String, nullable=True)
    preferred_fields = Column(JSON, default=[])
    degree_preference = Column(String, nullable=True)
    percentile = Column(Float, nullable=True)
    budget = Column(Integer, nullable=True)
    
    additional_info = Column(JSON, default={})
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User", back_populates="profile")

class ChatRecord(Base):
    """
    Unified table for both chat messages and college recommendations.
    """
    __tablename__ = "chat_records"

    # Common fields for both messages and recommendations
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    chat_id = Column(String, ForeignKey("chats.id"), index=True, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)

    # Fields specific to a chat message
    role = Column(String, nullable=True)  # 'user', 'assistant', or None for recommendations
    content = Column(Text, nullable=True) # The text of the chat message

    # Field specific to a college recommendation
    recommendation_data = Column(JSON, nullable=True)

    chat = relationship("Chat", back_populates="records")


# -------------------- Init Function --------------------
def init_db():
    """Creates all PostgreSQL tables."""
    Base.metadata.create_all(bind=engine)
    print("✅ PostgreSQL database tables created successfully")

if __name__ == "__main__":
    init_db()