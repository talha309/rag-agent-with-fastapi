from pydantic import BaseModel, EmailStr

class UserCreate(BaseModel):
    email: EmailStr
    password: str

class UserUpdate(BaseModel):
    password: str

class UserResponse(BaseModel):
    id: int
    email: EmailStr

    class Config:
        from_attributes = True  # Updated to use from_attributes for Pydantic v2 compatibility

class Token(BaseModel):
    access_token: str
    token_type: str

class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    response: str
    user_id: int
    timestamp: str  # Assuming you want to return the timestamp as a string in the response

    class Config:
        from_attributes = True  # Enable ORM mode for compatibility with SQLAlchemy