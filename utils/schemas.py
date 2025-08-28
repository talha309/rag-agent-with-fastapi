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
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str

class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    response: str
    user_id: int
    timestamp: str

    class Config:
        from_attributes = True