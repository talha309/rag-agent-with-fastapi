import os
import sqlite3
from fastapi import FastAPI
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain import hub
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph, END
from langgraph.prebuilt import tools_condition, ToolNode
from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables.config import RunnableConfig
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Alexandra Hotel Virtual Assistant", description="A FastAPI-based hotel assistant with memory")

# Initialize Google Generative AI model and embeddings
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Load and process hotel data
try:
    loader = TextLoader("./data/data.txt")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    db = FAISS.from_documents(texts, embeddings)
    retriever = db.as_retriever()
except Exception as e:
    print(f"Error loading hotel data: {e}")
    retriever = None

# Create retriever tool
info_retriever = create_retriever_tool(
    retriever,
    "hotel_information_sender",
    "Searches information about hotel from provided vector and returns accurate details"
) if retriever else None

tools = [info_retriever] if info_retriever else []

# Bind tools to LLM
llm_with_tools = llm.bind_tools(tools) if tools else llm

# System messages
HOTEL_SYSTEM_MESSAGE = (
    "You are Alexandra Hotel's virtual assistant, trained to assist customers with any queries related to the hotel. "
    "Your primary responsibility is to provide accurate, helpful, and friendly responses. "
    "You have access to a specialized tool for retrieving detailed and up-to-date information about the hotel, "
    "such as amenities, room availability, pricing, dining options, events, and policies. Use this tool effectively to provide precise answers. "
    "If a query is beyond your scope or requires external actions (e.g., booking confirmation, cancellations), "
    "politely inform the user and guide them to contact the hotel's staff for further assistance. "
    "Maintain a professional yet approachable tone at all times."
)

MEMORY_SYSTEM_MESSAGE = """You are a helpful assistant with memory that provides information about the user.
If you have memory for this user, use it to personalize your responses.
Here is the memory (it may be empty): {memory}"""

CREATE_MEMORY_INSTRUCTION = """You are collecting information about the user to personalize your responses.

CURRENT USER INFORMATION:
{memory}

INSTRUCTIONS:
1. Review the chat history below carefully.
2. Identify new information about the user, such as:
   - Personal details (name, location)
   - Preferences (likes, dislikes)
   - Interests and hobbies
   - Past experiences
   - Goals or future plans
3. Merge any new information with existing memory.
4. Format the memory as a clear, bulleted list.
5. If new information conflicts with existing memory, keep the most recent version.

Remember: Only include factual information directly stated by the user. Do not make assumptions or inferences.

Based on the chat history below, please update the user information:"""

# Database functions
def create_connection(db_file="memory.db"):
    """Creates a database connection and table if it doesn't exist."""
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        cursor.execute("""CREATE TABLE IF NOT EXISTS memories (
                          user_id TEXT PRIMARY KEY,
                          memory TEXT NOT NULL
                          );""")
        conn.commit()
    except sqlite3.Error as e:
        print(f"Database error: {e}")
    return conn

def get_connection():
    """Gets the database connection."""
    return create_connection()

def save_memory(user_id, memory):
    """Saves the memory for a user in the database."""
    conn = get_connection()
    if conn:
        cursor = conn.cursor()
        cursor.execute("INSERT OR REPLACE INTO memories (user_id, memory) VALUES (?, ?)", (user_id, memory))
        conn.commit()
        conn.close()

def retrieve_memory(user_id):
    """Retrieves the memory for a user from the database."""
    conn = get_connection()
    if conn:
        cursor = conn.cursor()
        cursor.execute("SELECT memory FROM memories WHERE user_id = ?", (user_id,))
        row = cursor.fetchone()
        conn.close()
        return row[0] if row else "No existing memory found."
    return "No existing memory found."

# Graph nodes
def call_model(state: MessagesState, config: RunnableConfig):
    """Handles the main assistant logic with memory and hotel tools."""
    user_id = config["configurable"]["user_id"]
    existing_memory = retrieve_memory(user_id)
    
    # Combine system messages
    system_msg = f"{HOTEL_SYSTEM_MESSAGE}\n\n{MEMORY_SYSTEM_MESSAGE.format(memory=existing_memory)}"
    
    # Use last 10 messages for context
    messages = [SystemMessage(content=system_msg)] + state["messages"][-10:]
    response = llm_with_tools.invoke(messages)
    
    return {"messages": [response]}

def write_memory(state: MessagesState, config: RunnableConfig):
    """Updates and saves user memory based on chat history."""
    user_id = config["configurable"]["user_id"]
    existing_memory = retrieve_memory(user_id)
    
    system_msg = CREATE_MEMORY_INSTRUCTION.format(memory=existing_memory)
    new_memory_response = llm.invoke([SystemMessage(content=system_msg)] + state['messages'])
    
    save_memory(user_id, new_memory_response.content)
    return {"messages": state["messages"]}

# Define StateGraph
builder = StateGraph(MessagesState)
builder.add_node("assistant", call_model)
if tools:
    builder.add_node("tools", ToolNode(tools))
builder.add_node("write_memory", write_memory)

# Define edges
builder.add_edge(START, "assistant")
if tools:
    builder.add_conditional_edges("assistant", tools_condition)
    builder.add_edge("tools", "assistant")
builder.add_edge("assistant", "write_memory")
builder.add_edge("write_memory", END)

# Compile graph with memory
memory = MemorySaver()
agent = builder.compile(checkpointer=memory)

# FastAPI endpoint
@app.get("/chat/{user_id}/{query}")
async def get_content(user_id: str, query: str):
    """Handles chat queries with user-specific memory."""
    try:
        config = {"configurable": {"thread_id": user_id, "user_id": user_id}}
        result = agent.invoke({"messages": [HumanMessage(content=query)]}, config)
        return {"response": result["messages"][-1].content}
    except Exception as e:
        return {"error": str(e)}

# Main loop for testing
if __name__ == "__main__":
    import uvicorn
    db_file_path = 'memory.db'
    if not os.path.exists(db_file_path):
        print(f"Database file '{db_file_path}' created.")
    uvicorn.run(app, host="0.0.0.0", port=8000)