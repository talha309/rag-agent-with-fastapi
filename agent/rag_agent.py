import os
import sqlite3
from fastapi import FastAPI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda
from langgraph.graph.message import AnyMessage, add_messages
from langchain_core.messages import ToolMessage, SystemMessage, HumanMessage
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from typing import Annotated
from typing_extensions import TypedDict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Solar Panel Virtual Assistant", description="A FastAPI-based solar panel assistant with memory")

# Initialize Google Generative AI model
google_api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

if not google_api_key:
    print("WARNING: GOOGLE_API_KEY not found in environment variables.")
    llm = None
else:
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=google_api_key)
    except Exception as e:
        print(f"Error initializing Google AI: {e}")
        llm = None

if not tavily_api_key:
    print("WARNING: TAVILY_API_KEY not found in environment variables.")

# Define compute_savings tool
def compute_savings(monthly_cost: float) -> dict:
    """
    Tool to compute the potential savings when switching to solar energy based on the user's monthly electricity cost.

    Args:
        monthly_cost (float): The user's current monthly electricity cost.

    Returns:
        dict: A dictionary containing:
            - 'number_of_panels': The estimated number of solar panels required.
            - 'installation_cost': The estimated installation cost.
            - 'net_savings_10_years': The net savings over 10 years after installation costs.
    """
    cost_per_kWh = 0.28
    cost_per_watt = 1.50
    sunlight_hours_per_day = 3.5
    panel_wattage = 350
    system_lifetime_years = 10

    monthly_consumption_kWh = monthly_cost / cost_per_kWh
    daily_energy_production = monthly_consumption_kWh / 30
    system_size_kW = daily_energy_production / sunlight_hours_per_day
    number_of_panels = system_size_kW * 1000 / panel_wattage
    installation_cost = system_size_kW * 1000 * cost_per_watt
    annual_savings = monthly_cost * 12
    total_savings_10_years = annual_savings * system_lifetime_years
    net_savings = total_savings_10_years - installation_cost

    return {
        "number_of_panels": round(number_of_panels),
        "installation_cost": round(installation_cost, 2),
        "net_savings_10_years": round(net_savings, 2)
    }

# Define tools
tools = []
if tavily_api_key:
    tools.append(TavilySearchResults(tavily_api_key=tavily_api_key))

# Bind tools to LLM
llm_with_tools = llm.bind_tools(tools) if tools and llm else llm

# System prompt for solar panel assistant
SOLAR_SYSTEM_MESSAGE = (
    "You are a helpful customer support assistant specializing in Solar Panels. "
    "Your role is to provide users with accurate information about solar panels and how they work by utilizing the Tavily search tool for web searches. "
    "For any questions about solar systems, provide to-the-point answers based on search results. "
    "When users inquire about savings, ask for their current monthly electricity cost. "
    "If the user's message doesn't include this information or it's unclear, kindly request clarification. Avoid making assumptions. "
    "Once you've gathered the necessary details, call the appropriate tool to assist the user."
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

# Error handling for tools
def handle_tool_error(state) -> dict:
    """
    Function to handle errors that occur during tool execution.

    Args:
        state (dict): The current state of the AI agent, which includes messages and tool call details.

    Returns:
        dict: A dictionary containing error messages for each tool that encountered an issue.
    """
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }

# Create tool node with fallback
def create_tool_node_with_fallback(tools: list) -> dict:
    """
    Function to create a tool node with fallback error handling.

    Args:
        tools (list): A list of tools to be included in the node.

    Returns:
        dict: A tool node that uses fallback behavior in case of errors.
    """
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)],
        exception_key="error"
    )

# Define state
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

# Assistant class
class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State, config: RunnableConfig):
        while True:
            result = self.runnable.invoke(state)
            if not result.tool_calls and (
                not result.content
                or isinstance(result.content, list)
                and not result.content[0].get("text")
            ):
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                break
        return {"messages": result}

# Define StateGraph
if llm:
    primary_assistant_prompt = ChatPromptTemplate.from_messages([
        ("system", f"{SOLAR_SYSTEM_MESSAGE}\n\n{MEMORY_SYSTEM_MESSAGE.format(memory='{memory}')}"),
        ("placeholder", "{messages}"),
    ])
    assistant_runnable = primary_assistant_prompt | llm_with_tools
    builder = StateGraph(State)
    builder.add_node("assistant", Assistant(assistant_runnable))
    if tools:
        builder.add_node("tools", create_tool_node_with_fallback(tools))
    builder.add_node("write_memory", write_memory)

    builder.add_edge(START, "assistant")
    if tools:
        builder.add_conditional_edges("assistant", tools_condition)
        builder.add_edge("tools", "assistant")
    builder.add_edge("assistant", "write_memory")
    builder.add_edge("write_memory", END)

    memory = MemorySaver()
    agent = builder.compile(checkpointer=memory)
else:
    agent = None

# Write memory node
def write_memory(state: State, config: RunnableConfig):
    """Updates and saves user memory based on chat history."""
    user_id = config["configurable"]["user_id"]
    existing_memory = retrieve_memory(user_id)
    
    if not llm:
        return {"messages": state["messages"]}
    
    system_msg = CREATE_MEMORY_INSTRUCTION.format(memory=existing_memory)
    new_memory_response = llm.invoke([SystemMessage(content=system_msg)] + state['messages'])
    
    save_memory(user_id, new_memory_response.content)
    return {"messages": state["messages"]}

# FastAPI endpoints
@app.get("/chat/{user_id}/{query}")
async def get_content(user_id: str, query: str):
    """Handles chat queries with user-specific memory."""
    if not agent:
        return {"error": "AI service is not available. Please check your Google API key configuration."}
    
    try:
        config = {"configurable": {"thread_id": user_id, "user_id": user_id}}
        result = agent.invoke({"messages": [HumanMessage(content=query)]}, config)
        return {"response": result["messages"][-1].content}
    except Exception as e:
        return {"error": str(e)}

@app.get("/")
async def root():
    """Root endpoint with status information."""
    status = {
        "status": "running",
        "ai_service": "available" if llm else "unavailable - missing GOOGLE_API_KEY",
        "tavily_search": "available" if tavily_api_key else "unavailable",
        "message": "Solar Panel Virtual Assistant API is running"
    }
    return status

# Main loop for testing
if __name__ == "__main__":
    import uvicorn
    db_file_path = 'memory.db'
    if not os.path.exists(db_file_path):
        print(f"Database file '{db_file_path}' created.")
    uvicorn.run(app, host="0.0.0.0", port=8000)