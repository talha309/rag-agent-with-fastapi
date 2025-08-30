# Alexandra Hotel Virtual Assistant

A FastAPI-based hotel assistant with memory capabilities using Google's Generative AI and LangChain.

## Features

- **Virtual Hotel Assistant**: AI-powered assistant for hotel queries
- **Memory System**: Remembers user preferences and past interactions
- **RAG (Retrieval-Augmented Generation)**: Uses vector search for accurate hotel information
- **RESTful API**: Easy-to-use endpoints for integration
- **User-specific Memory**: Each user has their own memory profile

## Prerequisites

- Python 3.11 or higher
- Google AI API key (Gemini)

## Installation

1. **Clone the repository** (if not already done):
   ```bash
   git clone <repository-url>
   cd backend_fastapi
   ```

2. **Install dependencies**:
   ```bash
   uv sync
   ```

3. **Set up environment variables**:
   Create a `.env` file in the `backend_fastapi` directory:
   ```env
   GOOGLE_API_KEY=your_google_api_key_here
   ```

   To get a Google AI API key:
   1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
   2. Sign in with your Google account
   3. Create a new API key
   4. Copy the key to your `.env` file

## Running the Application

### Development Mode
```bash
uv run uvicorn agent.rag_agent:app --reload
```

### Production Mode
```bash
uv run uvicorn agent.rag_agent:app --host 0.0.0.0 --port 8000
```

The application will be available at `http://localhost:8000`

## API Endpoints

### Health Check
- **GET** `/` - Check application status and AI service availability

### Chat Endpoint
- **GET** `/chat/{user_id}/{query}` - Send a message to the virtual assistant
  - `user_id`: Unique identifier for the user (for memory)
  - `query`: The message/question to ask the assistant

### Example Usage
```bash
# Check status
curl http://localhost:8000/

# Send a message
curl "http://localhost:8000/chat/user123/What%20are%20your%20room%20rates?"
```

## Troubleshooting

### Common Issues

1. **"No module named 'langchain_community'"**
   - Solution: Run `uv sync` to install dependencies

2. **"Your default credentials were not found"**
   - Solution: Set the `GOOGLE_API_KEY` environment variable in your `.env` file

3. **"AI service is not available"**
   - Check that your Google API key is valid and properly set
   - Verify you have internet connectivity

4. **"Error loading hotel data"**
   - Ensure the `data/data.txt` file exists and is readable
   - Check that the embeddings service is working

### API Key Issues

If you're having trouble with the Google API key:

1. **Verify the key format**: Should be a long string without spaces
2. **Check API quotas**: Ensure you haven't exceeded your daily limit
3. **Enable the API**: Make sure the Gemini API is enabled in your Google Cloud Console
4. **Test the key**: Try using it in the Google AI Studio playground first

## Project Structure

```
backend_fastapi/
├── agent/
│   └── rag_agent.py          # Main FastAPI application
├── data/
│   └── data.txt              # Hotel information database
├── database/                 # Database configuration
├── models/                   # Database models
├── routes/                   # API routes
├── utils/                    # Utility functions
├── pyproject.toml           # Project dependencies
└── README.md                # This file
```

## Features in Detail

### Memory System
- Each user has a unique memory profile stored in SQLite
- The assistant remembers user preferences, past interactions, and personal details
- Memory is automatically updated with each conversation

### RAG Implementation
- Uses FAISS vector database for efficient information retrieval
- Hotel data is chunked and embedded for semantic search
- Provides accurate, context-aware responses about hotel services

### Error Handling
- Graceful handling of missing API keys
- Fallback responses when services are unavailable
- Comprehensive error messages for debugging

## Development

### Adding New Features
1. Modify `agent/rag_agent.py` for core functionality
2. Update `data/data.txt` for new hotel information
3. Test with the API endpoints

### Customizing the Assistant
- Modify system messages in `rag_agent.py`
- Update hotel data in `data/data.txt`
- Adjust memory collection logic as needed

## License

This project is for educational and demonstration purposes.

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Verify your API key configuration
3. Check the application logs for detailed error messages
