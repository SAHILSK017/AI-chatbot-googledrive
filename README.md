# DriveAgent – AI Powered Google Drive Assistant

DriveAgent is an AI-based Google Drive search assistant built using FastAPI, LangChain, ChatGroq, and Google Drive API.

It allows users to search files using natural language queries like:

* “show my pics”
* “find pdf files”
* “show spreadsheets”
* “find resume”

## Features

* Natural language file search
* LangChain conversational agent
* Recursive folder search
* File type filtering
* Recent file search
* Shared Drive support
* React frontend + FastAPI backend

## Tech Stack

* Python
* FastAPI
* LangChain
* ChatGroq
* Google Drive API
* React
* Render

## Run Backend

```bash id="1x4m1o"
pip install -r requirements.txt
uvicorn main:app --reload
```

## Environment Variables

```env id="8bdp8e"
GROQ_API_KEY=your_key
GOOGLE_SERVICE_ACCOUNT_JSON=your_json
TARGET_FOLDER_ID=your_folder_id
```

## Deployment

Render Start Command:

```bash id="cq92zc"
uvicorn backend.main:app --host 0.0.0.0 --port $PORT
```

## Author

Sahil Kumar
