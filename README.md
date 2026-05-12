# Google Drive AI Agent

A conversational AI assistant for searching and browsing files in Google Drive using natural language.

Built with FastAPI, LangChain, Groq, and Streamlit.

---

## Features

- Natural language file search (name, type, date, content)
- Folder browsing — ask what's inside any folder
- Image thumbnail previews
- Typo-tolerant fuzzy matching
- Conversation memory for follow-up queries

## Tech Stack

| Layer    | Technology                          |
|----------|-------------------------------------|
| Backend  | FastAPI, LangChain, Groq (Llama 3.3)|
| Frontend | Streamlit                           |
| Storage  | Google Drive API v3 (Service Account)|
| Infra    | Docker, Docker Compose              |

---

## Setup

### 1. Google Cloud credentials

- Create a Service Account in Google Cloud Console
- Enable the **Google Drive API**
- Download the service account key as `credentials.json` and place it in the project root
- Share your target Drive folder with the service account email

### 2. Environment variables

```bash
cp .env.example .env
```

Fill in `.env`:

```
GROQ_API_KEY=your_groq_api_key
TARGET_FOLDER_ID=your_drive_folder_id
GOOGLE_APPLICATION_CREDENTIALS=credentials.json
```

Get a free Groq API key at [console.groq.com](https://console.groq.com).

### 3. Run with Docker

```bash
docker-compose up --build
```

- Backend: http://localhost:8000
- Frontend: http://localhost:8501

---

## Environment Variables

| Variable                        | Description                              |
|---------------------------------|------------------------------------------|
| `GROQ_API_KEY`                  | Groq API key for the LLM                 |
| `TARGET_FOLDER_ID`              | Google Drive folder ID to search within  |
| `GOOGLE_APPLICATION_CREDENTIALS`| Path to service account credentials file |

---

## Project Structure

```
.
├── backend/
│   ├── main.py          # FastAPI app, /chat and /thumbnail endpoints
│   ├── agent.py         # LangChain agent with conversation memory
│   ├── tools.py         # Drive search tools with multi-stage matching
│   ├── google_drive.py  # Drive API client and query cache
│   └── requirements.txt
├── frontend/
│   ├── app.py           # Streamlit chat interface
│   └── components/
│       ├── cards.py     # File result cards
│       ├── chat.py      # Chat message rendering
│       ├── sidebar.py   # Sidebar with suggested prompts
│       └── styles.py    # Global CSS
├── docker-compose.yml
├── .env.example
└── credentials.json     # (not committed)
```
