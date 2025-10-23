# Desktop Assistant

A small Python-based desktop assistant project with modular components for automation, core assistant logic, interfaces (speech and voice), and LLM integrations (Gemini). This repository provides a minimal structure to run a local assistant that can use various LLM and retrieval modules.

## Requirements

- Python 3.9+ recommended
- See `requirements.txt` for Python dependencies

## Installation (Windows PowerShell)

# Desktop Assistant (Gemini-powered)

A modular Python desktop assistant that combines:

- A conversational ChatBot (Gemini-powered) with multilingual support
- Desktop automation and GUI interaction utilities (screen analysis, GUI patterns, automation)
- Local and web data retrieval modules powered by Gemini AI
- Context memory stored in a local SQLite database for retrieval and insights

This README documents installation, configuration, running modes, and architecture based on the repository code.

## Requirements

- Python 3.9+ recommended
- See `requirements.txt` for a comprehensive list of dependencies (LLM client, GUI automation, voice, NLP, etc.)

Notes:
- Some features (voice, advanced NLP, GPU-accelerated models) are optional and guarded by imports. The assistant falls back to safe behavior when optional libraries are missing.

## Quick install (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Environment variables

Create a `.env` file or set environment variables in your shell. Important variables used by the project:

- `GOOGLE_AI_API_KEY` — required to use Gemini via the `google-generativeai` library. The `GeminiClient` will raise an error if this is missing.
- `USER_AGENT` — optional; set automatically by `core.config` if not present.
- Other variables used by optional modules (e.g., translation, debug flags) may exist; the code reads `.env` where appropriate.

If `GOOGLE_AI_API_KEY` is not set, `main.py` will prompt for it interactively and optionally save it to `.env`.

## Running

Run the assistant interactively:

```powershell
python main.py
```

On first run the CLI will:

- Call `core.config.setup_environment()` to set runtime environment variables
- Load environment variables (and prompt for `GOOGLE_AI_API_KEY` if missing)
- Optionally let you choose TTS voice
- Initialize `DesktopAssistant` and `ChatBot` which wire together Gemini client, context retrieval, and automation components

Basic CLI commands available in interactive mode include: `help`, `status`, `analyze`, `history`, `clear-chat`, `context search [query]`, and `exit`.

## Main components (what each folder/file does)

- `main.py` — Application entrypoint, CLI, model selection logic, voice selection, and interactive loop. Initializes `DesktopAssistant` and `ChatBot`.
- `core/config.py` — `AssistantConfig` dataclass and `setup_environment()` which sets process-level env vars used across the app.
- `core/desktop_assistant.py` — `DesktopAssistant` class that orchestrates Gemini client, GUI automator, voice handler, and task execution flow. Handles safe mode and confirmation settings.
- `core/chatbot.py` — `ChatBot` class that builds prompts, handles language detection/translation, context retrieval, and calls the Gemini client for conversational responses.

- `automation/`:
	- `os_detection.py` — `OSDetector` detects OS, installed apps, settings locations, and constructs a system blueprint used to generate OS-appropriate commands and UI guidance.
	- `storage.py` — `ContextMemoryManager` uses a local SQLite DB (`context_memory.db`) to persist conversations, GUI patterns, and task history. Provides retrieval, summarization, cleanup, and preference learning.

- `interfaces/`:
	- `voice.py` — `VoiceHandler` wraps speech recognition (SpeechRecognition) and TTS (pyttsx3). Supports single-shot and continuous listening modes and TTS playback.

- `llm/gemini/` — Gemini-powered LLM integration modules:
	- `client.py` — `GeminiClient` wrapper around `google.generativeai`, model selection with quota-aware fallbacks, screenshot analysis helper.
	- `context_retrieval.py` — `GeminiContextRetriever` to find relevant conversation history and generate insights using Gemini.
	- `intent_classifier.py` — `GeminiIntentClassifier` to classify intents into conversation, automation, local_data_retrieval, or web_data_retrieval.
	- `local_retrieval.py` — `GeminiLocalDataRetriever` to generate safe, read-only OS-specific commands and execute them to retrieve local data.
	- `web_retrieval.py` — `GeminiWebRetriever` to format web-style information retrieval using Gemini.
	- `translation.py` — `GeminiTranslator` wrapper for language detection and translation.

## Data & persistence

- Conversation history, GUI patterns, and task execution records are stored in `context_memory.db` (SQLite). The `ContextMemoryManager` handles schema creation, saving, retrieval, summarization, and cleanup.

## Voice / TTS

- The assistant uses `speech_recognition` + `pyttsx3` (if installed). `interfaces/voice.py` includes utilities to test, speak, listen_once, and start continuous listening with a callback.
- Voice is optional; the assistant can run in text-only mode if voice packages are missing.

## Safety & confirmation

- The assistant enforces several safety rules in code paths that execute system commands: read-only commands for local data retrieval, a confirmation flow before performing automation actions (if `confirmation_required` is enabled), and a `safe_mode` flag to reduce risk.
- `DesktopAssistant._confirm_action()` prompts the user with action details, confidence, and options to execute or skip.

## Troubleshooting

- If you see import errors for optional modules (like `SpeechRecognition`, `scikit-learn`, or `pyttsx3`), install them via `pip` or disable voice/NLP features.
- Ensure `GOOGLE_AI_API_KEY` is set before running to avoid interactive prompts. You can create a `.env` file with `GOOGLE_AI_API_KEY=your_key`.
- If Gemini model quota/rate-limit errors occur, the code attempts model fallbacks and shows a friendly message; you may need a higher-tier API key or wait for quota reset.

## Development notes

- The code contains defensive fallbacks for missing dependencies and modular client classes so you can test components independently. Key interfaces to look at when extending functionality:
	- `llm/gemini/client.py` — model selection and screenshot analysis
	- `automation/storage.py` — persistent memory and retrieval
	- `automation/os_detection.py` — system blueprint used for OS-aware automation

## Contributing

- Add tests where behavior is critical (context retrieval, command generation, storage). The project lists `pytest` in `requirements.txt`.
- When adding new automation commands, prefer read-only safe approaches and unit tests that validate command generation without running destructive commands.

## License

This project is licensed under the MIT License — see `LICENSE` for details.

## Contact / Next steps

- Want a simplified quickstart, docker container, or typed CLI? Tell me how you'd like to run it and I can add scripts, tasks, or a minimal `requirements.txt` pinning.
