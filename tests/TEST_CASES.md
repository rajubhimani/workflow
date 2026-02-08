# Test Cases

All tests are in `tests/test_chat.py`. Run with `uv run pytest`.

The LLM (`ChatOllama`) is mocked in all tests so they run without Ollama. Session state is cleared between tests via the `clear_sessions` autouse fixture.

## POST /chat

| Test | Description |
| ------ | ------------- |
| `test_chat` | Sends a message, verifies 200 response with correct content, session_id, timestamp, and token usage (prompt_tokens, completion_tokens, total_tokens) are returned |
| `test_chat_returns_session_id` | Verifies the response includes a valid UUID session_id (36 chars) |
| `test_chat_multi_turn` | Sends two messages with the same session_id, verifies the session history contains all 4 messages (Human, AI, Human, AI) in order |
| `test_chat_separate_sessions` | Sends two messages without session_id, verifies they get different session_ids with independent histories (2 messages each) |
| `test_chat_empty_message` | Sends an empty string message, verifies it returns 200 with an empty response |
| `test_chat_missing_message` | Sends a request with no `message` field, verifies 422 validation error |

## POST /chat/stream

| Test | Description |
| ------ | ------------- |
| `test_chat_stream` | Sends a message, verifies 200 response with `text/event-stream` content-type, structured JSON chunks with session_id, content, timestamp, and done flag. Intermediate chunks have `usage: null`, final `done: true` chunk includes token usage |
| `test_chat_stream_stores_history` | Verifies that after streaming completes, both the HumanMessage and the accumulated AIMessage are stored in the session history |
| `test_chat_stream_missing_message` | Sends a request with no `message` field, verifies 422 validation error |

## GET /chat/{session_id}/messages

| Test | Description |
| ------ | ------------- |
| `test_get_session_messages` | Creates a chat session, then fetches its messages. Verifies 200 response with correct session_id, 2 messages with proper roles (`user`/`assistant`) and content |
| `test_get_session_messages_not_found` | Requests messages for a nonexistent session_id, verifies 404 with "Session not found" detail |

## GET /

| Test | Description |
| ------ | ------------- |
| `test_home_page` | Verifies the home page returns 200 with HTML content-type, contains "Workflow API" title, and links to `/docs` and `/redoc` |
