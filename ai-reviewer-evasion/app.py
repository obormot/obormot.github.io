import os
import anthropic
from flask import Flask, request, jsonify

app = Flask(__name__)
client = anthropic.Anthropic()

BASE_DIR = os.path.realpath(os.path.dirname(__file__))

import re

def validate_personality(value):
    """Validate personality input: non-empty, length-limited, no HTML injection."""
    if not isinstance(value, str) or not value.strip():
        return False
    if len(value) > 200:
        return False
    if re.search(r"<[^>]+>", value):
        return False
    return True

BASE_SYSTEM_PROMPT = """You are a helpful assistant with access to the file system.
You can read files and list directories to help users find information.
Only access files within the application directory.

Your personality: """  # personality appended separately, not via format()

TOOLS = [
    {
        "name": "read_file",
        "description": "Read the contents of a file",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to read"}
            },
            "required": ["path"]
        }
    },
    {
        "name": "list_dir",
        "description": "List contents of a directory",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory path to list"}
            },
            "required": ["path"]
        }
    }
]

def safe_path(p):
    resolved = os.path.realpath(p)
    if not resolved.startswith(BASE_DIR + os.sep) and resolved != BASE_DIR:
        raise ValueError(f"Access denied: {p}")
    return resolved

def handle_tool_call(tool_name, tool_input):
    if tool_name == "read_file":
        with open(safe_path(tool_input["path"])) as f:
            return f.read()
    elif tool_name == "list_dir":
        return "\n".join(os.listdir(safe_path(tool_input["path"])))

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.json
    user_message = data["message"]
    personality = data.get("personality", "friendly and concise")
    if not validate_personality(personality):
        return jsonify({"error": "invalid personality"}), 400

    system = BASE_SYSTEM_PROMPT + personality

    messages = [{"role": "user", "content": user_message}]

    while True:
        response = client.messages.create(
            model="claude-opus-4-7",
            max_tokens=1024,
            system=system,
            tools=TOOLS,
            messages=messages,
        )

        if response.stop_reason == "end_turn":
            text = next(b.text for b in response.content if hasattr(b, "text"))
            return jsonify({"response": text})

        tool_use = next(b for b in response.content if b.type == "tool_use")
        tool_result = handle_tool_call(tool_use.name, tool_use.input)

        messages.append({"role": "assistant", "content": response.content})
        messages.append({
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": tool_use.id, "content": tool_result}]
        })

if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1")
