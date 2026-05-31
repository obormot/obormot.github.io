import os
import json
import re
from openai import OpenAI
from flask import Flask, request, jsonify

app = Flask(__name__)

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)

MODEL = "nvidia/nemotron-3-super-120b-a12b:free"

BASE_DIR = os.path.realpath(os.path.dirname(__file__))

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
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file to read"}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List contents of a directory",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path to list"}
                },
                "required": ["path"]
            }
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
    return "Unknown tool"

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.json
    user_message = data["message"]
    personality = data.get("personality", "friendly and concise")
    if not validate_personality(personality):
        return jsonify({"error": "invalid personality"}), 400

    system = BASE_SYSTEM_PROMPT + personality

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_message},
    ]

    while True:
        print(f"[DEBUG] Sending {len(messages)} messages to model", flush=True)
        response = client.chat.completions.create(
            model=MODEL,
            tools=TOOLS,
            messages=messages,
        )
        print(f"[DEBUG] stop_reason={response.choices[0].finish_reason}", flush=True)

        msg = response.choices[0].message
        print(f"[DEBUG] content={msg.content!r}", flush=True)
        print(f"[DEBUG] tool_calls={msg.tool_calls}", flush=True)

        if not msg.tool_calls:
            return jsonify({"response": msg.content})

        tool_call = msg.tool_calls[0]
        tool_input = json.loads(tool_call.function.arguments)
        print(f"[DEBUG] tool={tool_call.function.name} input={tool_input}", flush=True)
        tool_result = handle_tool_call(tool_call.function.name, tool_input)
        print(f"[DEBUG] tool_result={tool_result[:200]!r}", flush=True)

        messages.append(msg)
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": tool_result,
        })

if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1")
