# assistant.py

import json
import urllib.error
import urllib.request

from ai_tools import (
    count_entries_today,
    find_repeat_vehicles,
    summarize_today,
    count_entries_for_plate,
    find_busiest_hour_today,
)


OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
MODEL = "llama3.1:8b"

# Only these predefined tools are available to the model.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "count_entries_today",
            "description": "Count the vehicle entries recorded today.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_repeat_vehicles",
            "description": (
                "Find vehicle registration numbers that have been recorded "
                "more than once across all dates in the database."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_today",
            "description": (
                "Summarize today's vehicle activity, including total entries, "
                "unique vehicles, repeated vehicles, and first and last entries."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "count_entries_for_plate",
            "description": (
                "Count how many times a specified registration number "
                "appears in the database and return its recorded entries."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "plate_number": {
                        "type": "string",
                        "description": "The vehicle registration number.",
                    }
                },
                "required": ["plate_number"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_busiest_hour_today",
            "description": (
                "Find the hour or hours with the most vehicle entries today."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]


# Allowlist: the model cannot call arbitrary Python functions.
TOOL_FUNCTIONS = {
    "count_entries_today": count_entries_today,
    "find_repeat_vehicles": find_repeat_vehicles,
    "summarize_today": summarize_today,
    "count_entries_for_plate": count_entries_for_plate,
    "find_busiest_hour_today": find_busiest_hour_today,
}


SYSTEM_PROMPT = """
You are the AI assistant for PlateWatch, a vehicle-entry recording system.

Rules:
1. Use the available tools to answer questions about actual vehicle records.
2. Never invent vehicle counts, registration numbers, dates, or times.
3. If the user asks about today's activity, use the relevant today tool.
4. If the user asks about a specific registration number, use the
   count_entries_for_plate tool with that number.
5. You may call more than one tool if a question requires multiple results.
6. Explain results clearly in natural language.
7. If the database has no matching records, say so.
8. Do not claim that the database contains information that the tools
   did not return.
9. If a question cannot be answered with the available tools, explain
   what information is currently supported.
10. Do not generate SQL or attempt to access the database directly.
"""


def call_ollama(messages):
    """Send a chat request to the local Ollama API."""
    payload = {
        "model": MODEL,
        "messages": messages,
        "tools": TOOLS,
        "stream": False,
    }

    request = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            result = json.loads(response.read().decode("utf-8"))

        return result["message"]

    except urllib.error.HTTPError as exc:
        # Show the actual error returned by Ollama.
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Ollama API error (HTTP {exc.code}): {details}"
        ) from exc

    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Cannot connect to Ollama at {OLLAMA_URL}. "
            f"Details: {exc.reason}"
        ) from exc

    except (KeyError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Ollama returned an unexpected response: {exc}"
        ) from exc


def ask_platewatch(question):
    """
    Ask a natural-language question.

    The LLM chooses tools dynamically. This function does not match
    the user's question against predefined question phrases.
    """
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": question,
        },
    ]

    # Prevent an unlimited tool-calling loop.
    max_tool_rounds = 5

    for _ in range(max_tool_rounds):
        assistant_message = call_ollama(messages)
        messages.append(assistant_message)

        tool_calls = assistant_message.get("tool_calls", [])

        # No more tool calls: the model has produced its final answer.
        if not tool_calls:
            return assistant_message.get(
                "content",
                "I could not generate an answer.",
            )

        for tool_call in tool_calls:
            function_info = tool_call.get("function", {})
            tool_name = function_info.get("name")
            arguments = function_info.get("arguments", {})

            if tool_name not in TOOL_FUNCTIONS:
                tool_result = {
                    "error": "This tool is not permitted."
                }
            else:
                try:
                    if isinstance(arguments, str):
                        arguments = json.loads(arguments)

                    if not isinstance(arguments, dict):
                        raise ValueError(
                            "Tool arguments must be a JSON object."
                        )

                    tool_result = TOOL_FUNCTIONS[tool_name](**arguments)

                except Exception as exc:
                    # Report the error without exposing a full traceback.
                    tool_result = {
                        "error": str(exc)
                    }

            messages.append(
                {
                    "role": "tool",
                    "content": json.dumps(
                        tool_result,
                        ensure_ascii=False,
                        default=str,
                    ),
                }
            )

    return (
        "I couldn't complete the request within the allowed number "
        "of tool calls. Please try a more specific question."
    )


def main():
    print("=" * 55)
    print("PlateWatch AI Assistant")
    print(f"Model: {MODEL}")
    print("Ask about your vehicle-entry records.")
    print("Type 'exit' to quit.")
    print("=" * 55)

    while True:
        try:
            question = input("\nYou: ").strip()

            if question.lower() in {"exit", "quit"}:
                print("Goodbye!")
                break

            if not question:
                continue

            answer = ask_platewatch(question)
            print(f"\nPlateWatch AI: {answer}")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break

        except Exception as exc:
            print(f"\nError: {exc}")


if __name__ == "__main__":
    main()