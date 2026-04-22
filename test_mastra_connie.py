"""
test_mastra_connie.py
─────────────────────
Tests calling Connie via Mastra API with multi-turn conversation history.
Uses the exact format seen in live production traces.

Run:
    python test_mastra_connie.py
"""

import requests
import json
import uuid

MASTRA_HOST = "https://mastra.staging.forgehg.cloud"
AGENT_ID    = "connie-agent"
URL         = f"{MASTRA_HOST}/api/agents/{AGENT_ID}/generate"

def call_connie(messages: list, thread_id: str = None) -> str:
    """
    Call Connie via Mastra API.
    
    messages: list of {"role": "user"|"assistant", "content": "..."}
    thread_id: unique ID to maintain conversation state across turns
    
    Returns Connie's text response.
    """
    if thread_id is None:
        thread_id = str(uuid.uuid4())

    payload = {
        "messages":   messages,
        "threadId":   thread_id,
        "resourceId": "eval-runner"
    }

    resp = requests.post(URL, json=payload, timeout=30)
    resp.raise_for_status()

    data = resp.json()
    return data.get("text", str(data)), thread_id


def run_multi_turn_conversation(turns: list) -> list:
    """
    Run a full multi-turn conversation with Connie.
    
    turns: list of user messages to send in sequence
    Returns: list of (user_message, connie_response) pairs
    """
    thread_id = str(uuid.uuid4())
    messages  = []
    results   = []

    for user_msg in turns:
        # Add user message to history
        messages.append({"role": "user", "content": user_msg})

        print(f"\n  User: {user_msg}")
        response, _ = call_connie(messages, thread_id)
        print(f"  Connie: {response}")

        # Add Connie's response to history for next turn
        messages.append({"role": "assistant", "content": response})
        results.append((user_msg, response))

    return results


# ── Test 1: Single turn ───────────────────────────────────────────────
print("=" * 60)
print("TEST 1: Single turn")
print("=" * 60)
response, tid = call_connie([
    {"role": "user", "content": "Hi, I'm looking for a holiday cottage"}
])
print(f"Response: {response}")
print(f"Thread ID: {tid}")

# ── Test 2: Multi-turn slot-filling ──────────────────────────────────
print("\n" + "=" * 60)
print("TEST 2: Multi-turn slot-filling conversation")
print("=" * 60)
results = run_multi_turn_conversation([
    "I'd like a cottage in Cornwall",
    "July 15th to 22nd",
    "Just 2 of us",
    "Budget around £800",
    "No other requirements, please search"
])

# ── Test 3: Multi-turn from dataset item ─────────────────────────────
print("\n" + "=" * 60)
print("TEST 3: Replay from dataset item (with conversation history)")
print("=" * 60)

# Simulates what the eval pipeline does — feeds in history + new message
conversation_history = [
    {"role": "ai",   "content": "Hello! Where are you thinking of visiting?"},
    {"role": "user", "content": "Cornwall"},
    {"role": "ai",   "content": "Cornwall is a brilliant choice! When are you planning to visit?"},
    {"role": "user", "content": "July 15-22"},
    {"role": "ai",   "content": "How many of you will be travelling?"},
]
user_message = "2 adults and a dog"

# Convert ai/user roles to assistant/user for Mastra
messages = []
for msg in conversation_history:
    role = "assistant" if msg["role"] == "ai" else "user"
    messages.append({"role": role, "content": msg["content"]})
messages.append({"role": "user", "content": user_message})

print(f"  Sending {len(messages)} messages (history + new turn)")
response, tid = call_connie(messages)
print(f"  Connie: {response}")

print("\n✓ All tests complete")
