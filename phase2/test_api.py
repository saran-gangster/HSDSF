#!/usr/bin/env python3
"""Quick test to verify Cerebras API key works."""

import os
import sys
from pathlib import Path

# Add parent to path to import from phase2
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

# Load .env
load_dotenv()

api_key = os.environ.get("CEREBRAS_API_KEY")
if not api_key:
    print("ERROR: CEREBRAS_API_KEY not set in environment or .env file")
    sys.exit(1)

print(f"API key found: {api_key[:20]}...{api_key[-4:]}")
print("Testing Cerebras API connection...")

try:
    from cerebras.cloud.sdk import Cerebras

    client = Cerebras(api_key=api_key)
    
    print("Sending test request to zai-glm-4.6...")
    resp = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say 'Hello from Cerebras' in JSON format with key 'message'"},
        ],
        model="zai-glm-4.6",
        response_format={
            "type": "json_object",
        },
        max_completion_tokens=100,
        temperature=0.2,
    )
    
    print(f"Response received: {resp.choices[0].message.content}")
    print("\n✅ API connection successful!")
    
except Exception as e:
    print(f"\n❌ API test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
