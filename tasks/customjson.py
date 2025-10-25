"""
CustomJSON task for loading conversations from JSONL files.
Each line in the JSONL file should be a JSON array of messages.
"""

import os
import json
from tasks.common import Task


class CustomJSON(Task):
    """
    Load conversations from a JSONL file.
    Each line should be a JSON array of message objects with 'role' and 'content' fields.
    Example line: [{"role":"user","content":"Hi"},{"role":"assistant","content":"Hello"}]
    """

    def __init__(self, filepath, **kwargs):
        super().__init__(**kwargs)
        self.filepath = filepath
        self.conversations = []

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File {filepath} does not exist.")

        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                messages = json.loads(line)
                # validate messages
                assert isinstance(
                    messages, list
                ), f"expected list of messages, got {type(messages)}"
                assert (
                    len(messages) >= 2
                ), "each conversation must have at least one message"

                for i, msg in enumerate(messages):
                    assert "role" in msg, f"message {i} is missing 'role' field"
                    assert "content" in msg, f"message {i} is missing 'content' field"
                    expected_role = "user" if i % 2 == 0 else "assistant"
                    assert (
                        msg["role"] == expected_role
                    ), f"message {i} has role {msg['role']}, expected {expected_role}"
                    assert isinstance(
                        msg["content"], str
                    ), f"message {i} content must be a string"

                self.conversations.append(messages)
        self.length = len(self.conversations)

    def num_examples(self):
        return self.length

    def get_example(self, index):
        messages = self.conversations[index]
        conversation = {"messages": messages}
        return conversation
