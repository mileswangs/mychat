"""
SmolTalk by HuggingFace. Good "general" conversational dataset.
https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk
We use the "smol" version, which is more appropriate for smaller models.
"""

from datasets import load_dataset
from tasks.common import Task


class SmolTalk(Task):
    """smol-smoltalk dataset. train is 460K rows, test is 24K rows."""

    def __init__(self, split, **kwargs):
        super().__init__(**kwargs)
        assert split in ["train", "test"]
        self.ds = load_dataset("HuggingFaceTB/smol-smoltalk", split=split).shuffle(seed=42)
        self.length = len(self.ds)

    def num_examples(self):
        return self.length

    def get_example(self, index):
        row = self.ds[index]
        messages = row["messages"]
        # ---------------------------------------------------------------------
        # sanity checking asserts here
        # TODO: we could remove these asserts later, for now just don't want any footguns
        # there is an optional system message at the beginning

        assert len(messages) >= 1
        first_message = messages[0]
        if first_message["role"] == "system":
            rest_messages = messages[1:]
        else:
            rest_messages = messages

        assert len(rest_messages) >= 2, "SmolTalk messages must have at least 2 messages"

        for i, msg in enumerate(rest_messages):
            expected_role = "user" if i % 2 == 0 else "assistant"
            assert msg["role"] == expected_role, f"SmolTalk message {i} role must be {expected_role}"
            assert isinstance(msg["content"], str), f"content must be a string"

        conversation = {"messages": messages}
        return conversation
