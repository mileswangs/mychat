"""
GSM8K evaluation.
https://huggingface.co/datasets/openai/gsm8k

Example problem instance:

Question:
Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?
Answer:
Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute.
Working 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10.
#### 10

Notice that GSM8K uses tool calls inside << >> tags.
"""

import re
from datasets import load_dataset
from tasks.common import Task

GSM_RE = re.compile(r"#### (\-?[0-9\.\,]+)")


def extract_answer(completion: str) -> str:
    match = GSM_RE.search(completion)
    if not match:
        return None
    match_str = match.group(1).strip()
    match_str = match_str.replace(",", "")
    return match_str


class GSM8K(Task):
    def __init__(self, subset, split, **kwargs):
        super().__init__(**kwargs)
        assert subset in ["main", "socratic"]
        assert split in ["train", "validation", "test"]
        self.ds = load_dataset("openai/gsm8k", subset, split=split).shuffle(seed=42)

    @property
    def eval_type(self):
        return "generative"

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        row = self.ds[index]
        question = row["question"]
        answer = row["answer"]
        # create and return conversation
        # extract tool use from answer
        assistant_message_parts = []
        parts = re.split(r"(<<[^>]+>>)", answer)
        for part in parts:
            if part.startswith("<<") and part.endswith(">>"):
                # tool use, convert to inline code block
                inner = part[2:-2]
                if "=" in inner:
                    expr, result = inner.split("=", 1)
                else:
                    expr, result = inner, ""
                assistant_message_parts.append(
                    {
                        "type": "python",
                        "text": expr,
                    }
                )
                assistant_message_parts.append(
                    {
                        "type": "python_output",
                        "text": result,
                    }
                )
            else:
                assistant_message_parts.append(
                    {
                        "type": "text",
                        "text": part,
                    }
                )
        messages = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": assistant_message_parts},
        ]
        conversation = {"messages": messages}
        return conversation

    def evaluate(self, conversation, assistant_response):
        """
        Given (conversation, completion), return evaluation outcome (0 = wrong, 1 = correct)
        Note that:
        - the conversation has both user AND assistant message (containing the ground truth answer)
        - the assistant_response is usually the alternative assistant message achieved via sampling

        TODO: Technically, assistant_response should be a Message (either a string or a list of parts)
              We can handle this later possibly. For now just assume string.
        """
        assert isinstance(assistant_response, str), f"Expected assistant_response to be str but got {type(assistant_response)}"

        assistant_message = conversation["messages"][-1]
        assert assistant_message["role"] == "assistant", f"Expected last message to be assistant but got {assistant_message['role']}"

        assert isinstance(assistant_message["content"], list), "Expected assistant message content to be list of parts"
        last_text_part = assistant_message["content"][-1]["text"]
        ref_num = extract_answer(last_text_part)
        pred_num = extract_answer(assistant_response)
        is_correct = int(ref_num == pred_num)
        return is_correct

    def reward(self, conversation, assistant_response):
        is_correct = self.evaluate(conversation, assistant_response)
        return float(is_correct)
