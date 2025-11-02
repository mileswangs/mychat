"""
The MMLU dataset.
https://huggingface.co/datasets/cais/mmlu
"""

from datasets import load_dataset
from tasks.common import Task, render_mc


mmlu_groups = (
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
)


class MMLU(Task):
    letters = ("A", "B", "C", "D")
    groups = mmlu_groups

    def __init__(self, subset, split, **kwargs):
        super().__init__(**kwargs)
        assert subset in [
            "all",
            "auxiliary_train",
        ], f"subset {subset} must be 'all' or 'auxiliary_train'"
        assert split in [
            "train",
            "validation",
            "dev",
            "test",
        ], f"split {split} must be one of 'train', 'validation', 'dev', 'test'"
        if subset == "auxiliary_train":
            assert split == "train", "auxiliary_train subset only has train split"
        self.subset = subset
        self.split = split
        self.ds = load_dataset("cais/mmlu", name=subset, split=split).shuffle(seed=42)
        if subset == "auxiliary_train":
            # I don't understand why but the auxiliary_train rows have some weird additional 'train' wrapper
            self.ds = self.ds.map(lambda row: row["train"], remove_columns=["train"])

    @property
    def eval_type(self):
        return "categorical"

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        row = self.ds[index]
        question = row["question"]
        choices = row["choices"]
        answer_string = row["answer"]
        subject = row["subject"]
        # create and return conversation
        user_message = render_mc(question, self.letters, choices)
        messages = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": answer_string},
        ]
        conversation = {
            "messages": messages,
            "subject": subject,
            "letters": self.letters,
        }
        return conversation

    def evaluate(self, conversation, assistant_response):
        assert assistant_response in self.letters, f'MMLU answer "{assistant_response}" not in choices {self.letters}'
        assistant_message = conversation["messages"][-1]["content"]
        return assistant_message == assistant_response
