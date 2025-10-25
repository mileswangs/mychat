import random


class Task:
    def __init__(self, start=0, stop=None, step=1):
        assert start >= 0, f"start must be non-negative but got {start}"
        assert (
            stop is None or stop > start
        ), f"stop must be None or greater than start but got {stop}"
        assert step > 0, f"step must be positive but got {step}"
        self.start = start
        self.stop = stop
        self.step = step

    @property
    def eval_type(self):
        # one of "generative", "categorial"
        raise NotImplementedError

    def num_examples(self):
        raise NotImplementedError

    def get_example(self, index):
        raise NotImplementedError

    def __len__(self):
        start = self.start
        stop = self.num_examples() if self.stop is None else self.stop
        step = self.step
        span = stop - start
        num = (span + step - 1) // step  # ceil_div(span, step)
        assert num >= 0, f"Negative number of examples???: {num}"  # prevent footguns
        return num

    def __getitem__(self, index: int):
        assert isinstance(index, int), f"Index must be int but got {type(index)}"
        physical_index = self.start + index * self.step
        conversations = self.get_example(physical_index)
        return conversations

    def evaluate(self, problem, completion):
        raise NotImplementedError


class TaskMixture(Task):
    """
    For SFT Training it becomes useful to train on a tax mixture of datasets.
    Fun trick: if you wish to oversample any task, just pass it in multiple times in the list.
    """

    def __init__(self, tasks, **kwargs):
        super().__init__(**kwargs)
        self.tasks = tasks
        self.lengths = [len(task) for task in self.tasks]
        self.num_conversations = sum(self.lengths)
        # build list of (task_idx, local_idx) pairs
        self.index_map = []
        for task_idx, task_length in enumerate(self.lengths):
            for local_idx in range(task_length):
                self.index_map.append((task_idx, local_idx))
        # shuffle to mix tasks
        rng = random.Random(42)
        rng.shuffle(self.index_map)

    def num_examples(self):
        return self.num_conversations

    def get_conversation(self, index):
        assert (
            index < self.num_conversations
        ), f"Index {index} out of mixture range {self.num_conversations}"
        task_idx, local_idx = self.index_map[index]
        return self.tasks[task_idx].get_example(local_idx)
