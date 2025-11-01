"""
Evaluate compression ratio of the tokenizer.
"""

from mychat.tokenizer import get_tokenizer, RustBPETokenizer
from mychat.dataset import parquets_iter_batched

eng_text = """
Trump said while travelling to Asia on Saturday that he was "increasing the Tariff on Canada by 10% over and above what they are paying now". Tariffs are paid by the companies that import foreign products, not the exporters themselves.

Three-quarters of Canadian exports are sold to the US, and Ontario is home to the bulk of Canada's automobile manufacturing.

US-Canada trade minister Dominic LeBlanc said of the tariff increase: "We stand ready to build on the progress made in constructive discussions with American counterparts over the course of recent weeks.

"We will remain focused on achieving results that benefit workers and families in both the United States and Canada, and that progress is best achieved through direct engagement with the US administration."

Trump's decision comes after Ontario Premier Doug Ford said on Friday that he would pause the anti-tariff advertising campaign "so that trade talks can resume", after discussions with Prime Minister Mark Carney.

But he said it would still appear during games for the World Series, including between the Toronto Blue Jays and the Los Angeles Dodgers.

Trump responded that the advert should have been pulled down "IMMEDIATELY". A spokesperson for Ford stood by his statement on Friday.

The advert, sponsored by the Ontario government, quotes former US President Ronald Reagan, a Republican and icon of US conservatism, saying tariffs "hurt every American".

The video takes excerpts from a 1987 national radio address that focused on foreign trade.
""".strip()

cn_text = """
一组盲人患者在眼底植入了一个物件后，如今能够重新閱读。

一位在伦敦摩尔菲尔德眼科医院为五名患者植入晶片的外科医生表示，这项国际试验的结果“令人震惊”。

70岁的注册盲人希拉·欧文（Sheila Irvine）告诉BBC，能够再次閱读和玩填字游戏真是“不可思议”。“这太美妙了，太棒了。这让我非常高兴。”

这项技术为地图样萎缩症（GA）患者带来了希望。 “地图样萎缩症”是一种晚期干性老年黄斑部病变（AMD），据估计英国约有35万人患有这种疾病。
""".strip()

code_text = """
class Humaneval(Task):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ds = load_dataset("openai/openai_humaneval", split="test").shuffle(seed=42)

    @property
    def eval_type(self):
        return "generative"

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        row = self.ds[index]
        prompt = row["prompt"]
        solution = row["canonical_solution"]
        entry_point = row["entry_point"]
        test = row["test"]
        complete_solution = f"{prompt}\n{solution}"
        # create and return conversation
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": complete_solution},
        ]
        conversation = {
            "messages": messages,
            "entry_point": entry_point,
            "test": test,
        }
        return conversation

    def evaluate(self, conversation, completion):
        # the prompt will contain the imports and the function signature
        imports = extract_imports(conversation["messages"][-1]["content"])
        # the completion will usually contain the whole function
        # but not always with the needed imports, so we manually append them
        completion_code = extract_program(completion)
        program = (
            imports
            + "\n\n"
            + completion_code
            + "\n\n"
            + conversation["test"]
            + "\n"
            + f"check({conversation['entry_point']})"
        )
        result = execute_code(program, timeout=5)
        return result.success
""".strip()

math_text = r"""
\documentclass[12pt]{article}
\usepackage{amsmath,amsthm,amssymb}
\usepackage[margin=1in]{geometry}

\newtheorem{theorem}{Theorem}
\newtheorem*{remark}{Remark}

\begin{document}

\begin{center}
{\Large A Cute Identity: The Sum of Cubes is a Square}
\end{center}

\begin{theorem}
For every integer $n \ge 1$,
\[
\sum_{k=1}^{n} k^{3} \;=\; \left(\frac{n(n+1)}{2}\right)^{2}.
\]
\end{theorem}

\begin{proof}[Proof 1 (Induction)]
Let $S(n) = \sum_{k=1}^{n} k^3$. For $n=1$, $S(1)=1=(1\cdot 2/2)^2$, so the base case holds.

Assume $S(n)=\big(\tfrac{n(n+1)}{2}\big)^2$ for some $n\ge 1$.
Then
\[
S(n+1)
= S(n) + (n+1)^3
= \left(\frac{n(n+1)}{2}\right)^2 + (n+1)^3.
\]
Factor out $(n+1)^2$:
\[
S(n+1)
= (n+1)^2\left( \frac{n^2}{4} + (n+1) \right)
= (n+1)^2\left( \frac{n^2 + 4n + 4}{4} \right)
= (n+1)^2\left( \frac{(n+2)^2}{4} \right).
\]
Thus
\[
S(n+1)=\left(\frac{(n+1)(n+2)}{2}\right)^2,
\]
which matches the claimed formula with $n$ replaced by $n+1$. By induction, the identity holds for all $n\ge 1$.
\end{proof}

\begin{proof}[Proof 2 (Algebraic telescoping)]
Recall the binomial identity
\[
(k+1)^4 - k^4 = 4k^3 + 6k^2 + 4k + 1.
\]
Summing both sides from $k=0$ to $n$ telescopes:
\[
(n+1)^4 - 0^4
= \sum_{k=0}^{n}\big(4k^3 + 6k^2 + 4k + 1\big)
= 4\sum_{k=1}^{n}k^3 + 6\sum_{k=1}^{n}k^2 + 4\sum_{k=1}^{n}k + (n+1).
\]
Using the standard sums
\[
\sum_{k=1}^{n}k = \frac{n(n+1)}{2}
\quad\text{and}\quad
\sum_{k=1}^{n}k^2 = \frac{n(n+1)(2n+1)}{6},
\]
solve for $\sum_{k=1}^{n}k^3$ to get
\[
\sum_{k=1}^{n}k^3 = \left(\frac{n(n+1)}{2}\right)^2.
\]
\end{proof}

\begin{remark}
Geometrically, the identity says: ``adding up $1^3,2^3,\dots,n^3$ builds a perfect square’’—namely the square of the $n$th triangular number. This is why one sometimes calls it the \emph{sum-of-cubes is a square} phenomenon.
\end{remark}

\end{document}
""".strip()

math_text = r"""
\documentclass[12pt]{article}
\usepackage{amsmath,amsthm,amssymb}
\usepackage[margin=1in]{geometry}

\newtheorem{theorem}{Theorem}
\newtheorem*{remark}{Remark}

\begin{document}

\begin{center}
{\Large A Cute Identity: The Sum of Cubes is a Square}
\end{center}

\begin{theorem}
For every integer $n \ge 1$,
\[
\sum_{k=1}^{n} k^{3} \;=\; \left(\frac{n(n+1)}{2}\right)^{2}.
\]
\end{theorem}

\begin{proof}[Proof 1 (Induction)]
Let $S(n) = \sum_{k=1}^{n} k^3$. For $n=1$, $S(1)=1=(1\cdot 2/2)^2$, so the base case holds.

Assume $S(n)=\big(\tfrac{n(n+1)}{2}\big)^2$ for some $n\ge 1$.
Then
\[
S(n+1)
= S(n) + (n+1)^3
= \left(\frac{n(n+1)}{2}\right)^2 + (n+1)^3.
\]
Factor out $(n+1)^2$:
\[
S(n+1)
= (n+1)^2\left( \frac{n^2}{4} + (n+1) \right)
= (n+1)^2\left( \frac{n^2 + 4n + 4}{4} \right)
= (n+1)^2\left( \frac{(n+2)^2}{4} \right).
\]
Thus
\[
S(n+1)=\left(\frac{(n+1)(n+2)}{2}\right)^2,
\]
which matches the claimed formula with $n$ replaced by $n+1$. By induction, the identity holds for all $n\ge 1$.
\end{proof}

\begin{proof}[Proof 2 (Algebraic telescoping)]
Recall the binomial identity
\[
(k+1)^4 - k^4 = 4k^3 + 6k^2 + 4k + 1.
\]
Summing both sides from $k=0$ to $n$ telescopes:
\[
(n+1)^4 - 0^4
= \sum_{k=0}^{n}\big(4k^3 + 6k^2 + 4k + 1\big)
= 4\sum_{k=1}^{n}k^3 + 6\sum_{k=1}^{n}k^2 + 4\sum_{k=1}^{n}k + (n+1).
\]
Using the standard sums
\[
\sum_{k=1}^{n}k = \frac{n(n+1)}{2}
\quad\text{and}\quad
\sum_{k=1}^{n}k^2 = \frac{n(n+1)(2n+1)}{6},
\]
solve for $\sum_{k=1}^{n}k^3$ to get
\[
\sum_{k=1}^{n}k^3 = \left(\frac{n(n+1)}{2}\right)^2.
\]
\end{proof}
""".strip()

science_text = r"""
Photosynthesis is a photochemical energy transduction process in which light-harvesting pigment–protein complexes within the thylakoid membranes of oxygenic phototrophs absorb photons and initiate charge separation at the reaction center, driving the linear electron transport chain from water to NADP⁺ via photosystem II, the cytochrome b₆f complex, and photosystem I, concomitantly generating a trans-thylakoid proton motive force utilized by chloroplastic ATP synthase. The light-dependent reactions produce ATP and NADPH, which fuel the Calvin–Benson–Bassham cycle in the stroma, wherein ribulose-1,5-bisphosphate is carboxylated by ribulose-1,5-bisphosphate carboxylase/oxygenase (RuBisCO) to form 3-phosphoglycerate, subsequently reduced and regenerated through a series of enzymatic steps, enabling net assimilation of CO₂ into triose phosphates and ultimately carbohydrates. This process is tightly regulated by photoprotective mechanisms, redox feedback, and metabolite flux, representing a central biochemical pathway coupling solar energy capture to the biosphere’s primary productivity.
""".strip()

train_docs = next(parquets_iter_batched(split="train"))
train_text = "\n".join(train_docs).strip()
val_docs = next(parquets_iter_batched(split="val"))
val_text = "\n".join(val_docs).strip()

all_text = [
    ("news", eng_text),
    ("chinese", cn_text),
    ("code", code_text),
    ("math", math_text),
    ("science", science_text),
    ("fwe-train", train_text),
]

if val_text:
    all_text.append(("fwe-val", val_text))
tokenizer_results = {}
vocab_sizes = {}

for tokenizer_name in ["gpt2", "gpt4", "ours"]:
    if tokenizer_name == "gpt2":
        tokenizer = RustBPETokenizer.from_pretrained("gpt2")
    elif tokenizer_name == "gpt4":
        tokenizer = RustBPETokenizer.from_pretrained("cl100k_base")
    else:
        tokenizer = get_tokenizer()

    vocab_sizes[tokenizer_name] = tokenizer.get_vocab_size()
    tokenizer_results[tokenizer_name] = {}

    for name, text in all_text:
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        assert decoded == text, f"Decoded text does not match original for {tokenizer_name}"

        encoded_bytes = text.encode("utf-8")
        ratio = len(encoded_bytes) / len(encoded)
        tokenizer_results[tokenizer_name][name] = {
            "bytes": len(encoded_bytes),
            "tokens": len(encoded),
            "ratio": ratio,
        }

# ANSI color codes
GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"

# Print vocab sizes
print(f"\nVocab sizes:")
print(f"GPT-2: {vocab_sizes['gpt2']}")
print(f"GPT-4: {vocab_sizes['gpt4']}")
print(f"Ours: {vocab_sizes['ours']}")


def print_comparison(baseline_name, baseline_results, ours_results, all_text):
    """Print comparison table between baseline tokenizer and ours."""
    print(f"\nComparison with {baseline_name}:")
    print("=" * 95)
    print(
        f"{'Text Type':<10} {'Bytes':<8} {baseline_name:<15} {'Ours':<15} {'Relative':<12} {'Better':<10}"
    )
    print(f"{'':10} {'':8} {'Tokens':<7} {'Ratio':<7} {'Tokens':<7} {'Ratio':<7} {'Diff %':<12}")
    print("-" * 95)

    for name, text in all_text:
        baseline_data = baseline_results[name]
        ours_data = ours_results[name]

        # Calculate relative difference (positive means ours is better, negative means worse)
        # Using tokens: fewer tokens is better, so we calculate (baseline_tokens - ours_tokens) / baseline_tokens
        relative_diff = (
            (baseline_data["tokens"] - ours_data["tokens"]) / baseline_data["tokens"]
        ) * 100

        # Determine which has better compression (higher ratio = better)
        if baseline_data["ratio"] > ours_data["ratio"]:
            baseline_color, ours_color = GREEN, RED
            better = baseline_name
            diff_color = RED
        elif ours_data["ratio"] > baseline_data["ratio"]:
            baseline_color, ours_color = RED, GREEN
            better = "Ours"
            diff_color = GREEN
        else:
            baseline_color, ours_color = "", ""
            better = "Tie"
            diff_color = ""

        print(
            f"{name:<10} {baseline_data['bytes']:<8} "
            f"{baseline_color}{baseline_data['tokens']:<7}{RESET} "
            f"{baseline_color}{baseline_data['ratio']:<7.2f}{RESET} "
            f"{ours_color}{ours_data['tokens']:<7}{RESET} "
            f"{ours_color}{ours_data['ratio']:<7.2f}{RESET} "
            f"{diff_color}{relative_diff:+7.1f}%{RESET}     "
            f"{better:<10}"
        )


# Print comparisons
print_comparison("GPT-2", tokenizer_results["gpt2"], tokenizer_results["ours"], all_text)
print_comparison("GPT-4", tokenizer_results["gpt4"], tokenizer_results["ours"], all_text)

# Log to report
from mychat.report import get_report

lines = []
for baseline_name in ["GPT-2", "GPT-4"]:
    baseline_key = baseline_name.lower().replace("-", "")
    baseline_results = tokenizer_results[baseline_key]
    ours_results = tokenizer_results["ours"]
    lines.append(f"### Comparison with {baseline_name}")
    lines.append("")
    lines.append(
        "| Text Type | Bytes | "
        + baseline_name
        + " Tokens | "
        + baseline_name
        + " Ratio | Ours Tokens | Ours Ratio | Relative Diff % |"
    )
    lines.append(
        "|-----------|-------|--------------|--------------|-------------|------------|-----------------|"
    )
    for name, text in all_text:
        baseline_data = baseline_results[name]
        ours_data = ours_results[name]
        relative_diff = (
            (baseline_data["tokens"] - ours_data["tokens"]) / baseline_data["tokens"]
        ) * 100
        lines.append(
            f"| {name} | {baseline_data['bytes']} | {baseline_data['tokens']} | {baseline_data['ratio']:.2f} | {ours_data['tokens']} | {ours_data['ratio']:.2f} | {relative_diff:+.1f}% |"
        )
    lines.append("")
report_markdown = "\n".join(lines)
get_report().log(
    section="Tokenizer evaluation",
    data=[
        report_markdown,
    ],
)
