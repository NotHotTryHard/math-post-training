"""Published Qwen2.5-Math evaluation prompt protocols.

The Base prompts below are transcribed from Appendix B of the Qwen2.5-Math
technical report. They are deliberately kept in Python: these are fixed pieces
of an evaluation protocol, not experiment prose that should drift in YAML.
"""

# ruff: noqa: E501 -- line wrapping would change the frozen prompt text

QWEN_INSTRUCT_SYSTEM = "Please reason step by step, and put your final answer within \\boxed{}."


def build_evaluation_prompt(tokenizer, problem, benchmark, protocol):
    """Return the exact string sent to the model for one benchmark problem."""

    if protocol == "qwen2_5_math_instruct":
        messages = [
            {"role": "system", "content": QWEN_INSTRUCT_SYSTEM},
            {"role": "user", "content": problem},
        ]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    if protocol == "qwen2_5_math_base":
        if benchmark in {"gsm8k", "gsm1k"}:
            return GSM8K_BASE_PREFIX + problem + "\nLet's think step by step"
        if benchmark == "math":
            return MATH_BASE_PREFIX + problem + "\nSolution:"
        raise ValueError(f"No Qwen Base prompt is defined for benchmark {benchmark!r}")

    raise ValueError(f"Unknown evaluation protocol: {protocol!r}")


def get_evaluation_settings(protocol, benchmark):
    """Return generation and answer-parsing settings fixed by the protocol."""

    if protocol == "qwen2_5_math_instruct":
        return {
            "num_shots": 0,
            "answer_format": "boxed",
            "stop_strings": None,
        }
    if protocol == "qwen2_5_math_base":
        if benchmark in {"gsm8k", "gsm1k"}:
            return {
                "num_shots": 8,
                "answer_format": None,
                "stop_strings": ["Question:"],
            }
        if benchmark == "math":
            return {
                "num_shots": 4,
                "answer_format": None,
                "stop_strings": ["Problem:"],
            }
    raise ValueError(f"Unknown protocol/benchmark pair: {protocol!r}/{benchmark!r}")


# Fixed demonstrations from the paper live below the executable prompt logic.

GSM8K_BASE_PREFIX = r"""Question: In 2004, there were 60 kids at a cookout. In 2005, half the number of kids came to the cookout as compared to 2004. In 2006, 2/3 as many kids came to the cookout as in 2005. How many kids came to the cookout in 2006?
Let's think step by step
In 2005, 60/2=30 kids came to the cookout.
In 2006, 30/3*2=20 kids came to the cookout.
The answer is 20
Question: Zilla spent 7% of her monthly earnings on rent, half of it on her other monthly expenses, and put the rest in her savings. If she spent $133 on her rent, how much does she deposit into her savings account in a month?
Let's think step by step
Since $133 is equal to 7% of her earnings, then 1% is equal to $133/7 = $19.
The total monthly earning of Zilla is represented by 100%, so $19 x 100 = $1900 is her monthly earnings.
So, $1900/2 = $950 is spent on her other monthly expenses.
The total amount spent on the rent and other monthly expenses is $133 + $950 = $1083.
Hence, she saves $1900 -$1083 = $817 per month.
The answer is 817
Question: If Buzz bought a pizza with 78 slices at a restaurant and then decided to share it with the waiter in the ratio of 5:8, with Buzz's ratio being 5, what's twenty less the number of slices of pizza that the waiter ate?
Let's think step by step
The total ratio representing the slices of pizza that Buzz bought is 5+8=13
If he shared the slices of pizza with the waiter, the waiter received a fraction of 8/13 of the total number of slices, which totals 8/13 * 78 = 48 slices
Twenty less the number of slices of pizza that the waiter ate is 48-20 = 28
The answer is 28
Question: Jame gets a raise to $20 per hour and works 40 hours a week.  His old job was $16 an hour for 25 hours per week.  How much more money does he make per year in his new job than the old job if he works 52 weeks a year?
Let's think step by step
He makes 20*40=$800 per week
He used to make 16*25=$400 per week
So his raise was 800-400=$400 per week
So he makes 400*52=$20,800 per year more
The answer is 20800
Question: Mr. Gardner bakes 20 cookies, 25 cupcakes, and 35 brownies for his second-grade class of 20 students. If he wants to give each student an equal amount of sweet treats, how many sweet treats will each student receive?
Let's think step by step
Mr. Gardner bakes a total of 20 + 25 + 35 = 80 sweet treats
Each student will receive 80 / 20 = 4 sweet treats
The answer is 4
Question: A used car lot has 24 cars and motorcycles (in total) for sale. A third of the vehicles are motorcycles, and a quarter of the cars have a spare tire included. How many tires are on the used car lot’s vehicles in all?
Let's think step by step
The used car lot has 24 / 3 = 8 motorcycles with 2 tires each.
The lot has 24 -8 = 16 cars for sale
There are 16 / 4 = 4 cars with a spare tire with 5 tires each.
The lot has 16 -4 = 12 cars with 4 tires each.
Thus, the used car lot’s vehicles have 8 * 2 + 4 * 5 + 12 * 4 = 16 + 20 + 48 = 84 tires in all.
The answer is 84
Question: Norma takes her clothes to the laundry. She leaves 9 T-shirts and twice as many sweaters as T-shirts in the washer. When she returns she finds 3 sweaters and triple the number of T-shirts. How many items are missing?
Let's think step by step
Norma left 9 T-shirts And twice as many sweaters, she took 9 * 2= 18 sweaters
Adding the T-shirts and sweaters, Norma left 9 + 18 = 27 clothes
When she came back, she found 3 sweaters And triple the number of T-shirts, she found 3 * 3 = 9 T-shirts
Adding the T-shirts and sweaters, Norma found 3 + 9 = 12 clothes
Subtracting the clothes she left from the clothes she found, 27 -12 = 15 clothes are missing
The answer is 15
Question: Adam has an orchard. Every day for 30 days he picks 4 apples from his orchard. After a month, Adam has collected all the remaining apples, which were 230. How many apples in total has Adam collected from his orchard?
Let's think step by step
During 30 days Adam picked 4 * 30 = 120 apples.
So in total with all the remaining apples, he picked 120 + 230 = 350 apples from his orchard.
The answer is 350
Question: """

MATH_BASE_PREFIX = r"""Problem: Find the domain of the expression $\frac{\sqrt{x-2}}{\sqrt{5-x}}$.
Solution: The expressions inside each square root must be non-negative. Therefore, $x-2 \ge 0$, so $x\ge2$, and $5-x \ge 0$, so $x \le 5$. Also, the denominator cannot be equal to zero, so $5-x>0$, which gives $x<5$. Therefore, the domain of the expression is $\boxed{[2,5)}$. The answer is: $[2,5)$.
Problem: If $\det \mathbf{A} = 2$ and $\det \mathbf{B} = 12,$ then find $\det (\mathbf{A} \mathbf{B}).$
Solution: We have that $\det (\mathbf{A} \mathbf{B}) = (\det \mathbf{A})(\det \mathbf{B}) = (2)(12) = \boxed{24}.$ The answer is: $24$.
Problem: Terrell usually lifts two 20-pound weights 12 times. If he uses two 15-pound weights instead, how many times must Terrell lift them in order to lift the same total weight?
Solution: If Terrell lifts two 20-pound weights 12 times, he lifts a total of $2\cdot 12\cdot20=480$ pounds of weight. If he lifts two 15-pound weights instead for $n$ times, he will lift a total of $2\cdot15\cdot n=30n$ pounds of weight. Equating this to 480 pounds, we can solve for $n$: $30n=480 \Rightarrow n=480/30=\boxed{16}$. The answer is: $16$.
Problem: If the system of equations $6x-4y=a,$ $6y-9x=b$ has a solution $(x,y)$ where $x$ and $y$ are both nonzero, find $\frac{a}{b},$ assuming $b$ is nonzero.
Solution: If we multiply the first equation by $-\frac{3}{2}$, we obtain $6y-9x=-\frac{3}{2}a$. Since we also know that $6y-9x=b$, we have $-\frac{3}{2}a=b \Rightarrow \frac{a}{b}=\boxed{-\frac{2}{3}}$. The answer is: $-\frac{2}{3}$.
Problem: """
