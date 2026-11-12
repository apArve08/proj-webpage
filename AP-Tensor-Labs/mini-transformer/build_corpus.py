"""Builds the training text for the AP Tensor Lab mini transformer.

Everything the model will ever "know" is written here. The output is a list of short
lowercase sentences, split into a training set and a small held-out set of phrasings
the model never sees (used to check it learned patterns, not only whole sentences).
"""
import json, random, sys

random.seed(7)
train, heldout = [], []          # heldout items: (prompt, expected_next_word, family)


def add(sentence, weight=1):
    train.extend([sentence] * weight)


# ---------------------------------------------------------------- capitals
CAPITALS = [
    ("france", "paris"), ("germany", "berlin"), ("italy", "rome"), ("spain", "madrid"), ("japan", "tokyo"),
    ("china", "beijing"), ("malaysia", "kuala lumpur"), ("thailand", "bangkok"), ("indonesia", "jakarta"),
    ("vietnam", "hanoi"), ("korea", "seoul"), ("england", "london"), ("russia", "moscow"), ("egypt", "cairo"),
    ("canada", "ottawa"), ("australia", "canberra"), ("brazil", "brasilia"), ("portugal", "lisbon"),
    ("greece", "athens"), ("turkey", "ankara"), ("norway", "oslo"), ("sweden", "stockholm"),
    ("finland", "helsinki"), ("ireland", "dublin"), ("poland", "warsaw"), ("austria", "vienna"),
    ("belgium", "brussels"), ("philippines", "manila"), ("kenya", "nairobi"), ("peru", "lima"),
    ("cuba", "havana"), ("iran", "tehran"), ("iraq", "baghdad"), ("nepal", "kathmandu"),
    ("denmark", "copenhagen"), ("hungary", "budapest"), ("india", "delhi"), ("netherlands", "amsterdam"),
]
held_caps = set(random.sample(range(len(CAPITALS)), 8))
for i, (c, k) in enumerate(CAPITALS):
    add(f"the capital of {c} is {k} .", 3)
    add(f"{k} is the capital of {c} .", 2)
    add(f"{k} is a city in {c} .", 2)
    add(f"what is the capital of {c} ? {k} .", 2)
    if i in held_caps:
        heldout.append((f"the capital city of {c} is", k.split()[0], "capital (new phrasing)"))
    else:
        add(f"the capital city of {c} is {k} .", 2)

# ---------------------------------------------------------------- opposites
OPPOSITES = [
    ("hot", "cold"), ("big", "small"), ("up", "down"), ("fast", "slow"), ("happy", "sad"), ("day", "night"),
    ("open", "closed"), ("wet", "dry"), ("old", "young"), ("early", "late"), ("hard", "soft"), ("high", "low"),
    ("long", "short"), ("full", "empty"), ("rich", "poor"), ("strong", "weak"), ("thick", "thin"),
    ("loud", "quiet"), ("near", "far"), ("left", "right"), ("in", "out"), ("on", "off"), ("true", "false"),
    ("good", "bad"), ("first", "last"), ("start", "stop"), ("win", "lose"), ("push", "pull"), ("buy", "sell"),
    ("love", "hate"), ("clean", "dirty"), ("easy", "difficult"), ("inside", "outside"), ("above", "below"),
    ("before", "after"), ("top", "bottom"), ("black", "white"), ("yes", "no"),
]
held_opp = set(random.sample(range(len(OPPOSITES)), 8))
for i, (a, b) in enumerate(OPPOSITES):
    for x, y in ((a, b), (b, a)):
        add(f"the opposite of {x} is {y} .", 3)
        add(f"{x} is the opposite of {y} .", 2)
        add(f"what is the opposite of {x} ? {y} .", 2)
    add(f"{a} and {b} are opposites .", 2)
    if i in held_opp:
        heldout.append((f"not {a} means", b, "opposite (new phrasing)"))
    else:
        add(f"not {a} means {b} .", 2)
        add(f"not {b} means {a} .", 2)

# ---------------------------------------------------------------- numbers
NUM = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
       "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty"]
pairs = [(a, b) for a in range(11) for b in range(11)]
held_sum = set(random.sample(pairs, 14))
for a, b in pairs:
    c = NUM[a + b]
    add(f"{NUM[a]} plus {NUM[b]} is {c} .", 2)
    add(f"what is {NUM[a]} plus {NUM[b]} ? {c} .", 2)
    if (a, b) in held_sum:
        heldout.append((f"{NUM[a]} plus {NUM[b]} equals", c, "sum (new phrasing)"))
    else:
        add(f"{NUM[a]} plus {NUM[b]} equals {c} .", 2)
for a in range(11):
    for b in range(a + 1):
        add(f"{NUM[a]} minus {NUM[b]} equals {NUM[a - b]} .", 2)
        add(f"{NUM[a]} minus {NUM[b]} is {NUM[a - b]} .", 1)
for n in range(20):
    add(f"the number after {NUM[n]} is {NUM[n + 1]} .", 2)
    add(f"the number before {NUM[n + 1]} is {NUM[n]} .", 2)

# ---------------------------------------------------------------- days and months
DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
          "november", "december"]
for seq, unit in ((DAYS, "day"), (MONTHS, "month")):
    for i, d in enumerate(seq):
        nxt = seq[(i + 1) % len(seq)]
        add(f"after {d} comes {nxt} .", 3)
        add(f"the {unit} after {d} is {nxt} .", 3)
        add(f"the {unit} before {nxt} is {d} .", 3)
        add(f"{d} is a {unit} .", 2)

# ---------------------------------------------------------------- colours, animals, everyday facts
COLOURS = [("the sky", "blue"), ("grass", "green"), ("the sun", "yellow"), ("snow", "white"), ("coal", "black"),
           ("blood", "red"), ("a banana", "yellow"), ("the sea", "blue"), ("a leaf", "green"), ("a tomato", "red"),
           ("milk", "white"), ("chocolate", "brown"), ("a carrot", "orange"), ("a lemon", "yellow"),
           ("a strawberry", "red"), ("a frog", "green")]
for thing, col in COLOURS:
    add(f"{thing} is {col} .", 4)
    add(f"the color of {thing} is {col} .", 3)
for col in ["blue", "green", "yellow", "white", "black", "red", "brown", "orange"]:
    add(f"{col} is a color .", 3)

SOUNDS = [("dog", "barks"), ("cat", "meows"), ("cow", "moos"), ("duck", "quacks"), ("lion", "roars"),
          ("bird", "sings"), ("bee", "buzzes"), ("snake", "hisses"), ("horse", "neighs"), ("sheep", "bleats")]
BABIES = [("dog", "puppy"), ("cat", "kitten"), ("cow", "calf"), ("duck", "duckling"), ("sheep", "lamb"),
          ("horse", "foal"), ("lion", "cub"), ("bird", "chick")]
LEGS = [("dog", "four"), ("cat", "four"), ("cow", "four"), ("horse", "four"), ("bird", "two"), ("duck", "two"),
        ("spider", "eight"), ("bee", "six"), ("ant", "six")]
for a, s in SOUNDS:
    add(f"a {a} {s} .", 4)
    add(f"a {a} is an animal .", 3)
for a, b in BABIES:
    add(f"a baby {a} is called a {b} .", 4)
    add(f"a {b} is a baby {a} .", 3)
for a, n in LEGS:
    add(f"a {a} has {n} legs .", 4)

EVERYDAY = """
the sun rises in the east .
the sun sets in the west .
water freezes into ice .
ice melts into water .
a week has seven days .
a year has twelve months .
an hour has sixty minutes .
a minute has sixty seconds .
a triangle has three sides .
a square has four sides .
the earth goes around the sun .
the moon goes around the earth .
fire is hot .
ice is cold .
we see with our eyes .
we hear with our ears .
we smell with our nose .
we eat with our mouth .
we walk with our legs .
we think with our brain .
a fish lives in water .
a bird lives in a nest .
a bee lives in a hive .
plants need water and light .
rain falls from clouds .
""".strip().split("\n")
for s in EVERYDAY:
    add(s, 5)

# ---------------------------------------------------------------- AI and computer science, in the lab's voice
AI = """
a tensor is a container of numbers .
a tensor can be a number , a list or a table .
a vector is a list of numbers .
a matrix is a table of numbers .
a scalar is a single number .
a neuron is a brain cell .
a neuron adds up its inputs and decides to fire .
the axon carries the signal to the next cell .
the axon is the output wire of a neuron .
dendrites bring signals into the neuron .
a synapse connects two neurons .
a perceptron is the first artificial neuron .
a perceptron is a copy of a neuron made of math .
a perceptron multiplies inputs by weights and adds them up .
frank rosenblatt invented the perceptron .
ap stands for axon and perceptron .
ap tensor lab explains how ai works .
ap tensor lab is a place to learn ai .
a weight tells how much an input matters .
a bias shifts the output of a neuron .
an activation function decides if a neuron fires .
relu turns negative numbers into zero .
a neural network is made of many neurons .
a neural network is made of layers .
a layer is a group of neurons .
deep learning uses many layers .
machine learning learns from data .
ai stands for artificial intelligence .
ml stands for machine learning .
dl stands for deep learning .
gpu stands for graphics processing unit .
cpu stands for central processing unit .
llm stands for large language model .
rag stands for retrieval augmented generation .
a gpu does many small calculations at the same time .
a cpu is the brain of a computer .
a token is a small piece of text .
a tokenizer cuts text into tokens .
each token has an id number .
an embedding turns a token into a vector .
an embedding is a list of numbers that stands for meaning .
similar words have similar embeddings .
a transformer is a stack of layers .
a transformer is a neural network that uses attention .
each layer has attention and a feed forward network .
attention lets tokens look at other tokens .
attention decides which tokens matter most .
a query asks what a token is looking for .
a key tells what a token has .
a value is what a token gives .
the model predicts the next token .
a language model predicts the next token .
a language model writes one token at a time .
softmax turns scores into probabilities .
temperature controls how random the output is .
a low temperature makes the output more certain .
a high temperature makes the output more random .
training means adjusting the weights .
training teaches the model with data .
the loss measures how wrong the model is .
a low loss means the model is good .
a high loss means the model is wrong .
backpropagation updates the weights to reduce the loss .
the gradient shows which way to change the weights .
the learning rate controls how big each step is .
an epoch is one pass over the data .
a parameter is a number the model learns .
a large model has billions of parameters .
this tiny model runs in your browser .
this tiny model only knows a small world .
the context window is how much text the model can read .
a prompt is the text you give the model .
the output is the text the model writes .
inference means using a trained model .
data is the fuel of ai .
an algorithm is a list of steps .
a program is a list of instructions for a computer .
a bug is a mistake in a program .
code is text that tells a computer what to do .
python is a programming language .
rust is a programming language .
a computer stores numbers as bits .
a bit is a zero or a one .
a byte is eight bits .
memory stores data for a short time .
a disk stores data for a long time .
a database stores data in tables .
a network connects computers .
the internet is a network of networks .
a server answers requests from other computers .
a chip is made of transistors .
a transistor is a tiny switch .
a logic gate is made of transistors .
rag gives the model documents to read .
an agent is a model that can use tools .
a vector database stores embeddings .
fine tuning teaches a trained model a new task .
quantization makes a model smaller .
a dataset is a collection of data .
a label is the correct answer for an example .
overfitting means the model only remembers the data .
a hallucination is a wrong answer that sounds right .
""".strip().split("\n")
for s in AI:
    add(s, 8)
    words = s.split()
    if words[0] in ("a", "an") and "is" in words[1:4]:                      # "a tensor is ..." -> question form
        subject = " ".join(words[:words.index("is")])
        add(f"what is {subject} ? {s}", 4)
    if "stands for" in s:                                                    # "gpu stands for ..." -> question form
        short = words[0]
        add(f"what does {short} stand for ? {' '.join(words[3:])}", 4)

random.shuffle(train)
out = sys.argv[1]
json.dump({"train": train, "heldout": heldout}, open(out, "w"), indent=0)
uniq = sorted(set(train))
words = sorted({w for s in uniq for w in s.split()})
print(f"{len(train)} training lines, {len(uniq)} unique sentences, {len(words)} unique words, {len(heldout)} held-out prompts")
print("longest sentence:", max(len(s.split()) for s in uniq), "words")
