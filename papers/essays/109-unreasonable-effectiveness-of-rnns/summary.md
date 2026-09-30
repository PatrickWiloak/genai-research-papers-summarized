---
title: "The Unreasonable Effectiveness of Recurrent Neural Networks (char-rnn)"
slug: "109-unreasonable-effectiveness-of-rnns"
number: 109
category: "essays"
authors: "Andrej Karpathy (Stanford University)"
published: "May 2015 (blog post, karpathy.github.io)"
year: 2015
url: "https://karpathy.github.io/2015/05/21/rnn-effectiveness/"
tags: ["essay", "language-model", "history"]
---

# The Unreasonable Effectiveness of Recurrent Neural Networks (char-rnn)

**Authors:** Andrej Karpathy (Stanford University)
**Published:** May 2015 (blog post, karpathy.github.io)
**Post:** [karpathy.github.io/2015/05/21/rnn-effectiveness](https://karpathy.github.io/2015/05/21/rnn-effectiveness/)

---

## Why This Matters

This blog post is where a generation of engineers first watched a neural network learn to write. Karpathy trained small recurrent networks to predict the next *character* of a text file, then sampled from them, and out came fake Shakespeare, fake Wikipedia markup, fake algebraic geometry in LaTeX and fake Linux kernel C. None of it made sense, but all of it had the right *shape*, learned from raw characters with nothing built in.

- **It made next-token prediction feel powerful.** The same objective that trains every modern language model was shown, vividly, to pick up spelling, syntax, nesting, formatting and style on its own.
- **It shipped code.** The accompanying `char-rnn` (Lua/Torch) and a roughly 100-line numpy version let anyone reproduce the results, and many people did.
- **It showed interpretable neurons.** Individual cells turned out to track things like "am I inside a quote?" or "am I inside a URL?", with no one telling them to.
- **It called attention early.** In its research roundup, the post says: "The concept of attention is the most interesting recent architectural innovation in neural networks." Two years later, attention replaced recurrence entirely.

**The insight:** a model trained only to guess the next character, given enough data and capacity, is forced to learn whatever structure in the text helps it guess. Structure is not programmed in; it is extracted.

---

## The Problem It Addressed

In 2015 the common wisdom was that recurrent networks were hard to train and finicky. Karpathy opens by saying that "the common wisdom was that RNNs were supposed to be difficult to train (with more experience I've in fact reached the opposite conclusion)". Language modelling was mostly word-level and statistical, and generated text from neural models was not something most practitioners had seen up close.

There was also a gap in intuition. Feed-forward and convolutional networks take a fixed-size input and give a fixed-size output. It was not obvious to many people why a network that reads sequences one step at a time should be qualitatively more capable.

---

## The Argument

### 1. RNNs are programs, not just functions

An ordinary network maps a fixed input to a fixed output. An RNN keeps a hidden state and updates it on every step, so its output depends on everything it has seen so far. Karpathy's framing: "If training vanilla neural nets is optimization over functions, training recurrent nets is optimization over programs."

The whole forward pass of a vanilla RNN fits in three lines:

```python
# one step of a vanilla RNN (from the post)
h = np.tanh(W_hh @ h + W_xh @ x)   # update hidden state from old state + new input
y = W_hy @ h                        # read out a prediction
```

Stack two or three of these, or swap in an LSTM (a recurrent cell with gates that make long-range memory easier to learn), and you have the models in the post.

### 2. Character-level language modelling

Train the network to output a probability for every possible next character. The post's toy example uses the vocabulary "h, e, l, o" and the word "hello":

```
input:   h    e    l    l
target:  e    l    l    o

Note the two "l" inputs have different targets ("l" then "o").
The network can only get both right by remembering context
in its hidden state.
```

At generation time, sample a character from the predicted distribution, feed it back in, and repeat. A **temperature** knob on the softmax trades off diversity against mistakes: at very low temperature the Paul Graham model gets stuck in a loop ("is that they were all the same thing that was a startup is that they were all the same thing...").

### 3. The experiments

| Dataset | Size | Model (as stated) | What it learned |
|---|---|---|---|
| Paul Graham essays | about 1 MB | 2-layer LSTM, 512 hidden units, about 3.5M parameters | English spelling, punctuation, startup-flavoured nonsense |
| Shakespeare | 4.4 MB | 3-layer RNN, 512 hidden units per layer | Speaker names, verse layout, archaic register |
| Wikipedia (Hutter Prize) | 100 MB, first 96 MB for training | LSTM | Wiki markup, links, nested XML with made-up ids and timestamps |
| Algebraic geometry LaTeX | 16 MB | multilayer LSTM | Output that "almost compiles"; lemmas, proofs, even "Proof omitted." |
| Linux source | 474 MB of C | 3-layer LSTMs, about 10M parameters | Brace matching, indentation, comments, a GNU licence header |
| Baby names | 8,000 names | (small) | Plausible new names, most not in the training set |

The errors are as instructive as the successes. The LaTeX model opens `\begin{proof}` and closes with `\end{lemma}`. The C model uses undeclared variables and returns values from `void` functions. Karpathy attributes these to dependencies that are "too long-term": by the time the model reaches the end of a block, it has forgotten how the block started.

He also noticed the fake Wikipedia output contained a Yahoo URL that does not exist: "the model just hallucinated it." That is an early, casual use of the word for what language models do when they produce confident fabrications.

### 4. Looking inside

Two experiments peek under the hood:

- **Training dynamics.** Sampling a War and Peace model every 100 iterations shows it first learns the word/space pattern, then short common words, then longer words, and only much later topics and longer-range structure.
- **Individual neurons.** Some cells fire inside URLs, some inside `[[ ]]` wiki links, one tracks position within a link like a clock. Karpathy estimates that "about 5%" of cells learned interesting, interpretable behaviour. The punchline is the quote-detection cell: nobody told the model that tracking open quotes was useful; "one of its cells gradually tuned itself during training to become a quote detection cell, since this helps it better perform the final task."

### 5. Where the field was going

The post ends with a survey: RNNs in speech, translation, captioning and visual attention; Neural Turing Machines with differentiable memory; soft versus hard attention. It observes that word-level models currently beat character-level ones but that this "is surely a temporary thing", and notes a key RNN weakness: they "unnecessarily couple their representation size to the amount of computation per step".

---

## Key Claims

1. **Simple models plus raw data go surprisingly far.** Next-character prediction alone learns syntax, formatting and style.
2. **Recurrence is a general computational primitive** ("optimization over programs"), useful even for non-sequential data.
3. **End-to-end training discovers useful internal features** that nobody specified.
4. **Long-range dependencies are the weak spot**, and attention and external memory look like the fix.
5. **RNNs "will become a pervasive and critical component to intelligent systems."**

---

## How It Has Aged

As of September 2026:

**What came true:**

- **The objective won completely.** Every major language model, from [GPT-1](../../language-models/93-gpt1/summary.md) through [GPT-2](../../language-models/64-gpt2/summary.md), [GPT-3](../../language-models/04-gpt3-few-shot-learners/summary.md) and today's frontier systems, is trained first on next-token prediction. The post is the best informal demonstration of why that objective is so rich, and [The Scaling Hypothesis](../112-scaling-hypothesis/summary.md) builds its whole argument on the same "learn what helps you predict" intuition.
- **Attention really was the most interesting innovation.** [Bahdanau attention](../../architectures/66-bahdanau-attention/summary.md), which the post cites, led to the [Transformer](../../architectures/01-attention-is-all-you-need/summary.md) (2017), which dropped recurrence and fixed exactly the long-range-dependency failures shown here.
- **Interpretable units were real.** OpenAI's "sentiment neuron" work (Radford et al., 2017) found a single unit tracking sentiment in a character-level LSTM trained on product reviews, a direct echo of the quote cell. Interpretability is now a major field (see [Sparse Autoencoders](../../techniques/82-sparse-autoencoders/summary.md) and [Induction Heads](../../techniques/126-induction-heads/summary.md)).
- **"Hallucination" stuck** as the standard term for confident fabrication.
- **Small, readable, runnable code as teaching.** The char-rnn pattern (one file, a text dataset, sample as you train) became Karpathy's signature. His later minGPT and nanoGPT repositories reuse the same idea, including a character-level Shakespeare example.

**What did not:**

- **RNNs did not become the pervasive component.** Transformers displaced LSTMs for language within a few years. The coupling of state size to per-step compute, which the post flags, plus sequential training that is hard to parallelise, were decisive.
- **Character-level did not beat word-level in the way the post implied.** The field settled on *subword* tokens (see [BPE](../../techniques/136-bpe-subword-units/summary.md)), a compromise between the two.
- **Recurrence is back, in a new form.** State-space models such as [Mamba](../../architectures/20-mamba/summary.md) revived fixed-size recurrent state for efficient long sequences, often in hybrids with attention. So the essay's enthusiasm for recurrence is partly vindicated, though not for LSTMs.

---

## Criticisms

- **Cherry-picking and anthropomorphism.** The samples are selected for fun, and phrases like "the model decided" invite readers to credit understanding where there is surface pattern-matching. The post itself shows the model has no grip on meaning.
- **Hand-wavy interpretability.** Karpathy says so himself: the hidden state is "a huge, high-dimensional and largely distributed representation", and most cells were not interpretable. Picking out the few that are can overstate how legible networks are.
- **It is a demonstration, not a measurement.** There are few quantitative comparisons with other methods, so it persuades rather than proves.

None of these undercut its role: it was never meant to be a paper.

---

## Key Takeaways for Practitioners

1. **Next-token prediction is a curriculum.** Models learn easy structure first and long-range structure last. If your model gets formatting right but content wrong, that is the expected order.
2. **Temperature is a real control.** Low temperature gives safe, repetitive output; high temperature gives variety and errors.
3. **Long-range consistency is where models break.** Unclosed brackets and undefined variables in 2015 are the ancestors of today's long-context failures. Test for them.
4. **Build the small version first.** A 100-line model on one text file teaches more than reading about a large one.

---

## Further Reading

- **Original post:** [karpathy.github.io/2015/05/21/rnn-effectiveness](https://karpathy.github.io/2015/05/21/rnn-effectiveness/)
- **Code:** [github.com/karpathy/char-rnn](https://github.com/karpathy/char-rnn)
- **Karpathy, Johnson, Fei-Fei, "Visualizing and Understanding Recurrent Networks" (2015):** [arxiv.org/abs/1506.02078](https://arxiv.org/abs/1506.02078)
- **In this collection:** [Seq2Seq](../../architectures/55-seq2seq/summary.md), [Bahdanau Attention](../../architectures/66-bahdanau-attention/summary.md), [Attention Is All You Need](../../architectures/01-attention-is-all-you-need/summary.md), [GPT-2](../../language-models/64-gpt2/summary.md), [Software 2.0](../110-software-2/summary.md)

## Citation

```bibtex
@misc{karpathy2015unreasonable,
  title={The Unreasonable Effectiveness of Recurrent Neural Networks},
  author={Karpathy, Andrej},
  year={2015},
  month={May},
  howpublished={\url{https://karpathy.github.io/2015/05/21/rnn-effectiveness/}}
}
```

<!-- related:start -->

---

## Related in This Collection

- [Attention Is All You Need](../../architectures/01-attention-is-all-you-need/summary.md)
- [Language Models are Few-Shot Learners (GPT-3)](../../language-models/04-gpt3-few-shot-learners/summary.md)
- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](../../architectures/20-mamba/summary.md)
- [Sequence to Sequence Learning with Neural Networks (Seq2Seq)](../../architectures/55-seq2seq/summary.md)
- [Language Models are Unsupervised Multitask Learners (GPT-2)](../../language-models/64-gpt2/summary.md)
- [Neural Machine Translation by Jointly Learning to Align and Translate (Bahdanau Attention)](../../architectures/66-bahdanau-attention/summary.md)
- [Sparse Autoencoders and Monosemanticity: Reading the Features Inside a Model](../../techniques/82-sparse-autoencoders/summary.md)
- [Improving Language Understanding by Generative Pre-Training (GPT-1)](../../language-models/93-gpt1/summary.md)

<!-- related:end -->
