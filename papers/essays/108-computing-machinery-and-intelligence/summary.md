---
title: "Computing Machinery and Intelligence (The Imitation Game / Turing Test)"
slug: "108-computing-machinery-and-intelligence"
number: 108
category: "essays"
authors: "Alan M. Turing (University of Manchester)"
published: "October 1950 (Mind, Vol. 59, No. 236, pp. 433-460)"
year: 1950
url: "https://academic.oup.com/mind/article/LIX/236/433/986238"
tags: ["essay", "history", "evaluation"]
---

# Computing Machinery and Intelligence (The Imitation Game / Turing Test)

**Authors:** Alan M. Turing (University of Manchester)
**Published:** October 1950 (Mind, Vol. 59, No. 236, pp. 433-460)
**Essay:** [academic.oup.com/mind/article/LIX/236/433/986238](https://academic.oup.com/mind/article/LIX/236/433/986238)

---

## Why This Matters

This is the founding essay of artificial intelligence as a question people could argue about productively. Before it, "can machines think?" was a philosophical parlour game with no way to settle it. Turing replaced it with a concrete, testable game, and in doing so gave the field its first benchmark, its first list of objections to answer, and a surprisingly modern sketch of how a thinking machine might actually be built.

- **It swapped a vague question for an operational one.** Instead of defining "think", ask whether a machine can hold up its end of a typed conversation well enough that a judge cannot reliably tell it from a person.
- **It made a dated, falsifiable prediction.** In about fifty years, machines with about 10^9 bits of storage would fool an average interrogator often enough that the judge would be right no more than 70 percent of the time after five minutes.
- **It pre-answered nine objections** that critics still raise in 2026, from "machines can only do what we tell them" to "machines cannot be conscious".
- **Its final section describes machine learning.** Turing argues that hand-programming a mind is hopeless, and proposes building a "child machine" and educating it with rewards and punishments. That is, in outline, the approach that won.

**The insight:** you do not need to agree on what thinking *is* to make progress. Pick a behaviour that everyone agrees requires intelligence when a human does it, make it measurable, and let the argument move from definitions to evidence.

---

## The Problem: "Can Machines Think?" Had No Answer

Turing opens by noting that if you try to answer the question by examining how people normally use the words "machine" and "think", you end up needing "a statistical survey such as a Gallup poll". He calls that absurd. The words were too loaded and too vague, so the debate could never end.

In 1950 there were only a handful of stored-program computers in the world (Turing was working with one at Manchester). Nobody had built anything that looked remotely intelligent. The question was genuinely open, and genuinely stuck.

---

## The Argument

### 1. The imitation game

Turing starts with a party game. A man (A) and a woman (B) sit in another room. An interrogator (C) exchanges typed messages with both and must work out which is which. A tries to mislead; B tries to help.

Then he asks what happens when a machine takes the part of A:

```
          +-------------------+
          |   Interrogator C  |
          +---------+---------+
                    | typed text only (teleprinter)
         +----------+----------+
         |                     |
   +-----+-----+         +-----+-----+
   |  "X"      |         |  "Y"      |
   | machine A |         | human B   |
   +-----------+         +-----------+

   C's job: say which of X and Y is the human.
   The machine "does well" if C is wrong about as often
   as C would be when A was a human trying to deceive.
```

"These questions replace our original, 'Can machines think?'" The typed channel is deliberate: it strips away appearance and voice, so the test is about intellectual capacity only.

### 2. Which machines count

Turing restricts the contestants to digital computers, and then makes a crucial move (section 5, "Universality of Digital Computers"): a digital computer with enough storage and speed can imitate any other discrete-state machine. So the question collapses to one machine: can a single computer, given "adequate storage", more speed and "an appropriate programme", play the game well? The problem becomes, in his words, "mainly one of programming".

### 3. The prediction

> "I believe that in about fifty years' time it will be possible, to programme computers, with a storage capacity of about 10^9, to make them play the imitation game so well that an average interrogator will not have more than 70 per cent chance of making the right identification after five minutes of questioning."

He adds that the original question "I believe to be too meaningless to deserve discussion", but predicts that by the end of the century educated opinion will have shifted so much "that one will be able to speak of machines thinking without expecting to be contradicted".

For scale, 10^9 binary digits is about 125 megabytes. Turing notes that the 11th edition of the Encyclopaedia Britannica is about 2 x 10^9.

### 4. Nine objections, answered

Section 6 is the longest part of the essay. Turing lists and rebuts:

| # | Objection | Turing's reply, in brief |
|---|---|---|
| 1 | Theological: thinking belongs to the immortal soul | Unconvincing on its own terms, and it limits God's power |
| 2 | "Heads in the sand": the consequences would be too dreadful | A wish, not an argument |
| 3 | Mathematical: Godel shows machines have limits | Humans have limits too; nobody has shown humans escape them |
| 4 | Consciousness: a machine would not *feel* anything | Taken strictly, this leads to solipsism; we grant other people minds from their behaviour |
| 5 | "Various disabilities": a machine could never be kind, fall in love, enjoy strawberries... | Mostly unexamined induction from the limited machines people had seen |
| 6 | Lady Lovelace: a machine can only do "whatever we know how to order it to perform" | Machines already surprise their programmers, and a *learning* machine need not be limited to what it was told |
| 7 | The nervous system is continuous, not discrete | In the game, a discrete machine can imitate a continuous one well enough |
| 8 | Informality of behaviour: no rule book covers all of life | No complete *rules of conduct* does not mean no *laws of behaviour* |
| 9 | Extrasensory perception | Turing takes it oddly seriously and suggests a "telepathy-proof room" |

Objection 6 matters most for what came later, because Turing's real answer to it is the next section.

### 5. Learning machines

Turing admits he has "no very convincing arguments of a positive nature". What he offers instead is an engineering estimate and a plan.

The estimate: at his own rate of about a thousand digits of program a day, "about sixty workers, working steadily through the fifty years might accomplish the job, if nothing went into the wastepaper basket. Some more expeditious method seems desirable."

The plan: do not program the adult mind. Program a child's, then educate it.

```
adult mind  =  initial state (at birth)
             + education
             + other experience

So build:   child programme  +  education process

Evolution analogy (Turing's own):
  structure of the child machine = hereditary material
  changes of the child machine   = mutation
  natural selection              = judgment of the experimenter
```

He goes on to propose reward and punishment signals, a random element in learning (arguing that random search beats systematic search when there are many acceptable solutions), and a warning that reads like a description of modern interpretability research: "An important feature of a learning machine is that its teacher will often be very largely ignorant of quite what is going on inside."

He closes by asking where to start: something abstract like chess, or a machine given "the best sense organs that money can buy" and taught to "understand and speak English" the way a child is taught. "I do not know what the right answer is, but I think both approaches should be tried."

---

## Key Claims

1. **Behaviour is the right test.** Conversation indistinguishable from a human's is enough to warrant the word "thinking", in the same way we grant it to other people.
2. **Digital computers are universal**, so hardware is not the fundamental barrier; the problem is the program.
3. **The program is too big to write by hand**, so it must be learned.
4. **Learning needs a teacher signal**, some randomness, and tolerance for not understanding the result.
5. **Language will follow capability**: once machines behave intelligently, people will talk about them thinking.

---

## How It Has Aged

As of September 2026, this is the essay in the collection that has aged best, with some important caveats.

**What came true:**

- **Learning beat programming.** Turing's back-of-envelope case that nobody could hand-write a mind, and that a learned "child machine" was the realistic route, is the thesis of [The Bitter Lesson](../111-bitter-lesson/summary.md) and [Software 2.0](../110-software-2/summary.md), seventy years early. Today's systems are trained, not written.
- **Reward and punishment became a field.** Turing's reward and punishment signals are an ancestor of reinforcement learning, which now does much of the post-training of language models (see [InstructGPT / RLHF](../../language-models/05-instructgpt-rlhf/summary.md) and [RLVR](../../techniques/39-rlvr/summary.md)).
- **The teacher really is ignorant of what is inside.** Understanding what a trained network has learned is now its own research program (see [Sparse Autoencoders](../../techniques/82-sparse-autoencoders/summary.md) and [Induction Heads](../../techniques/126-induction-heads/summary.md)).
- **The test was passed, roughly on his terms.** In a pre-registered study posted in March 2025, Cameron Jones and Benjamin Bergen (UC San Diego) ran the standard three-party version with five-minute conversations. GPT-4.5, prompted to adopt a humanlike persona, was judged to be the human 73 percent of the time, more often than the real humans it was paired with. LLaMa-3.1-405B with the same prompt reached 56 percent, while the baselines ELIZA and GPT-4o scored 23 and 21 percent. The authors describe it as the first empirical evidence of any system passing a standard three-party Turing test. Turing's five-minute window even survives in the protocol.
- **"Thinking" is ordinary vocabulary.** People routinely say a model is "thinking" while it produces a reasoning trace. The change in usage Turing predicted happened, a couple of decades after his end-of-century date.

**What did not:**

- **The timeline slipped.** Turing's fifty years pointed at 2000. The test was convincingly passed around 2025, by systems many orders of magnitude larger than 10^9 bits. His storage estimate was far too low, and his "mainly one of programming" framing hid a problem that turned out to be mainly one of data and compute.
- **Passing settled less than he hoped.** The essay assumes a machine that converses like a human would be recognised as intelligent. In practice, by the time models passed, the field had largely stopped using the Turing test as its yardstick, and the debate had moved to reasoning, agency, reliability and economic usefulness (see [ARC](../../techniques/138-arc-agi/summary.md) and [SWE-bench](../../techniques/84-swe-bench/summary.md)). Fooling a judge for five minutes turned out to be a narrow slice of what people mean by intelligence.
- **The ESP section** has aged exactly as badly as you would expect.

---

## Criticisms

- **Behaviour is not understanding.** John Searle's "Chinese Room" argument (1980) holds that a system could pass by manipulating symbols with no understanding of them. This is objection 4 in new clothes, and it has not gone away in the language-model era; if anything it is louder.
- **The test rewards human-likeness and deception, not capability.** A system scores better by feigning ignorance and adopting a persona, which is exactly what the successful 2025 prompt did. A system that is obviously a machine because it knows too much would "fail". Critics argue this measures humanlike style more than intelligence.
- **It is judge-dependent and gameable.** Chatbot tricks fooled some judges in short conversations long before language models existed. The length of the conversation, who the judges are and what they are told all move the result.
- **An anthropocentric target.** Making "indistinguishable from a human" the goal rewards building mimics rather than useful or trustworthy systems.
- **The gender-game framing** at the start is read in conflicting ways, and scholars still debate whether the machine was meant to imitate a woman specifically or a human generally. Most modern versions use the latter.

Turing anticipated some of this. He called his own positive arguments "recitations tending to produce belief" rather than proof, and said the only really satisfactory support would come from waiting and running the experiment.

---

## Key Takeaways for Practitioners

1. **Operationalise the question.** When "is it intelligent / good / safe?" stalls, replace it with a measurable task most people agree would count. Every modern benchmark descends from this move.
2. **Expect the benchmark to fall before the argument ends.** Passing a test does not settle what the test was a proxy for. Decide in advance what a result would and would not show.
3. **If writing it by hand would take sixty people fifty years, learn it instead.** Turing's estimate is still a good heuristic for when a problem wants a trained model.
4. **Budget for opacity.** The teacher of a learned system does not know what is inside it. Evaluation and interpretability are part of the job.

---

## Further Reading

- **Original essay:** [academic.oup.com/mind/article/LIX/236/433/986238](https://academic.oup.com/mind/article/LIX/236/433/986238) (DOI 10.1093/mind/LIX.236.433)
- **Jones & Bergen, "Large Language Models Pass the Turing Test" (2025):** [arxiv.org/abs/2503.23674](https://arxiv.org/abs/2503.23674)
- **Stanford Encyclopedia of Philosophy, "The Turing Test":** [plato.stanford.edu/entries/turing-test](https://plato.stanford.edu/entries/turing-test/)
- **In this collection:** [The Bitter Lesson](../111-bitter-lesson/summary.md), [Software 2.0](../110-software-2/summary.md), [On the Measure of Intelligence (ARC)](../../techniques/138-arc-agi/summary.md), [GPT-3](../../language-models/04-gpt3-few-shot-learners/summary.md)

## Citation

```bibtex
@article{turing1950computing,
  title={Computing Machinery and Intelligence},
  author={Turing, Alan M.},
  journal={Mind},
  volume={59},
  number={236},
  pages={433--460},
  year={1950},
  doi={10.1093/mind/LIX.236.433}
}
```

<!-- related:start -->

---

## Related in This Collection

- [Language Models are Few-Shot Learners (GPT-3)](../../language-models/04-gpt3-few-shot-learners/summary.md)
- [Training Language Models to Follow Instructions with Human Feedback (InstructGPT)](../../language-models/05-instructgpt-rlhf/summary.md)
- [GPT-4 Technical Report](../../language-models/36-gpt4/summary.md)
- [RLVR: Reinforcement Learning from Verifiable Rewards](../../techniques/39-rlvr/summary.md)
- [GPT-4o: The First Omni Model](../../language-models/40-gpt4o/summary.md)
- [Sparse Autoencoders and Monosemanticity: Reading the Features Inside a Model](../../techniques/82-sparse-autoencoders/summary.md)
- [SWE-bench: Can Language Models Resolve Real-World GitHub Issues?](../../techniques/84-swe-bench/summary.md)
- [Software 2.0](../../essays/110-software-2/summary.md)

<!-- related:end -->
