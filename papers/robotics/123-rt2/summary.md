---
title: "RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control (RT-2)"
slug: "123-rt2"
number: 123
category: "robotics"
authors: "Anthony Brohan, Noah Brown, ... Brianna Zitkovich (54 authors, Google DeepMind)"
published: "July 2023 (CoRL 2023)"
year: 2023
url: "https://arxiv.org/abs/2307.15818"
tags: ["robotics", "multimodal", "transfer-learning"]
---

# RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control (RT-2)

**Authors:** Anthony Brohan, Noah Brown, ... Brianna Zitkovich (54 authors, Google DeepMind)
**Published:** July 2023 (CoRL 2023)
**Paper:** [arxiv.org/abs/2307.15818](https://arxiv.org/abs/2307.15818)

---

## Why This Paper Matters

Robots learn from demonstrations, and demonstrations are expensive: a person has to teleoperate a physical arm, one episode at a time. Even the largest robot datasets of 2022 held a few hundred thousand episodes - tiny next to the billions of image-text pairs that vision-language models learn from. So robots were good at the exact tasks they had practised and poor at anything new.

RT-2 asked a simple question: **what if the robot's brain was a vision-language model that already understood the web, and robot actions were just another kind of text it learned to output?** The answer was the first **vision-language-action (VLA)** model, and it defined how the field has built robot foundation models since.

- **Actions as tokens.** Joint movements are discretised into numbers and emitted as text tokens by the same model that answers questions about images.
- **Web knowledge transferred to control.** The robot could follow instructions involving objects, symbols and concepts that never appeared in its robot training data.
- **Generalisation roughly doubled** on unseen scenarios compared with its predecessor RT-1.
- **"VLA" became the name of a model class**, followed by Open X-Embodiment, OpenVLA, [pi0](../124-pi0/summary.md) and others.

**The insight:** a large vision-language model already knows what a "Taylor Swift picture", an "extinct animal" or "something you could use as an improvised hammer" looks like. If you fine-tune it to also output robot actions - without letting it forget the web data - that semantic knowledge carries over into what the robot does.

---

## Background: RT-1

RT-2 builds on **RT-1** (Robotics Transformer 1, December 2022), a 35M-parameter model that turned camera images and an instruction into actions at 3 Hz. RT-1 was trained on about **130,000 demonstrations** of **700+ instructions**, collected with **13 robots over 17 months** in office kitchens, and reached **97% success on the instructions it was trained on**. It was a strong specialist - but its understanding stopped where its training data did.

---

## The Core Innovation: Actions as Language

### 1. Start from a vision-language model
RT-2 used two backbones:
- **PaLI-X** at 5B and 55B parameters
- **PaLM-E** at 12B parameters

Both take images plus text and produce text.

### 2. Turn actions into tokens
Each robot action is an end-effector movement - position and rotation changes (6 degrees of freedom), gripper opening, and a flag for "episode finished". Each continuous dimension is split into **256 bins**, so an action becomes a short string of integers:

```
action = [terminate, dx, dy, dz, droll, dpitch, dyaw, gripper]
       ->  "1 128 91 241 5 101 127 217"
```

For PaLI-X, whose vocabulary already contains tokens for small integers, those are reused. For PaLM-E, the 256 least-used tokens in the vocabulary are overwritten to mean action bins. At inference, decoding is **constrained** to action tokens when the robot is asked to act, so it cannot emit prose instead of a movement.

### 3. Co-fine-tune on web data and robot data together
Fine-tuning only on robot trajectories would erode what the model knew about the world. RT-2 **mixes the original web vision-language data back in** during fine-tuning. The ablations show why:

| 5B model trained... | Generalisation score |
|---|---|
| from scratch on robot data | 9 |
| fine-tuned on robot data | 42 |
| co-fine-tuned (robot + web) | 44 |

| 55B model trained... | Generalisation score |
|---|---|
| fine-tuned on robot data | 52 |
| co-fine-tuned (robot + web) | 63 |

Pretraining matters enormously; scale and co-fine-tuning each add more.

### 4. Run it from the cloud
A 55B model does not fit on a robot. RT-2 ran on a cloud TPU service and streamed actions back: the 55B model at **1-3 Hz**, the 5B model at about **5 Hz**.

---

## Key Results

Around **6,000 evaluation trials** on real robots.

| Model | Seen tasks | Unseen tasks (average) |
|---|---|---|
| RT-1 | 92% | 32% |
| VC-1 | 63% | 10% |
| R3M | 45% | 12% |
| MOO | 75% | 35% |
| **RT-2-PaLI-X-55B** | **91%** | **62%** |
| **RT-2-PaLM-E-12B** | **93%** | **62%** |

On seen tasks RT-2 matched RT-1. On unseen objects, backgrounds and environments it nearly doubled it.

**Emergent capabilities** - tasks needing semantic understanding, symbol recognition or reasoning that the robot data never taught:
- RT-1: 17% average; VC-1: 11%
- RT-2-PaLI-X: **60%**; RT-2-PaLM-E: **40%**

Examples in the paper include moving a can to a picture of a specific person, placing an object on a number written on paper, and picking an object that could serve as an improvised hammer. With a chain-of-thought variant, the model could plan in words ("I need a rock to hammer") before emitting actions.

In the **Language-Table** simulation benchmark, a smaller RT-2 (PaLI 3B) scored **90%**, against 72-77% for earlier methods.

---

## Why This Was Revolutionary

- **It unified perception, language and control in one model** instead of a pipeline of separate modules.
- **It showed robot learning can ride on web-scale pretraining**, sidestepping part of the robot-data shortage.
- **It gave the field a recipe:** start from a VLM, tokenise actions, co-train. Most VLAs since follow some version of it.

---

## Real-World Impact

- **[Open X-Embodiment](../125-open-x-embodiment/summary.md)** (late 2023) pooled robot data across 22 robot types and trained RT-2-X, showing cross-robot transfer.
- **OpenVLA** (2024) released an open 7B VLA trained on Open X-Embodiment data.
- **[pi0](../124-pi0/summary.md)** (2024) kept the VLM backbone but replaced token-by-token actions with a flow-matching "action expert", reaching much higher control frequencies.
- **Robot foundation models** became a major investment area for large labs and start-ups alike.

---

## Key Takeaways for Practitioners

1. **Pretraining is the biggest lever.** Training from scratch on robot data scored 9 where a pretrained model scored 42 or more.
2. **Keep the web data in the mix** during fine-tuning, or the model forgets the knowledge you wanted to transfer.
3. **Discretised action tokens are simple but slow.** Emitting one token per action dimension limits control rate; later work moved to continuous action heads.
4. **Semantic generalisation is not motor generalisation.** RT-2 could apply new *concepts* to known skills, but not invent new *movements*.

---

## Limitations & Future Directions

- **No new skills.** The paper states plainly that RT-2 does not learn new physical motions from web data; it recombines the motions in its robot data.
- **Compute and latency.** Running a 55B model at 1-3 Hz needs a data centre and limits tasks to slow, quasi-static manipulation.
- **Single-embodiment data.** Trained on one robot type; cross-robot transfer was left to later work.
- **Closed model.** RT-2 was not released, which is part of why open successors followed quickly.

---

## Further Reading

- **Original Paper:** [arxiv.org/abs/2307.15818](https://arxiv.org/abs/2307.15818)
- **Project page:** [robotics-transformer2.github.io](https://robotics-transformer2.github.io/)
- **RT-1:** Brohan et al. 2022, [arxiv.org/abs/2212.06817](https://arxiv.org/abs/2212.06817)
- **In this collection:** [pi0](../124-pi0/summary.md), [Open X-Embodiment](../125-open-x-embodiment/summary.md), [PaLM](../../language-models/94-palm/summary.md), [Vision Transformer](../../architectures/11-vision-transformer/summary.md), [Chain-of-Thought](../../techniques/09-chain-of-thought/summary.md)

## Citation

```bibtex
@inproceedings{brohan2023rt2,
  title={RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control},
  author={Brohan, Anthony and Brown, Noah and Carbajal, Justice and Chebotar, Yevgen and Chen, Xi and Choromanski, Krzysztof and Ding, Tianli and Driess, Danny and Dubey, Avinava and Finn, Chelsea and others},
  booktitle={Conference on Robot Learning (CoRL)},
  pages={2165--2183},
  year={2023}
}
```

<!-- related:start -->

---

## Related in This Collection

- [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](../../techniques/09-chain-of-thought/summary.md)
- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (Vision Transformer)](../../architectures/11-vision-transformer/summary.md)
- [PaLM: Scaling Language Modeling with Pathways](../../language-models/94-palm/summary.md)
- [pi0: A Vision-Language-Action Flow Model for General Robot Control (pi0)](../../robotics/124-pi0/summary.md)
- [Open X-Embodiment: Robotic Learning Datasets and RT-X Models (Open X-Embodiment / RT-X)](../../robotics/125-open-x-embodiment/summary.md)

<!-- related:end -->
