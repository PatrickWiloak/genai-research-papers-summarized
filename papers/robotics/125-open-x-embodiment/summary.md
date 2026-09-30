---
title: "Open X-Embodiment: Robotic Learning Datasets and RT-X Models (Open X-Embodiment / RT-X)"
slug: "125-open-x-embodiment"
number: 125
category: "robotics"
authors: "Open X-Embodiment Collaboration - more than 290 researchers from 21 institutions, including Google DeepMind, Stanford, UC Berkeley and many others"
published: "October 2023 (ICRA 2024, Best Conference Paper)"
year: 2023
url: "https://arxiv.org/abs/2310.08864"
tags: ["robotics", "scaling", "benchmarks"]
---

# Open X-Embodiment: Robotic Learning Datasets and RT-X Models (Open X-Embodiment / RT-X)

**Authors:** Open X-Embodiment Collaboration - more than 290 researchers from 21 institutions, including Google DeepMind, Stanford, UC Berkeley and many others
**Published:** October 2023 (ICRA 2024, Best Conference Paper)
**Paper:** [arxiv.org/abs/2310.08864](https://arxiv.org/abs/2310.08864)

---

## Why This Paper Matters

Computer vision took off when ImageNet gave everyone one large shared dataset. Language models took off when the web gave them trillions of tokens. Robotics had neither. Every lab collected its own demonstrations, on its own robot, in its own format, and trained its own model. A policy for one arm was useless on another, and no single lab could collect enough data to find out whether scale would help.

Open X-Embodiment was a deliberate attempt to build robotics' shared dataset. Thirty-four robotics labs pooled their data into one standard format, and the paper asked the key question: **does training one model on many different robots make it better on each of them?** The answer was yes - most clearly for the robots with the least data of their own.

- **More than 1 million robot trajectories** from **22 robot embodiments**, 60 existing datasets, 34 labs and 21 institutions.
- **527 skills** across **160,266 tasks**, converted to one format (RLDS).
- **RT-1-X** raised success by about **50% on average** over each lab's own method on small-data robots.
- **RT-2-X** roughly **tripled** performance on emergent skills that came from other robots' data.
- **Won Best Conference Paper at ICRA 2024**, and became the default pretraining pool for open robot models.

**The insight:** robots differ in their bodies, but the world they manipulate is shared. Cups, drawers and tables look the same to a Franka arm and a WidowX. A single model trained on all of them can pick up regularities no single robot's dataset contains.

---

## The Problem: Fragmented, Small, Incompatible Data

Before this project:

- **Datasets were small.** Tens of thousands of episodes was a big robot dataset.
- **Formats differed.** Different camera setups, action spaces (joint angles vs end-effector deltas), control rates and file formats.
- **The received wisdom was that cross-robot training would hurt**, because each robot's dynamics differ.

Nobody could test "X-embodiment" (cross-embodiment) transfer properly without first doing the unglamorous work of unifying the data.

---

## The Core Innovation: Pool the Data, Train One Model

### 1. The dataset
Each contributing dataset was converted into **RLDS** (a standard episode format built on TensorFlow Datasets). Observations were aligned to a common camera view where possible, and actions to a common 7-dimensional end-effector representation (position change, rotation change, gripper). The pooled result: over **one million trajectories**, spanning single arms, bi-manual setups and quadrupeds with arms.

### 2. The models
The authors retrained two existing architectures on a mixture from **9 manipulators**:

- **RT-1-X**: the 35M-parameter RT-1 architecture ([background in RT-2](../123-rt2/summary.md)).
- **RT-2-X**: the 55B-parameter vision-language-action model [RT-2](../123-rt2/summary.md), co-trained with web data as before.

No robot-specific heads were used: the same model outputs the same kind of action for every robot, and the camera image implicitly tells it which robot it is controlling.

---

## Key Results

### Small-data robots benefit most
On five robots with small datasets, **RT-1-X beat the original method from each lab on four of the five**, with a mean success rate about **50% higher** than either the original methods or RT-1 trained on the same robot's data alone. Data from other robots filled in what each small dataset lacked.

### Large-data robots need a bigger model
On robots with plenty of their own data, the small RT-1-X **underfit** - it could not absorb everything:

| Evaluation | Original method | RT-1 | RT-1-X | RT-2-X (55B) |
|---|---|---|---|---|
| Bridge tasks, WidowX at Stanford | 13% | 40% | 27% | **50%** |
| Bridge tasks, WidowX at UC Berkeley | 13% | **30%** | 27% | **30%** |
| RT-1 paper's 6 skills, Google Robot | - | **92%** | 73% | 91% |

The 55B RT-2-X matched or beat the specialists, suggesting cross-embodiment training helps if the model has enough capacity.

### Transfer of skills across robots
The team tested the Google robot on "emergent" skills that existed only in *other* robots' data - for instance, relations and object placements seen in the Bridge dataset. **RT-2-X scored 75.8% against 27.3% for RT-2**, about three times better. Removing the Bridge data from training dropped RT-2-X to **42.8%**, confirming the skills really came from the other robot.

---

## Why This Was Revolutionary

- **It overturned the default assumption** that robot data from other bodies is noise.
- **It created a commons.** A shared, standardised dataset lowered the entry cost for every robotics lab.
- **It showed the scaling pattern familiar from language**: more diverse data helps, provided the model is big enough to use it.

---

## Real-World Impact

- **Octo** (2024) - an open generalist policy trained on 800,000 Open X-Embodiment trajectories.
- **OpenVLA** (2024) - an open 7B vision-language-action model trained on about 970,000 demonstrations, which reported beating RT-2-X by 16.5 percentage points on its evaluation.
- **[pi0](../124-pi0/summary.md)** (2024) - included Open X-Embodiment data in its pretraining mixture.
- **Later datasets** such as DROID continued the collaborative, standardised approach.

---

## Key Takeaways for Practitioners

1. **Pretrain on pooled data, then fine-tune on your robot.** Small datasets gain the most.
2. **Capacity matters.** Cross-embodiment mixtures can make small models worse on data-rich robots.
3. **Standardise action spaces.** A shared end-effector representation is what made pooling possible.
4. **Watch the mixture.** Which datasets you include changes which skills transfer.

---

## Limitations & Future Directions

The authors name these gaps:
- **No study of generalisation to entirely new robots** - all evaluation robots were in the training mix.
- **No criterion for when transfer is positive.** Why some robots benefit and others do not is not explained.
- **Mostly single-arm manipulation.** Locomotion, dexterous hands and mobile manipulation are under-represented.
- **Heterogeneous quality.** Pooled data from 34 labs varies widely in quality and task definition.

---

## Further Reading

- **Original Paper:** [arxiv.org/abs/2310.08864](https://arxiv.org/abs/2310.08864)
- **Project page:** [robotics-transformer-x.github.io](https://robotics-transformer-x.github.io/)
- **Octo:** [arxiv.org/abs/2405.12213](https://arxiv.org/abs/2405.12213)
- **OpenVLA:** [arxiv.org/abs/2406.09246](https://arxiv.org/abs/2406.09246)
- **In this collection:** [RT-2](../123-rt2/summary.md), [pi0](../124-pi0/summary.md), [Scaling Laws](../../techniques/12-scaling-laws/summary.md)

## Citation

```bibtex
@inproceedings{oxe2024,
  title={Open X-Embodiment: Robotic Learning Datasets and RT-X Models},
  author={{Open X-Embodiment Collaboration}},
  booktitle={IEEE International Conference on Robotics and Automation (ICRA)},
  year={2024}
}
```

<!-- related:start -->

---

## Related in This Collection

- [RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control (RT-2)](../../robotics/123-rt2/summary.md)
- [pi0: A Vision-Language-Action Flow Model for General Robot Control (pi0)](../../robotics/124-pi0/summary.md)

<!-- related:end -->
