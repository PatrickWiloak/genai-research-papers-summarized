---
title: "pi0: A Vision-Language-Action Flow Model for General Robot Control (pi0)"
slug: "124-pi0"
number: 124
category: "robotics"
authors: "Kevin Black, Noah Brown, Danny Driess, ... Ury Zhilinsky (24 authors, Physical Intelligence)"
published: "October 2024 (RSS 2025)"
year: 2024
url: "https://arxiv.org/abs/2410.24164"
tags: ["robotics", "flow-matching", "multimodal"]
---

# pi0: A Vision-Language-Action Flow Model for General Robot Control (pi0)

**Authors:** Kevin Black, Noah Brown, Danny Driess, ... Ury Zhilinsky (24 authors, Physical Intelligence)
**Published:** October 2024 (RSS 2025)
**Paper:** [arxiv.org/abs/2410.24164](https://arxiv.org/abs/2410.24164)

---

## Why This Paper Matters

[RT-2](../123-rt2/summary.md) showed that a vision-language model could be taught to output robot actions as text tokens. That worked for slow pick-and-place tasks, but real dexterity - folding laundry, bussing a table, assembling a box - needs smooth, fast control, dozens of times a second, often with two arms at once. Emitting actions one discrete token at a time is too slow and too coarse for that.

pi0 (pronounced "pi-zero") kept the part of RT-2 that worked - a pretrained vision-language model as the robot's knowledge base - and replaced the action output with a **flow-matching "action expert"** that generates whole chunks of continuous motion at once. Trained on over 10,000 hours of robot data across seven robot configurations, it performed long, dexterous tasks that earlier generalist models could not.

- **Continuous actions at up to 50 Hz**, generated as 50-step chunks.
- **One model, many robots:** single-arm, dual-arm and mobile manipulators trained together.
- **Pretrain then fine-tune**, like an LLM: a broad base model, then task-specific post-training for hard skills such as laundry folding.
- **Open weights.** Physical Intelligence released pi0 through its openpi repository, making it a common base for robotics research.

**The insight:** language is discrete, but motion is continuous. Use the VLM for what it is good at - understanding the scene and the instruction - and let a generative model built for continuous data (flow matching, the technique behind [Stable Diffusion 3](../../image-generation/72-flow-matching-sd3/summary.md)) produce the motion.

---

## The Core Innovation: A VLM Plus a Flow-Matching Action Expert

### 1. The backbone
pi0 starts from **PaliGemma**, a 3B-parameter open vision-language model from Google. It reads camera images (several views) and the language instruction.

### 2. The action expert
A separate set of Transformer weights, about **300M parameters**, handles the robot's state and the actions. The two parts share attention - action tokens can attend to the image and text tokens - but have their own weights, a mixture-of-experts-style split. Total: about **3.3B parameters**.

```
 images + instruction  --->  [ PaliGemma VLM weights ]
                                     |  shared attention
 robot state + noisy actions --->  [ action expert weights ]  ---> denoised action chunk
```

### 3. Flow matching over action chunks
Instead of predicting one action, pi0 generates an **action chunk of H = 50 future steps** at once. Training uses **flow matching**: take a real action chunk, mix it with random noise, and train the action expert to predict the direction ("velocity") from the noisy version back towards the real one. At inference, start from pure noise and integrate that velocity field for **10 steps** to get a clean chunk.

Chunks are executed open-loop before the model replans - for example, every 0.5 s on a 50 Hz robot.

### 4. Many robots, one action space
Different robots have different numbers of joints. pi0 **zero-pads every action to 18 dimensions** so one model can control them all, from a single arm to a mobile manipulator with two arms.

---

## The Training Recipe

### Pre-training mixture
- More than **10,000 hours** of robot data in total.
- **903 million timesteps** of Physical Intelligence's own data - 106M single-arm and 797M dual-arm - spanning **7 robot configurations** and **68 tasks**.
- Open datasets (Open X-Embodiment, Bridge v2, DROID) make up **9.1%** of the mixture.
- Task weighting scales with `n^0.43` of each task's data size, so large tasks do not drown small ones.
- **700,000 training steps.**

### Post-training
For hard tasks, fine-tune on curated, high-quality demonstrations, typically 5 to 100 hours per task. The analogy the authors draw is LLM pretraining followed by alignment: broad data gives robustness and recovery behaviours; curated data gives fluent execution.

### Inference cost
On a consumer RTX 4090 GPU, generating an action chunk takes about **73 ms** on-board, or **86 ms** with the model off-board.

---

## Key Results

- **Out-of-the-box tasks** (shirt folding, bussing, grocery bagging, taking toast out of a toaster): pi0 outperformed **OpenVLA** (7B, token-based actions), **Octo** (93M) and a **pi0-small** variant without VLM initialisation. OpenVLA in particular struggled because its discrete actions could not produce high-frequency chunks.
- **Language following** improved clearly with the VLM backbone compared with pi0-small.
- **Complex multi-stage tasks** after fine-tuning - folding laundry from a hamper, assembling a cardboard box, packing eggs, bussing a table - were completed with partial or full success where baselines largely failed. These were among the longest and most dexterous tasks shown by a single generalist robot policy at the time.
- **Pretraining helped fine-tuning:** fine-tuned from the pre-trained base, pi0 generally beat the same architecture trained from scratch on the task data.

---

## Why This Was Revolutionary

- **It settled the action-representation question for many labs.** Continuous generative action heads on top of a VLM became the common VLA design.
- **It brought the LLM playbook to robotics:** large, diverse pretraining, then targeted post-training.
- **It was released.** Open weights and code turned pi0 into shared infrastructure.

---

## Real-World Impact

- **pi0-FAST** (January 2025) introduced a better action tokeniser, making autoregressive VLAs competitive again for some settings.
- **pi0.5** (April 2025, [arXiv 2504.16054](https://arxiv.org/abs/2504.16054)) aimed at open-world generalisation, cleaning kitchens and bedrooms in homes not seen during training. The openpi repository ships pi0, pi0-FAST and pi0.5 checkpoints.
- **Research groups** fine-tune pi0 on their own robots, the way NLP researchers fine-tune open LLMs.

---

## Key Takeaways for Practitioners

1. **Use a pretrained VLM for understanding and a continuous generator for motion.**
2. **Predict chunks, not single steps.** Chunking gives smoother motion and makes high control rates feasible.
3. **Post-training data quality matters more than quantity** for dexterous skills.
4. **A consumer GPU is enough to run it**, which lowers the barrier for real-robot experiments.

---

## Limitations & Future Directions

The authors list these themselves:
- **Data composition is not understood.** Which data helps which skill, and how to weight it, remains largely empirical.
- **Reliability.** Even the best results are not near the reliability needed for unsupervised deployment in homes.
- **Transfer across domains.** It is not yet clear how far positive transfer extends to very different robots and tasks, such as legged locomotion or driving.
- **Open-loop chunks** mean the robot does not react to surprises within a chunk.

---

## Further Reading

- **Original Paper:** [arxiv.org/abs/2410.24164](https://arxiv.org/abs/2410.24164)
- **Code and weights:** [github.com/Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi)
- **FAST action tokenizer:** [arxiv.org/abs/2501.09747](https://arxiv.org/abs/2501.09747)
- **In this collection:** [RT-2](../123-rt2/summary.md), [Open X-Embodiment](../125-open-x-embodiment/summary.md), [Flow Matching / SD3](../../image-generation/72-flow-matching-sd3/summary.md), [Mixture of Experts](../../architectures/37-mixture-of-experts/summary.md)

## Citation

```bibtex
@article{black2024pi0,
  title={$\pi_0$: A Vision-Language-Action Flow Model for General Robot Control},
  author={Black, Kevin and Brown, Noah and Driess, Danny and Esmail, Adnan and Equi, Michael and Finn, Chelsea and Fusai, Niccolo and Groom, Lachy and Hausman, Karol and Ichter, Brian and others},
  journal={arXiv preprint arXiv:2410.24164},
  year={2024}
}
```

<!-- related:start -->

---

## Related in This Collection

- [High-Resolution Image Synthesis with Latent Diffusion Models (Stable Diffusion)](../../image-generation/07-stable-diffusion/summary.md)
- [Mixtral of Experts (and the Mixture-of-Experts Architecture)](../../architectures/37-mixture-of-experts/summary.md)
- [Flow Matching and Rectified Flow: The New Default for Image Generation (Stable Diffusion 3)](../../image-generation/72-flow-matching-sd3/summary.md)
- [RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control (RT-2)](../../robotics/123-rt2/summary.md)
- [Open X-Embodiment: Robotic Learning Datasets and RT-X Models (Open X-Embodiment / RT-X)](../../robotics/125-open-x-embodiment/summary.md)

<!-- related:end -->
