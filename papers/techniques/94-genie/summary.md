# Genie: Generative Interactive Environments

**Authors:** Jake Bruce, Michael Dennis, Ashley Edwards, Jack Parker-Holder, Yuge (Jimmy) Shi, Edward Hughes, Matthew Lai, Aditi Mavalankar, Richie Steigerwald, Chris Apps, Yusuf Aytar, Sarah Bechtle, Feryal Behbahani, Stephanie Chan, Nicolas Heess, Lucy Gonzalez, Simon Osindero, Sherjil Ozair, Scott Reed, Jingwei Zhang, Konrad Zolna, Jeff Clune, Nando de Freitas, Satinder Singh, Tim Rocktaeschel (Google DeepMind)
**Published:** February 2024
**Paper:** [arxiv.org/abs/2402.15391](https://arxiv.org/abs/2402.15391)
**Successor:** Genie 2 — [DeepMind blog, December 2024](https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/)

---

## Why This Paper Matters

Genie was the first **foundation world model**: a single neural network trained on internet video that can take a single image — a sketch, a photo, a frame from a real game — and turn it into a fully **playable, controllable 2D environment**. You hit a key; the model generates the next frame consistent with your action. There were no action labels in the training data. The model figured out, on its own, what actions exist in this universe of videos.

That capability — generating not just pixels but a **playable world** from a prompt — pointed at a future where game-like environments, training simulators, and embodied AI testbeds can be created on demand by a generative model. Within ten months DeepMind released Genie 2, which scaled the same idea to photorealistic 3D worlds with minute-long horizons, and Google's later Veo and Genie family extended it further. Genie established the template for the entire "world model from video" research program.

---

## The Problem

To train an embodied agent — a robot, a game-playing RL agent, a self-driving stack — you need an environment to act in. Building environments is brutally expensive:

**1. Hand-built simulators** (MuJoCo, Unreal Engine scenes, GTA-style worlds) take engineer-years and cover narrow domains.

**2. Real-world data collection** is slow, dangerous, and not scalable.

**3. Existing generative video models** produce nice clips but you can't *act* in them — they're passive footage, not environments.

Meanwhile, the internet has millions of hours of video of people playing games, performing tasks, driving, manipulating objects. Crucially, almost none of it is labeled with the actions that produced it. You see the screen but not the keystrokes.

The question Genie asks: **can we learn a controllable world model from unlabeled video alone?**

---

## The Core Innovation

Genie's central trick is the **Latent Action Model (LAM)**. Instead of requiring action labels, the model *invents* its own discrete action vocabulary by watching consecutive frame pairs.

Three components, trained jointly:

```
1. Video Tokenizer:        frames     -> discrete visual tokens
2. Latent Action Model:    (f_t, f_t+1) -> discrete latent action a_t
3. Dynamics Model:          (tokens_<=t, a_t) -> tokens_t+1
```

The brilliant part is the bottleneck. The LAM is trained so that an action `a_t` must be sufficient to predict frame `t+1` from frame `t`. But `a_t` is constrained to a small discrete vocabulary (e.g. 8 codes). The only way for the model to compress all the variation between consecutive frames into 3 bits is to discover the **actual underlying control signal** — the thing that *caused* one frame to lead to the next.

At inference, you ignore the LAM and let the user supply the action token directly. The dynamics model rolls the world forward. You have a playable game.

---

## How It Works

### Training data

200,000 hours of publicly available 2D platformer gameplay video scraped from the internet. **No action labels. No reward labels. No metadata.** Just RGB frames at 10 FPS, 16x16 tokenized to 256 visual codes per frame.

### Architecture

All three components are spatio-temporal transformers using **ST-Transformer blocks** — alternating attention over space (within a frame) and time (across frames). Parameters are scaled to **11B** for the largest Genie model.

```
   Input video frames (T frames)
              |
       Video Tokenizer (VQ-VAE)
              |
       discrete tokens [T x H x W]
              |
     +--------+--------+
     |                 |
Latent Action      Dynamics
   Model            Model
     |                 |
  action a_t       predict tokens_{t+1}
                  conditioned on a_t
```

### The LAM bottleneck

The Latent Action Model is a small transformer that takes `(frame_t, frame_{t+1})` and emits an action token. The dynamics model receives this action plus past tokens and must reconstruct frame `t+1`. The action vocabulary is tiny — **8 codes in the released model**.

If the LAM could pass arbitrarily rich information, the dynamics model would have a trivial job and no useful action would emerge. The bottleneck forces the LAM to encode only the parts of the frame change that the dynamics model *can't* predict from the past — which turns out to be the user's intent.

### Inference

```
1. Start with a prompt image (sketch, photo, painting, real game frame).
2. Tokenize it. This is frame 0.
3. User presses an action key (one of 8 latent action IDs).
4. Dynamics model predicts tokens for frame 1.
5. Detokenize -> render.
6. Repeat indefinitely.
```

The user controls a world the model has never seen, with actions the model never saw labeled.

---

## Key Results

### Emergent controls

The 8 learned actions, when probed, correspond to **intuitive game controls**: move left, move right, jump, no-op, and variants. This emerged with zero supervision — purely from compressing inter-frame change.

### Out-of-distribution prompts

Genie generates playable worlds from prompts wildly outside its training distribution:

- **Hand-drawn sketches:** a child's crayon scribble becomes a navigable platformer.
- **Real photographs:** a photo of a beach becomes a 2D environment you can move around in.
- **AI-generated art:** Imagen outputs become Genie levels.

This is the foundation-model property: train on enough variety and the model generalizes far beyond what was in the training set.

### Latent actions transfer to real robots

In a striking follow-up experiment, the authors trained Genie on **robot manipulation video** without action labels. The emergent latent actions could then be used to drive a real robot — proving the LAM was discovering genuine control structure, not artifacts of platformer rendering.

### Scaling holds

Loss decreases predictably with parameters and data. The scaling-laws story that worked for language models also works for world models. This is the key empirical justification for pushing the program further to 3D and to longer horizons.

---

## Genie 2 (Forward Look)

In December 2024 DeepMind announced **Genie 2**: the same recipe applied at much larger scale and in **3D**. Given a single generated image (often from Imagen), Genie 2 produces a playable, photorealistic 3D environment with first/third-person controls, object permanence across occlusions, basic physics (water, smoke, gravity), NPC behavior, and **up to a minute of consistent rollout**.

DeepMind positions Genie 2 as a tool for training general embodied agents — SIMA, their general game-playing agent, can be deployed and evaluated inside Genie-2-generated worlds. The vision: unlimited diverse training environments for embodied AI, generated on demand by a world model.

The line from Genie 1 to Genie 2 is the line from "proof of concept" to "foundation model for environments." The same recipe — tokenize video, learn latent actions from frame transitions, predict the next tokens — scales straight through.

---

## Impact and Legacy

Genie crystallized the **foundation world model** as a research category alongside foundation language models and foundation vision models. Subsequent work has followed the same broad recipe:

- **DIAMOND**, **GameNGen** (2024): neural simulations of specific games (DOOM, Atari) from video.
- **Oasis** (Decart, 2024): a real-time Minecraft world model in a browser.
- **World Labs / Fei-Fei Li**: photorealistic generated 3D scenes from images.
- **Wayve GAIA-1 / GAIA-2**: generative world models for autonomous driving.
- **Genie 2** and **SIMA**: DeepMind's continuing program for training embodied agents in generated worlds.

Genie also reframes the relationship between video models and RL. A sufficiently good generative video model that can be *conditioned on an action* is a simulator. A simulator is what RL has been bottlenecked on for a decade. World models close that loop.

There is even a credible path where general-purpose AI agents are trained primarily in generated environments — using a world model as the substrate the way LLMs use text corpora. Genie was the first paper to make that path look plausible.

---

## Connections to Other Papers

- **Sora / DiT (#44):** Genie shares Sora's diffusion-transformer family for video generation but adds action conditioning and a learned action space — turning passive video into an interactive environment.
- **Scaling Laws (#12):** The Genie paper explicitly demonstrates that the language-model scaling story carries over to world models.
- **DreamerV3 (#95):** The classical model-based RL world-model line of work. DreamerV3 learns a world model from agent experience for planning; Genie learns one from internet video for interactive generation. Different goals, converging future.
- **Voyager (#86):** An LLM-driven Minecraft agent — sibling work on agents that operate in rich environments, the kind of environment Genie can now generate.
- **AlphaZero (#89):** Earlier world-model + planning system; learned its model from self-play in a fixed game. Genie inverts that: the world model is the *output*, learned from observation.
- **Generative Agents (#58):** Agents inhabiting a simulated world. Genie provides a path to generating that world from a single image.

---

## Key Takeaways

1. **A world model can be learned from unlabeled video.** Genie introduced the Latent Action Model, which discovers the underlying action vocabulary by bottlenecking inter-frame predictions.
2. **Foundation world models work.** Train on enough varied video and the model generates playable environments from out-of-distribution prompts — sketches, photos, AI art.
3. **The 8-bit bottleneck is the trick.** Forcing the latent action to fit in a tiny vocabulary is what makes the model discover real control structure.
4. **Scale carries to 3D.** Genie 2 showed the recipe extends to photorealistic 3D worlds with minute-scale consistency — a substrate for training embodied agents.
5. **It opens a new training regime for embodied AI.** If environments can be generated on demand, the historical bottleneck for RL — having something to act in — starts to dissolve.
