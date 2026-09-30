# A Timeline of AI, 1950 to 2026

**In one line:** AI has swung between two ideas - write the knowledge in by hand, or learn it from data - and this page walks through each era of that swing, with dated milestones that link to the paper summaries in this collection.
**Last reviewed:** 2026-09-30

---

## The short version

- **Two families of ideas have taken turns leading.** Symbolic AI (hand-written rules and logic) led from the 1950s to the late 1980s. Learning from data (statistics, then neural networks) has led since the 1990s and completely since 2012.
- **Hype and "winters" alternate.** Twice, in the 1970s and again in the late 1980s, promises ran ahead of results and funding collapsed. The current boom is the longest run without a winter.
- **Three enablers explain the deep learning era:** big labelled datasets, GPU compute, and a few architectural ideas that let networks get deeper and longer. Rich Sutton's [Bitter Lesson](../../papers/essays/111-bitter-lesson/summary.md) is the one-page version of this argument.
- **The Transformer (2017) is the hinge of modern AI.** Almost every system you hear about in 2026, for text, images, code or robots, is built on it.
- **ChatGPT (November 2022) was a product moment, not a research one.** The underlying ideas were already published; what changed was that hundreds of millions of people could use them.
- **Since late 2024 the frontier has been "reasoning" models that think before answering, and "agents" that take actions.** By 2026 the biggest story is as much about who is allowed to use the strongest models, and when, as about what they can do.

```
1950   1956        1974-80    1980-87       1987-93     1990s-2000s     2012       2017         2022         2024-2026
 |      |             |          |             |            |             |          |            |              |
Turing Dartmouth  1st winter  expert      2nd winter   statistical    AlexNet   Transformer  ChatGPT   reasoning + agents
test   workshop               systems                  ML, SVMs       deep learning                    + release controls
       (symbolic AI) ------------------------|   (learning from data) --------------------------------------------->
```

## Era 1: The founding ideas and symbolic AI (1950-1973)

The story usually starts with Alan Turing's 1950 paper, [Computing Machinery and Intelligence](../../papers/essays/108-computing-machinery-and-intelligence/summary.md). Rather than argue about what "thinking" means, Turing proposed a test: if a machine can hold a text conversation that a judge cannot tell apart from a human's, we should take seriously the claim that it thinks. He also sketched the idea of a "child machine" that learns, which reads like a preview of the whole field.

The name "artificial intelligence" comes from a 1955 proposal by John McCarthy, Marvin Minsky, Nathaniel Rochester and Claude Shannon for a summer workshop at Dartmouth College, held in 1956. Their bet was that "every aspect of learning or any other feature of intelligence can in principle be so precisely described that a machine can be made to simulate it." For the next two decades most researchers pursued this through **symbolic AI**: represent the world as symbols (words, facts, rules) and manipulate them with logic and search. Programs proved theorems, played checkers, and solved puzzles in toy worlds. Joseph Weizenbaum's ELIZA (1966) mimicked a therapist with simple pattern matching, and people confided in it anyway, an early warning that fluent text is easy to mistake for understanding.

A second, quieter line of work tried to copy the brain instead. Frank Rosenblatt's **perceptron** (1958) was a single artificial neuron that learned its weights from examples. In 1969 Minsky and Seymour Papert's book *Perceptrons* showed the limits of single-layer networks, and neural network research lost most of its funding for more than a decade.

| Date | Milestone |
|---|---|
| 1950 | Turing publishes [Computing Machinery and Intelligence](../../papers/essays/108-computing-machinery-and-intelligence/summary.md) (the "imitation game") |
| Aug 1955 | Dartmouth proposal coins "artificial intelligence"; the workshop runs in summer 1956 |
| 1958 | Rosenblatt's perceptron, the first learning neural network |
| 1966 | ELIZA chatbot (Weizenbaum, MIT) |
| 1969 | Minsky and Papert, *Perceptrons*, shows single-layer network limits |

## Era 2: The first AI winter (1974-1980)

Early AI promised general problem solvers within a generation. What it delivered worked in small, clean "microworlds" and broke on the messiness of the real one. Two problems kept recurring: the **combinatorial explosion** (search spaces grow faster than any computer can explore) and **common sense** (a useful program needs a vast amount of everyday knowledge no one had written down).

In the UK, the 1973 Lighthill Report told the government that AI had failed to meet its "grandiose objectives", and funding was cut. In the US, DARPA shifted toward narrow, mission-oriented projects. The period from roughly 1974 to 1980 is now called the **first AI winter**: not a stop in research, but a collapse in money and reputation.

| Date | Milestone |
|---|---|
| 1973 | Lighthill Report criticises AI research in the UK; funding cut |
| 1974-1980 | First AI winter: US and UK funding shrinks sharply |

## Era 3: Expert systems, and the second winter (1980-1993)

AI came back by narrowing its ambition. An **expert system** encoded one domain's knowledge as hundreds or thousands of if-then rules written with human experts: diagnosing blood infections (MYCIN, Stanford, 1970s) or configuring computer orders (XCON, used by Digital Equipment Corporation from 1980). These saved companies real money, and a commercial industry grew around them, including specialised "Lisp machines". Japan's Fifth Generation Computer project (1982) prompted matching government programmes in the US and UK.

The rules approach had a ceiling. Each system was expensive to build, brittle outside its domain, and hard to maintain as the rule base grew. When cheaper desktop workstations undercut the Lisp machine market around 1987, the industry contracted fast. That began the **second AI winter**, which lasted into the early 1990s.

Meanwhile the neural network line revived. In 1986 David Rumelhart, Geoffrey Hinton and Ronald Williams popularised **backpropagation**, a way to train networks with several layers by passing the error backward and nudging every weight. Yann LeCun used it to train convolutional networks that read handwritten zip codes (1989). The idea that would win was on the table; it lacked data and compute.

| Date | Milestone |
|---|---|
| 1980 | XCON expert system in production at Digital Equipment Corporation |
| 1982 | Japan launches the Fifth Generation Computer project |
| Oct 1986 | Rumelhart, Hinton and Williams, "Learning representations by back-propagating errors" (Nature) |
| 1987 | Lisp machine market collapses; second AI winter begins |
| 1989 | LeCun et al. train a convolutional network to read handwritten zip codes |

## Era 4: Statistical machine learning (1993-2011)

The field recovered by becoming more modest and more mathematical. Instead of writing rules, researchers let algorithms fit **statistical models** to data and measured them on shared benchmarks. Support vector machines (1995), decision-tree ensembles, and probabilistic models powered spam filters, search ranking, recommendation and speech recognition. Machine translation moved from grammar rules to statistics learned from parallel texts. Much of this was not called "AI" at the time, partly to avoid the word's reputation.

Two moments reached the public. IBM's Deep Blue beat world chess champion Garry Kasparov in May 1997, mostly through fast search and a hand-tuned evaluation function rather than learning. IBM's Watson won *Jeopardy!* in February 2011 by combining many statistical components. Also in 1997, Sepp Hochreiter and Jurgen Schmidhuber published the **LSTM**, a recurrent network that could remember across long sequences and became the workhorse for text and speech until the Transformer.

The pieces for the next era were assembled quietly. Hinton's group showed in 2006 that deep networks could be trained layer by layer. Fei-Fei Li's team released **ImageNet** in 2009, a dataset of millions of labelled images. And GPUs, built for video games, turned out to be very good at the matrix arithmetic neural networks need.

| Date | Milestone |
|---|---|
| 1995 | Cortes and Vapnik, support vector networks |
| May 1997 | Deep Blue defeats Garry Kasparov at chess |
| Nov 1997 | Hochreiter and Schmidhuber, Long Short-Term Memory (LSTM) |
| 2006 | Hinton, Osindero and Teh, fast learning for deep belief nets |
| 2009 | ImageNet dataset published (CVPR 2009) |
| Feb 2011 | IBM Watson wins *Jeopardy!* |

## Era 5: The deep learning revival (2012-2016)

In 2012 a convolutional network from Alex Krizhevsky, Ilya Sutskever and Hinton, later called **AlexNet**, won the ImageNet competition with a top-5 error of 15.3 percent against 26.2 percent for the next entry. It was trained on two consumer GPUs. The margin convinced the computer vision community almost overnight, and the big technology companies began hiring neural network researchers and buying their startups.

Progress then came quickly on every front. [Word2vec](../../papers/techniques/53-word2vec/summary.md) (2013) showed that words could be represented as vectors where meaning becomes geometry. The [VAE](../../papers/image-generation/57-vae/summary.md) (2013) and [GANs](../../papers/image-generation/02-generative-adversarial-networks/summary.md) (2014) made neural networks generate images, not just classify them. [Seq2seq](../../papers/architectures/55-seq2seq/summary.md) and [Bahdanau attention](../../papers/architectures/66-bahdanau-attention/summary.md) (2014) rebuilt machine translation around neural networks, and attention would become the core of the next era. The [Adam optimizer](../../papers/techniques/142-adam/summary.md) (2014) made training more forgiving. [ResNet](../../papers/architectures/73-resnet/summary.md) (2015) used "skip connections" to train networks over a hundred layers deep, a trick every large model still uses.

Reinforcement learning (learning by trial, error and reward) had its showcase when DeepMind's AlphaGo beat Lee Sedol 4-1 at Go in March 2016, a decade earlier than many experts expected. Andrej Karpathy's 2015 essay [The Unreasonable Effectiveness of Recurrent Neural Networks](../../papers/essays/109-unreasonable-effectiveness-of-rnns/summary.md) captured the mood: a simple network trained to predict the next character was learning surprising structure.

| Date | Milestone |
|---|---|
| 2012 | AlexNet wins ImageNet (15.3% vs 26.2% top-5 error) |
| Jan 2013 | [Word2vec](../../papers/techniques/53-word2vec/summary.md) |
| Dec 2013 | [Variational autoencoders](../../papers/image-generation/57-vae/summary.md) |
| Jun 2014 | [Generative adversarial networks](../../papers/image-generation/02-generative-adversarial-networks/summary.md) |
| Sep 2014 | [Seq2seq](../../papers/architectures/55-seq2seq/summary.md) and [Bahdanau attention](../../papers/architectures/66-bahdanau-attention/summary.md) |
| Dec 2014 | [Adam optimizer](../../papers/techniques/142-adam/summary.md) |
| 2015 | [Knowledge distillation](../../papers/techniques/134-knowledge-distillation/summary.md), [U-Net](../../papers/architectures/74-unet/summary.md), [BPE subword tokens](../../papers/techniques/136-bpe-subword-units/summary.md), Karpathy's [RNN essay](../../papers/essays/109-unreasonable-effectiveness-of-rnns/summary.md) |
| Dec 2015 | [ResNet](../../papers/architectures/73-resnet/summary.md) |
| Mar 2016 | AlphaGo defeats Lee Sedol 4-1 |

## Era 6: The Transformer era (2017-2022)

In June 2017 eight Google researchers published [Attention Is All You Need](../../papers/architectures/01-attention-is-all-you-need/summary.md). The **Transformer** dropped recurrence entirely: every word looks at every other word through attention, all in parallel. That parallelism mattered more than any accuracy gain, because it let models use far more data and GPUs. (For how it works, see the sibling repo's [transformer architecture](https://github.com/PatrickWiloak/cloud-data-ai-security-zero-to-hero/blob/main/learn/concepts/transformer-architecture.md) page.)

Two recipes followed. **Pretrain, then fine-tune**: [GPT-1](../../papers/language-models/93-gpt1/summary.md) (2018) and [BERT](../../papers/language-models/03-bert/summary.md) (2018) trained on raw text first and adapted cheaply to tasks. **Scale up**: [GPT-2](../../papers/language-models/64-gpt2/summary.md) (2019) was big enough that OpenAI staged its release over misuse worries, and [GPT-3](../../papers/language-models/04-gpt3-few-shot-learners/summary.md) (2020, 175 billion parameters) could do new tasks from a few examples in the prompt. [Scaling laws](../../papers/techniques/12-scaling-laws/summary.md) (2020) showed loss falling predictably as models, data and compute grew, and [Chinchilla](../../papers/techniques/18-chinchilla/summary.md) (2022) corrected the recipe toward more data per parameter. Karpathy's [Software 2.0](../../papers/essays/110-software-2/summary.md), Sutton's [Bitter Lesson](../../papers/essays/111-bitter-lesson/summary.md) and Gwern's [Scaling Hypothesis](../../papers/essays/112-scaling-hypothesis/summary.md) argued, from different angles, that general methods plus compute beat hand-built cleverness.

The Transformer then spread beyond text. [Vision Transformers](../../papers/architectures/11-vision-transformer/summary.md) (2020) and [CLIP](../../papers/multimodal/08-clip/summary.md) (2021) linked images and words. [Diffusion models](../../papers/image-generation/06-diffusion-models/summary.md) (2020) led to [DALL-E 2](../../papers/image-generation/119-dalle2-unclip/summary.md), [Imagen](../../papers/image-generation/91-imagen/summary.md) and [Stable Diffusion](../../papers/image-generation/07-stable-diffusion/summary.md) in 2022, making text-to-image generation mainstream. [AlphaFold 2](../../papers/techniques/68-alphafold/summary.md) effectively solved protein structure prediction. [Codex](../../papers/language-models/56-codex/summary.md) (2021) powered GitHub Copilot. And two techniques that would matter enormously were published in early 2022: [chain-of-thought prompting](../../papers/techniques/09-chain-of-thought/summary.md) and [InstructGPT](../../papers/language-models/05-instructgpt-rlhf/summary.md), which used reinforcement learning from human feedback (RLHF) to make a model follow instructions.

In March 2019 Hinton, LeCun and Yoshua Bengio received the Turing Award for deep learning, a formal sign that the neural network line had won.

| Date | Milestone |
|---|---|
| Jun 2017 | [Transformer](../../papers/architectures/01-attention-is-all-you-need/summary.md) |
| Jul 2017 | [PPO](../../papers/techniques/63-ppo/summary.md), later the RL workhorse for RLHF |
| Nov 2017 | Karpathy, [Software 2.0](../../papers/essays/110-software-2/summary.md) |
| Dec 2017 | [AlphaZero](../../papers/techniques/102-alphazero/summary.md) masters chess, shogi and Go by self-play |
| Jun 2018 | [GPT-1](../../papers/language-models/93-gpt1/summary.md) |
| Oct 2018 | [BERT](../../papers/language-models/03-bert/summary.md) |
| Feb 2019 | [GPT-2](../../papers/language-models/64-gpt2/summary.md) |
| Mar 2019 | Sutton, [The Bitter Lesson](../../papers/essays/111-bitter-lesson/summary.md); Turing Award to Hinton, LeCun and Bengio |
| Oct 2019 | [T5](../../papers/language-models/65-t5/summary.md); [ZeRO and Megatron-LM](../../papers/techniques/76-zero-megatron/summary.md) for training at scale |
| Nov 2019 | Chollet, [On the Measure of Intelligence (ARC)](../../papers/techniques/138-arc-agi/summary.md) |
| Jan 2020 | [Scaling laws](../../papers/techniques/12-scaling-laws/summary.md) |
| May 2020 | [GPT-3](../../papers/language-models/04-gpt3-few-shot-learners/summary.md); [RAG](../../papers/techniques/13-rag/summary.md) |
| Jun 2020 | [Diffusion models (DDPM)](../../papers/image-generation/06-diffusion-models/summary.md) |
| Sep 2020 | [MMLU](../../papers/techniques/137-mmlu/summary.md) benchmark |
| Oct 2020 | [Vision Transformer](../../papers/architectures/11-vision-transformer/summary.md) |
| 2021 | [CLIP](../../papers/multimodal/08-clip/summary.md), [LoRA](../../papers/techniques/10-lora/summary.md), [Codex](../../papers/language-models/56-codex/summary.md), [AlphaFold 2](../../papers/techniques/68-alphafold/summary.md) in Nature |
| Jan 2022 | [Chain-of-thought prompting](../../papers/techniques/09-chain-of-thought/summary.md) |
| Mar 2022 | [InstructGPT / RLHF](../../papers/language-models/05-instructgpt-rlhf/summary.md); [Chinchilla](../../papers/techniques/18-chinchilla/summary.md) |
| Apr 2022 | [PaLM](../../papers/language-models/94-palm/summary.md); [DALL-E 2](../../papers/image-generation/119-dalle2-unclip/summary.md) |
| 2022 | [Stable Diffusion](../../papers/image-generation/07-stable-diffusion/summary.md) released openly |

## Era 7: The ChatGPT moment (November 2022 to mid-2024)

OpenAI released ChatGPT on 30 November 2022 as a "research preview". The model was a GPT-3.5 series model tuned with the InstructGPT recipe; the research was not new. The interface was: a free chat window anyone could use. It became one of the fastest-adopted consumer products on record, and within months every large technology company had reorganised around generative AI.

The race that followed had three threads. **Frontier models** got larger and multimodal: [GPT-4](../../papers/language-models/36-gpt4/summary.md) (March 2023, the same day Anthropic released its first Claude), [GPT-4V](../../papers/multimodal/23-gpt4v/summary.md), and Google's Gemini (December 2023). **Open-weight models** (models whose trained weights anyone can download and run) arrived with Meta's [LLaMA](../../papers/language-models/15-llama/summary.md) and [Llama 2](../../papers/language-models/17-llama2/summary.md), then [Mistral 7B](../../papers/language-models/95-mistral-7b/summary.md) and [Mixtral](../../papers/architectures/37-mixture-of-experts/summary.md), and cheap adaptation methods like [QLoRA](../../papers/techniques/22-qlora/summary.md) let hobbyists fine-tune them. **Alignment and safety** became mainstream research: [Constitutional AI](../../papers/language-models/14-constitutional-ai/summary.md), [DPO](../../papers/language-models/19-dpo/summary.md), [red teaming](../../papers/techniques/129-red-teaming-lms/summary.md), [adversarial jailbreaks](../../papers/techniques/127-gcg-adversarial-attacks/summary.md), [weak-to-strong generalisation](../../papers/techniques/128-weak-to-strong/summary.md) and [sleeper agents](../../papers/techniques/83-sleeper-agents/summary.md).

The first agent ideas also appeared here, well before they worked reliably: [ReAct](../../papers/techniques/21-react/summary.md), [Toolformer](../../papers/techniques/24-toolformer/summary.md), [Generative Agents](../../papers/techniques/58-generative-agents/summary.md), [Voyager](../../papers/techniques/100-voyager/summary.md) and [Reflexion](../../papers/techniques/78-reflexion/summary.md). [SWE-bench](../../papers/techniques/84-swe-bench/summary.md) (October 2023) gave the field a realistic yardstick for coding agents, on which the best systems then scored under 2 percent.

| Date | Milestone |
|---|---|
| 30 Nov 2022 | ChatGPT launches |
| Dec 2022 | [Constitutional AI](../../papers/language-models/14-constitutional-ai/summary.md); [Whisper](../../papers/multimodal/49-whisper/summary.md) paper |
| Feb 2023 | [LLaMA](../../papers/language-models/15-llama/summary.md); [Toolformer](../../papers/techniques/24-toolformer/summary.md) |
| 14 Mar 2023 | [GPT-4](../../papers/language-models/36-gpt4/summary.md); first Claude model |
| Apr 2023 | [LLaVA](../../papers/multimodal/46-llava/summary.md); [Generative Agents](../../papers/techniques/58-generative-agents/summary.md) |
| May 2023 | [DPO](../../papers/language-models/19-dpo/summary.md); [QLoRA](../../papers/techniques/22-qlora/summary.md); [process reward models](../../papers/techniques/51-process-reward-models/summary.md) |
| Jun 2023 | [phi-1, "Textbooks Are All You Need"](../../papers/language-models/135-phi-1-textbooks/summary.md) |
| Jul 2023 | [Llama 2](../../papers/language-models/17-llama2/summary.md); [GCG jailbreaks](../../papers/techniques/127-gcg-adversarial-attacks/summary.md); [RT-2](../../papers/robotics/123-rt2/summary.md) |
| Sep 2023 | [Mistral 7B](../../papers/language-models/95-mistral-7b/summary.md); [vLLM](../../papers/techniques/52-pagedattention-vllm/summary.md) |
| Oct 2023 | [SWE-bench](../../papers/techniques/84-swe-bench/summary.md) |
| Dec 2023 | Gemini 1.0; [Mamba](../../papers/architectures/20-mamba/summary.md); [weak-to-strong](../../papers/techniques/128-weak-to-strong/summary.md) |
| Feb 2024 | [Sora and DiT](../../papers/image-generation/44-sora-dit/summary.md) |
| May 2024 | [GPT-4o](../../papers/language-models/40-gpt4o/summary.md); [Scaling Monosemanticity](../../papers/techniques/82-sparse-autoencoders/summary.md); [AlphaFold 3](../../papers/techniques/101-alphafold3/summary.md); [DeepSeek-V2 / MLA](../../papers/architectures/141-multi-head-latent-attention/summary.md) |

## Era 8: Reasoning and agents (late 2024 to September 2026)

The next shift was about **when** a model does its work. Until 2024, a model's quality was fixed at training time. OpenAI's [o1](../../papers/language-models/31-openai-o1/summary.md) (September 2024) was trained with reinforcement learning to produce a long hidden chain of thought before answering, and it got better the longer it was allowed to think. This is **test-time compute** (see [the test-time compute paper](../../papers/techniques/50-test-time-compute/summary.md) and the [reasoning models explainer](../concepts/reasoning-models.md)). In January 2025 [DeepSeek-R1](../../papers/language-models/26-deepseek-r1/summary.md) published an open recipe for the same idea, built on [GRPO](../../papers/techniques/38-grpo/summary.md) and [verifiable rewards](../../papers/techniques/39-rlvr/summary.md), and matched o1 on major math and code benchmarks. Its release briefly shook markets, because it suggested frontier capability no longer required frontier budgets.

At the same time, models learned to **act**. Anthropic's [computer use](../../papers/language-models/30-claude-3.5-sonnet/summary.md) (October 2024) let a model operate a desktop through screenshots and clicks, and the [Model Context Protocol](../../papers/techniques/59-model-context-protocol/summary.md) (November 2024) standardised how models connect to tools. Coding became the first domain where agents paid for themselves: Claude Code launched in February 2025, and coding assistants from every major lab followed. Anthropic's [Building Effective Agents](../../papers/essays/115-building-effective-agents/summary.md) became the practical reference. Through 2025, reasoning, tool use and agents merged into single products: [Claude 4](../../papers/language-models/43-claude4/summary.md), [Gemini 2.5](../../papers/multimodal/29-gemini-2.5/summary.md), o3, [GPT-5](../../papers/language-models/42-gpt5/summary.md) and [Gemini 3](../../papers/multimodal/47-gemini3/summary.md). In July 2025, experimental systems from OpenAI and Google DeepMind each scored 35 of 42 points at the International Mathematical Olympiad, a gold-medal standard. Open-weight models from China ([DeepSeek-V3](../../papers/language-models/27-deepseek-v3/summary.md), [Qwen3](../../papers/language-models/28-qwen3/summary.md)) closed much of the gap with closed ones.

In 2026 a new theme took over: **controlled release**. In April, Anthropic announced Claude Mythos Preview and withheld it from general release, saying it could find and exploit software vulnerabilities better than all but the most skilled humans; it went instead to critical-software maintainers through Project Glasswing. In June, a US executive order set up a voluntary framework for the government to get early access to "covered frontier models". Anthropic released Claude Fable 5, a Mythos-class model with added safeguards, to the public on 9 June; on 12 June US export controls forced it to restrict access for foreign nationals until the controls were lifted at the end of the month. OpenAI released its GPT-5.6 family first to a small group of trusted partners on 26 June at the government's request, and publicly on 9 July. In Europe, the AI Act's first obligations took effect in 2025, and a 2026 amendment deferred most high-risk system rules to December 2027. For the details, see the [US AI policy](../policy/us-ai-policy.md), [EU AI Act](../policy/eu-ai-act.md) and [frontier safety frameworks](../policy/frontier-safety-frameworks.md) explainers.

| Date | Milestone |
|---|---|
| 1 Aug 2024 | EU AI Act enters into force |
| 12 Sep 2024 | [OpenAI o1](../../papers/language-models/31-openai-o1/summary.md), the first widely available reasoning model |
| Oct 2024 | Nobel Prizes: Physics to Hopfield and Hinton, Chemistry half to Hassabis and Jumper (AlphaFold); Amodei, [Machines of Loving Grace](../../papers/essays/114-machines-of-loving-grace/summary.md) |
| 22 Oct 2024 | [Claude 3.5 Sonnet computer use](../../papers/language-models/30-claude-3.5-sonnet/summary.md) |
| 25 Nov 2024 | [Model Context Protocol](../../papers/techniques/59-model-context-protocol/summary.md) |
| Dec 2024 | [DeepSeek-V3](../../papers/language-models/27-deepseek-v3/summary.md); Anthropic, [Building Effective Agents](../../papers/essays/115-building-effective-agents/summary.md) |
| 20 Jan 2025 | [DeepSeek-R1](../../papers/language-models/26-deepseek-r1/summary.md) |
| 24 Feb 2025 | Claude 3.7 Sonnet (hybrid reasoning) and Claude Code preview |
| Mar 2025 | [Gemini 2.5](../../papers/multimodal/29-gemini-2.5/summary.md) |
| Apr 2025 | [Llama 4](../../papers/language-models/41-llama4/summary.md) (5 Apr); OpenAI o3 and o4-mini (16 Apr); [Qwen3](../../papers/language-models/28-qwen3/summary.md); Silver and Sutton, [Era of Experience](../../papers/essays/116-era-of-experience/summary.md) |
| May 2025 | [AlphaEvolve](../../papers/techniques/62-alphaevolve/summary.md); [Claude 4](../../papers/language-models/43-claude4/summary.md) (22 May) |
| Jul 2025 | OpenAI and Google DeepMind systems reach IMO gold-medal standard (35/42) |
| 2 Aug 2025 | EU AI Act obligations for general-purpose AI models apply |
| 7 Aug 2025 | [GPT-5](../../papers/language-models/42-gpt5/summary.md) |
| 18 Nov 2025 | [Gemini 3](../../papers/multimodal/47-gemini3/summary.md) |
| 5 Feb 2026 | Claude Opus 4.6 (1M-token context) |
| 7 Apr 2026 | Claude Mythos Preview withheld from public release; Project Glasswing |
| 23-24 Apr 2026 | GPT-5.5; DeepSeek-V4 preview (open weights, 1M context) |
| 19 May 2026 | Gemini 3.5 Flash at Google I/O |
| 28 May 2026 | Claude Opus 4.8 |
| 2 Jun 2026 | US Executive Order 14409 on frontier AI innovation and security |
| 9 Jun 2026 | Claude Fable 5 (public) and Mythos 5 (restricted) |
| 12-30 Jun 2026 | US export controls restrict Fable 5 / Mythos 5 access for foreign nationals, then lifted |
| 26 Jun / 9 Jul 2026 | GPT-5.6 (Luna, Terra, Sol): trusted-partner preview, then public release |
| 27 Jul 2026 | EU AI Omnibus in force; Annex III high-risk obligations deferred to 2 Dec 2027 |

## What to watch

- **Release gating.** As of September 2026 both leading US labs have shipped their strongest models to limited groups first, and the US government has a formal pre-release access process. Whether this stays voluntary, and whether other countries copy it, will shape who gets frontier AI and when.
- **Agents on long tasks.** The measure that matters is shifting from single-question benchmarks to how long a model can work unsupervised. See the [agent benchmarks explainer](../benchmarks/agents-and-computer-use.md) and [OSWorld](../../papers/techniques/139-osworld/summary.md).
- **Open-weight parity.** DeepSeek-V4 was released in April 2026 claiming open-model leadership on agentic coding; how close open models stay to closed ones is covered in [open vs closed weights](../concepts/open-vs-closed-weights.md).
- **Does scaling keep paying?** Pretraining gains, data limits and the shift to reinforcement learning are argued out in [scaling limits](../open-questions/scaling-limits.md) and [synthetic data and model collapse](../open-questions/synthetic-data-and-model-collapse.md).
- **2 December 2027:** the deferred EU high-risk obligations begin to apply.

## Read next: where to go from here

This page is the entry point. Pick a thread:

**Model families** - how each lab's line evolved, release by release:
[GPT](../model-families/gpt.md) · [Claude](../model-families/claude.md) · [Gemini](../model-families/gemini.md) · [Llama](../model-families/llama.md) · [DeepSeek](../model-families/deepseek.md) · [Qwen](../model-families/qwen.md) · [Mistral](../model-families/mistral.md)

**Benchmarks** - how progress is measured, and how the measures break:
[Knowledge and reasoning](../benchmarks/knowledge-and-reasoning.md) · [Math and code](../benchmarks/math-and-code.md) · [Agents and computer use](../benchmarks/agents-and-computer-use.md) · [Human preference arenas](../benchmarks/human-preference-arenas.md) · [Contamination and saturation](../benchmarks/contamination-and-saturation.md)

**Open questions** - where informed people disagree:
[Scaling limits](../open-questions/scaling-limits.md) · [Do LLMs reason?](../open-questions/do-llms-reason.md) · [Synthetic data and model collapse](../open-questions/synthetic-data-and-model-collapse.md)

**Also useful:**
- How the pieces work: [tokenization](../concepts/tokenization.md), [context windows](../concepts/context-windows.md), [reasoning models](../concepts/reasoning-models.md); the sibling repo's [LLM basics](https://github.com/PatrickWiloak/cloud-data-ai-security-zero-to-hero/blob/main/learn/concepts/llm-basics.md)
- Who builds it and what it costs: [labs landscape](../ecosystem/labs-landscape.md), [AI hardware](../compute/ai-hardware-landscape.md), [cost of training](../compute/cost-of-training.md)
- Big-picture essays: [Situational Awareness](../../papers/essays/113-situational-awareness/summary.md), [Machines of Loving Grace](../../papers/essays/114-machines-of-loving-grace/summary.md), [Era of Experience](../../papers/essays/116-era-of-experience/summary.md)
- Beyond text: [robotics with pi0](../../papers/robotics/124-pi0/summary.md), [Open X-Embodiment](../../papers/robotics/125-open-x-embodiment/summary.md), [NeRF](../../papers/image-generation/117-nerf/summary.md), [3D Gaussian Splatting](../../papers/image-generation/118-3d-gaussian-splatting/summary.md)

## Sources

- Turing, A. M. (1950). Computing Machinery and Intelligence. *Mind* 59(236). https://doi.org/10.1093/mind/LIX.236.433
- McCarthy, Minsky, Rochester, Shannon (1955). A Proposal for the Dartmouth Summer Research Project on Artificial Intelligence. http://jmc.stanford.edu/articles/dartmouth/dartmouth.pdf
- Rosenblatt, F. (1958). The perceptron. *Psychological Review* 65(6). https://doi.org/10.1037/h0042519
- Weizenbaum, J. (1966). ELIZA. *Communications of the ACM* 9(1). https://doi.org/10.1145/365153.365168
- Lighthill, J. (1973). Artificial Intelligence: A General Survey. http://www.chilton-computing.org.uk/inf/literature/reports/lighthill_report/p001.htm
- Rumelhart, Hinton, Williams (1986). Learning representations by back-propagating errors. *Nature* 323. https://doi.org/10.1038/323533a0
- LeCun et al. (1989). Backpropagation Applied to Handwritten Zip Code Recognition. *Neural Computation* 1(4). https://doi.org/10.1162/neco.1989.1.4.541
- Cortes and Vapnik (1995). Support-vector networks. *Machine Learning* 20. https://doi.org/10.1007/BF00994018
- Hochreiter and Schmidhuber (1997). Long Short-Term Memory. *Neural Computation* 9(8). https://doi.org/10.1162/neco.1997.9.8.1735
- IBM, Deep Blue. https://www.ibm.com/history/deep-blue ; IBM, Watson and Jeopardy!. https://www.ibm.com/history/watson-jeopardy
- Hinton, Osindero, Teh (2006). A fast learning algorithm for deep belief nets. *Neural Computation* 18(7). https://doi.org/10.1162/neco.2006.18.7.1527
- Deng et al. (2009). ImageNet: A large-scale hierarchical image database. CVPR. https://doi.org/10.1109/CVPR.2009.5206848
- Krizhevsky, Sutskever, Hinton (2012). ImageNet Classification with Deep Convolutional Neural Networks. NeurIPS. https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html
- Google DeepMind, AlphaGo. https://deepmind.google/research/breakthroughs/alphago/
- ACM, 2018 Turing Award (announced March 2019). https://awards.acm.org/about/2018-turing
- OpenAI, Introducing ChatGPT (30 November 2022). https://openai.com/index/chatgpt/
- Nobel Prize in Physics 2024. https://www.nobelprize.org/prizes/physics/2024/press-release/ ; Nobel Prize in Chemistry 2024. https://www.nobelprize.org/prizes/chemistry/2024/press-release/
- European Commission, AI Act regulatory framework and timeline. https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai
- Anthropic, Claude 3.7 Sonnet and Claude Code (24 February 2025). https://www.anthropic.com/news/claude-3-7-sonnet
- OpenAI, Introducing OpenAI o3 and o4-mini (16 April 2025). https://openai.com/index/introducing-o3-and-o4-mini/
- Anthropic, Introducing Claude 4 (22 May 2025). https://www.anthropic.com/news/claude-4
- Google DeepMind, Gemini Deep Think achieves gold-medal standard at the IMO (July 2025). https://deepmind.google/blog/advanced-version-of-gemini-with-deep-think-officially-achieves-gold-medal-standard-at-the-international-mathematical-olympiad/ ; OpenAI announcement: https://x.com/OpenAI/status/1946594928945148246
- InfoQ, Google Announces Gemini 3 (November 2025). https://www.infoq.com/news/2025/11/google-gemini-3/
- Anthropic, Claude Opus 4.6 system card (February 2026). https://www-cdn.anthropic.com/0dd865075ad3132672ee0ab40b05a53f14cf5288.pdf
- Anthropic, Project Glasswing (7 April 2026). https://www.anthropic.com/glasswing
- OpenAI, Introducing GPT-5.5 (23 April 2026). https://openai.com/index/introducing-gpt-5-5/
- DeepSeek, DeepSeek V4 Preview Release (24 April 2026). https://api-docs.deepseek.com/news/news260424/
- Google, Google I/O 2026 announcements (19 May 2026). https://blog.google/innovation-and-ai/technology/developers-tools/google-io-2026-collection/
- Anthropic, Introducing Claude Opus 4.8 (28 May 2026). https://www.anthropic.com/news/claude-opus-4-8
- The White House, Promoting Advanced Artificial Intelligence Innovation and Security (2 June 2026; EO 14409). https://www.whitehouse.gov/presidential-actions/2026/06/promoting-advanced-artificial-intelligence-innovation-and-security/
- Anthropic, Claude Fable 5 and Claude Mythos 5 (9 June 2026). https://www.anthropic.com/news/claude-fable-5-mythos-5
- Anthropic, Redeploying Claude Fable 5 (30 June 2026). https://www.anthropic.com/news/redeploying-fable-5
- OpenAI, Previewing GPT-5.6 Sol. https://openai.com/index/previewing-gpt-5-6-sol/ ; TechCrunch, OpenAI limits GPT-5.6 rollout after government request (26 June 2026). https://techcrunch.com/2026/06/26/openai-limits-gpt-5-6-rollout-after-government-request-says-restrictions-shouldnt-be-the-norm/ ; GCN, GPT-5.6 launches publicly. https://gcn.com/gpt-5-6-openai-launches-publicly/20242/
- White & Case, EU AI Omnibus enters into force (Regulation (EU) 2026/1744, in force 27 July 2026). https://www.whitecase.com/insight-alert/eu-ai-omnibus-enters-force-amending-ai-act ; Gibson Dunn, EU AI Act Omnibus agreement. https://www.gibsondunn.com/eu-ai-act-omnibus-agreement-postponed-high-risk-deadlines-and-other-key-changes/
- Paper summaries linked above, each with its own primary citation.
