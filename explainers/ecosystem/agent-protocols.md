# Agent Protocols and Interop Standards

**In one line:** As of September 2026, AI agents connect to the world through a small set of layered standards (a model API to talk to the model, MCP to reach tools and data, A2A to reach other agents, and Agent Skills plus AGENTS.md to carry instructions), and most of them now sit under neutral Linux Foundation governance.
**Last reviewed:** 2026-09-30

---

## The short version

- **The OpenAI Chat Completions format is the de facto model API.** Anthropic, Google, Ollama, vLLM and many others accept it, so one client library can talk to many models. OpenAI's newer **Responses API** (March 2025) is designed for agents, and an open specification of it, **Open Responses**, launched in January 2026.
- **MCP (Model Context Protocol)** connects an agent to tools and data. Anthropic released it in November 2024; OpenAI, Google, Microsoft and others adopted it; it moved to the Linux Foundation's **Agentic AI Foundation** in December 2025. Its July 2026 revision made the protocol stateless.
- **A2A (Agent2Agent)** connects agents to other agents. Google announced it in April 2025, donated it to the Linux Foundation in June 2025, and v1.0 shipped in March 2026.
- **Agent Skills** package reusable instructions and scripts in a folder with a `SKILL.md` file. Anthropic launched them in October 2025 and published the format as an open standard in December 2025.
- **AGENTS.md** is a plain Markdown file in a code repository telling coding agents how to build, test and follow conventions. It is now also stewarded by the Agentic AI Foundation.
- **Commerce and UI protocols** (AP2, the Agentic Commerce Protocol, A2UI and others) are a newer, less settled layer.

## The mental model: a layered stack

```
   +-----------------------------------------------------------+
   |                  YOUR AGENT / APPLICATION                 |
   |   reads: AGENTS.md (repo guidance), SKILL.md (skills)     |
   +-----------------------------------------------------------+
        |                  |                      |
        | model API        | MCP                  | A2A
        v                  v                      v
   +-----------+   +-----------------+   +--------------------+
   |   MODEL   |   |  TOOLS & DATA   |   |   OTHER AGENTS     |
   | Chat Comp.|   |  (MCP servers:  |   | (discovered via an |
   | Responses |   |  GitHub, DBs,   |   |  Agent Card, work  |
   | native API|   |  files, SaaS)   |   |  exchanged as tasks)|
   +-----------+   +-----------------+   +--------------------+
```

Each arrow solves a different "M x N" problem. Without a shared standard, M applications times N models (or tools, or agents) means M x N custom integrations. With one, each side implements the standard once. The [Model Context Protocol summary](../../papers/techniques/59-model-context-protocol/summary.md) tells this story for tools in detail.

The layers are complementary, not competing. The A2A project describes itself explicitly as the agent-to-agent layer alongside MCP's agent-to-tool layer.

## 1. The model API: Chat Completions and Responses

### Chat Completions as a de facto standard

OpenAI's Chat Completions API (`POST /v1/chat/completions`, with a list of `messages` and optional `tools`) was never formally standardised, but it became the lingua franca because everyone copied it:

- **Anthropic** offers an OpenAI SDK compatibility layer for the Claude API, while noting it is intended mainly for testing and comparison, and that its native API exposes features the compatibility layer does not.
- **Google** makes Gemini models callable from the OpenAI Python and JavaScript libraries by changing three lines; its docs label the support beta.
- **Open-source servers** such as vLLM and SGLang, and local runners such as Ollama, expose the same endpoint (see [Open-Source Stack](open-source-stack.md)).

The practical benefit is portability: you can swap providers by changing a base URL and model name. The cost is lowest-common-denominator features. Provider-specific capabilities (Anthropic's prompt caching controls, extended thinking, citations; Google's grounding) usually require the native API.

How tool calling works inside these APIs, in general terms, is covered by the sibling repo's [Tool Use and Function Calling](https://github.com/PatrickWiloak/cloud-data-ai-security-zero-to-hero/blob/main/learn/concepts/tool-use-and-function-calling.md).

### The Responses API and Open Responses

Chat Completions was designed for turn-by-turn chat. In March 2025 OpenAI released the **Responses API**, built for agents: a single call can include reasoning, several tool calls (including hosted tools), and multi-step loops, with optional server-side conversation state. OpenAI announced in August 2025 that its older Assistants API was scheduled for removal on August 26, 2026, with Responses as the replacement.

In January 2026 OpenAI initiated **Open Responses**, an open specification of the Responses format for multi-provider use, backed by the Hugging Face ecosystem. Its stated aim is one schema for messages, tool calls, reasoning and streaming that can run against OpenAI, Anthropic, Gemini or local models. Ollama's OpenAI-compatible server already accepts stateless Responses requests.

Whether Responses-style APIs replace Chat Completions as the common denominator is still open as of September 2026. Chat Completions has the installed base; Responses has the agent features.

## 2. MCP: connecting agents to tools and data

The [Model Context Protocol](../../papers/techniques/59-model-context-protocol/summary.md) defines how an AI application (the client) discovers and calls capabilities exposed by a server: **tools** (functions the model can call), **resources** (data it can read) and **prompts** (templates). A GitHub MCP server, a database MCP server or a browser MCP server each works with any MCP-capable client.

Timeline:

| Date | Event |
|---|---|
| November 25, 2024 | Anthropic releases MCP as an open protocol |
| 2025 | OpenAI, Google, Microsoft and major developer tools add MCP support |
| November 25, 2025 | Spec revision `2025-11-25` |
| December 9, 2025 | Linux Foundation forms the **Agentic AI Foundation (AAIF)** with MCP, Block's goose and OpenAI's AGENTS.md as founding projects. Platinum members: AWS, Anthropic, Block, Bloomberg, Cloudflare, Google, Microsoft, OpenAI. The Linux Foundation cited over 10,000 published MCP servers |
| July 28, 2026 | Spec revision `2026-07-28` becomes current |

The July 2026 revision is the largest change since launch:

- **Stateless by design.** The `initialize` handshake and protocol-level sessions are gone. Every request carries its protocol version and client capabilities, and servers must implement a `server/discover` call instead. This makes MCP servers much easier to run behind ordinary load balancers.
- **Multi round-trip requests** replace server-initiated requests: when a server needs more input (for example, asking the user a question), it returns an "input required" result and the client retries with the answer.
- **Deprecations** under a new twelve-month deprecation policy: the Roots, Sampling and Logging features, and OAuth Dynamic Client Registration in favour of Client ID Metadata Documents.
- **Long-running tasks** moved from the core protocol to an official extension.

The sibling repo's [MCP Explained](https://github.com/PatrickWiloak/cloud-data-ai-security-zero-to-hero/blob/main/learn/concepts/mcp-explained.md) is the hands-on companion, including how to build and connect a server.

**The main criticism of MCP** is security. An MCP server's tool descriptions and outputs flow straight into the model's context, which makes prompt injection and over-broad permissions real risks, and early versions shipped before authorisation was fully specified. The spec has added OAuth-based authorisation and tightened it in each revision, but the risk of connecting an agent to an untrusted server is inherent. See [Guardrails and Safety](https://github.com/PatrickWiloak/cloud-data-ai-security-zero-to-hero/blob/main/learn/concepts/guardrails-and-safety.md) and [AI Threat Modeling](https://github.com/PatrickWiloak/cloud-data-ai-security-zero-to-hero/blob/main/learn/concepts/ai-threat-modeling.md) in the sibling repo.

## 3. A2A: connecting agents to other agents

MCP treats the other side as a tool: a function with a schema. **A2A** treats it as a peer agent that may work for minutes or hours, ask questions back, and return files. Its core concepts:

- **Agent Card:** a JSON document an agent publishes describing who it is, what it can do and how to authenticate. Other agents use it for discovery. Version 1.0 added signed Agent Cards.
- **Task:** the unit of work, with a lifecycle (submitted, working, input required, completed and so on).
- **Messages and artifacts:** the conversation about the task, and the outputs it produces.
- **Transport:** JSON-RPC over HTTP, and gRPC.

Timeline:

| Date | Event |
|---|---|
| April 9, 2025 | Google announces A2A with over 50 partners |
| June 23, 2025 | Linux Foundation launches the Agent2Agent project; founding members include AWS, Cisco, Google, Microsoft, Salesforce, SAP and ServiceNow |
| March 2026 | A2A v1.0, described by Google as the first stable, production-ready version |
| April 2026 | Google reports over 100 supporting companies at the one-year mark |

**The honest caveat:** agent-to-agent delegation across companies is still early. Most production agent systems as of 2026 are one organisation's agents calling tools, where MCP suffices. A2A's value grows when agents from different vendors must cooperate, for example a travel agent from one company booking through an airline's agent.

## 4. Instructions as files: Agent Skills and AGENTS.md

Two standards tackle a different problem: not "how does the agent call something" but "how does the agent know what to do here".

### Agent Skills

A **skill** is a folder with a `SKILL.md` file (a name and description in frontmatter, then instructions), plus optional scripts, reference documents and templates:

```
my-skill/
  SKILL.md        # required: name, description, instructions
  scripts/        # optional: code the agent can run
  references/     # optional: docs loaded only when needed
  assets/         # optional: templates, resources
```

Skills use **progressive disclosure**: at startup the agent reads only each skill's name and description; the full instructions load only when a task matches. This lets an agent carry many skills without filling its context window (see [Context Windows](../concepts/context-windows.md)).

Anthropic introduced skills on October 16, 2025 and published the format as an open standard at agentskills.io on December 18, 2025. As of September 2026 the standard's site lists support in, among others, OpenAI's Codex and ChatGPT, Google's Gemini CLI, GitHub Copilot, VS Code, Cursor, goose and Claude.

### AGENTS.md

**AGENTS.md** is simpler still: a Markdown file at the root of a code repository, like a README for agents. It holds build commands, test commands and conventions. There are no required fields. It emerged in August 2025 from collaboration between OpenAI Codex, Amp, Google's Jules, Cursor and Factory, and became a founding AAIF project. The AAIF announcement cited adoption by more than 60,000 open-source projects.

Skills and AGENTS.md overlap in spirit (both are "context engineering" through files), but differ in scope: AGENTS.md is always-on guidance for one repository; skills are on-demand capabilities that travel between projects.

## 5. Newer layers: commerce and interfaces

Several protocols target agents that buy things or render interfaces. They are younger and less settled, so this page lists only what could be verified:

- **Agentic Commerce Protocol (ACP)**, developed by Stripe and OpenAI, Apache-2.0 licensed, for agent-initiated checkout with merchants keeping control of the customer relationship.
- **AP2 (Agent Payments Protocol)**, **A2UI (Agent to User Interface)** and **UCP (Universal Commerce Protocol)**, which Google groups with A2A as a family of related protocols.

Expect consolidation here, as happened with MCP for tools.

## How to choose, in practice

| You want to... | Use |
|---|---|
| Call a model and stay portable across providers | Chat Completions-compatible client (or Open Responses as it matures) |
| Use one provider's full feature set | That provider's native API |
| Give an agent access to a tool, database or SaaS app | MCP |
| Let your agent delegate to another organisation's agent | A2A |
| Teach an agent a repeatable procedure | An Agent Skill |
| Tell coding agents how your repo works | AGENTS.md |

For the design question of whether you need an agent at all, see [Building Effective Agents](../../papers/essays/115-building-effective-agents/summary.md). For how agents are measured, see [Agents and Computer Use benchmarks](../benchmarks/agents-and-computer-use.md).

## What to watch

- **MCP `2026-07-28` adoption.** How quickly SDKs, clients and hosted servers move to the stateless protocol, and how smoothly the deprecated features are retired over the twelve-month window.
- **AAIF governance.** Whether more projects join the foundation and how spec decisions are made now that Anthropic and OpenAI share stewardship.
- **Open Responses versus Chat Completions.** Whether open servers and non-OpenAI providers converge on the Responses shape.
- **A2A in production.** Real cross-vendor deployments, not just framework support.
- **Skills portability.** Whether skills written for one agent reliably work in another, which the standard promises but cannot guarantee.
- **Security incidents.** Prompt injection through tool outputs and malicious servers or skills remain the main risk of all of these layers.

## Read next

- [Model Context Protocol (MCP)](../../papers/techniques/59-model-context-protocol/summary.md) - the paper-style summary of the tools layer
- [Toolformer](../../papers/techniques/24-toolformer/summary.md) and [ReAct](../../papers/techniques/21-react/summary.md) - the research roots of tool-using agents
- [Building Effective Agents](../../papers/essays/115-building-effective-agents/summary.md)
- [Open-Source Stack](open-source-stack.md) and [Labs Landscape](labs-landscape.md)
- Sibling repo: [MCP Explained](https://github.com/PatrickWiloak/cloud-data-ai-security-zero-to-hero/blob/main/learn/concepts/mcp-explained.md) and [Tool Use and Function Calling](https://github.com/PatrickWiloak/cloud-data-ai-security-zero-to-hero/blob/main/learn/concepts/tool-use-and-function-calling.md)

## Sources

- Anthropic, "Introducing the Model Context Protocol" (November 25, 2024): https://www.anthropic.com/news/model-context-protocol
- Model Context Protocol, versioning (current revision 2026-07-28): https://modelcontextprotocol.io/specification/versioning
- Model Context Protocol, 2026-07-28 changelog: https://modelcontextprotocol.io/specification/2026-07-28/changelog
- Linux Foundation, "Linux Foundation Announces the Formation of the Agentic AI Foundation (AAIF)" (December 9, 2025): https://www.linuxfoundation.org/press/linux-foundation-announces-the-formation-of-the-agentic-ai-foundation
- OpenAI, "OpenAI co-founds the Agentic AI Foundation under the Linux Foundation": https://openai.com/index/agentic-ai-foundation/
- Linux Foundation, "Linux Foundation Launches the Agent2Agent Protocol Project" (June 23, 2025): https://www.linuxfoundation.org/press/linux-foundation-launches-the-agent2agent-protocol-project-to-enable-secure-intelligent-communication-between-ai-agents
- Google Developers Blog, "Google Cloud donates A2A to Linux Foundation": https://developers.googleblog.com/en/google-cloud-donates-a2a-to-linux-foundation/
- Google Open Source Blog, "A year of open collaboration: Celebrating the anniversary of A2A" (April 16, 2026): https://opensource.googleblog.com/2026/04/a-year-of-open-collaboration-celebrating-the-anniversary-of-a2a.html
- A2A Protocol documentation: https://a2a-protocol.org/latest/
- Anthropic (Claude blog), "Introducing Agent Skills" (October 16, 2025; updated December 18, 2025): https://claude.com/blog/skills
- Agent Skills specification and client list: https://agentskills.io/
- AGENTS.md: https://agents.md/
- OpenAI, Responses API reference: https://developers.openai.com/api/reference/python/resources/responses
- OpenAI, deprecations page (Assistants API removal on August 26, 2026; Responses API released March 2025): https://developers.openai.com/api/docs/deprecations
- Hugging Face, "Open Responses: What you need to know" (January 15, 2026): https://huggingface.co/blog/open-responses
- Open Responses specification: https://www.openresponses.org/specification
- Anthropic, OpenAI SDK compatibility: https://platform.claude.com/docs/en/api/openai-sdk
- Google, Gemini API OpenAI compatibility: https://ai.google.dev/gemini-api/docs/openai
- Ollama, OpenAI compatibility: https://docs.ollama.com/api/openai-compatibility
- Agentic Commerce Protocol: https://www.agenticcommerce.dev/
