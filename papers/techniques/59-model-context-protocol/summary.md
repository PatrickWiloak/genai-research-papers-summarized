---
title: "Model Context Protocol (MCP): An Open Standard for AI Tool Integration"
slug: "59-model-context-protocol"
number: 59
category: "techniques"
authors: "Anthropic"
published: "November 25, 2024"
year: 2024
url: "https://www.anthropic.com/news/model-context-protocol"
tags: [techniques]
---

# Model Context Protocol (MCP): An Open Standard for AI Tool Integration

**Organization:** Anthropic
**Published:** November 25, 2024
**Announcement:** [anthropic.com/news/model-context-protocol](https://www.anthropic.com/news/model-context-protocol)
**Specification:** [modelcontextprotocol.io](https://modelcontextprotocol.io)
**Reference Implementations:** [github.com/modelcontextprotocol](https://github.com/modelcontextprotocol)

---

## Why This Matters

Model Context Protocol (MCP) is **the integration standard that turned LLMs from chatbots into agents**:

- **Open protocol** - Vendor-neutral standard for connecting AI models to tools, data, and prompts
- **Solves the M x N problem** - Replaces N-squared bespoke integrations with a single shared interface
- **USB-C for AI** - One plug for every model and every tool, instead of custom wiring per pair
- **Universal adoption** - Anthropic, OpenAI (March 2025), Microsoft Copilot Studio, Google DeepMind, Cursor, Windsurf, Cline, Zed, Replit, Sourcegraph, and thousands of community servers
- **Agent-native by design** - First-class support for tool use, resource exposure, prompt templates, and sampling

**Real-world impact:**
- Within 6 months of release, more than 1,000 community MCP servers existed
- OpenAI adopted MCP in ChatGPT and the Agents SDK in March 2025, ending the "function calling" vs. "MCP" cold war
- Microsoft Copilot Studio, Windows 11, and GitHub Copilot all consume MCP
- Replaced ad-hoc plugin systems (OpenAI Plugins, ChatGPT Actions) as the default agent integration story
- Made local-first agentic IDEs (Cursor, Windsurf, Cline, Claude Code) practical by giving them a tool ecosystem

**The insight:** **Connecting an LLM to your tools should be a configuration step, not an engineering project.** Every model already speaks tool use. Every tool already has an API. The missing piece was a standard wire format between them. MCP fills that gap.

---

## The Problem

### The M x N Integration Explosion

```
Before MCP, every AI app shipped its own integrations:

   Models (M)              Tools / Data Sources (N)
   --------                ------------------------
   Claude        x         GitHub
   GPT-4         x         Slack
   Gemini        x         Postgres
   Llama         x         Notion
   Mistral       x         Google Drive
   ...                     Filesystem
                           Jira
                           Salesforce
                           ...

   M models x N tools = M*N custom integrations

   Each pair needed:
     - Bespoke auth wiring
     - Bespoke schema definitions
     - Bespoke serialization
     - Bespoke error handling
     - Bespoke deployment

   Result: an O(M*N) explosion of nearly-identical glue code,
   none of it reusable across vendors.
```

### Why Function Calling Alone Wasn't Enough

```
OpenAI Function Calling (2023):
  - Defines: how a model declares it wants to call a tool
  - Does NOT define: how the tool itself is hosted, discovered, or invoked
  - Each app rewrites:
      tool registry, dispatch loop, auth, rate limiting,
      streaming, schema versioning, error semantics

ChatGPT Plugins / Actions (2023):
  - Vendor-locked to OpenAI
  - Required hosted OpenAPI manifests
  - No local-tool story
  - Deprecated in early 2024

LangChain / LlamaIndex tool wrappers:
  - Library-level, not protocol-level
  - Lock you to one framework
  - Tools written for LangChain don't work in Claude Desktop,
    don't work in Cursor, don't work in your CLI agent.
```

### The Real Pain Point: Local Context

```
Most useful agent context lives on YOUR machine:
  - Your filesystem
  - Your local Postgres
  - Your git repos
  - Your private API keys
  - Your IDE state

Cloud-only plugin architectures couldn't reach any of it
without exposing data through an internet-facing endpoint.

What was missing: a way for a desktop app (Claude Desktop, Cursor)
to spawn a local subprocess that exposes tools to the model
over a well-defined wire protocol.

That is exactly the gap MCP fills.
```

---

## How MCP Works

### The Architecture in One Picture

```
+--------------------+        +---------------------+        +--------------------+
|   MCP Host         |        |   MCP Client        |        |   MCP Server       |
|   (Claude Desktop, | <----> |  (one per server,   | <----> |  (GitHub, Postgres,|
|    Cursor, Cline,  |        |   embedded in host) |        |   Filesystem, etc) |
|    ChatGPT, ...)   |        |                     |        |                    |
+--------------------+        +---------------------+        +--------------------+
        |                                                              |
        |  Hosts the LLM, owns the UI, gates user approval             |
        |                                                              |
        |                                       Exposes tools / resources / prompts
        |                                       Talks to the actual API or system
        |
   Approves tool calls, displays results, manages multiple servers at once.

The HOST is the trust boundary. The MODEL never talks to a server directly;
it asks the host, the host asks the user (when needed), and the host calls
the appropriate client which speaks JSON-RPC to the server.
```

### The Three Primitives

MCP servers expose exactly three kinds of capability. This minimal vocabulary is the protocol's most important design choice.

```
1. TOOLS         - Model-invoked actions with side effects
                   Example: github.create_issue, postgres.query, fs.write_file
                   The model decides when to call them (with host approval).

2. RESOURCES     - Read-only data the host can attach to context
                   Example: file://README.md, postgres://schema/users
                   The USER (or host) decides what to include - the model just reads.

3. PROMPTS       - Reusable, parameterized prompt templates the user can invoke
                   Example: /summarize-pr, /explain-table
                   Surface as slash commands or buttons in the host UI.

This three-way split maps cleanly to:
  - Tools     -> agent action
  - Resources -> retrieval / RAG context
  - Prompts   -> user-driven workflows

Three primitives. That is the entire surface area.
```

### The Wire Protocol: JSON-RPC 2.0

```
MCP is JSON-RPC 2.0 with a fixed message vocabulary.

Initialize:
  -> {"jsonrpc":"2.0","id":1,"method":"initialize",
      "params":{"protocolVersion":"2024-11-05","capabilities":{...}}}
  <- {"jsonrpc":"2.0","id":1,"result":
      {"capabilities":{"tools":{},"resources":{},"prompts":{}}}}

List tools:
  -> {"jsonrpc":"2.0","id":2,"method":"tools/list"}
  <- {"jsonrpc":"2.0","id":2,"result":{"tools":[
       {"name":"create_issue","description":"...",
        "inputSchema":{"type":"object","properties":{...}}}
     ]}}

Call a tool:
  -> {"jsonrpc":"2.0","id":3,"method":"tools/call",
      "params":{"name":"create_issue","arguments":{"title":"..."}}}
  <- {"jsonrpc":"2.0","id":3,"result":
      {"content":[{"type":"text","text":"Issue #42 created"}]}}

Server-initiated notifications:
  <- {"jsonrpc":"2.0","method":"notifications/tools/list_changed"}
```

### Three Transports for Three Deployment Stories

```
1. stdio                    Local subprocess
   - Host spawns server as a child process
   - Communicates via stdin/stdout
   - Use when: server runs on the user's machine
   - Examples: filesystem, git, local Postgres, Docker

2. SSE (Server-Sent Events) Remote, streaming, original 2024 transport
   - Server hosts an HTTP endpoint
   - Use when: server is networked
   - Superseded by Streamable HTTP in 2025 spec revisions

3. Streamable HTTP          Remote, modern (2025+)
   - Single HTTP endpoint with optional upgrade to SSE
   - Better for serverless, load balancers, auth proxies
   - Now the recommended remote transport
```

### A Minimal Server in Python

```python
# pip install mcp
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("weather")

@mcp.tool()
def get_forecast(city: str) -> str:
    """Get the weather forecast for a city."""
    return f"Sunny, 72F in {city}"

@mcp.resource("weather://stations")
def list_stations() -> str:
    """List all known weather stations."""
    return "KSEA, KJFK, KLAX, ..."

@mcp.prompt()
def daily_briefing(city: str) -> str:
    """Generate a daily weather briefing prompt."""
    return f"Give me today's weather in {city} as a 3-bullet brief."

if __name__ == "__main__":
    mcp.run(transport="stdio")
```

A user adds this to Claude Desktop's config:

```json
{
  "mcpServers": {
    "weather": {
      "command": "python",
      "args": ["/path/to/weather_server.py"]
    }
  }
}
```

That is the entire integration. No SDK lock-in. No hosted endpoint. No vendor manifest. The same server now works in Claude Desktop, Cursor, Windsurf, Cline, Zed, ChatGPT, and any other MCP host.

---

## Key Innovations

### 1. The Host / Client / Server Split

```
The host is the trust boundary, NOT the model.

This split lets MCP servers be:
  - Untrusted by default
  - Approved per-tool, per-call by the user
  - Sandboxed to specific resources
  - Run as ordinary subprocesses with normal OS permissions

The model never sees raw server credentials. The host injects
auth on the way out, scrubs secrets on the way back in, and
mediates every side-effecting call with user approval.
```

### 2. Capability Negotiation

```
Initialize handshake declares which primitives each side supports:

  Server says:  "I have tools and resources, no prompts."
  Host says:    "I support sampling and roots."

This means a server can target the protocol once and degrade
gracefully across hosts with different feature support. No
"works in Cursor but not in Claude Desktop" surprises.
```

### 3. Sampling: Reverse Tool Use

```
Most striking inversion in the spec.

Normal flow:  Model -> calls tool -> server returns data
Sampling:     Server -> asks host -> host runs the model

A server can request that the host run an LLM completion
on its behalf. The host stays in control (it picks the model,
shows the user what is being asked, can deny). This lets a
server like "summarize-this-document" leverage whatever model
the user already pays for, instead of bringing its own API key.
```

### 4. Roots and Resources as First-Class Context

```
Other tool-use protocols treat everything as a function call.
MCP separates "things the model can DO" (tools) from "things
the model can READ" (resources).

This matches how humans use IDEs: you OPEN a file (resource)
and you RUN a command (tool). Conflating them is the original
sin of plugin-style architectures.
```

### 5. Schema-First, Not Code-First

```
Tool schemas are JSON Schema. Resources are URIs. Prompts are
templates. There is no SDK requirement - you can implement an
MCP server in 200 lines of any language that can speak JSON
over a pipe.

This is why community servers proliferated so fast: writing one
is the same effort as writing a small CLI tool.
```

---

## The Ecosystem

### Reference Servers (maintained by Anthropic / community)

```
Filesystem        Read/write files in user-approved roots
Git               Branches, diffs, commits, log
GitHub            Issues, PRs, releases, code search
GitLab            Same surface for GitLab
Postgres          Schema introspection + parameterized queries
SQLite            Local DB read/write
Slack             Channels, messages, search
Notion            Pages, databases, blocks
Google Drive      Search and read documents
Brave Search      Web search
Puppeteer         Browser automation
Playwright        Browser automation, more modern
Memory            Persistent KV store across sessions
Sequential Thinking  Multi-step reasoning helper
Time              Timezone-aware date/time math
Fetch             HTTP fetch with markdown extraction
```

### Hosts That Ship MCP Support

```
Anthropic
  Claude Desktop (Nov 2024 - launch host)
  Claude Code CLI (early 2025)
  claude.ai web (2025)

OpenAI
  ChatGPT (March 2025 announcement)
  OpenAI Agents SDK (native MCP client)

Microsoft
  Copilot Studio (May 2025)
  Windows 11 (system-level MCP runtime, 2025)
  GitHub Copilot (VS Code, JetBrains)

Google
  Gemini CLI / Gemini Code Assist
  Google DeepMind official SDK contributions

IDE / dev-tool ecosystem
  Cursor, Windsurf, Cline, Zed, Replit Agent,
  Sourcegraph Cody, Continue, Aider

Agent frameworks
  LangChain (mcp-adapters), LlamaIndex,
  Pydantic AI, Mastra, Vercel AI SDK
```

### MCP vs. OpenAI Function Calling vs. Plugins

```
                          Function       ChatGPT          MCP
                          Calling        Plugins
                          ----------     ----------       ----------
Scope                     Wire format    Hosted action    Full protocol
                          for ONE call   manifest         (tools+resources+prompts)

Vendor lock-in            OpenAI only    OpenAI only      Open, multi-vendor
Local subprocess          No             No               Yes (stdio)
Remote HTTP               Implicit       Yes              Yes (Streamable HTTP)
Resources / files         No             No               Yes (first-class)
Prompt templates          No             No               Yes
User approval boundary    App-defined    Hosted           Built into spec
Tool discovery            Per-app        OpenAPI manifest tools/list at runtime
Server-initiated msgs     No             No               Yes (notifications, sampling)
Adopted by competitors    No             No               Yes (OpenAI, Google, MS)

Function calling is a SUBSET of what MCP does. Plugins were
a vendor-specific point solution. MCP is the protocol-level
generalization both efforts implied but neither delivered.
```

---

## Security Model

```
MCP's threat model assumes:
  - Servers are untrusted code from third parties
  - Tools may have side effects on user data
  - Resources may contain prompt-injection payloads
  - The model itself may be manipulated

Defenses baked into the spec:

1. HOST-MEDIATED APPROVAL
   No tool call reaches a server without the host's say-so.
   Hosts SHOULD prompt the user (or apply a policy) per call.

2. EXPLICIT ROOTS
   Filesystem-style servers receive an allowlist of paths
   ("roots") at session start. They cannot access anything
   outside that allowlist.

3. CAPABILITY-SCOPED CONNECTIONS
   Each server connection declares exactly what it can do.
   A read-only server cannot suddenly start writing.

4. NO IMPLICIT NETWORK
   stdio servers run as subprocesses. They have whatever
   network access the host's OS gives them - nothing more.

5. AUDIT-FRIENDLY WIRE FORMAT
   JSON-RPC over stdio means every call is loggable, replayable,
   and inspectable. Hosts like Claude Desktop ship a built-in
   inspector for exactly this reason.

Caveats and ongoing work:
  - Prompt injection through resource content is still possible
    (defense lives in the host, not the protocol).
  - Remote MCP servers need OAuth - the 2025 spec adds an
    OAuth 2.1 profile.
  - Supply-chain risk for community servers is real; signed
    server registries are an active area of standardization.
```

---

## Real-World Applications

### IDE Agents

```
Cursor / Windsurf / Cline / Claude Code:

  Built-in MCP servers give the agent:
    - filesystem access (scoped to the workspace)
    - git operations
    - shell command execution
    - browser automation (Playwright)

  User-installed MCP servers add:
    - the company Postgres
    - internal API wrappers
    - design-system documentation lookup
    - Notion / Linear / Jira ticket access

  The same agent loop works against any combination because
  every tool is just another MCP server.
```

### Desktop / OS-Level Agents

```
Claude Desktop:           Original MCP host. Local-first.
Windows 11 (2025):        OS surfaces MCP servers as system capabilities.
ChatGPT desktop:          Reads / edits files via MCP filesystem server.

This is the use case OpenAI Plugins tried and failed at:
agents that act on YOUR machine, not just on the cloud.
```

### Internal Enterprise Agents

```
Pattern that emerged in 2025:

  Company runs internal MCP servers for:
    - their Salesforce / Hubspot
    - their data warehouse (Snowflake, BigQuery)
    - their docs (Confluence, Notion)
    - their ticketing (Jira, Linear)

  Employees use ANY MCP-capable client (Claude, ChatGPT, Cursor)
  and get the same internal toolset.

  The integration work is done ONCE per tool, not once per
  AI vendor x once per tool.

  This is the M+N replacement for the M*N problem.
```

### Personal Agent Stacks

```
A typical power-user stack in 2026:

  Host: Claude Desktop (or Cursor, or Claude Code)

  Servers:
    filesystem        - work directories
    git + github      - source control
    postgres          - personal databases
    notion / obsidian - notes
    google-calendar   - schedule
    fetch + brave     - web context
    memory            - cross-session state
    custom            - personal scripts wrapped as tools

  Same servers, same config, work across every host.
```

---

## Connections to Other Papers

### ReAct (Yao et al., 2022) - Paper 21

```
ReAct introduced the THOUGHT -> ACTION -> OBSERVATION loop:
the model interleaves reasoning with tool calls.

MCP is the missing piece: ReAct described WHAT the loop should
look like; MCP describes HOW the actions and observations
actually move between model and world.

Every MCP-driven agent is, at the inner loop, running ReAct.
```

### Toolformer (Schick et al., 2023) - Paper 24

```
Toolformer fine-tuned a model to learn WHEN to call tools.

Modern instruction-tuned models (Claude 3+, GPT-4+, Llama 3.1+)
already know when to call tools - that capability is baked in
during post-training. What was missing was a uniform way to
EXPOSE tools to those models.

Toolformer answered: "the model should be able to call tools."
MCP answers: "...and here is the standard plug for the tools."
```

### Reflexion (Shinn et al., 2023) - Paper 57

```
Reflexion-style self-correcting agents need a reliable loop
for executing actions and reading results. MCP gives that loop
a stable transport, so reflection logic doesn't have to reinvent
schema validation, error handling, and retries per tool.
```

### Claude 4 / Claude Code (Anthropic, 2025)

```
Claude 4 Opus and Sonnet are explicitly tuned for agent loops:
parallel tool use, long-horizon task execution, and faithful
adherence to tool schemas. Claude Code (the CLI agent shipped
alongside) uses MCP as its plugin system - every "skill",
"hook", and external service is an MCP server underneath.

MCP is the protocol that made Claude 4's agentic capabilities
useful in the wild rather than just impressive in a demo.
```

### Retrieval-Augmented Generation (Lewis et al., 2020) - Paper 13

```
RAG is the special case of MCP where the only primitive used
is RESOURCES, the only operation is "read", and the host
auto-injects results into context.

MCP generalizes RAG: instead of one retrieval pipeline hardcoded
into the app, ANY server can expose ANY resource URI scheme,
and the host can mix and match.
```

---

## Limitations and Open Problems

### 1. Discovery and Trust
```
There is no canonical registry yet. Users find servers via
GitHub, blog posts, and word of mouth. A signed registry with
permission manifests is an active area of standardization.
```

### 2. Authentication for Remote Servers
```
The 2024 launch spec assumed local stdio. Remote servers need
OAuth 2.1 - added in 2025 spec revisions but still maturing
in client implementations.
```

### 3. Prompt Injection via Resources
```
A malicious file content or web page returned through a server
can hijack the model. MCP cannot solve this at the protocol
level - it is a model-side and host-side problem. Hosts are
adding "untrusted content" markers and the spec encourages
servers to label resource provenance.
```

### 4. Tool Sprawl and Context Bloat
```
A user with 30 servers each exposing 20 tools puts 600 tool
schemas into every prompt. Hosts are responding with:
  - Tool selection / search before invocation
  - Hierarchical / lazy tool listing
  - Per-tool enable toggles in the UI
The spec leaves this as a host concern, which is correct
but means UX varies wildly across hosts.
```

### 5. Versioning
```
Servers and the protocol both version. Capability negotiation
helps, but breaking changes to a server's tool schemas can
silently break agent workflows. Best practice (semver-style
tool versioning, deprecation warnings) is still informal.
```

---

## Key Takeaways

1. **MCP is the USB-C moment for AI** - one connector replaces M x N integrations with M + N.
2. **Three primitives: tools, resources, prompts** - the minimal vocabulary that covers agent action, retrieval context, and user workflows.
3. **JSON-RPC over stdio / Streamable HTTP** - boring, debuggable, language-agnostic transport with no SDK lock-in.
4. **Host is the trust boundary** - the model never holds credentials, every side-effecting call is host-mediated and user-approvable.
5. **Universal adoption** - within 18 months MCP went from Anthropic announcement to Anthropic + OpenAI + Microsoft + Google + every major IDE, ending the "function calling protocol war" before it really started.
6. **Local-first agents finally work** - subprocess transport made desktop-class AI agents (Claude Desktop, Cursor, Windsurf, Cline, Claude Code) practical.

**Bottom line:** Function calling told the model how to ask for a tool. MCP defined the rest of the stack: how tools are hosted, discovered, approved, transported, and composed across hosts. By choosing a deliberately small surface area (three primitives, JSON-RPC, two-and-a-half transports) and shipping reference servers + reference hosts on day one, Anthropic produced an open protocol that competitors found cheaper to adopt than to fight. MCP is now the default integration substrate for the agent era - the "TCP/IP of AI tools."

---

## Further Reading

### Primary Sources
- **Announcement:** https://www.anthropic.com/news/model-context-protocol
- **Specification:** https://modelcontextprotocol.io
- **GitHub Org:** https://github.com/modelcontextprotocol

### SDKs
- **TypeScript:** https://github.com/modelcontextprotocol/typescript-sdk
- **Python:** https://github.com/modelcontextprotocol/python-sdk
- **Rust, Go, Java, C#, Swift, Kotlin** - community SDKs in the org

### Reference Servers
- **Servers repo:** https://github.com/modelcontextprotocol/servers
- **Inspector (debug tool):** https://github.com/modelcontextprotocol/inspector

### Key Adoption Milestones
- **Anthropic announcement:** November 25, 2024
- **OpenAI Agents SDK + ChatGPT:** March 2025
- **Microsoft Copilot Studio:** May 2025
- **Windows 11 system-level MCP:** 2025

### Related Work in This Repo
- **ReAct:** Paper 21
- **Toolformer:** Paper 24
- **RAG:** Paper 13
- **Reflexion:** Paper 57

---

**Published:** November 25, 2024 (Anthropic blog + open-source release)
**Impact:** 🔥🔥🔥🔥🔥 **CRITICAL** - The default integration protocol for AI agents
**Adoption:** Universal across major model vendors, IDEs, and agent frameworks
**Current Relevance:** Foundational - virtually every production agent stack in 2026 speaks MCP
**Legacy:** Made tool integration a configuration step rather than an engineering project, unlocking the multi-vendor agent ecosystem

**Modern Status (April 2026):** MCP is the de facto standard for AI tool integration. The 2025 spec revisions added Streamable HTTP transport, OAuth 2.1 for remote servers, and elicitation primitives for richer host-server interaction. Thousands of community servers exist; a signed server registry is in active development. Every major model vendor (Anthropic, OpenAI, Google, Microsoft, Meta) ships first-party MCP support. The protocol's success has prompted comparisons to TCP/IP and HTTP - boring, ubiquitous infrastructure that future work simply assumes.
