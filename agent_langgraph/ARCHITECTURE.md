# SonarQube Remediation Agent Architecture

This document describes how the LangGraph-based SonarQube remediation agent coordinates MCP servers, the MCP client, Poolside inline editing, and git/PR automation under the guidance of an LLM orchestrator.

## High-level view

- **LLM orchestrator:** A DataRobot-hosted LLM steers the workflow via LangGraph nodes, following the SonarQube system prompt and the tool hints derived from the active MCP servers.
- **LangGraph workflow:** Four nodes (`entry_node`, `checkout_node`, `sonarqube_autorun`, and `sonarqube_orchestrator`) share state through a `TypedDict` that tracks repository metadata, available tools, SonarQube results, and PR status.
- **MCP client + servers:** `custom_model/mcp_client.py` loads MCP servers from `mcp_servers.json` (GitHub, SonarQube, and others) and exposes their tools to the graph. The client filters tools into categories (SonarQube search, repo editing, git commit/PR, inline editing) so the orchestrator knows what capabilities are present for the current run.
- **Poolside inline editing:** When the `poolside_cli` tool is available, the checkout node retargets it to the freshly cloned repository and triggers a one-time indexing pass. The autorun node translates SonarQube issues into Poolside CLI calls to apply patches with full-file context.
- **Git/PR automation:** After Poolside patches are applied, the agent commits, pushes to a derived branch (`auto/<repo>`), and attempts a single pull request through the MCP GitHub helper. The orchestrator is reminded not to open additional PRs in the same session.

## Component responsibilities

| Layer | Responsibilities |
| --- | --- |
| LangGraph nodes | Enforce the deterministic flow (entry → checkout → autorun → orchestrator) and carry the shared `AgentState`. |
| LLM orchestrator | Provides reasoning and fallback fixes via `create_react_agent`, using tool hints and approval mode to decide whether to call MCP or Poolside tools. |
| MCP client | Builds connections to declared servers, instantiates LangChain tools, and categorizes them for SonarQube, repo edits, git commits, PRs, and inline edits. |
| MCP servers | Offer the concrete actions: cloning via GitHub MCP, SonarQube issue search, git commit/PR helpers, and any extra repo-editing utilities. |
| Poolside CLI | Performs repo-scoped inline edits. The agent rewrites `POOLSIDE_CWD` to the cloned repository and runs indexing before applying SonarQube patches. |
| Git/PR helpers | Use MCP tools to push changes and create a single PR; state tracks `pr_created` to avoid duplicates. |

## Local environment setup

1. **Install prerequisites**
   - Python 3.10–3.12 (tested with 3.11) and `uv` package manager.
   - Git, Taskfile, and a C++ toolchain (per the root README prerequisites).
2. **Sync dependencies**
   ```bash
   cd agent_langgraph
   uv sync
   ```
3. **Configure MCP servers**
   - Copy `custom_model/mcp_servers.json.example` to `custom_model/mcp_servers.json`.
   - Export the required secrets (for example `GITHUB_PERSONAL_ACCESS_TOKEN`, `SONAR_HOST_URL`, `SONAR_TOKEN`).
   - Set `MCP_SERVERS_CONFIG_PATH` if you keep the config outside the repo.
4. **Enable Poolside (optional but recommended)**
   - Place the Poolside tarball in the repo root or set `POOLSIDE_TARBALL`.
   - Ensure `POOLSIDE_TOKEN`/`POOLSIDE_API_URL` are available; the agent will seed `~/.config/poolside` automatically.
5. **Run the LangGraph CLI**
   ```bash
   cd agent_langgraph
   uv run python cli.py --require_approval false --repository_url <GIT_URL>
   ```
   The agent clones the target repo into `agent_langgraph/workspace`, applies Poolside fixes for `MAJOR` and `CRITICAL` SonarQube issues when auto-approval is on, then commits, pushes, and attempts a PR.

## Execution flow and state

```mermaid
flowchart TD
    entry[entry_node\nSeed AgentState] --> checkout[checkout_node\nClone repo + retarget Poolside]
    checkout --> autorun[sonarqube_autorun\nFetch and apply SonarQube issues]
    autorun --> orchestrator[sonarqube_orchestrator\nLLM-driven fallback + extra fixes]
    orchestrator --> end((END))
```

### State handoffs

- **`entry_node`:** Initializes `AgentState` keys (`repo_path`, `repository_url`, `working_branch`, `messages`, `available_tools`).
- **`checkout_node`:** Resolves the repository URL, clones to `workspace/<repo>`, retargets `poolside_cli`, runs indexing, and sets `working_branch` plus tool summary messages.
- **`sonarqube_autorun`:** Computes the project key (`CloudIQ:<repo>`), fetches SonarQube issues via MCP, applies `MAJOR`/`CRITICAL` fixes with Poolside, commits/pushes, and records `pr_created` along with the raw SonarQube response in state.
- **`sonarqube_orchestrator`:** Invokes the LLM-based ReAct loop with MCP and Poolside tools for any remaining work, respecting approval mode and the one-PR-per-run policy.

With this flow, the LLM orchestrator always has access to the current repository path, available tools, SonarQube findings, and whether a PR has already been attempted, enabling reliable, automated remediation.
