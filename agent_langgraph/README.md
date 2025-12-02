# SonarQube remediation agent (LangGraph)

This LangGraph-based agent automates SonarQube remediation for CloudIQ repositories. It clones the target repo,
derives the SonarQube project key (`CloudIQ:<repo_name>`), fetches issues via MCP, applies fixes with inline tools
such as Poolside, and raises a pull request on the suggested working branch.

The repository still follows the DataRobot agent template conventions; see the
[DataRobot Agent Templates](https://github.com/datarobot-community/datarobot-agent-templates) reference for the
underlying deployment model.

## MCP configuration

LangGraph agents now read MCP server definitions from `agent_langgraph/custom_model/mcp_servers.json`
(see the adjacent `.example` file for structure). Set `MCP_SERVERS_CONFIG_PATH` if you want to point at
an alternate location. Enabled servers are exposed to the LangGraph nodes as LangChain tools, so the
GitHub MCP clone tool can be invoked from the checkout node automatically.

The example configuration now includes GitHub and SonarQube servers; copy it to `mcp_servers.json`, set
`GITHUB_PERSONAL_ACCESS_TOKEN`, `SONAR_HOST_URL`, and `SONAR_TOKEN` in your environment, and the agent will
load both servers at startup. The orchestrator detects tool capabilities (SonarQube, repo editing, commit,
and PR helpers) and surfaces them to the LLM as system context.

For SonarQube issue enumeration, the orchestrator relies on the `search_sonar_issues_in_projects` MCP tool
provided by the SonarQube server. The project key is derived from the repository name using the rule
`CloudIQ:<REPOSITORY_NAME>` (for example, `CloudIQ:psiz-acp-advisor-plugin`), so no project discovery call is
needed.

### End-to-end orchestration

- **Approval model**: auto-approval is enabled by default so unattended CLI runs never stall. Pass
  `--require_approval` to revert to interactive confirmation.
- **Project key rule**: the SonarQube project key is always computed as `CloudIQ:<REPO_NAME>`; the agent does **not**
  call project-discovery tools.
- **Issue retrieval**: uses the SonarQube MCP tool `search_sonar_issues_in_projects` with the computed key.
- **Fixes**: when auto-approval is on, `sonarqube_autorun` immediately fetches issues, calls `poolside_cli` for each
  issue, and then runs git commit/push + a single PR creation attempt; the orchestrator also prefers Poolside for
  inline edits when available.
- **Git/PR**: commits land on the derived working branch (for example `auto/psiz-acp-advisor-plugin`) followed by one
  pull request via GitHub MCP helpers; subsequent nodes are instructed not to open additional PRs in the same run.

Mermaid view of the flow:

```mermaid
flowchart TD
    A[task agent:cli] --> B[Poolside bootstrap from tarball]
    B --> C[Load MCP servers & derive repo name]
    C --> D[checkout_node clones repository]
    D --> E[Poolside retarget + context indexing]
    E --> F[sonarqube_autorun derives CloudIQ key, fetches issues, applies Poolside patches]
    F --> G[Commit + Push + one-time PR automation]
    G --> H[sonarqube_orchestrator (fallback/interactive fixes, no extra PRs)]
```

## Poolside CLI integration for inline fixes

If you have the Poolside inline coding assistant installed locally, the agent can call it directly as a LangGraph
tool to apply SonarQube fixes with full-repo context. Set `POOLSIDE_CLI_CMD` if the binary is not on `PATH`
(defaults to `pool`, falling back to `poolside`) and optionally `POOLSIDE_CWD` to force a working directory. Any
environment variables that start with `POOLSIDE_` (for example `POOLSIDE_HOST`, `POOLSIDE_TOKEN`, `POOLSIDE_API_URL`,
or a custom `POOLSIDE_CLI_CMD`) are passed through to the subprocess so your locally configured Poolside instance is
used. When both `POOLSIDE_API_URL` and `POOLSIDE_TOKEN` are set, the agent will write a `~/.config/poolside/credentials.json`
entry automatically if one is not present.

When the binary is detected, an extra tool named `poolside_cli` is registered and exposed to the orchestrator; it
executes the command with the argument list you provide and defaults the working directory to the repository root
after checkout. The SonarQube system prompt instructs the LLM to run a Poolside indexing command from the repo root,
then pass each Sonar issue (rule, message, component, line range) as a JSON-formatted tool call so Poolside can
generate and apply patches automatically.

### Installing the Poolside CLI for local runs

The install and CLI tasks run the provided tarball installer (expects `pool-v0.2.105-linux-amd64.tar.gz` in the repo
root) to guarantee Poolside is present before execution:

1. Installs the `pool` binary to `/usr/local/bin/pool` if missing.
2. Seeds `~/.config/poolside/credentials.json` and `~/.config/poolside/settings.yaml` with the default API
   endpoint/token (`http://poolside.rr2caitest01.amer.dell.com`).
3. Verifies the binary with `pool --version` and runs a hello prompt.

Override the tarball path with `POOLSIDE_TARBALL` if needed; the installer runs from `task agent_langgraph:install`
and `task agent_langgraph:cli` so every CLI invocation boots Poolside automatically.

### How the checkout is indexed with Poolside

After the repository is cloned, the checkout node retargets the Poolside tool to the repo root and—when
auto-approval is on—runs a one-time indexing prompt to give Poolside full context. This happens before any SonarQube
issues are processed and uses the repo root as `working_dir` so subsequent fixes operate on the cloned files:

```python
# agent_langgraph/custom_model/agent.py (checkout_node)
self._retarget_poolside_tool(repo_path)
self._index_repo_with_poolside(repo_path)

def _index_repo_with_poolside(self, repo_path: str) -> None:
    if not self.config.auto_approve_tools:
        return
    tool = next((t for t in self._mcp_tools if isinstance(t, PoolsideCLITool)), None)
    if not tool:
        return
    args = ["--unsafe-auto-allow", "-p", "Index this repository for context before applying SonarQube fixes", "--", "."]
    tool.invoke({"args": args, "working_dir": repo_path})
```

With this flow, the Poolside agent indexes the freshly cloned repository in `agent_langgraph/custom_model/workspace` and
has full-file context when applying patches generated from SonarQube issue details.

### How SonarQube issues are auto-applied (auto-approval path)

When auto-approval is enabled, the `sonarqube_autorun` node applies the issues returned by
`search_sonar_issues_in_projects` before the orchestrator runs. Each issue is translated into a `poolside_cli` call with
the relative file path and line range, then the agent performs git commit, push, and a single PR creation attempt (when
the GitHub MCP tool is available):

```python
issues = self._extract_sonar_issues(issues_result)
poolside_applied = self._apply_sonar_issues_with_poolside(repo_path, issues)
commit_status = self._git_commit_push_pr(
    repo_path=repo_path,
    branch=working_branch,
    repository_url=repository_url,
    issue_count=len(poolside_applied),
)
```

This ensures Poolside receives the SonarQube issue context (rule, message, component, and line numbers), applies
patches from the checkout root, and then the agent pushes the branch and attempts a PR automatically.
