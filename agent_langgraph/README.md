# DataRobot Agent Templates: Agent agent_langgraph

The DataRobot agent template provides a starting point for building custom agents that can be deployed in DataRobot.
This template can be modified to support various frameworks, including CrewAI, LangGraph, Llama-Index, or
a generic base framework that can be customized to use any other agentic framework.

For additional information, examples, and documentation on developing an agent, please see the
[DataRobot Agent Templates](https://github.com/datarobot-community/datarobot-agent-templates).

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
provided by the SonarQube server. You should see that tool listed in the startup log under "SonarQube tools"
once the server is enabled and loaded.

The default workflow is streamlined for SonarQube automation while keeping an LLM in the loop:
`START → entry_node → checkout_node → sonarqube_autorun → sonarqube_orchestrator → END`.
After cloning the repository through the GitHub MCP tool, the `sonarqube_autorun` node immediately calls
`search_my_sonarqube_projects` and then `search_sonar_issues_in_projects` (when auto-approval is enabled) to
seed the graph state with discovered project keys and issue lists. The subsequent SonarQube orchestrator uses
the LLM plus MCP tools to apply fixes, commit, and open pull requests via GitHub MCP helpers on the suggested
working branch. Tool calls are auto-approved by default to ensure non-interactive runs do not stall; the agent
still announces and summarizes each call. Use `--require_approval` if you want to force manual approvals instead.
