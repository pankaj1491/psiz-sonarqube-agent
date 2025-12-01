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

The default workflow is streamlined for SonarQube automation while keeping an LLM in the loop:
`START → entry_node → checkout_node → sonarqube_orchestrator → END`.
After cloning the repository through the GitHub MCP tool, the SonarQube orchestrator uses an LLM configured
with MCP tools to plan and execute SonarQube remediation. The orchestrator’s system prompt instructs the LLM
to ask the user for approval **before every MCP tool call**, to use SonarQube tools to fetch and verify code
smells, to apply fixes with repository-editing tools, and to commit and open pull requests via GitHub MCP
tools on the suggested working branch.
