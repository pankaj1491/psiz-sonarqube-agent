# Copyright 2025 DataRobot, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import subprocess
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, TypedDict, cast

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool
from langchain_litellm.chat_models import ChatLiteLLM
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command

from config import Config
from datarobot_genai.core.agents import make_system_prompt
from datarobot_genai.langgraph.agent import LangGraphAgent

from mcp_client import build_mcp_client, load_mcp_tools


class AgentState(TypedDict, total=False):
    messages: list[Any]
    repo_path: str | None
    repository_url: str | None
    working_branch: str | None
    sonarqube_project_key: str | None
    sonarqube_result: Any
    available_tools: dict[str, list[str]]


class MyAgent(LangGraphAgent):
    """LangGraph agent that orchestrates SonarQube automation via MCP tools and an LLM."""

    config = Config()

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._mcp_client = build_mcp_client()
        self._mcp_tools = self._load_mcp_tools()
        self._tool_capabilities = self._categorize_tools(self._mcp_tools)

    @property
    def mcp_client(self):  # type: ignore[override]
        return self._mcp_client

    @property
    def mcp_tools(self) -> list[BaseTool]:  # type: ignore[override]
        return self._mcp_tools

    @property
    def workflow(self) -> StateGraph[AgentState]:
        langgraph_workflow = StateGraph[
            AgentState, None, AgentState, AgentState
        ](AgentState)
        langgraph_workflow.add_node("entry_node", self.entry_node)
        langgraph_workflow.add_node("checkout_node", self.checkout_node)
        langgraph_workflow.add_node("sonarqube_orchestrator", self.agent_sonarqube_orchestrator)
        langgraph_workflow.add_edge(START, "entry_node")
        langgraph_workflow.add_edge("entry_node", "checkout_node")
        langgraph_workflow.add_edge("checkout_node", "sonarqube_orchestrator")
        langgraph_workflow.add_edge("sonarqube_orchestrator", END)
        return langgraph_workflow  # type: ignore[return-value]

    def entry_node(self, state: AgentState) -> AgentState:
        """Normalize the incoming state before invoking MCP tools."""

        state.setdefault("repo_path", None)
        state.setdefault("repository_url", None)
        state.setdefault("working_branch", None)
        state.setdefault("messages", [])
        state.setdefault("available_tools", self._tool_capabilities)
        return state

    def checkout_node(self, state: AgentState) -> Command[Any]:
        """Clone the target repository using the GitHub MCP tool."""

        repository_url = self._resolve_repository_url(state)
        repo_path = self._clone_repository(repository_url)
        working_branch = self._derive_working_branch(state, repo_path)

        messages = list(state.get("messages", []))
        tool_hint = self._tool_capabilities
        tool_summary_lines = []
        if tool_hint["sonarqube"]:
            tool_summary_lines.append(
                "SonarQube tools: " + ", ".join(sorted(tool_hint["sonarqube"]))
            )
        if tool_hint["repo_edit"]:
            tool_summary_lines.append(
                "Repo editing tools: " + ", ".join(sorted(tool_hint["repo_edit"]))
            )
        if tool_hint["git_commit"]:
            tool_summary_lines.append(
                "Commit tools: " + ", ".join(sorted(tool_hint["git_commit"]))
            )
        if tool_hint["git_pr"]:
            tool_summary_lines.append(
                "PR tools: " + ", ".join(sorted(tool_hint["git_pr"]))
            )

        tool_summary = "; ".join(tool_summary_lines) if tool_summary_lines else "No tool hints"

        messages.append(
            SystemMessage(
                content=(
                    "Repository prepared for SonarQube remediation. "
                    f"Repo: {repository_url} cloned to {repo_path}. "
                    f"Working branch suggestion: {working_branch}. "
                    f"Available tooling -> {tool_summary}."
                )
            )
        )

        return Command(
            update={
                "repo_path": repo_path,
                "repository_url": repository_url,
                "working_branch": working_branch,
                "messages": messages,
                "available_tools": tool_hint,
            }
        )

    def agent_sonarqube_orchestrator(self) -> Any:
        """LLM-backed LangGraph node that orchestrates SonarQube fixes via MCP tools."""

        return create_react_agent(
            self.llm(preferred_model="datarobot/datarobot-deployed-llm"),
            tools=self._tools_with_approval_instructions(),
            prompt=make_system_prompt(self._sonarqube_system_prompt),
            post_model_hook=partial(self._add_the_last_message_and_go_to_next_node, END),
        )

    def _resolve_repository_url(self, state: AgentState) -> str:
        repository_url = state.get("repository_url") or os.environ.get("REPOSITORY_URL")
        if repository_url:
            return cast(str, repository_url)

        try:
            repository_url = (
                subprocess.check_output(
                    ["git", "config", "--get", "remote.origin.url"],
                    text=True,
                    stderr=subprocess.DEVNULL,
                )
                .strip()
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            repository_url = None

        if not repository_url:
            raise ValueError(
                "A repository URL is required for checkout but could not be determined."
            )

        return repository_url

    def _clone_repository(self, repository_url: str) -> str:
        workspace_dir = Path.cwd() / "workspace"
        workspace_dir.mkdir(parents=True, exist_ok=True)

        target_dir = workspace_dir / self._repo_directory_name(repository_url)

        github_tools = [
            tool
            for tool in self.mcp_tools
            if "github" in getattr(tool, "name", "").lower()
            and "clone" in getattr(tool, "name", "").lower()
        ]

        if not github_tools:
            raise RuntimeError("No GitHub MCP clone tool is available for checkout.")

        github_tool = github_tools[0]
        if hasattr(github_tool, "invoke"):
            github_tool.invoke(
                {
                    "repo_url": repository_url,
                    "target_dir": str(target_dir),
                }
            )
        elif callable(github_tool):
            github_tool(
                {
                    "repo_url": repository_url,
                    "target_dir": str(target_dir),
                }
            )
        else:
            raise RuntimeError("The GitHub MCP tool cannot be invoked.")

        return str(target_dir.resolve())

    def _derive_working_branch(self, state: AgentState, repo_path: str) -> str:
        issue_identifier = state.get("issue_id") or state.get("issue_key")
        if issue_identifier:
            slug = str(issue_identifier).replace(" ", "-").lower()
        else:
            slug = Path(repo_path).name.lower()
        return f"auto/{slug}"

    def _repo_directory_name(self, repository_url: str) -> str:
        repo_name = repository_url.rstrip("/").split("/")[-1]
        return repo_name.removesuffix(".git")

    def _load_mcp_tools(self) -> list[BaseTool]:
        try:
            return load_mcp_tools(client=self._mcp_client)
        except FileNotFoundError as exc:
            msg = (
                "MCP server configuration is missing. "
                "Provide MCP_SERVERS_CONFIG_PATH or place mcp_servers.json next to mcp_client.py "
                "(see mcp_servers.json.example)."
            )
            raise RuntimeError(msg) from exc

    def _categorize_tools(self, tools: list[BaseTool]) -> dict[str, list[str]]:
        """Group MCP tools by capability hints for the orchestrator prompt."""

        capabilities = {
            "sonarqube": [],
            "repo_edit": [],
            "git_commit": [],
            "git_pr": [],
        }

        for tool in tools:
            name = getattr(tool, "name", "").lower()
            if "sonar" in name:
                capabilities["sonarqube"].append(tool.name)
            if any(key in name for key in ["write", "file", "patch", "apply"]):
                capabilities["repo_edit"].append(tool.name)
            if "commit" in name:
                capabilities["git_commit"].append(tool.name)
            if "pull" in name or "pr" in name:
                capabilities["git_pr"].append(tool.name)

        return capabilities

    def llm(
        self,
        preferred_model: str | None = None,
        auto_model_override: bool = True,
    ) -> ChatLiteLLM:
        """Returns the ChatLiteLLM to use for a given model."""

        api_base = self.litellm_api_base(self.config.llm_deployment_id)
        model = preferred_model
        if preferred_model is None:
            model = self.config.llm_default_model
        if auto_model_override and not self.config.use_datarobot_llm_gateway:
            model = self.config.llm_default_model
        if self.verbose:
            print(f"Using model: {model}")
        return ChatLiteLLM(
            model="datarobot/datarobot-deployed-llm",
            api_base=api_base,
            api_key=self.api_key,
            timeout=self.timeout,
            streaming=True,
            max_retries=3,
        )

    def _add_the_last_message_and_go_to_next_node(
        self, node_name: str, result: AgentState
    ) -> Command[Any]:
        last_msg = result["messages"][-1]
        tool_calls = getattr(last_msg, "tool_calls", [])
        if tool_calls:
            return Command()

        result["messages"][-1] = HumanMessage(
            content=result["messages"][-1].content, name=node_name
        )

        checkout_summary = None
        repo_path = result.get("repo_path")
        working_branch = result.get("working_branch")
        if repo_path:
            checkout_summary = SystemMessage(
                content=(
                    f"Repository cloned to {repo_path}. "
                    f"Use branch '{working_branch or 'auto/<repo>'}' for commits and PRs."
                )
            )

        messages = result["messages"]
        if checkout_summary:
            messages.append(checkout_summary)

        return Command(
            update={
                "messages": messages,
            },
        )

    @property
    def prompt_template(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages(
            [
                (
                    "user",
                    f"The topic is {{topic}}. Make sure you find any interesting and relevant information given the current year is {datetime.now().year}.",
                ),
            ]
        )

    @property
    def _sonarqube_system_prompt(self) -> str:
        return (
            "You are an AI release engineer fixing SonarQube issues via MCP tools. "
            "Always ask the user for explicit approval before invoking any MCP tool. "
            "When you plan to call a tool, first summarize the exact command, inputs, and effect, "
            "ask 'Approve? (yes/no)' and wait for the user to respond with 'yes' before executing. "
            "Do not auto-approve or guess approvals."
            "\n"
            "Workflow guidance:\n"
            "1) Use the repo path from shared state to run SonarQube scans and apply fixes.\n"
            "2) Prefer the GitHub MCP clone tool output path and derived working branch for changes.\n"
            "3) Use SonarQube MCP tools (see available tool names in the latest system message) to fetch code smells, suggest remediation, and re-check after changes.\n"
            "4) Use repository-editing tools (write/apply/patch) to apply fixes only after user approval.\n"
            "5) After fixes, use GitHub MCP tools to commit changes on the suggested working branch and open a pull request when approved.\n"
            "6) Keep the user informed after each tool run with concise summaries, including what changed and any remaining issues.\n"
            "7) Keep conversations focused on SonarQube remediation and PR creation."
        )

    def _tools_with_approval_instructions(self) -> list[BaseTool]:
        """Return MCP tools with names/descriptions annotated for approval etiquette."""

        annotated_tools: list[BaseTool] = []
        for tool in self.mcp_tools:
            tool.description = (
                (tool.description or "")
                + "\n(Require user approval: describe call, ask Approve?, wait for yes.)"
            )
            annotated_tools.append(tool)
        return annotated_tools
