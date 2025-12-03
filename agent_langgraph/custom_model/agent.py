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
import logging
from typing import Any, Dict, List, Optional, TypedDict

import asyncio

import json
import os
import shutil
import subprocess
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, TypedDict, cast
from urllib.parse import urlparse
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool
from langchain_litellm.chat_models import ChatLiteLLM
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command
from pydantic import BaseModel, Field
from config import Config
from datarobot_genai.core.agents import make_system_prompt
from datarobot_genai.langgraph.agent import LangGraphAgent
from mcp_client import build_mcp_client, load_mcp_tools
from functools import cached_property

class PoolsideCLIArgs(BaseModel):
    """Inputs for invoking the Poolside CLI helper."""
    args: list[str] = Field(
        ..., description="Arguments passed to the poolside CLI (excluding the binary itself)."
    )
    working_dir: str | None = Field(
        None, description="Optional working directory; defaults to the repo path or current directory."
    )

class PoolsideCLITool(BaseTool):
    """Lightweight wrapper for a locally installed Poolside CLI."""
    name: str = "poolside_cli"
    description: str = (
        "Run the Poolside CLI for repo-aware code edits. Provide the exact argument list you would "
        "pass to the `poolside` binary; working_dir should be the repository root when applying patches."
    )
    args_schema: type[PoolsideCLIArgs] = PoolsideCLIArgs
    def __init__(self, command: str, default_cwd: str | None, env_overrides: dict[str, str]) -> None:
        super().__init__()
        self._command = command
        self._default_cwd = default_cwd
        self._env_overrides = env_overrides
    def retarget_working_dir(self, cwd: str) -> None:
        """Update the default working directory after checkout."""
        self._default_cwd = cwd
        # Always constrain Poolside to the checked-out repository so it doesn't index
        # or edit files from this automation repo. We also update POOLSIDE_CWD in both
        # the live environment and the tool's env overrides to keep subprocess calls
        # aligned with the latest checkout path.
        self._env_overrides["POOLSIDE_CWD"] = cwd
        os.environ["POOLSIDE_CWD"] = cwd
    def _run(
        self, args: list[str], working_dir: str | None = None, **_: Any
    ) -> dict[str, Any]:
        cwd = working_dir or self._default_cwd or os.getcwd()
        env = os.environ.copy()
        env.update(self._env_overrides)
        result = subprocess.run(
            [self._command, *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            env=env,
        )
        return {
            "exit_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "cwd": cwd,
            "command": [self._command, *args],
            "env_keys": sorted(self._env_overrides.keys()),
        }

class AgentState(TypedDict, total=False):
    messages: list[Any]
    repo_path: str | None
    repository_url: str | None
    working_branch: str | None
    pr_created: bool | None
    sonarqube_project_key: str | None
    sonarqube_result: Any
    available_tools: dict[str, list[str]]

class MyAgent(LangGraphAgent):
    """LangGraph agent that orchestrates SonarQube automation via MCP tools and an LLM."""
    config = Config()
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._prompt_template: ChatPromptTemplate | None = None
        # Handle configuration passed from CLI
        if "config" in kwargs:
            self.config = kwargs["config"]

        self.verbose_logging = getattr(self.config, "verbose_logging", False)
        self.verbose = self.verbose_logging
        print("Initializing MCP client...")
        mcp_config_path = os.environ.get("MCP_SERVERS_CONFIG_PATH")
        self._mcp_client = build_mcp_client(mcp_config_path)
        connection_names = ", ".join(sorted(self._mcp_client.connections.keys()))
        print(f"MCP client initialized with servers: {connection_names}")
        self._mcp_tools = self._load_mcp_tools()
        print(f"Loaded MCP tools: {len(self._mcp_tools)}")
        self._tool_capabilities = self._categorize_tools(self._mcp_tools)
        # ------------------------------------
        # 🔧 Enable autonomous execution mode
        # ------------------------------------
        if self.config.auto_approve_tools:
            self.allowed_tools = [tool.name for tool in self._mcp_tools]
            self.interrupt_before = None  # do NOT pause before tools
            print(f"🤖 Auto-exec ON — Allowed tools: {self.allowed_tools}")
        else:
            self.allowed_tools = []
            self.interrupt_before = ["tool"]  # require confirmation
            print("🧑‍💻 Manual mode — tool calls need user approval.")
        if not self._tool_capabilities.get("sonarqube"):
            print(
                "Warning: no SonarQube tools detected. Check that the 'sonarqube' MCP server is enabled in the config."
            )

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
        langgraph_workflow.add_node("sonarqube_autorun", self.sonarqube_autorun)
        langgraph_workflow.add_node("sonarqube_orchestrator", self.agent_sonarqube_orchestrator)
        langgraph_workflow.add_edge(START, "entry_node")
        langgraph_workflow.add_edge("entry_node", "checkout_node")
        langgraph_workflow.add_edge("checkout_node", "sonarqube_autorun")
        langgraph_workflow.add_edge("sonarqube_autorun", "sonarqube_orchestrator")
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
        self._retarget_poolside_tool(repo_path)
        self._index_repo_with_poolside(repo_path)
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
        if tool_hint["inline_edit"]:
            tool_summary_lines.append(
                "Inline editing tools: " + ", ".join(sorted(tool_hint["inline_edit"]))
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

    
    def _find_tool(self, name: str):
        name = name.lower().strip()

        for tool in self._mcp_tools:
            # check direct match
            if tool.name.lower().strip() == name:
                return tool
            
            # check suffix match (mcp prefixing)
            if tool.name.lower().endswith(name):
                return tool
            
            # check arg schema name fallback (common in MCP adapters)
            if hasattr(tool, "args_schema") and hasattr(tool.args_schema, "__name__"):
                if tool.args_schema.__name__.lower() == name:
                    return tool

        return None

    def _invoke_tool_sync(self, tool: BaseTool, args: dict) -> Any:
        """Invoke a LangChain tool that may only support async invocation.

        Tries tool.invoke(...) first; if the tool is async-only (e.g. MCP StructuredTool),
        falls back to running tool.ainvoke(...) in a fresh asyncio event loop.
        """
        try:
            return tool.invoke(args)
        except NotImplementedError as e:
            # MCP tools from mcp-langchain are often async-only and raise this
            if "does not support sync invocation" not in str(e):
                raise

            async def _call():
                return await tool.ainvoke(args)

            return asyncio.run(_call())

    def _repo_hint(self, state: AgentState) -> str | None:
        """Derive a repo identifier for matching SonarQube project keys."""
        if state.get("repo_path"):
            return Path(cast(str, state["repo_path"])).name.lower()
        if state.get("repository_url"):
            parsed = urlparse(cast(str, state["repository_url"]))
            if parsed.path:
                return Path(parsed.path).stem.lower()
        return None
    def _repo_name_for_sonar(self, state: AgentState) -> str | None:
        """Return the repository name used to derive the SonarQube project key."""
        if state.get("repo_path"):
            return Path(cast(str, state["repo_path"])).name
        if state.get("repository_url"):
            parsed = urlparse(cast(str, state["repository_url"]))
            if parsed.path:
                return Path(parsed.path).stem
        return None
    def _match_project_key(self, projects: Any, hint: str | None) -> str | None:
        """Best-effort match of SonarQube project key using repo hint."""
        if hint is None:
            return None
        candidates: list[Any] = []
        if isinstance(projects, list):
            candidates = projects
        if isinstance(projects, dict):
            for key in ["projects", "components", "results", "items"]:
                if isinstance(projects.get(key), list):
                    candidates = projects[key]
                    break
        hint = hint.lower()
        for item in candidates:
            if not isinstance(item, dict):
                continue
            key = item.get("key") or item.get("projectKey") or item.get("id")
            name = item.get("name") or item.get("project")
            if key and hint in str(key).lower():
                return str(key)
            if name and hint in str(name).lower():
                return str(key or name)
        return None

    def sonarqube_autorun(self, state: AgentState) -> Command[Any]:
        """Eagerly fetch SonarQube issues when auto-approval is enabled."""
        self.ensure_tools_loaded()
        messages = list(state.get("messages", []))
        if not self.config.auto_approve_tools:
            return Command(update={"messages": messages})
        issues_tool = self._find_tool("search_sonar_issues_in_projects")
        if not issues_tool:
            messages.append(
                SystemMessage(
                    content=(
                        "Auto-approval is on, but the SonarQube issue tool is missing. "
                        "The orchestrator will continue without the eager fetch step."
                    )
                )
            )
            return Command(update={"messages": messages})
        repo_name = self._repo_name_for_sonar(state)
        if not repo_name:
            messages.append(
                SystemMessage(
                    content=(
                        "Could not derive repository name to compute the SonarQube project key. "
                        "Proceeding without autorun issue fetch."
                    )
                )
            )
            return Command(update={"messages": messages})
        project_key = f"CloudIQ:{repo_name}"
        issues_result = self._invoke_tool_sync(
            issues_tool,
            {"projects": [project_key], "severities": ["INFO", "LOW", "MEDIUM", "HIGH", "BLOCKER"]}
        )

        # 🔥 Normalize response to dict (SonarQube MCP often returns JSON as text)
        if isinstance(issues_result, str):
            try:
                issues_result = json.loads(issues_result)
            except json.JSONDecodeError:
                self._log(f"❌ Unexpected non-JSON tool output:\n{issues_result}")
                return []

        messages.append(
            SystemMessage(
                content=(
                    "Auto SonarQube issues fetch using computed project key "
                    f"'{project_key}': {issues_result}."
                )
            )
        )
        repo_path = cast(str | None, state.get("repo_path"))
        working_branch = cast(str | None, state.get("working_branch"))
        repository_url = cast(str | None, state.get("repository_url"))
        issues_json = json.loads(issues_result) if isinstance(issues_result, str) else issues_result
        raw_issues = issues_json.get("issues", [])

        # filter only MAJOR and CRITICAL
        issues = [
            i for i in raw_issues
            if i.get("severity", "").upper() in {"MAJOR", "CRITICAL"}
        ]

        print(
            f"SonarQube autorun: {len(issues_result)} issue(s) returned, "
            f"{len(issues)} selected for Poolside patching."
        )
        pr_created = False
        if repo_path and working_branch and issues:
            poolside_applied = self._apply_sonar_issues_with_poolside(
                repo_path=repo_path, issues=issues
            )
            if poolside_applied:
                messages.append(
                    SystemMessage(
                        content=(
                            "Applied SonarQube issues via poolside_cli "
                            f"for {len(poolside_applied)} issue(s): {poolside_applied}."
                        )
                    )
                )
                commit_status = self._git_commit_push_pr(
                    repo_path=repo_path,
                    branch=working_branch,
                    repository_url=repository_url,
                    issue_count=len(poolside_applied),
                )
                pr_created = bool(commit_status.get("pull_request"))
                messages.append(
                    SystemMessage(
                        content=(
                            "Git/PR automation summary after Poolside fixes: "
                            f"{json.dumps(commit_status)}"
                        )
                    )
                )
                messages.append(
                    SystemMessage(
                        content=(
                            "PR policy: only one PR per end-to-end run. "
                            "A PR has already been attempted; do not open another."
                        )
                    )
                )
        return Command(
            update={
                "messages": messages,
                "sonarqube_project_key": project_key,
                "sonarqube_result": issues_result,
                "pr_created": pr_created,
                "pending_fix_work": True
            }
        )

    def post_model_hook(self, state: AgentState):
        """
        Post-model hook for the SonarQube react agent.

        The DataRobot LLM returns a JSON blob like:
          { "tool": "<name>", "args": { ... } }

        This hook:
        - looks at the last AI message
        - if the content is such a JSON object with "tool" and "args"
        - finds that tool by name (from self._sonar_tools_by_name)
        - executes it immediately for its side effects.

        For now it does NOT modify graph state; it just runs the tool.
        """
        from langchain_core.messages import AIMessage

        try:
            messages = state.get("messages") or []
            # Find the last AI message from the model
            last_ai = next(
                m for m in reversed(messages) if isinstance(m, AIMessage)
            )

            raw = (last_ai.content or "").strip()
            # We only handle the simple "one JSON object" pattern
            if not (raw.startswith("{") and raw.endswith("}")):
                # Not a JSON tool call – nothing to do
                return {}

            call = json.loads(raw)
            tool_name = call.get("tool")
            args = call.get("args", {})

            if not tool_name:
                print("[post_model_hook] JSON has no 'tool' field, ignoring.")
                return {}

            tools_by_name = getattr(self, "_sonar_tools_by_name", {})
            tool = tools_by_name.get(tool_name)
            if not tool:
                print(f"[post_model_hook] Tool '{tool_name}' not found, ignoring.")
                return {}

            print(f"[post_model_hook] Executing tool '{tool_name}' with args: {args!r}")
            # Run the LangChain tool synchronously
            result = self._invoke_tool_sync(tool, args)
            print(f"[post_model_hook] Tool '{tool_name}' completed with result: {result!r}")

            # We are only interested in side-effects (poolside, Sonar, etc.)
            # so we do not alter the LangGraph state here.
            return {}

        except StopIteration:
            # No AIMessage yet – nothing to do
            return {}
        except Exception as e:
            print(f"[post_model_hook] ERROR: {e}")
            return {}

    def agent_sonarqube_orchestrator(self, state: AgentState) -> Any:
        """LLM-backed LangGraph node that orchestrates SonarQube fixes via MCP tools."""

        self.ensure_tools_loaded() 
        # ---- Debug instrumentation (helps confirm issue #2 is solved) ----
        print(f"🔧 Tools loaded before run: {len(self._mcp_tools)}")
        print(f"📌 Sonar tool available? {self._find_tool('search_sonar_issues_in_projects') is not None}")
        print(f"📌 Allowed tool execution mode: {'AUTO' if self.config.auto_approve_tools else 'MANUAL'}")

        return create_react_agent(
            self.llm(preferred_model="datarobot/datarobot-deployed-llm"),
            tools=self._tools_with_approval_instructions(),
            interrupt_before=self.interrupt_before,
            prompt=make_system_prompt(self._sonarqube_system_prompt),
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
        # Check if MCP GitHub tool is available
        github_tools = [
            tool
            for tool in self._mcp_tools
            if any(keyword in getattr(tool, "name", "").lower() for keyword in ["github", "fork", "clone", "repo"])
        ]
        if not github_tools:
            # Fallback to git clone with authentication
            try:
                # Get GitHub credentials from MCP config
                github_server_config = self._mcp_client.connections.get("github", {})
                token = github_server_config.get("env", {}).get("GITHUB_PERSONAL_ACCESS_TOKEN")
                username = github_server_config.get("env", {}).get("GITHUB_USERNAME")
                
                if token:
                    # Construct authenticated URL
                    repo_parts = repository_url.split("://")
                    if len(repo_parts) == 2:
                        if username:
                            # Use username:token format for authentication
                            repo_url_with_token = f"https://{username}:{token}@{repo_parts[1]}"
                        else:
                            # Fallback to token-only format
                            repo_url_with_token = f"https://{token}@{repo_parts[1]}"
                    else:
                        repo_url_with_token = repository_url
                else:
                    repo_url_with_token = repository_url
                    
                print(f"Cloning repository with URL: {repo_url_with_token}")
                print(f"Target directory: {target_dir}")
                
                # Check if target directory already exists and remove it if it does
                if target_dir.exists():
                    print(f"Removing existing directory: {target_dir}")
                    import shutil
                    shutil.rmtree(target_dir)
                
                # Run git clone with detailed error output
                result = subprocess.run([
                    "git", "clone", repo_url_with_token, str(target_dir)
                ], capture_output=True, text=True)
                
                if result.returncode != 0:
                    print(f"Git clone failed with return code: {result.returncode}")
                    print(f"Stdout: {result.stdout}")
                    print(f"Stderr: {result.stderr}")
                    raise subprocess.CalledProcessError(result.returncode, result.args, result.stdout, result.stderr)
                else:
                    print(f"Git clone succeeded")
            except Exception as e:
                print(f"Git clone failed: {e}")
                raise
        else:
            # Use the first available GitHub tool
            github_tool = github_tools[0]
            tool_name = getattr(github_tool, "name", "")
            print(f"Using GitHub tool: {tool_name}")
            
            if hasattr(github_tool, "invoke"):
                self._invoke_tool_sync(
                    github_tool,
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
    def _retarget_poolside_tool(self, repo_path: str) -> None:
        tool = next((t for t in self._mcp_tools if isinstance(t, PoolsideCLITool)), None)
        if not tool:
            return
        tool.retarget_working_dir(repo_path)
        os.environ["POOLSIDE_CWD"] = repo_path
        print(f"Poolside CLI working directory set to {repo_path}")
    def _index_repo_with_poolside(self, repo_path: str) -> None:
        if not self.config.auto_approve_tools:
            return
        tool = next((t for t in self._mcp_tools if isinstance(t, PoolsideCLITool)), None)
        if not tool:
            return
        repo_root = str(Path(repo_path).resolve())
        args = [
            "--unsafe-auto-allow",
            "-p",
            "Index this repository for context before applying SonarQube fixes",
            "--",
            ".",
        ]
        result = self._invoke_tool_sync(
            tool, {"args": args, "working_dir": repo_root}
        )
        print(
            "Poolside indexing run",
            json.dumps(
                {
                    "command": result.get("command"),
                    "exit_code": result.get("exit_code"),
                    "cwd": result.get("cwd"),
                },
                indent=2,
            ),
        )
    def _extract_sonar_issues(self, issues_result: Any) -> list[dict[str, Any]]:
        """
        Extract a list of SonarQube issues from MCP response.
        SonarQube returns format:
        
        {
        "issues": [...],
        "paging": {...}
        }
        """
        if not issues_result:
            return []

        # FIRST priority: SonarQube's real format
        if isinstance(issues_result, dict) and "issues" in issues_result:
            issues = issues_result.get("issues")
            return issues if isinstance(issues, list) else []

        # SECOND: fallback pattern for other MCP tools
        for key in ["results", "items", "components"]:
            maybe = issues_result.get(key) if isinstance(issues_result, dict) else None
            if isinstance(maybe, list):
                return [i for i in maybe if isinstance(i, dict)]

        # LAST: direct list response fallback
        if isinstance(issues_result, list):
            return [i for i in issues_result if isinstance(i, dict)]

        return []

    def _apply_sonar_issues_with_poolside(self, repo_path: str, issues: list[dict[str, Any]]):
        """Apply SonarQube fixes using Poolside CLI."""

        if not self.config.auto_approve_tools:
            print("🛑 Auto approval disabled — skipping Poolside.")
            return []

        # FIX: match tool by name instead of class
        tool = next((t for t in self._mcp_tools if getattr(t, "name", "") == "poolside_cli"), None)

        if not tool:
            print("⚠ Poolside CLI tool NOT found — cannot apply fixes.")
            return []

        repo_root = str(Path(repo_path).resolve())
        applied = []

        for issue in issues:
            component = issue.get("component", "")
            relative_path = component.split(":", 1)[1].replace("psiz-acp-advisor-plugin:", "") if ":" in component else component


            prompt = (
                f"Fix this SonarQube issue:\n"
                f"- Rule: {issue.get('rule')}\n"
                f"- Message: {issue.get('message')}\n"
                f"- File: {relative_path}\n\n"
                f"Make minimal changes to resolve the issue cleanly."
            )

            args = ["-p", prompt, "--", relative_path,"--unsafe-auto-allow", "--non-interactive"]

            print(f"🔧 Running Poolside: {args}")  # NEW DEBUG

            result = self._invoke_tool_sync(
                tool, {"args": args, "working_dir": repo_root}
            )

            stdout = result.get("stdout", "").strip() if isinstance(result, dict) else str(result)
            stderr = result.get("stderr", "").strip() if isinstance(result, dict) else ""

            print("\n🛠️ Pool CLI Execution Result:")
            print(f"Exit Code: {result.get('exit_code', '?')}")
            if stdout:
                print(f"📤 STDOUT:\n{stdout}")
            if stderr:
                print(f"⚠️ STDERR:\n{stderr}")
            print("────────────────────────────────────────\n")

            applied.append({
                "issue": issue.get("key"),
                "file": relative_path,
                "exit_code": result.get("exit_code"),
                "stdout": result.get("stdout"),
                "stderr": result.get("stderr"),
            })

        return applied



    def _git_commit_push_pr(
        self,
        repo_path: str,
        branch: str,
        repository_url: str | None,
        issue_count: int,
    ) -> dict[str, Any]:
        """Commit and push changes only if diffs exist, then create PR."""
        summary: dict[str, Any] = {}
        def _run(cmd: list[str]) -> dict[str, Any]:
            result = subprocess.run(cmd, cwd=repo_path, capture_output=True, text=True)
            return {
                "cmd": cmd,
                "code": result.returncode,
                "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip(),
            }
        # Check if there are any changes first
        changes_check = _run(["git", "status", "--porcelain"])
        summary["pre_check"] = changes_check
        if not changes_check["stdout"]:
            summary["skipped"] = "No file changes detected — skipping commit, push, and PR creation."
            return summary
        # Continue with normal Git flow
        checkout = _run(["git", "checkout", "-B", branch])
        add_res = _run(["git", "add", "-A"])
        commit_res = _run(["git", "commit", "-m", f"Fix {issue_count} SonarQube issue(s)"])
        push_res = _run(["git", "push", "-u", "origin", branch])
        summary.update({
            "checkout": checkout,
            "add": add_res,
            "commit": commit_res,
            "push": push_res,
        })
        # Create PR only if commit succeeded
        if commit_res["code"] == 0 and repository_url:
            pr_tool = self._find_tool("create_pull_request")
            if pr_tool:
                pr_args = {
                    "repository": repository_url,
                    "title": f"Fix SonarQube issues ({branch})",
                    "body": "Automated remediation using Poolside CLI and SonarQube guidance.",
                    "head": branch,
                    "base": "main",
                }
                try:
                    pr_result = pr_tool.invoke(pr_args)
                    summary["pull_request"] = pr_result
                except Exception as exc:
                    summary["pull_request_error"] = str(exc)
        return summary

    def _build_poolside_cli_tool(self) -> BaseTool | None:
        """Register Poolside CLI tool if installed system-wide."""
        # Detect Pool CLI in system PATH (/usr/local/bin/pool in your case)
        pool_bin = shutil.which("pool")
        if not pool_bin:
            print("⚠ Poolside CLI not found — skipping registration.")
            return None
        print(f"✔ Poolside CLI detected at: {pool_bin}")
        class Args(BaseModel):
            args: list[str]
            working_dir: str | None = None
        class PoolTool(BaseTool):
            name: str = "poolside_cli"
            description: str = (
                "Run Poolside CLI to automatically apply patches based on SonarQube feedback. "
                "Uses repository context. Works best with --unsafe-auto-allow."
            )
            args_schema: type[Args] = Args
            def _run(self, args, working_dir=None, **_):
                cwd = working_dir or os.getcwd()
                cwd = str(Path(cwd).resolve())
                env = os.environ.copy()
                env["POOLSIDE_CWD"] = cwd
                # ⬅ If auto approve is ON and user didn't include it, enforce it
                if os.getenv("AUTO_APPROVE_TOOLS", "false").lower() == "true":
                    if "--unsafe-auto-allow" not in args:
                        args = ["--unsafe-auto-allow"] + args
                result = subprocess.run(
                    [pool_bin] + args,
                    cwd=cwd,
                    capture_output=True,
                    text=True,
                    env=env
                )
                return {
                    "exit_code": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "cwd": cwd,
                    "ran": [pool_bin] + args
                }
        return PoolTool()

    def _load_mcp_tools(self) -> list[BaseTool]:
        try:
            print(f"Loading MCP tools with client: {self._mcp_client}")
            tools = load_mcp_tools(client=self._mcp_client)
            print(f"Loaded {len(tools)} tools")
            github_tools = []
            clone_tools = []
            all_tool_names = []
            for i, tool in enumerate(tools):
                name = getattr(tool, "name", "").lower()
                all_tool_names.append(name)
                if "github" in name:
                    github_tools.append((i, name))
                if "clone" in name:
                    clone_tools.append((i, name))
                if i < 10:  # Print first 10 tools for debugging
                    print(f"Tool {i}: {name}")
            print(f"Found {len(github_tools)} GitHub-related tools")
            print(f"Found {len(clone_tools)} clone-related tools")
            print(f"All tool names: {all_tool_names}")
            poolside_tool = self._build_poolside_cli_tool()
            if poolside_tool:
                tools.append(poolside_tool)
                print("Registered local Poolside CLI tool")
            return tools
        except FileNotFoundError as exc:
            msg = (
                "MCP server configuration is missing. "
                "Provide MCP_SERVERS_CONFIG_PATH or place mcp_servers.json next to mcp_client.py "
                "(see mcp_servers.json.example)."
            )
            raise RuntimeError(msg) from exc
        except Exception as e:
            print(f"Error loading MCP tools: {e}")
            raise
    def ensure_tools_loaded(self) -> list[BaseTool]:
        """
        Ensure MCP tools are loaded before an LLM or REACT step executes.
        Prevents empty tool registry during first execution cycle.
        """

        # Case 1: Tools have never been loaded
        if not hasattr(self, "_mcp_tools") or self._mcp_tools is None:
            print("⚠️ Tools missing — loading for the first time...")
            self._mcp_tools = self._load_mcp_tools()
            return self._mcp_tools

        # Case 2: Tools list exists but is empty (race condition after startup)
        if isinstance(self._mcp_tools, list) and len(self._mcp_tools) == 0:
            print("⚠️ Tools list was empty — reloading MCP tools...")
            self._mcp_tools = self._load_mcp_tools()
            return self._mcp_tools

        # Case 3: Tools are loaded already — nothing to do
        return self._mcp_tools

    def _categorize_tools(self, tools: list[BaseTool]) -> dict[str, list[str]]:
        """Group MCP tools by capability hints for the orchestrator prompt."""
        capabilities = {
            "sonarqube": [],
            "repo_edit": [],
            "git_commit": [],
            "git_pr": [],
            "inline_edit": [],
        }
        for tool in tools:
            name = getattr(tool, "name", "").lower()
            if "poolside" in name:
                capabilities["inline_edit"].append(tool.name)
                capabilities["repo_edit"].append(tool.name)
                capabilities.setdefault("fix_issues", []).append(tool.name)
                continue
            if "sonar" in name:
                capabilities["sonarqube"].append(tool.name)
            if any(key in name for key in ["write", "file", "patch", "apply"]):
                capabilities["repo_edit"].append(tool.name)
            if "commit" in name:
                capabilities["git_commit"].append(tool.name)
            if "pull" in name or "pr" in name:
                capabilities["git_pr"].append(tool.name)
        return capabilities

    def _log(self, message: str) -> None:
        """Emit debug logs only when verbose logging is enabled."""
        if self.verbose_logging:
            print(message)

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
            self._log(f"Using model: {model}")
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

    @cached_property
    def prompt_template(self) -> ChatPromptTemplate:  # type: ignore[override]
        """
        Required override of the LangGraphAgent abstract property.
        Defines the base conversational template for the orchestrator.
        Uses cached_property so the template is only built once and satisfies the abstract contract.
        """
        return self._build_prompt_template()

    def _build_prompt_template(self) -> ChatPromptTemplate:
        approval_line = (
            "Auto-approval is enabled. Execute tools without asking."
            if self.config.auto_approve_tools
            else "Ask for confirmation before executing tools."
        )
        return ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You are an automated SonarQube remediation agent. "
                        "Your job is to detect issues, apply fixes using Poolside CLI, "
                        "validate using SonarQube, commit changes, and create a pull request.\n"
                        f"{approval_line}\n"
                        "Always be explicit about each step you take."
                    )
                ),
                (
                    "human",
                    "{input}"
                ),
            ]
        )

    @property
    def _sonarqube_system_prompt(self) -> str:
        if self.config.auto_approve_tools:
            approval_guidance = (
                "Auto-approval is enabled. Invoke MCP tools without asking for confirmation. "
                "Before each tool call, briefly announce the intent in one short sentence. "
                "NEVER ask 'Approve?'."
            )
        else:
            approval_guidance = (
                "Approval required. Before invoking a tool, describe the exact action and wait "
                "for a 'yes' confirmation from the user. Do NOT execute without explicit approval."
            )

        return f"""
    You are an AUTONOMOUS SonarQube Remediation Agent.

    Your goals:
    1. Detect SonarQube issues
    2. Fix them using Poolside CLI
    3. Validate fixes by re-running SonarQube checks
    4. Commit changes and create a Pull Request

    {approval_guidance}

    ### EXECUTION RULES

    - You MUST respond **only** using JSON tool calls.
    - NEVER output plain text, analysis, or reasoning.
    - NEVER explain what you are doing.
    - NEVER output anything outside the required JSON format shown below.
    - Every response after a tool output MUST trigger another tool call until the workflow is complete.
    - If no more actions are needed, call `create_pull_request`.

    ### REQUIRED JSON FORMAT

    Each action MUST be formatted like:

    {{
    "tool": "<tool_name>",
    "args": {{
        ... tool parameters ...
    }}
    }}

    ### WORKFLOW:

    1. Compute the SonarQube project key as: **CloudIQ:<repository_name>**
    - Example: CloudIQ:psiz-acp-advisor-plugin
    - Never call `search_my_sonarqube_projects`.

    2. Ensure the repository is indexed by calling:
    - `poolside_cli` with working_dir = repo root and command "index".

    3. Fetch SonarQube issues using:
    - `search_sonar_issues_in_projects`
    - With severities: ["INFO", "LOW", "MEDIUM", "HIGH", "BLOCKER"] (in this priority order)

    4. For each issue found:
    - Call `poolside_cli` providing:
        - working_dir
        - rule, component path, start/end line
        - Command instructing Poolside to generate/apply a patch

    5. After each fix round:
    - Re-run `search_sonar_issues_in_projects`
    - Stop only when **zero issues remain** or retries are exhausted.

    6. Once fixes complete:
    - Commit changes using relevant git MCP tools.
    - Then create a PR ***exactly once***.

    ### TERMINATION RULE:

    - When no more fixes are required, the LAST response must be a `create_pull_request` tool call.

    FOLLOW THE FORMAT STRICTLY.
    DO NOT CHAT.
    ONLY RETURN JSON TOOL CALLS.
    """


    def _tools_with_approval_instructions(self) -> list[BaseTool]:
        """Return MCP tools with names/descriptions annotated for approval etiquette."""
        if self.config.auto_approve_tools:
            return self.mcp_tools
        annotated_tools: list[BaseTool] = []
        for tool in self.mcp_tools:
            tool.description = (
                (tool.description or "")
                + "\n(Require user approval: describe call, ask Approve?, wait for yes.)"
            )
            annotated_tools.append(tool)
        return annotated_tools