"""
Terminal Command Agent - Natural Language Terminal Interface
===========================================================

A comprehensive demonstration of the Delegant library that creates an intelligent
terminal agent capable of:

1. Converting natural language to terminal commands
2. Executing commands safely in devenv.sh containers
3. Storing command history via Atuin integration
4. Comprehensive logging of all stdio/stderr
5. Recursive help command tree building
6. Context-aware command suggestions

This serves as the MVP demonstration of Delegant's capabilities.
"""

import asyncio
import json
import re
import shlex
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import logging

# Delegant imports
from delegant import (
    Agent, 
    TerminalServer, 
    AtuinServer, 
    FileSystemServer,
    chain,
    configure
)

logger = logging.getLogger(__name__)


class NaturalLanguageProcessor:
    """Processes natural language input and converts to terminal commands."""
    
    def __init__(self):
        # Command pattern mappings
        self.command_patterns = {
            # File operations
            r"(?i)list.*files?.*in (.+)": "ls -la {0}",
            r"(?i)show.*contents?.*of (.+)": "cat {0}",
            r"(?i)find.*files?.*named (.+)": "find . -name '{0}'",
            r"(?i)find.*files?.*containing (.+)": "grep -r '{0}' .",
            r"(?i)create.*(?:file|document) (.+)": "touch {0}",
            r"(?i)delete.*(?:file|remove) (.+)": "rm {0}",
            r"(?i)copy (.+) to (.+)": "cp {0} {1}",
            r"(?i)move (.+) to (.+)": "mv {0} {1}",
            r"(?i)make.*director(?:y|ies) (.+)": "mkdir -p {0}",
            
            # System information
            r"(?i)show.*current.*director(?:y|ies)": "pwd",
            r"(?i)what.*director(?:y|ies).*am.*i.*in": "pwd",
            r"(?i)show.*disk.*usage": "df -h",
            r"(?i)show.*memory.*usage": "free -h",
            r"(?i)show.*processes": "ps aux",
            r"(?i)who.*am.*i": "whoami",
            r"(?i)show.*system.*info": "uname -a",
            
            # Git operations
            r"(?i)git.*status": "git status",
            r"(?i)git.*add.*all": "git add .",
            r"(?i)git.*add (.+)": "git add {0}",
            r"(?i)git.*commit.*message (.+)": "git commit -m '{0}'",
            r"(?i)git.*push": "git push",
            r"(?i)git.*pull": "git pull",
            r"(?i)git.*log": "git log --oneline -10",
            r"(?i)git.*diff": "git diff",
            r"(?i)git.*branch": "git branch",
            r"(?i)(?:create|make).*git.*branch (.+)": "git checkout -b {0}",
            r"(?i)switch.*to.*branch (.+)": "git checkout {0}",
            
            # Package management
            r"(?i)install.*package (.+)": "sudo apt-get install {0}",
            r"(?i)update.*packages": "sudo apt-get update",
            r"(?i)upgrade.*packages": "sudo apt-get upgrade",
            r"(?i)search.*package (.+)": "apt search {0}",
            
            # Network operations
            r"(?i)ping (.+)": "ping -c 4 {0}",
            r"(?i)download (.+)": "wget {0}",
            r"(?i)curl (.+)": "curl {0}",
            r"(?i)check.*port (.+)": "netstat -tuln | grep {0}",
            
            # Text processing
            r"(?i)count.*lines.*in (.+)": "wc -l {0}",
            r"(?i)count.*words.*in (.+)": "wc -w {0}",
            r"(?i)sort.*file (.+)": "sort {0}",
            r"(?i)unique.*lines.*in (.+)": "sort {0} | uniq",
            r"(?i)head.*of (.+)": "head {0}",
            r"(?i)tail.*of (.+)": "tail {0}",
            r"(?i)follow.*logs.*of (.+)": "tail -f {0}",
            
            # System control
            r"(?i)kill.*process (.+)": "pkill {0}",
            r"(?i)restart.*service (.+)": "sudo systemctl restart {0}",
            r"(?i)start.*service (.+)": "sudo systemctl start {0}",
            r"(?i)stop.*service (.+)": "sudo systemctl stop {0}",
            r"(?i)status.*of.*service (.+)": "systemctl status {0}",
            
            # Archive operations
            r"(?i)compress (.+)": "tar -czf {0}.tar.gz {0}",
            r"(?i)extract (.+)": "tar -xzf {0}",
            r"(?i)zip (.+)": "zip -r {0}.zip {0}",
            r"(?i)unzip (.+)": "unzip {0}",
            
            # Help and documentation
            r"(?i)help.*with (.+)": "man {0}",
            r"(?i)manual.*for (.+)": "man {0}",
            r"(?i)info.*about (.+)": "info {0}",
            r"(?i)which (.+)": "which {0}",
            r"(?i)where.*is (.+)": "whereis {0}",
        }
        
        # Common command aliases
        self.aliases = {
            "list": "ls",
            "show": "cat",
            "edit": "nano",
            "remove": "rm",
            "copy": "cp",
            "move": "mv",
            "link": "ln",
            "search": "grep",
            "find": "find",
            "locate": "locate",
        }
        
        # Safety patterns (commands that need confirmation)
        self.dangerous_patterns = [
            r"rm.*-rf",
            r"sudo.*rm",
            r"mkfs",
            r"dd.*if=",
            r":(){ :|:& };:",  # Fork bomb
            r"chmod.*777",
            r"chmod.*-R.*777",
        ]
    
    def parse_natural_language(self, text: str) -> List[Dict[str, Any]]:
        """Parse natural language input into potential terminal commands.
        
        Args:
            text: Natural language input
            
        Returns:
            List of potential command interpretations with confidence scores
        """
        text = text.strip()
        candidates = []
        
        # Try pattern matching
        for pattern, command_template in self.command_patterns.items():
            match = re.match(pattern, text)
            if match:
                try:
                    # Extract groups and format command
                    groups = match.groups()
                    command = command_template.format(*groups)
                    
                    candidates.append({
                        "command": command,
                        "confidence": 0.9,
                        "method": "pattern_match",
                        "pattern": pattern,
                        "explanation": f"Matched pattern: {pattern}"
                    })
                except (IndexError, KeyError):
                    # Pattern match failed, skip
                    continue
        
        # Try direct command detection
        if self._looks_like_command(text):
            candidates.append({
                "command": text,
                "confidence": 0.8,
                "method": "direct_command",
                "explanation": "Appears to be a direct command"
            })
        
        # Try alias expansion
        first_word = text.split()[0].lower() if text.split() else ""
        if first_word in self.aliases:
            expanded = text.replace(first_word, self.aliases[first_word], 1)
            candidates.append({
                "command": expanded,
                "confidence": 0.7,
                "method": "alias_expansion", 
                "explanation": f"Expanded alias '{first_word}' to '{self.aliases[first_word]}'"
            })
        
        # Sort by confidence
        candidates.sort(key=lambda x: x["confidence"], reverse=True)
        
        # Add safety analysis
        for candidate in candidates:
            candidate["is_safe"] = self._analyze_safety(candidate["command"])
            candidate["safety_warnings"] = self._get_safety_warnings(candidate["command"])
        
        return candidates
    
    def _looks_like_command(self, text: str) -> bool:
        """Determine if text looks like a direct terminal command."""
        # Check if it starts with a common command
        common_commands = [
            "ls", "cd", "pwd", "cat", "grep", "find", "ps", "top", "df", "du",
            "git", "npm", "pip", "docker", "kubectl", "ssh", "scp", "rsync",
            "vim", "nano", "emacs", "curl", "wget", "tar", "zip", "unzip"
        ]
        
        first_word = text.split()[0].lower() if text.split() else ""
        return first_word in common_commands
    
    def _analyze_safety(self, command: str) -> bool:
        """Analyze if a command is safe to execute."""
        for pattern in self.dangerous_patterns:
            if re.search(pattern, command, re.IGNORECASE):
                return False
        return True
    
    def _get_safety_warnings(self, command: str) -> List[str]:
        """Get safety warnings for a command."""
        warnings = []
        
        for pattern in self.dangerous_patterns:
            if re.search(pattern, command, re.IGNORECASE):
                warnings.append(f"Potentially dangerous pattern detected: {pattern}")
        
        # Additional safety checks
        if "sudo" in command:
            warnings.append("Command requires elevated privileges")
        
        if any(word in command for word in ["rm", "delete", "remove"]):
            warnings.append("Command may delete files or directories")
        
        if any(word in command for word in ["format", "mkfs", "dd"]):
            warnings.append("Command may modify or destroy data")
        
        return warnings
    
    def suggest_alternatives(self, text: str, failed_command: str) -> List[str]:
        """Suggest alternative commands when parsing fails or commands fail."""
        suggestions = []
        
        # Common typo corrections
        typo_corrections = {
            "lis": "ls",
            "cta": "cat", 
            "grpe": "grep",
            "finde": "find",
            "cehck": "check",
            "stauts": "status",
        }
        
        words = text.lower().split()
        for i, word in enumerate(words):
            if word in typo_corrections:
                corrected = words.copy()
                corrected[i] = typo_corrections[word]
                suggestions.append(" ".join(corrected))
        
        # Suggest help commands
        if "help" not in text.lower():
            potential_command = text.split()[0] if text.split() else ""
            if potential_command:
                suggestions.extend([
                    f"man {potential_command}",
                    f"{potential_command} --help",
                    f"which {potential_command}",
                    f"apropos {potential_command}"
                ])
        
        return suggestions


class HelpTreeBuilder:
    """Builds comprehensive help trees by recursively calling help commands."""
    
    def __init__(self, terminal_server: TerminalServer):
        self.terminal = terminal_server
        self.help_cache: Dict[str, Dict[str, Any]] = {}
    
    async def build_help_tree(self, command: str, max_depth: int = 3) -> Dict[str, Any]:
        """Build a comprehensive help tree for a command.
        
        Args:
            command: Base command to build help tree for
            max_depth: Maximum recursion depth
            
        Returns:
            Hierarchical help information
        """
        if command in self.help_cache:
            return self.help_cache[command]
        
        help_tree = {
            "command": command,
            "help_sources": {},
            "subcommands": {},
            "related_commands": [],
            "examples": [],
            "depth": 0
        }
        
        # Try different help sources
        help_commands = [
            f"man {command}",
            f"{command} --help",
            f"{command} -h", 
            f"info {command}",
            f"help {command}",
            f"which {command}",
            f"apropos {command}"
        ]
        
        for help_cmd in help_commands:
            try:
                result = await self.terminal.execute_command(
                    help_cmd, 
                    timeout=10,
                    capture_output=True
                )
                
                if result["success"] and result["stdout"].strip():
                    help_type = help_cmd.split()[0]
                    help_tree["help_sources"][help_type] = {
                        "command": help_cmd,
                        "output": result["stdout"][:2000],  # Limit size
                        "success": True
                    }
                    
                    # Extract subcommands and examples
                    if help_type in ["man", "help"]:
                        subcommands = self._extract_subcommands(result["stdout"])
                        help_tree["subcommands"].update(subcommands)
                        
                        examples = self._extract_examples(result["stdout"])
                        help_tree["examples"].extend(examples)
                        
            except Exception as e:
                logger.debug(f"Help command failed: {help_cmd} - {e}")
                continue
        
        # Recursively build help for subcommands (if depth allows)
        if max_depth > 0:
            for subcmd in list(help_tree["subcommands"].keys())[:5]:  # Limit to 5 subcommands
                try:
                    sub_tree = await self.build_help_tree(f"{command} {subcmd}", max_depth - 1)
                    sub_tree["depth"] = max_depth - 1
                    help_tree["subcommands"][subcmd] = sub_tree
                except Exception as e:
                    logger.debug(f"Failed to build subtree for {command} {subcmd}: {e}")
        
        # Find related commands
        help_tree["related_commands"] = await self._find_related_commands(command)
        
        # Cache result
        self.help_cache[command] = help_tree
        
        return help_tree
    
    def _extract_subcommands(self, help_text: str) -> Dict[str, str]:
        """Extract subcommands from help text."""
        subcommands = {}
        
        # Look for common subcommand patterns
        patterns = [
            r"^\s*(\w+)\s+(.+)$",  # command description
            r"^\s*(\w+):\s*(.+)$",  # command: description
            r"Available commands:\s*\n((?:\s*\w+.*\n)*)",  # Available commands section
        ]
        
        lines = help_text.split('\n')
        in_commands_section = False
        
        for line in lines:
            line = line.strip()
            
            # Check for command sections
            if any(keyword in line.lower() for keyword in ["commands", "subcommands", "operations"]):
                in_commands_section = True
                continue
            
            if in_commands_section and line and not line.startswith(' '):
                in_commands_section = False
            
            if in_commands_section or any(re.match(pattern, line) for pattern in patterns):
                # Try to extract command and description
                match = re.match(r'^\s*(\w+)\s+(.+)$', line)
                if match:
                    cmd, desc = match.groups()
                    if len(cmd) > 1 and len(desc) > 5:  # Basic quality filter
                        subcommands[cmd] = desc[:100]  # Limit description length
        
        return subcommands
    
    def _extract_examples(self, help_text: str) -> List[str]:
        """Extract usage examples from help text."""
        examples = []
        
        # Look for example sections
        lines = help_text.split('\n')
        in_examples = False
        
        for line in lines:
            line_lower = line.lower().strip()
            
            # Check for examples section
            if any(keyword in line_lower for keyword in ["example", "usage", "synopsis"]):
                in_examples = True
                continue
            
            if in_examples:
                # Stop at next section
                if line and not line.startswith(' ') and ':' in line:
                    in_examples = False
                    continue
                
                # Extract command-like lines
                stripped = line.strip()
                if stripped and (stripped.startswith('$') or stripped.startswith('#') or 
                               any(stripped.startswith(cmd) for cmd in ['ls', 'cd', 'git', 'cp', 'mv'])):
                    examples.append(stripped.lstrip('$# '))
        
        return examples[:10]  # Limit number of examples
    
    async def _find_related_commands(self, command: str) -> List[str]:
        """Find commands related to the given command."""
        related = []
        
        try:
            # Use apropos to find related commands
            result = await self.terminal.execute_command(
                f"apropos {command}",
                timeout=5,
                capture_output=True
            )
            
            if result["success"]:
                lines = result["stdout"].split('\n')
                for line in lines[:10]:  # Limit results
                    match = re.match(r'^(\w+(?:\.\d+)?)\s*\([^)]+\)\s*-', line)
                    if match:
                        related_cmd = match.group(1)
                        if related_cmd != command:
                            related.append(related_cmd)
        except Exception:
            pass
        
        return related


class CommandLogger:
    """Comprehensive logging of all command executions and outputs."""
    
    def __init__(self, log_directory: str = None):
        self.log_dir = Path(log_directory or Path.home() / ".delegant" / "terminal_logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create session-specific log file
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"terminal_session_{self.session_id}.jsonl"
        self.stdio_file = self.log_dir / f"stdio_{self.session_id}.log"
        
        # Initialize log files
        with open(self.log_file, 'w') as f:
            f.write(json.dumps({
                "event": "session_start",
                "session_id": self.session_id,
                "timestamp": datetime.now().isoformat()
            }) + '\n')
    
    async def log_command_execution(
        self,
        natural_language: str,
        parsed_commands: List[Dict[str, Any]],
        selected_command: str,
        execution_result: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> None:
        """Log complete command execution cycle."""
        log_entry = {
            "event": "command_execution",
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "natural_language": natural_language,
            "parsed_commands": parsed_commands,
            "selected_command": selected_command,
            "execution_result": {
                "success": execution_result.get("success"),
                "exit_code": execution_result.get("exit_code"),
                "execution_time": execution_result.get("execution_time"),
                "stdout_length": len(execution_result.get("stdout", "")),
                "stderr_length": len(execution_result.get("stderr", "")),
            },
            "context": context or {}
        }
        
        # Write to main log
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
        
        # Write stdio to separate file for easy analysis
        with open(self.stdio_file, 'a') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"Time: {datetime.now().isoformat()}\n")
            f.write(f"Input: {natural_language}\n")
            f.write(f"Command: {selected_command}\n")
            f.write(f"Exit Code: {execution_result.get('exit_code', 'N/A')}\n")
            f.write(f"{'='*80}\n")
            
            if execution_result.get("stdout"):
                f.write("STDOUT:\n")
                f.write(execution_result["stdout"])
                f.write("\n")
            
            if execution_result.get("stderr"):
                f.write("STDERR:\n")
                f.write(execution_result["stderr"])
                f.write("\n")
    
    async def log_help_tree(self, command: str, help_tree: Dict[str, Any]) -> None:
        """Log help tree building results."""
        log_entry = {
            "event": "help_tree_built",
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "command": command,
            "help_sources": list(help_tree.get("help_sources", {}).keys()),
            "subcommands_count": len(help_tree.get("subcommands", {})),
            "examples_count": len(help_tree.get("examples", [])),
            "related_commands_count": len(help_tree.get("related_commands", []))
        }
        
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
        
        # Save detailed help tree
        help_file = self.log_dir / f"help_tree_{command.replace(' ', '_')}_{self.session_id}.json"
        with open(help_file, 'w') as f:
            json.dump(help_tree, f, indent=2)
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of current session."""
        summary = {
            "session_id": self.session_id,
            "log_file": str(self.log_file),
            "stdio_file": str(self.stdio_file),
            "commands_executed": 0,
            "help_trees_built": 0,
            "total_stdout_size": 0,
            "total_stderr_size": 0,
            "successful_commands": 0,
            "failed_commands": 0
        }
        
        try:
            with open(self.log_file, 'r') as f:
                for line in f:
                    entry = json.loads(line)
                    if entry["event"] == "command_execution":
                        summary["commands_executed"] += 1
                        if entry["execution_result"]["success"]:
                            summary["successful_commands"] += 1
                        else:
                            summary["failed_commands"] += 1
                        summary["total_stdout_size"] += entry["execution_result"]["stdout_length"]
                        summary["total_stderr_size"] += entry["execution_result"]["stderr_length"]
                    elif entry["event"] == "help_tree_built":
                        summary["help_trees_built"] += 1
        except Exception:
            pass
        
        return summary


@chain(TerminalServer, AtuinServer, FileSystemServer)
class TerminalCommandAgent(Agent):
    """Intelligent terminal agent with natural language processing.
    
    This agent demonstrates the full capabilities of the Delegant library by providing
    a natural language interface to terminal operations with comprehensive logging,
    devenv.sh integration, and Atuin history management.
    """
    
    instruction: str = """
    I am an intelligent terminal command agent that converts natural language 
    instructions into safe terminal commands. I can execute commands in secure 
    devenv.sh containers, maintain comprehensive command history through Atuin, 
    and provide detailed help trees for any command.
    
    I prioritize safety and will warn about potentially dangerous operations
    while providing helpful suggestions and alternatives when commands fail.
    """
    
    # Server configurations will be set during initialization
    terminal: TerminalServer
    atuin: AtuinServer  
    files: FileSystemServer
    
    def __init__(self, **kwargs):
        # Configure servers with safe defaults
        self.configure_server("terminal", 
            shell="/bin/bash",
            allowed_commands=None,  # Allow all commands but with safety warnings
            log_all_commands=True,
            use_devenv=kwargs.get("use_devenv", False),
            devenv_config=kwargs.get("devenv_config"),
            timeout_seconds=300
        )
        
        self.configure_server("atuin",
            enable_sync=kwargs.get("enable_atuin_sync", False)
        )
        
        self.configure_server("files",
            root_dir=kwargs.get("working_directory", "/tmp/delegant_workspace"),
            readonly=False,
            enable_search=True
        )
        
        super().__init__(**kwargs)
        
        # Initialize components
        self.nlp = NaturalLanguageProcessor()
        self.help_builder = None  # Will be initialized after connection
        self.logger = CommandLogger(kwargs.get("log_directory"))
        self.session_history: List[Dict[str, Any]] = []
    
    async def __aenter__(self):
        """Initialize the agent and its components."""
        result = await super().__aenter__()
        
        # Initialize help builder after terminal connection
        self.help_builder = HelpTreeBuilder(self.terminal)
        
        # Create workspace directory
        workspace = Path(self.files.root_dir)
        workspace.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Terminal Command Agent initialized (session: {self.logger.session_id})")
        return result
    
    async def process_natural_language(
        self, 
        user_input: str,
        auto_execute: bool = False,
        build_help_tree: bool = False
    ) -> Dict[str, Any]:
        """Process natural language input and optionally execute commands.
        
        Args:
            user_input: Natural language command description
            auto_execute: Whether to automatically execute the best command
            build_help_tree: Whether to build comprehensive help tree
            
        Returns:
            Processing results including parsed commands and execution results
        """
        start_time = datetime.now()
        
        try:
            # Parse natural language
            parsed_commands = self.nlp.parse_natural_language(user_input)
            
            if not parsed_commands:
                return {
                    "success": False,
                    "error": "Could not parse natural language input",
                    "suggestions": self.nlp.suggest_alternatives(user_input, ""),
                    "user_input": user_input,
                    "processing_time": (datetime.now() - start_time).total_seconds()
                }
            
            result = {
                "success": True,
                "user_input": user_input,
                "parsed_commands": parsed_commands,
                "best_command": parsed_commands[0]["command"],
                "processing_time": (datetime.now() - start_time).total_seconds(),
                "execution_result": None,
                "help_tree": None,
                "atuin_imported": False
            }
            
            # Auto-execute if requested and command is safe
            if auto_execute and parsed_commands[0]["is_safe"]:
                execution_result = await self.execute_command(parsed_commands[0]["command"])
                result["execution_result"] = execution_result
                
                # Import to Atuin if successful
                if execution_result["success"]:
                    try:
                        await self.atuin.import_command(
                            command=parsed_commands[0]["command"],
                            directory=self.files.root_dir,
                            exit_code=execution_result["exit_code"],
                            duration=int(execution_result["execution_time"] * 1000)
                        )
                        result["atuin_imported"] = True
                    except Exception as e:
                        logger.warning(f"Failed to import command to Atuin: {e}")
            
            # Build help tree if requested
            if build_help_tree:
                base_command = parsed_commands[0]["command"].split()[0]
                try:
                    help_tree = await self.help_builder.build_help_tree(base_command)
                    result["help_tree"] = help_tree
                    await self.logger.log_help_tree(base_command, help_tree)
                except Exception as e:
                    logger.warning(f"Failed to build help tree: {e}")
            
            # Log the complete interaction
            await self.logger.log_command_execution(
                natural_language=user_input,
                parsed_commands=parsed_commands,
                selected_command=parsed_commands[0]["command"],
                execution_result=result.get("execution_result", {}),
                context={
                    "auto_execute": auto_execute,
                    "build_help_tree": build_help_tree,
                    "session_id": self.logger.session_id
                }
            )
            
            # Add to session history
            self.session_history.append(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing natural language: {e}")
            return {
                "success": False,
                "error": str(e),
                "user_input": user_input,
                "processing_time": (datetime.now() - start_time).total_seconds()
            }
    
    async def execute_command(self, command: str) -> Dict[str, Any]:
        """Execute a terminal command with comprehensive logging.
        
        Args:
            command: Terminal command to execute
            
        Returns:
            Execution result with full details
        """
        try:
            result = await self.terminal.execute_command(
                command=command,
                working_directory=self.files.root_dir,
                capture_output=True
            )
            
            logger.info(f"Executed command: {command} (exit code: {result['exit_code']})")
            return result
            
        except Exception as e:
            logger.error(f"Command execution failed: {command} - {e}")
            return {
                "success": False,
                "command": command,
                "exit_code": -1,
                "stdout": "",
                "stderr": str(e),
                "execution_time": 0.0,
                "error": str(e)
            }
    
    async def get_command_suggestions(self, partial_input: str) -> List[Dict[str, Any]]:
        """Get command suggestions based on partial input and history.
        
        Args:
            partial_input: Partial natural language input
            
        Returns:
            List of suggested commands with explanations
        """
        suggestions = []
        
        # Get basic NLP suggestions
        parsed = self.nlp.parse_natural_language(partial_input)
        for parse_result in parsed[:3]:  # Top 3 suggestions
            suggestions.append({
                "command": parse_result["command"],
                "confidence": parse_result["confidence"],
                "explanation": parse_result["explanation"],
                "source": "nlp_parser"
            })
        
        # Get suggestions from Atuin history
        try:
            history_results = await self.atuin.search_history(
                query=partial_input,
                limit=5
            )
            
            for hist in history_results:
                suggestions.append({
                    "command": hist["command"],
                    "confidence": 0.6,
                    "explanation": f"From history (used {hist.get('count', 1)} times)",
                    "source": "atuin_history",
                    "last_used": hist.get("formatted_timestamp")
                })
        except Exception as e:
            logger.debug(f"Failed to get Atuin suggestions: {e}")
        
        # Remove duplicates and sort by confidence
        seen_commands = set()
        unique_suggestions = []
        for suggestion in suggestions:
            if suggestion["command"] not in seen_commands:
                seen_commands.add(suggestion["command"])
                unique_suggestions.append(suggestion)
        
        unique_suggestions.sort(key=lambda x: x["confidence"], reverse=True)
        return unique_suggestions[:10]  # Top 10 suggestions
    
    async def analyze_command_history(self, period: str = "week") -> Dict[str, Any]:
        """Analyze command usage patterns from Atuin history.
        
        Args:
            period: Time period for analysis
            
        Returns:
            Analysis results with patterns and insights
        """
        try:
            # Get statistics from Atuin
            stats = await self.atuin.get_statistics(period=period, include_details=True)
            
            # Get top commands
            top_commands = await self.atuin.get_top_commands(limit=20, period=period)
            
            # Analyze patterns
            analysis = {
                "period": period,
                "basic_stats": stats["basic_stats"],
                "top_commands": top_commands,
                "patterns": {
                    "most_used_directories": stats.get("top_directories", [])[:10],
                    "common_failures": [],
                    "command_categories": self._categorize_commands(top_commands),
                    "efficiency_score": self._calculate_efficiency_score(stats["basic_stats"])
                },
                "recommendations": []
            }
            
            # Generate recommendations
            analysis["recommendations"] = self._generate_recommendations(analysis)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze command history: {e}")
            return {
                "error": str(e),
                "period": period
            }
    
    def _categorize_commands(self, top_commands: List[Dict[str, Any]]) -> Dict[str, int]:
        """Categorize commands by type."""
        categories = {
            "file_operations": 0,
            "git_operations": 0,
            "system_info": 0,
            "package_management": 0,
            "network": 0,
            "text_processing": 0,
            "development": 0,
            "other": 0
        }
        
        category_patterns = {
            "file_operations": ["ls", "cat", "cp", "mv", "rm", "mkdir", "find", "grep"],
            "git_operations": ["git"],
            "system_info": ["ps", "top", "df", "du", "free", "uname", "whoami"],
            "package_management": ["apt", "pip", "npm", "yarn", "brew"],
            "network": ["curl", "wget", "ping", "ssh", "scp"],
            "text_processing": ["sed", "awk", "sort", "uniq", "wc", "head", "tail"],
            "development": ["python", "node", "java", "gcc", "make", "docker", "kubectl"]
        }
        
        for cmd_data in top_commands:
            command = cmd_data["command"]
            categorized = False
            
            for category, patterns in category_patterns.items():
                if any(pattern in command.lower() for pattern in patterns):
                    categories[category] += cmd_data["count"]
                    categorized = True
                    break
            
            if not categorized:
                categories["other"] += cmd_data["count"]
        
        return categories
    
    def _calculate_efficiency_score(self, basic_stats: Dict[str, Any]) -> float:
        """Calculate efficiency score based on command patterns."""
        total = basic_stats.get("total_commands", 0)
        successful = basic_stats.get("successful_commands", 0)
        
        if total == 0:
            return 0.0
        
        success_rate = successful / total
        
        # Base score is success rate
        efficiency = success_rate * 100
        
        # Bonus for having good command diversity
        # (This would require more analysis in a real implementation)
        
        return round(efficiency, 1)
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on command analysis."""
        recommendations = []
        
        basic_stats = analysis["basic_stats"]
        success_rate = basic_stats.get("success_rate", 0)
        
        if success_rate < 80:
            recommendations.append(
                f"Consider using command validation - your success rate is {success_rate:.1f}%. "
                "Try using 'help <command>' or '<command> --help' before executing."
            )
        
        categories = analysis["patterns"]["command_categories"]
        total_commands = sum(categories.values())
        
        if total_commands > 0:
            file_ops_pct = (categories["file_operations"] / total_commands) * 100
            if file_ops_pct > 50:
                recommendations.append(
                    f"You use file operations commands {file_ops_pct:.1f}% of the time. "
                    "Consider learning advanced file manipulation tools like 'find', 'xargs', and 'awk'."
                )
            
            git_pct = (categories["git_operations"] / total_commands) * 100
            if git_pct > 20:
                recommendations.append(
                    f"You use git commands {git_pct:.1f}% of the time. "
                    "Consider setting up git aliases to speed up common operations."
                )
        
        return recommendations
    
    async def interactive_mode(self) -> None:
        """Run the agent in interactive mode for demonstration."""
        print(f"""
{'='*80}
Welcome to the Delegant Terminal Command Agent!
{'='*80}

This intelligent agent can:
• Convert natural language to terminal commands
• Execute commands safely in devenv.sh containers
• Build comprehensive help trees for any command
• Maintain detailed command history via Atuin
• Provide smart suggestions and safety warnings

Session ID: {self.logger.session_id}
Working Directory: {self.files.root_dir}
Log Directory: {self.logger.log_dir}

Type 'help' for commands, 'quit' to exit.
{'='*80}
        """)
        
        while True:
            try:
                user_input = input("\n🤖 What would you like me to do? ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    break
                
                if user_input.lower() == 'help':
                    self._show_help()
                    continue
                
                if user_input.lower() == 'history':
                    await self._show_history()
                    continue
                
                if user_input.lower() == 'analyze':
                    await self._show_analysis()
                    continue
                
                if user_input.lower().startswith('suggestions '):
                    partial = user_input[12:]
                    await self._show_suggestions(partial)
                    continue
                
                # Process the natural language input
                print(f"\n🔄 Processing: {user_input}")
                
                result = await self.process_natural_language(
                    user_input,
                    auto_execute=False,
                    build_help_tree=False
                )
                
                if not result["success"]:
                    print(f"❌ Error: {result['error']}")
                    if "suggestions" in result:
                        print("💡 Suggestions:")
                        for suggestion in result["suggestions"][:3]:
                            print(f"   • {suggestion}")
                    continue
                
                # Show parsed commands
                print(f"\n📝 Parsed Commands:")
                for i, cmd in enumerate(result["parsed_commands"][:3], 1):
                    safety = "✅ Safe" if cmd["is_safe"] else "⚠️  Unsafe"
                    print(f"   {i}. {cmd['command']} ({cmd['confidence']:.1%} confidence, {safety})")
                    if cmd["safety_warnings"]:
                        for warning in cmd["safety_warnings"]:
                            print(f"      ⚠️  {warning}")
                
                # Ask for execution confirmation
                best_cmd = result["best_command"]
                execute = input(f"\n❓ Execute '{best_cmd}'? [y/N/h for help tree]: ").strip().lower()
                
                if execute in ['y', 'yes']:
                    print(f"\n⚡ Executing: {best_cmd}")
                    exec_result = await self.execute_command(best_cmd)
                    
                    if exec_result["success"]:
                        print(f"✅ Command completed (exit code: {exec_result['exit_code']})")
                        if exec_result["stdout"]:
                            print(f"\n📤 Output:\n{exec_result['stdout']}")
                    else:
                        print(f"❌ Command failed (exit code: {exec_result['exit_code']})")
                        if exec_result["stderr"]:
                            print(f"\n🚨 Error output:\n{exec_result['stderr']}")
                    
                    # Import to Atuin
                    try:
                        await self.atuin.import_command(
                            command=best_cmd,
                            directory=self.files.root_dir,
                            exit_code=exec_result["exit_code"],
                            duration=int(exec_result["execution_time"] * 1000)
                        )
                        print("📚 Command saved to Atuin history")
                    except Exception as e:
                        print(f"⚠️  Could not save to Atuin: {e}")
                
                elif execute == 'h':
                    # Build help tree
                    base_cmd = best_cmd.split()[0]
                    print(f"\n🌳 Building help tree for '{base_cmd}'...")
                    
                    try:
                        help_tree = await self.help_builder.build_help_tree(base_cmd, max_depth=2)
                        self._display_help_tree(help_tree)
                    except Exception as e:
                        print(f"❌ Failed to build help tree: {e}")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Unexpected error: {e}")
                logger.exception("Interactive mode error")
        
        # Show session summary
        summary = self.logger.get_session_summary()
        print(f"""
{'='*80}
Session Summary
{'='*80}
Commands executed: {summary['commands_executed']}
Successful: {summary['successful_commands']}
Failed: {summary['failed_commands']}
Help trees built: {summary['help_trees_built']}

Logs saved to:
• Main log: {summary['log_file']}
• Stdio log: {summary['stdio_file']}
{'='*80}
        """)
    
    def _show_help(self) -> None:
        """Show help for interactive mode."""
        print("""
Available commands:
• help - Show this help message
• history - Show recent command history from Atuin
• analyze - Analyze command usage patterns
• suggestions <partial> - Get command suggestions
• quit/exit/bye - Exit the agent

For any other input, I'll try to convert it to terminal commands!

Examples:
• "list files in the current directory"
• "show me the contents of package.json"
• "find all Python files"
• "git status"
• "help with the ls command"
        """)
    
    async def _show_history(self) -> None:
        """Show recent command history."""
        try:
            recent = await self.atuin.get_recent_commands(limit=10)
            
            print(f"\n📚 Recent Commands:")
            for i, cmd in enumerate(recent, 1):
                timestamp = cmd.get("formatted_timestamp", "Unknown")
                exit_code = cmd.get("exit", "?")
                status = "✅" if exit_code == 0 else "❌"
                print(f"   {i:2d}. {status} {cmd['command']} ({timestamp})")
                
        except Exception as e:
            print(f"❌ Failed to get history: {e}")
    
    async def _show_analysis(self) -> None:
        """Show command usage analysis."""
        try:
            analysis = await self.analyze_command_history("week")
            
            if "error" in analysis:
                print(f"❌ Analysis failed: {analysis['error']}")
                return
            
            stats = analysis["basic_stats"]
            print(f"""
📊 Command Usage Analysis (Last Week)
{'='*50}
Total commands: {stats['total_commands']}
Success rate: {stats['success_rate']:.1f}%
Unique sessions: {stats['unique_sessions']}
Efficiency score: {analysis['patterns']['efficiency_score']}%

🏆 Top Commands:
            """)
            
            for i, cmd in enumerate(analysis["top_commands"][:5], 1):
                print(f"   {i}. {cmd['command']} ({cmd['count']} times)")
            
            if analysis["recommendations"]:
                print(f"\n💡 Recommendations:")
                for rec in analysis["recommendations"]:
                    print(f"   • {rec}")
            
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
    
    async def _show_suggestions(self, partial_input: str) -> None:
        """Show command suggestions for partial input."""
        try:
            suggestions = await self.get_command_suggestions(partial_input)
            
            print(f"\n💡 Suggestions for '{partial_input}':")
            for i, suggestion in enumerate(suggestions[:5], 1):
                confidence = suggestion["confidence"]
                source = suggestion["source"]
                print(f"   {i}. {suggestion['command']} ({confidence:.1%} from {source})")
                print(f"      {suggestion['explanation']}")
                
        except Exception as e:
            print(f"❌ Failed to get suggestions: {e}")
    
    def _display_help_tree(self, help_tree: Dict[str, Any]) -> None:
        """Display help tree in a readable format."""
        command = help_tree["command"]
        print(f"\n🌳 Help Tree for '{command}':")
        
        # Show available help sources
        sources = help_tree.get("help_sources", {})
        if sources:
            print(f"\n📖 Available help sources: {', '.join(sources.keys())}")
        
        # Show subcommands
        subcommands = help_tree.get("subcommands", {})
        if subcommands:
            print(f"\n🔧 Subcommands:")
            for subcmd, desc in list(subcommands.items())[:10]:
                print(f"   • {subcmd}: {desc}")
        
        # Show examples
        examples = help_tree.get("examples", [])
        if examples:
            print(f"\n💡 Examples:")
            for example in examples[:5]:
                print(f"   • {example}")
        
        # Show related commands
        related = help_tree.get("related_commands", [])
        if related:
            print(f"\n🔗 Related commands: {', '.join(related[:10])}")


# Main execution function
async def main():
    """Main function to run the Terminal Command Agent demo."""
    
    # Configure Delegant for the demo
    configure(
        max_retries=3,
        connection_timeout=30,
        debug_mode=True,
        context_extraction=True
    )
    
    print("🚀 Starting Delegant Terminal Command Agent Demo...")
    
    # Create the agent with demo configuration
    agent = TerminalCommandAgent(
        instruction="Intelligent terminal agent for natural language command processing",
        use_devenv=False,  # Set to True if devenv.sh is available
        enable_atuin_sync=False,  # Set to True if Atuin sync is configured
        working_directory="/tmp/delegant_demo",
        log_directory=None  # Uses default log directory
    )
    
    # Run the agent
    async with agent:
        # Demo some basic functionality first
        print("\n🧪 Running basic functionality tests...")
        
        # Test 1: Simple command parsing
        result1 = await agent.process_natural_language("list files in current directory")
        print(f"✅ Parsed: '{result1['user_input']}' → '{result1['best_command']}'")
        
        # Test 2: Command with help tree
        result2 = await agent.process_natural_language(
            "show git status", 
            build_help_tree=True
        )
        print(f"✅ Built help tree for git with {len(result2.get('help_tree', {}).get('subcommands', {}))} subcommands")
        
        # Test 3: Get suggestions
        suggestions = await agent.get_command_suggestions("find files")
        print(f"✅ Generated {len(suggestions)} suggestions for 'find files'")
        
        print("\n🎯 Basic tests completed! Starting interactive mode...\n")
        
        # Start interactive mode
        await agent.interactive_mode()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Demo terminated by user.")
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
