"""命令行型 LLM provider 客户端"""

import asyncio
import os
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import List

from ..config_loader import LLMParams, LLMProviderConfig
from ..exceptions import LLMInvocationError
from .base import BaseLLMClient, looks_like_token_limit_error


class BaseCLIClient(BaseLLMClient):
    """命令行型 LLM 客户端基类"""

    default_command = ""
    temp_dir_name = "llm-cli-inputs"

    def __init__(self, config: LLMProviderConfig, app_root):
        super().__init__(config, app_root)
        self._resolved_command = self._resolve_command()

    async def invoke(self, prompt: str, params: LLMParams) -> str:
        """异步调用 CLI"""
        return await asyncio.to_thread(self._invoke_sync, prompt, params)

    def _normalize_executable(self, executable: str) -> str:
        """清理可执行路径外层多余引号"""
        normalized = executable.strip()
        while len(normalized) >= 2 and normalized[0] == normalized[-1] and normalized[0] in {"'", '"'}:
            normalized = normalized[1:-1].strip()
        return normalized

    def _resolve_command(self) -> List[str]:
        """解析可用的 CLI 可执行命令"""
        configured_command = self.config.command or self.default_command

        if any(separator in configured_command for separator in ("\\", "/")) or Path(configured_command).suffix:
            candidates = [configured_command]
        else:
            candidates = [
                f"{configured_command}.exe",
                f"{configured_command}.cmd",
                configured_command,
            ]

        for candidate in candidates:
            resolved = shutil.which(candidate)
            if resolved:
                return [self._normalize_executable(resolved)]

        configured_path = Path(configured_command).expanduser()
        if configured_path.exists():
            return [self._normalize_executable(str(configured_path))]

        raise LLMInvocationError(f"[{self.describe()}] 未找到可执行命令: {configured_command}")

    def _get_prompt_temp_dir(self) -> Path:
        """返回临时 prompt 文件目录"""
        prompt_temp_dir = self.app_root / ".codex-runtime" / self.temp_dir_name / self.provider
        prompt_temp_dir.mkdir(parents=True, exist_ok=True)
        return prompt_temp_dir

    def _create_prompt_file(self, prompt: str) -> Path:
        """将 prompt 写入临时文件"""
        prompt_path = self._get_prompt_temp_dir() / f"{uuid.uuid4()}.txt"
        prompt_path.write_text(prompt, encoding="utf-8")
        return prompt_path

    def _build_env(self) -> dict:
        """构建 CLI 运行环境变量"""
        env = os.environ.copy()
        env.update({key: str(value) for key, value in self.config.env.items()})
        return env

    def _build_command(self, prompt_file: Path, prompt: str) -> List[str]:
        raise NotImplementedError

    def _extract_response_from_stdout(self, stdout_text: str) -> str:
        """从标准输出中提取最终回复"""
        return stdout_text.strip()

    def _format_error(self, stdout_text: str, stderr_text: str) -> str:
        """整合 CLI 输出，便于记录错误信息"""
        detail_parts = []
        for part in (stderr_text.strip(), stdout_text.strip()):
            if part:
                detail_parts.append(part)
        return "\n".join(detail_parts) if detail_parts else f"{self.provider} CLI 执行失败，但未返回错误信息"

    def _invoke_sync(self, prompt: str, params: LLMParams) -> str:
        """同步执行 CLI 调用"""
        prompt_file: Path | None = None
        try:
            prompt_file = self._create_prompt_file(prompt)
            command = self._build_command(prompt_file, prompt)
            result = subprocess.run(
                args=command,
                cwd=str(self.app_root),
                env=self._build_env(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=params.timeout,
                check=False,
            )

            stdout_text = result.stdout.decode("utf-8", errors="ignore")
            stderr_text = result.stderr.decode("utf-8", errors="ignore")

            if result.returncode != 0:
                error_message = self._format_error(stdout_text, stderr_text)
                raise LLMInvocationError(
                    f"[{self.describe()}] LLM调用失败: {error_message}",
                    is_token_limit_error=looks_like_token_limit_error(error_message),
                )

            response_text = self._extract_response_from_stdout(stdout_text)
            if response_text:
                return response_text

            raise LLMInvocationError(f"[{self.describe()}] 标准输出为空，未返回任何内容")
        except subprocess.TimeoutExpired as error:
            raise LLMInvocationError(f"[{self.describe()}] CLI 调用超时（>{params.timeout}秒）", error)
        except FileNotFoundError as error:
            if getattr(error, "winerror", None) == 206:
                error_message = (
                    f"[{self.describe()}] CLI 启动失败: 命令行参数过长（WinError 206）。"
                    " 当前调用已使用临时文件传递主要 prompt，请检查命令路径与附加参数。"
                )
            else:
                error_message = f"[{self.describe()}] 未找到 CLI 可执行文件"
            raise LLMInvocationError(error_message, error)
        except LLMInvocationError:
            raise
        except Exception as error:
            error_message = str(error)
            raise LLMInvocationError(
                f"[{self.describe()}] CLI 调用失败: {error_message}",
                error,
                looks_like_token_limit_error(error_message),
            )
        finally:
            if prompt_file is not None:
                try:
                    prompt_file.unlink(missing_ok=True)
                except OSError:
                    pass

    def _build_cmd_pipeline_command(self, prompt_file: Path, cli_command: List[str]) -> List[str]:
        """构建 Windows cmd 管道命令"""
        prompt_file_arg = subprocess.list2cmdline([str(prompt_file)])
        cli_command_text = subprocess.list2cmdline(cli_command)
        pipeline_command = f"type {prompt_file_arg} | {cli_command_text}"
        cmd_executable = self._normalize_executable(os.environ.get("ComSpec") or "cmd.exe")
        return [cmd_executable, "/d", "/s", "/c", pipeline_command]


class CodexCLIClient(BaseCLIClient):
    """Codex CLI 客户端"""

    default_command = "codex"

    def _resolve_codex_home(self) -> Path:
        """优先复用已登录的 Codex HOME，避免丢失认证态"""
        configured_codex_home = self.config.options.get("codex_home")
        if configured_codex_home:
            codex_home = Path(str(configured_codex_home)).expanduser()
            if not codex_home.is_absolute():
                codex_home = self.app_root / codex_home
            return codex_home

        env_codex_home = os.environ.get("CODEX_HOME")
        if env_codex_home:
            return Path(env_codex_home).expanduser()

        default_codex_home = Path.home() / ".codex"
        if (default_codex_home / "auth.json").exists():
            return default_codex_home

        return self.app_root / ".codex-runtime"

    def _build_env(self) -> dict:
        env = super()._build_env()
        codex_home = self._resolve_codex_home()
        codex_home.mkdir(parents=True, exist_ok=True)
        env["CODEX_HOME"] = str(codex_home)
        return env

    def _build_cli_args(self) -> List[str]:
        """构建 codex exec 参数"""
        cli_args: List[str] = []

        use_default_args = self.config.options.get("use_default_args", True)
        if use_default_args:
            cli_args.extend([
                "exec",
                "--ephemeral",
                "--sandbox",
                "read-only",
                "--color",
                "never",
            ])

        if self.config.model and self.config.options.get("use_model_arg", True):
            cli_args.extend(["--model", self.config.model])

        cli_args.extend(self.config.args)

        if use_default_args:
            cli_args.append("-")
        return cli_args

    def _build_command(self, prompt_file: Path, prompt: str) -> List[str]:
        cli_command = [*self._resolved_command, *self._build_cli_args()]
        return self._build_cmd_pipeline_command(prompt_file, cli_command)

    def _extract_response_from_stdout(self, stdout_text: str) -> str:
        """尽量提取 Codex 的最终回复"""
        text = stdout_text.strip()
        if not text:
            return ""

        for marker in ("\nassistant\n", "\nfinal\n"):
            marker_index = text.rfind(marker)
            if marker_index != -1:
                return text[marker_index + len(marker):].strip()

        return text


class ClaudeCLIClient(BaseCLIClient):
    """Claude CLI 客户端"""

    default_command = "claude"

    def _build_base_args(self) -> List[str]:
        """构建 claude 基础参数"""
        cli_args = [*self._resolved_command, "-p"]

        if self.config.model and self.config.options.get("use_model_arg", True):
            cli_args.extend(["--model", self.config.model])

        cli_args.extend(self.config.args)
        return cli_args

    def _build_command(self, prompt_file: Path, prompt: str) -> List[str]:
        prompt_transport = str(self.config.options.get("prompt_transport", "stdin")).lower()
        base_args = self._build_base_args()

        if prompt_transport == "inline":
            return [*base_args, prompt]

        stdin_prompt = str(
            self.config.options.get(
                "stdin_prompt",
                "请根据标准输入中的完整请求执行，并直接输出最终答案。",
            )
        )
        cli_command = [*base_args, stdin_prompt]
        return self._build_cmd_pipeline_command(prompt_file, cli_command)
