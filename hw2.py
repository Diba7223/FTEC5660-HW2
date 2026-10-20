#!/usr/bin/env python3
"""FTEC5660 HW2 student starter: build an agent that verifies CVs via MCP."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import re
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------
# 作业允许在 `### YOUR CODE HERE` 上方新增 import。
# 下面三个是我们实现 Agent 需要用到的库。
# ---------------------------------------------------------------------
from langchain.agents import create_agent
from langchain_deepseek import ChatDeepSeek


MCP_URL = "https://ftec5660.ngrok.app/mcp"
MODEL_NAME = "deepseek-v4-flash"
THRESHOLD = 0.5


def load_env_file(path: Path = Path(".env")) -> None:
    """Load the simple KEY=VALUE entries used by this homework."""
    if not path.is_file():
        return
    import os

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def cv_files(folder: Path) -> list[Path]:
    """Return PDFs directly inside *folder*, sorted numerically (CV_2 before CV_10)."""

    def key(path: Path) -> tuple[int, str]:
        digits = "".join(ch for ch in path.stem if ch.isdigit())
        return (int(digits) if digits else math.inf, path.name)

    return sorted(
        (p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"),
        key=key,
    )


def cv_text(path: Path) -> str:
    """Convert one CV PDF to markdown text."""
    from markitdown import MarkItDown

    return MarkItDown(enable_plugins=False).convert(str(path)).text_content


async def load_mcp_tools() -> list[Any]:
    """Connect to the course MCP server and return its tools as LangChain tools."""
    from langchain_mcp_adapters.client import MultiServerMCPClient

    client = MultiServerMCPClient(
        {
            "social_graph": {
                "transport": "http",
                "url": MCP_URL,
                "headers": {"ngrok-skip-browser-warning": "true"},
            }
        }
    )
    return await client.get_tools()


# =====================================================================
# Task 1：以下常量、提示词与辅助函数都可以自由修改。
# =====================================================================

MAX_CONCURRENCY = 3      # 全班共用 MCP 服务器，同时最多处理 3 份 CV
MAX_RETRIES = 2          # 单份 CV 出错时的重试次数
PER_CV_TIMEOUT = 300     # 单份 CV 的超时秒数，防止整体卡死超过 30 分钟
DEFAULT_SCORE = 0.5      # 彻底失败时的兜底分数（<=0.5 会被判为"有出入"）


VERIFY_SYSTEM_PROMPT = """You are a CV verification agent for a KYC-style screening task.

You receive the text of ONE CV. Decide whether its claims agree with the
candidate's real profiles, which you may look up ONLY through the SocialGraph
tools (mock LinkedIn + Facebook). Never use outside knowledge or the web.

STEP 1 - Extract the CV's claims: full name, city, work experience
(company, title, seniority, start year, end year), education
(degree, school, field, graduation year), and skills.

STEP 2 - Find the RIGHT person. Many people share a name, so a name search alone
is not enough.
- Use search_linkedin_people (optionally with the `location` and `industry`
  filters). Inspect candidates with get_linkedin_profile.
- Confirm identity by matching EDUCATION and EMPLOYERS, not just the name.
- LinkedIn is the primary source. Facebook (search_facebook_users then
  get_facebook_profile) agrees with LinkedIn; use it to confirm when unsure.
- The CV's city or employer may itself be the lie, so if a filtered search
  returns nothing, retry WITHOUT that filter.
- Work efficiently: prefer the search results and one or two profile lookups;
  do not look up every candidate.

STEP 3 - Compare ONLY these fields: name, city, jobs (company, title, seniority,
start and end years), education (degree, school, field, graduation year), skills.
IGNORE job descriptions, the headline, and the hometown: they are never a source
of discrepancy.

A difference in WORDING ONLY is NOT a discrepancy:
- "Bachelor of Science" vs "BSc"; "Master of Business Administration" vs "MBA".
- "UI/UX Design" vs "UI/UX".
- CV title "Senior Engineer" when the profile says "Engineer" with seniority
  "senior" (the seniority matches).
- The CV lists FEWER skills than the profile (a subset is fine).

A REAL discrepancy (a false claim) IS one of:
- Inflated job title, or seniority that does not match the profile
  (e.g. CV "Senior Analyst" vs profile "Analyst" with seniority "junior").
- Shifted employment or graduation years (even one year off).
- Upgraded degree (e.g. CV "PhD" vs profile "MSc").
- A school or employer that is not on the profile.
- A wrong city.
- A skill on the CV that is NOT on the profile (an extra skill).

STEP 4 - Score in [0, 1]:
- 0.8-1.0  every checked field agrees (valid);
- 0.4-0.6  borderline / you could not confirm the identity;
- 0.0-0.2  at least one clear false claim.
Finish your reply with one short sentence of reasoning, then a final line in
exactly this format (and nothing after it):
FINAL SCORE: <a number between 0 and 1>
"""

# 用前后缀拼接，避免 CV 文本里出现花括号时 .format() 报错。
VERIFY_USER_PREFIX = "Verify the following CV and report your score.\n\n<CV>\n"
VERIFY_USER_SUFFIX = "\n</CV>\n"

_SCORE_LABEL_RE = re.compile(
    r"(?:final\s*score|score)\s*[:：=]\s*\**\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE
)
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _clamp01(value: Any) -> float:
    """把任意数值裁剪到 [0, 1]。"""
    return max(0.0, min(1.0, float(value)))


def _extract_score(value: Any) -> float | None:
    """从文本或数值里稳健地取出一个 [0,1] 分数，取不到就返回 None。"""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return _clamp01(value)
    text = str(getattr(value, "content", value))
    match = _SCORE_LABEL_RE.search(text)
    if match:
        return _clamp01(match.group(1))
    numbers = _NUM_RE.findall(text)
    if len(numbers) == 1:
        return _clamp01(numbers[0])
    if numbers:
        return _clamp01(numbers[-1])
    return None


def _score_from_result(result: Any) -> float | None:
    """从 agent 的返回结果里取出分数（优先看最后一条消息）。"""
    if isinstance(result, dict):
        messages = result.get("messages")
        if messages:
            return _extract_score(messages[-1])
    return _extract_score(result)


def build_agent(tools: list[Any]) -> Any:
    """Create and return your agent once.

    ``tools`` are the six SocialGraph MCP tools (Facebook + LinkedIn search and
    profile lookup), already wrapped as LangChain tools. You may add your own
    local tools as well.

    Suggested imports:
        from langchain_deepseek import ChatDeepSeek
        from langchain.agents import create_agent

    Use the DeepSeek model named by ``MODEL_NAME``. The API key is loaded
    from .env.
    """
    model = ChatDeepSeek(model=MODEL_NAME, temperature=0)
    return create_agent(model, tools, system_prompt=VERIFY_SYSTEM_PROMPT)


async def score_cvs(agent: Any, cvs: dict[str, str]) -> dict[str, float | None]:
    """Run your agent and return one reliability score per CV.

    ``cvs`` maps each file name to its text, e.g. ``{"CV_1.pdf": "...", ...}``.
    Return a float in [0, 1] for every file name: higher means the CV is more
    likely consistent with the candidate's LinkedIn/Facebook data. A score
    above 0.5 counts as "valid", 0.5 or below counts as "has discrepancy".

        {"CV_1.pdf": 0.9, "CV_4.pdf": 0.1, ...}

    Catch errors per CV (e.g. a failed API call) and still return a score for
    it: an exception here means no results.csv, which scores zero.

    MCP tools are async, so call your agent with ``await agent.ainvoke(...)``.
    You may verify CVs in parallel (e.g. ``asyncio.gather``), but keep at most
    about 3 CVs in flight (e.g. with ``asyncio.Semaphore(3)``): the MCP server is
    shared by the whole class.
    """
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

    async def verify_one(name: str, text: str) -> tuple[str, float]:
        """核验一份 CV，返回 (文件名, 分数)。任何异常都在这里兜住。"""
        async with semaphore:
            prompt = VERIFY_USER_PREFIX + text + VERIFY_USER_SUFFIX
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    result = await asyncio.wait_for(
                        agent.ainvoke({"messages": [{"role": "user", "content": prompt}]}),
                        timeout=PER_CV_TIMEOUT,
                    )
                    score = _score_from_result(result)
                    if score is not None:
                        return name, score
                    print(f"[warn] {name}: 第 {attempt} 次没解析出分数，重试")
                except Exception as exc:  # 网络 / API / 超时等
                    print(f"[warn] {name}: 第 {attempt} 次核验出错 ({type(exc).__name__}: {exc})")
            print(f"[warn] {name}: 多次失败，使用兜底分数 {DEFAULT_SCORE}")
            return name, DEFAULT_SCORE

    pairs = await asyncio.gather(*(verify_one(name, text) for name, text in cvs.items()))
    return dict(pairs)


# Everything below is provided runner/scoring code. No edits are needed.

_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def parse_score(value: Any) -> float | None:
    """Accept a float/int, or text containing exactly one number, in [0, 1]."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        score = float(value)
    else:
        text = str(getattr(value, "content", value))
        matches = _NUMBER_RE.findall(text)
        if len(matches) != 1:
            return None
        score = float(matches[0])
    if math.isnan(score) or not 0.0 <= score <= 1.0:
        return None
    return score


def read_ground_truth(folder: Path) -> dict[str, dict[str, Any]]:
    """Read labels (1 = valid CV, 0 = has discrepancy) and reasons from the test folder."""
    path = folder / "ground_truth.json"
    if not path.is_file():
        return {}
    return {
        name: entry if isinstance(entry, dict) else {"label": entry}
        for name, entry in json.loads(path.read_text(encoding="utf-8")).items()
    }


def correctness_text(score: float | None, expected: dict[str, Any] | None) -> str:
    """Return `correct`, or an expected/predicted mismatch explanation."""
    if score is None:
        return "incorrect: score is missing or not a number in [0, 1]"
    if expected is None:
        return "not graded: no ground truth for this CV"
    label = int(expected["label"])
    predicted = 1 if score > THRESHOLD else 0
    if predicted == label:
        return "correct"
    reason = f" ({expected['reason']})" if expected.get("reason") else ""
    return f"incorrect: expected {label}{reason}, predicted {predicted}"


def write_results(names: list[str], scores: dict[str, Any], truth: dict[str, dict[str, Any]]) -> tuple[Path, int]:
    """Write the required results.csv file and return how many CVs were correct."""
    output = Path("results.csv")
    correct = 0
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["cv", "score", "correctness"])
        for name in names:
            score = parse_score(scores.get(name))
            verdict = correctness_text(score, truth.get(name))
            correct += verdict == "correct"
            writer.writerow([name, "" if score is None else f"{score:.4f}", verdict])
    return output, correct


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FTEC5660 HW2 on CV PDFs")
    parser.add_argument(
        "--cv-folder",
        required=True,
        type=Path,
        help="folder containing CV PDF files",
    )
    return parser.parse_args()


async def run(folder: Path) -> int:
    paths = cv_files(folder)
    if not paths:
        raise SystemExit(f"no PDF files found in {folder}")

    load_env_file()
    cvs = {path.name: cv_text(path) for path in paths}
    tools = await load_mcp_tools()
    agent = build_agent(tools)
    scores = await score_cvs(agent, cvs)
    if not isinstance(scores, dict):
        raise TypeError("score_cvs() must return a dictionary")

    truth = read_ground_truth(folder)
    output, correct = write_results(list(cvs), scores, truth)
    summary = f" Accuracy: {correct}/{len(cvs)}." if truth else ""
    print(f"Processed {len(cvs)} CV(s). Wrote {output}.{summary}")
    return 0


def main() -> int:
    args = parse_args()
    if not args.cv_folder.is_dir():
        raise SystemExit(f"not a folder: {args.cv_folder}")
    return asyncio.run(run(args.cv_folder))


if __name__ == "__main__":
    raise SystemExit(main())
