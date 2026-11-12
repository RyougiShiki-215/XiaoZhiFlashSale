"""Skill 注册表

三层渐进式加载策略：
  1. 启动时只加载元数据（skill.md 头部 frontmatter）
  2. Planner 规划时只把元数据描述给 LLM
  3. 执行 SKILL 步骤时才读取 skill.md 正文，资源文件由 LLM 通过工具按需读取
"""
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from app.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class SkillMetadata:
    """Skill 元数据"""
    skill_id: str
    name: str = ""
    description: str = ""
    tags: list[str] = field(default_factory=list)
    triggers: list[str] = field(default_factory=list)
    root: Path | None = None


@dataclass
class SkillContent:
    """Skill 正文内容"""
    metadata: SkillMetadata
    markdown_body: str = ""


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """解析 skill.md 头部 YAML frontmatter（简化解析，无 pyyaml 依赖）"""
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n", text, re.DOTALL)
    if not m:
        return {}, text
    meta: dict = {}
    for line in m.group(1).splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        value = value.strip().strip('"').strip("'")
        if key.strip() == "tags" or key.strip() == "triggers":
            meta[key.strip()] = [v.strip() for v in re.split(r"[,，]", value) if v.strip()]
        else:
            meta[key.strip()] = value
    return meta, text[m.end():]


class SkillRegistry:
    """Skill 注册表"""

    def __init__(self):
        self._skills: dict[str, SkillMetadata] = {}
        self._loaded_bodies: dict[str, str] = {}

    def load_all(self) -> None:
        """扫描所有 Skill 根目录"""
        for root in get_settings().skills_root_list:
            if not root.exists():
                continue
            for skill_dir in sorted(root.iterdir()):
                skill_md = skill_dir / "skill.md"
                if not skill_dir.is_dir() or not skill_md.exists():
                    continue
                try:
                    meta, _ = _parse_frontmatter(skill_md.read_text(encoding="utf-8"))
                    sid = meta.get("id", skill_dir.name)
                    self._skills[sid] = SkillMetadata(
                        skill_id=sid,
                        name=meta.get("name", skill_dir.name),
                        description=meta.get("description", ""),
                        tags=meta.get("tags", []),
                        triggers=meta.get("triggers", []),
                        root=skill_dir,
                    )
                except Exception as e:
                    logger.warning("⚠️ 解析 Skill %s 失败: %s", skill_dir, e)
        logger.info("🧰 SkillRegistry 加载完成：%s", list(self._skills))

    def list_metadata(self) -> list[SkillMetadata]:
        return list(self._skills.values())

    def find_by_id(self, skill_id: str) -> SkillMetadata | None:
        return self._skills.get(skill_id)

    def list_ids(self) -> list[str]:
        return list(self._skills)

    def match_by_keywords(self, objective: str) -> list[SkillMetadata]:
        """按关键词匹配"""
        matched = []
        for meta in self._skills.values():
            keywords = set(meta.triggers + meta.tags)
            if any(k and k in objective for k in keywords):
                matched.append(meta)
        return matched

    def ensure_content_loaded(self, skill_id: str) -> SkillContent | None:
        """懒加载 skill.md 正文"""
        meta = self._skills.get(skill_id)
        if meta is None or meta.root is None:
            return None
        if skill_id not in self._loaded_bodies:
            _, body = _parse_frontmatter((meta.root / "skill.md").read_text(encoding="utf-8"))
            self._loaded_bodies[skill_id] = body
        return SkillContent(metadata=meta, markdown_body=self._loaded_bodies[skill_id])

    def list_resource_paths(self, skill_id: str) -> list[str]:
        """列出 Skill 资源文件"""
        meta = self._skills.get(skill_id)
        if meta is None or meta.root is None:
            return []
        return sorted(
            str(p.relative_to(meta.root))
            for p in meta.root.rglob("*")
            if p.is_file() and p.name != "skill.md"
        )

    def read_resource(self, skill_id: str, file_path: str) -> str:
        """读取单个资源文件"""
        meta = self._skills.get(skill_id)
        if meta is None or meta.root is None:
            return f"错误：Skill[{skill_id}]不存在"
        target = (meta.root / file_path).resolve()
        # 防目录穿越
        if not str(target).startswith(str(meta.root.resolve())):
            return "错误：非法的文件路径"
        if not target.exists() or not target.is_file():
            return f"错误：文件不存在（可用资源：{self.list_resource_paths(skill_id)}）"
        return target.read_text(encoding="utf-8")


# 全局单例
skill_registry = SkillRegistry()
