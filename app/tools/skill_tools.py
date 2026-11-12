"""Skill 资源工具：给 LLM 按需读取 Skill 资源"""
import logging

from langchain_core.tools import tool

from app.skill.registry import skill_registry

logger = logging.getLogger(__name__)


@tool(description="列出指定Skill自带的资源文件清单（相对路径）")
def list_skill_resources(skill_id: str) -> str:
    paths = skill_registry.list_resource_paths(skill_id)
    if not paths:
        return "（该 Skill 无资源文件）"
    return "\n".join(f"{i}. {p}" for i, p in enumerate(paths, 1))


@tool(description="读取指定Skill下的某个资源文件文本内容（一次读一个）")
def read_skill_resource(skill_id: str, file_path: str) -> str:
    return skill_registry.read_resource(skill_id, file_path)
