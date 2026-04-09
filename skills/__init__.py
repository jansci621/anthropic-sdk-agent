"""Skill auto-discovery: scan skills/ directory and load all valid skills."""

from pathlib import Path

from .base import Skill


def discover_skills(skill_dir: str | None = None) -> dict[str, Skill]:
    """Scan the skills directory and return {name: Skill} for all valid skills.

    A valid skill is a subdirectory containing a SKILL.md file.
    """
    if skill_dir is None:
        skill_dir = Path(__file__).parent

    skill_dir = Path(skill_dir)
    registry: dict[str, Skill] = {}

    if not skill_dir.exists():
        return registry

    for child in sorted(skill_dir.iterdir()):
        if not child.is_dir():
            continue
        if child.name.startswith("_") or child.name.startswith("."):
            continue
        if not (child / "SKILL.md").exists():
            continue
        try:
            skill = Skill(str(child))
            registry[skill.name] = skill
        except Exception as e:
            print(f"[Skill] Failed to load {child.name}: {e}")

    return registry
