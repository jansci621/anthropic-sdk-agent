---
name: architecture
description: "Use when making architectural decisions, analyzing system design, evaluating trade-offs, or planning project structure. Covers requirements analysis, pattern selection, trade-off evaluation, and ADR documentation."
---

# Architecture Decision Framework

> "Requirements drive architecture. Trade-offs inform decisions. ADRs capture rationale."

## Core Principle

**"Simplicity is the ultimate sophistication."**

- Start simple
- Add complexity ONLY when proven necessary
- You can always add patterns later
- Removing complexity is MUCH harder than adding it

## When to Use

- Making architectural decisions for a new or evolving system
- Evaluating technology choices or design trade-offs
- Reviewing code structure and module boundaries
- Planning project organization and patterns
- Writing Architecture Decision Records (ADRs)

**Do NOT use for:** Simple one-off tasks, minor implementation details, routine maintenance.

## Process

1. **Assess** -- Gather context using the discovery questions
2. **Classify** -- Determine project type (MVP / SaaS / Enterprise)
3. **Select** -- Choose patterns using decision trees
4. **Document** -- Record decisions with trade-off analysis and ADRs
5. **Verify** -- Validate against requirements and constraints

## Selective Reading Guide

| File | Description | When to Read |
|------|-------------|--------------|
| [references/context-discovery.md](references/context-discovery.md) | Discovery questions, project classification matrix | Starting architecture design |
| [references/trade-off-analysis.md](references/trade-off-analysis.md) | ADR templates, trade-off framework, decision records | Documenting decisions |
| [references/pattern-selection.md](references/pattern-selection.md) | Decision trees, the 3 questions, anti-patterns | Choosing patterns |
| [references/patterns-reference.md](references/patterns-reference.md) | Quick lookup tables for data, domain, distributed, and API patterns | Pattern comparison |
| [references/examples.md](references/examples.md) | MVP, SaaS, Enterprise reference architectures | Reference implementations |

## Quick Reference

| Action | Description |
|--------|-------------|
| Scan structure | Check file organization and naming |
| Review logic | Verify correctness of core logic |
| Check quality | Evaluate readability, performance, security |

## The 3 Questions (Before ANY Pattern)

1. **Problem Solved**: What SPECIFIC problem does this pattern solve?
2. **Simpler Alternative**: Is there a simpler solution?
3. **Deferred Complexity**: Can we add this LATER when needed?

## Validation Checklist

Before finalizing architecture:

- [ ] Requirements clearly understood
- [ ] Constraints identified
- [ ] Each decision has trade-off analysis
- [ ] Simpler alternatives considered
- [ ] ADRs written for significant decisions
- [ ] Team expertise matches chosen patterns

## Related Skills

| Skill | Use For |
|-------|---------|
| `@[skills/database-design]` | Database schema design |
| `@[skills/api-patterns]` | API design patterns |
| `@[skills/deployment-procedures]` | Deployment architecture |
