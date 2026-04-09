---
name: api-design-principles
description: "Use when designing new REST or GraphQL APIs, refactoring existing APIs, reviewing API specifications, establishing API design standards, migrating between API paradigms, or optimizing APIs for mobile and third-party integrations."
---

# API Design Principles

Master REST and GraphQL API design to build intuitive, scalable, and maintainable APIs.

## When to Use

- Designing new REST or GraphQL APIs
- Refactoring existing APIs for better usability
- Establishing API design standards for a team
- Reviewing API specifications before implementation
- Migrating between API paradigms (REST to GraphQL, etc.)
- Optimizing APIs for specific use cases (mobile, third-party integrations)

**Do NOT use for:** Framework-specific implementation guidance, infrastructure-only work, or immutable public interfaces.

## Design Process

1. **Define** consumers, use cases, and constraints
2. **Choose** API style (REST or GraphQL) and model resources/types
3. **Specify** errors, versioning, pagination, and auth strategy
4. **Validate** with examples and review for consistency

## Quick Reference

| Topic | Key File |
|-------|----------|
| REST best practices, URLs, methods, pagination | [references/rest-best-practices.md](references/rest-best-practices.md) |
| GraphQL schema design, types, pagination, mutations | [references/graphql-schema-design.md](references/graphql-schema-design.md) |
| Implementation patterns, HATEOAS, DataLoaders | [references/implementation-playbook.md](references/implementation-playbook.md) |
| Pre-implementation review checklist (REST + GraphQL) | [references/api-design-checklist.md](references/api-design-checklist.md) |
| FastAPI REST API template (production-ready) | [references/rest-api-template.py](references/rest-api-template.py) |

## REST vs GraphQL Decision Guide

| Factor | Prefer REST | Prefer GraphQL |
|--------|------------|----------------|
| Simple CRUD | Yes | No |
| Many related entities | No | Yes |
| Mobile clients | No | Yes (small payloads) |
| Public API | Yes (widely understood) | Varies |
| Caching | HTTP caching built-in | Needs custom solution |
| File uploads | Straightforward | Complex |

## Common Mistakes

| Mistake | Fix |
|---------|-----|
| API mirrors DB schema | Model for consumers, not tables |
| Inconsistent error formats | Standardize error response across all endpoints |
| No versioning plan | Version from day one (URL versioning recommended) |
| Over-fetching (REST) | Support sparse fieldsets, pagination |
| N+1 queries (GraphQL) | Use DataLoaders for all relationships |
| Breaking changes | Use deprecation, evolve schema incrementally |

## Detailed References

- [references/rest-best-practices.md](references/rest-best-practices.md) -- URL structure, HTTP methods, filtering/sorting, pagination, versioning, rate limiting, caching, CORS, error responses, OpenAPI docs
- [references/graphql-schema-design.md](references/graphql-schema-design.md) -- Schema organization, type design, unions, interfaces, input types, pagination (Relay), mutations, subscriptions, custom scalars, directives, error handling, N+1 solutions
- [references/implementation-playbook.md](references/implementation-playbook.md) -- REST resource collection design, pagination/filtering patterns, error handling, HATEOAS, GraphQL schema/resolver patterns, DataLoader N+1 prevention, best practices summary
- [references/api-design-checklist.md](references/api-design-checklist.md) -- Full pre-implementation checklist covering resources, methods, status codes, pagination, filtering, versioning, errors, auth, rate limiting, docs, testing, security, performance, monitoring, plus GraphQL-specific checks
- [references/rest-api-template.py](references/rest-api-template.py) -- Production-ready FastAPI template with Pydantic models, pagination, error handling, CORS, and security middleware
