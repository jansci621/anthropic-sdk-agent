---
name: backend-patterns
description: "Use when building server-side applications with Node.js, Express, or Next.js API routes. Covers API design patterns, database optimization, caching strategies, authentication, error handling, rate limiting, background jobs, and structured logging for scalable backends."
---

# Backend Development Patterns

Architecture patterns and best practices for scalable server-side applications with Node.js, Express, and Next.js API routes.

## When to Use

- Building new API routes or backend services
- Implementing data access layers, caching, or auth in server-side code
- Designing error handling, rate limiting, or background job processing
- Optimizing database queries and preventing N+1 problems

## Quick Reference

| Pattern | Purpose | Key File |
|---------|---------|----------|
| Repository | Abstract data access | [api-design-patterns.md](references/api-design-patterns.md) |
| Service Layer | Separate business logic | [api-design-patterns.md](references/api-design-patterns.md) |
| Middleware | Request/response pipeline | [api-design-patterns.md](references/api-design-patterns.md) |
| Query Optimization | Efficient DB queries | [database-patterns.md](references/database-patterns.md) |
| N+1 Prevention | Batch fetching | [database-patterns.md](references/database-patterns.md) |
| Transactions | Atomic operations | [database-patterns.md](references/database-patterns.md) |
| Cache-Aside / Redis | Response caching | [caching-patterns.md](references/caching-patterns.md) |
| Error Handler | Centralized errors | [error-handling-and-auth.md](references/error-handling-and-auth.md) |
| JWT / RBAC | Auth and permissions | [error-handling-and-auth.md](references/error-handling-and-auth.md) |
| Rate Limiter | Throttle requests | [rate-limiting-and-jobs.md](references/rate-limiting-and-jobs.md) |
| Job Queue | Background processing | [rate-limiting-and-jobs.md](references/rate-limiting-and-jobs.md) |
| Structured Logger | JSON logging | [rate-limiting-and-jobs.md](references/rate-limiting-and-jobs.md) |

## Core Process

1. **Analyze** -- Understand requirements and constraints
2. **Select** -- Choose appropriate patterns from references below
3. **Implement** -- Apply patterns following code examples in reference files
4. **Verify** -- Confirm scalability, error handling, and test coverage

## Common Mistakes

| Mistake | Fix |
|---------|-----|
| Selecting `*` from DB | Select only needed columns |
| N+1 queries in loops | Batch-fetch related data |
| Business logic in route handlers | Extract to service layer |
| Storing secrets in JWT payload | JWT is not encrypted; store only identifiers |
| Missing error boundaries | Use centralized error handler |
| No rate limiting | Add per-IP or per-user rate limiter |

## Detailed References

- [api-design-patterns.md](references/api-design-patterns.md) -- RESTful structure, Repository pattern, Service layer, Middleware
- [database-patterns.md](references/database-patterns.md) -- Query optimization, N+1 prevention, Transaction patterns
- [caching-patterns.md](references/caching-patterns.md) -- Redis caching layer, Cache-aside pattern
- [error-handling-and-auth.md](references/error-handling-and-auth.md) -- Centralized error handler, retry with backoff, JWT validation, RBAC
- [rate-limiting-and-jobs.md](references/rate-limiting-and-jobs.md) -- In-memory rate limiter, job queue, structured logging
