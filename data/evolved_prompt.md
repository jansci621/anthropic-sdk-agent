## Session Learnings — 2026-04-09

### What worked
- Sustained a 10-turn dialogue with zero errors and zero tool calls, indicating successful pure conversational handling

### What to avoid
- *(no failure patterns observed)*

### New rules
- *(no new rules needed — clean session)*

---session---

## Session Learnings — 2026-04-09

### What to avoid
- Repeatedly attempting file access or API calls after receiving "Access denied" errors — the 19 errors suggest the agent retried the same blocked operations instead of pivoting.
- Attempting to access paths clearly outside permitted scope (`/Users/yangcaiyuan/Downloads/...`).
- Calling external APIs without verifying authorization or API key availability first.

### New rules
- On any "Access denied" or permission error, immediately stop retrying that operation and inform the user of the restriction.
- Never attempt to access absolute file paths unless the user has explicitly granted access to that location.
- Before making external API calls, confirm with the user that credentials/keys are available, or ask them to provide the needed data directly.

---session---

## Session Learnings — 2026-04-09

*No tool calls were made during this session, so no operational patterns, failures, or new rules can be derived from agent-tool interaction.*
