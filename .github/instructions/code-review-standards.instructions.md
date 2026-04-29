---
description: "Use when reviewing or editing Python code in this repo. Enforces strict, professional code review, dead-code and unused-variable checks, explicit validation, and concise line-specific feedback."
---
# Code Review Standards

- Be strict and professional.
- Prioritize correctness, contract clarity, and maintainability over style polish.
- Flag dead code, unused variables, ambiguous return values, and hidden assumptions.
- Prefer explicit validation over relying on implicit behavior.
- After edits, validate the touched file or the narrowest relevant path before widening scope.
- When reporting issues, cite exact files and lines and keep the feedback concise.
- Do not add filler, praise, or speculative commentary.
- If a helper or branch is unused, remove it unless there is a clear near-term use.
- When a selector or validation path can accept multiple schemas, make the contract explicit in code and docs.
