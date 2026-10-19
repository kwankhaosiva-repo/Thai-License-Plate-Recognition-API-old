# GLOBAL PROJECT RULES

## 1. Core Principles

- Prefer simple, maintainable, testable solutions.
- Do not over-engineer.
- Reuse existing code before creating new code.
- Keep responsibilities separated and dependencies explicit.

## 2. Security

- Never hardcode secrets.
- Use environment variables or secret managers.
- Never perform destructive operations without explicit user authorization —
  this includes database/file deletion, `rm -rf` or bulk file removal,
  and destructive terminal/git commands (e.g. `git push --force`,
  `git reset --hard`, deleting branches).
- If authorization is not given, prepare the command and explain what it
  will do; let the User run it.

## 3. Code Structure

- The application entrypoint is `main.py` or `app.py` (whichever the project uses)
  and serves as the orchestration layer.
- Business logic must not be placed directly in the entrypoint file.
- Keep reusable logic in appropriate modules/services/skills.
- Avoid circular dependencies.
- Prefer composition over deep inheritance.

## 4. Configuration

- Put project/runtime configuration in `config.py` or the existing config system.
- Do not scatter important configurable values throughout the codebase.
- Do not put actual secrets in configuration files.

## 5. Naming

- Functions and variables: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPER_SNAKE_CASE`
- Use descriptive names; avoid meaningless names such as `data1`, `data2`.

## 6. Error Handling

- Handle errors intentionally.
- Prefer specific exceptions.
- Do not silently swallow exceptions.
- Do not add unnecessary `try/except` blocks.

## 7. Dependencies

- Reuse existing dependencies when appropriate.
- New libraries must be added to `requirements.txt` (or the project's
  dependency file) with a pinned version.

## 8. Testing

- Validate the affected functionality after implementation.
- Perform regression checks for affected components.
- Use `src/scratch/` only for experimentation; permanent tests belong in `tests/`.

## 9. Skill Library

- Check `src/skills/` before implementing reusable logic.
- Reuse an existing skill when applicable.
- Create a new skill only when the capability is meaningfully reusable.
- Do not create skills for trivial or one-off code.
- Update the source skill when its reusable behavior changes.

## 10. Deployment

- Validate locally before deployment.
- Do not deploy automatically unless explicitly authorized by the User.

## 11. Version Control

- Never force-push, rewrite shared history, or push/merge directly to
  main/production branches without explicit authorization.
- Do not include secrets or credentials in commits.
- Prefer small, reviewable commits over large unexplained ones.

## 12. Communication

- Inspect the codebase before making assumptions.
- If required schema/configuration information cannot be determined safely,
  state the uncertainty instead of guessing.
- At delivery, summarize changed files and validation performed.
- When the User writes in English, correct significant grammar, spelling,
  or terminology mistakes before proceeding, to help the User improve.
  Keep corrections brief (one line) and do not block task execution.
  Skip correction when the meaning is already clear, the correction would
  consume excessive tokens, or it would interfere with completing the task.

## 13. Project Bootstrap (AGENTS.md)

- When starting work on any project that has no `AGENTS.md` at its root,
  create one automatically and copy these Global Project Rules into it,
  then add a "Project-Specific Notes" section describing the actual
  entrypoint file, how to run tests, and any project-specific conventions.
- If an `AGENTS.md` already exists, do not overwrite it; keep the existing
  rules and only suggest updates when they are clearly outdated.
- Use the filename `AGENTS.md` (not `AGENT.md`) at the repository root.
- `AGENTS.md` may be a symlink to this file (`GEMINI.md`) so that Gemini,
  Antigravity, Freebuff, and other agents share one source of truth.

---

## Project-Specific Notes

- **Entrypoint**: `src/api_server.py` (FastAPI backend server) and `src/main.py`.
- **Configuration**: `src/config.py` holds pipeline flags, model weights paths, and thresholds.
- **Python Environment**: Miniconda environment `thai-lpr` (`/Users/kwankhaos/miniconda3/envs/thai-lpr/bin/python`).
- **Tests**: Permanent unit/integration tests are in `tests/` (run with `pytest tests/`). Experiments in `scratch/`.
- **Model Versioning**: Current production models are `v5`. All next iterations must be tagged `v6` (e.g. `v6aug`, `v6frame`, `v6new`, `v6`).
