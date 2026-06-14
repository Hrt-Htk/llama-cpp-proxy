# Development Workflow

Rules for issues, branches, commits, and PRs. Designed for a solo dev working with AI agents — practical, not theoretical.

## Clean Working Tree

Never start new work while the tree is dirty. Before beginning a task, run `git status` — it must be clean. If it isn't, commit the work, stash it, or discard it first.

A dirty tree is how changes get lost: half-finished changes sit uncommitted, get entangled with new work, and vanish. A clean starting point means every change you make is attributable to the task at hand.

Transient scratch — `.tmp/`, `.pi/tasks/`, screenshots, build leftovers — should be gitignored so it never shows as dirty. If `git status` surfaces noise that isn't real work, add it to `.gitignore` rather than tolerating a permanently dirty tree.

## Issue Rules

- **Labels are mandatory.** Every issue needs at least one label (`bug`, `enhancement`, `docs`, `chore`, etc.). Unlabelled issues are unclear and get deprioritized.
- **No issue, no work.** If there isn't an open issue describing the change, create one first. The issue is the source of truth for the "why."
- **One issue, one PR.** Don't bundle unrelated changes. If a fix reveals a new problem, file a separate issue and handle it in a follow-up PR.
- **Issue lifecycle:** `open` → `in progress` (when work starts) → `PR` (linked pull request) → `closed` (only after the PR merges). Never close an issue before its PR is merged.

## Commit Rules

Use [Conventional Commits](https://www.conventionalcommits.org/) format:

```
type(scope): description

[optional body]

Closes #N
```

### Types

| Type       | When to use                                          |
|------------|------------------------------------------------------|
| `feat`     | New feature or user-visible capability               |
| `fix`      | Bug fix                                              |
| `docs`     | Documentation changes only                           |
| `refactor` | Code change that neither fixes a bug nor adds a feature |
| `style`    | Formatting, whitespace, UI styling — no behavior change |
| `test`     | Adding or updating tests                             |
| `chore`    | Build, tooling, dependency, or maintenance changes   |
| `perf`     | Performance improvement                              |
| `build`    | Changes to build system or external dependencies     |
| `ci`       | CI/CD configuration changes                          |

### Footer

Link the issue in the commit footer so `git log` traces back to the original discussion:

```
Closes #29
```

Use `Closes` for the final commit that completes the issue. Use `Refs` for intermediate commits that touch the same issue.

## Branch Rules

Name branches following `type/issueN-description`:

```
feat/issue29-workflow-docs
fix/issue11-empty-chevron
docs/issue15-update-readme
refactor/issue42-simplify-worker
```

- `type` matches the commit type table above
- `issueN` is the GitHub issue number
- `description` is a short kebab-case summary

## PR Rules

- The PR template includes a checklist — fill it out completely.
- Every PR must reference an open issue. The link can be in the PR body or in a commit message.
- The PR title should follow conventional commit format (e.g. `feat: add workflow checks to CI`).

## Size Limits

| Limit      | Value          | When it applies        |
|------------|----------------|------------------------|
| Files      | 10 (soft cap)  | Per PR                 |
| Lines      | 300 (soft cap) | Per PR (added + deleted) |

Soft caps — exceeding them triggers a CI warning, not a failure. If a PR legitimately needs more, add a note in the PR body explaining why.

## Enforcement

### Hard failures (block the PR)

The `workflow-checks` CI job enforces these automatically:

- **Conventional commit format** — every commit message must match `^(feat|fix|chore|docs|refactor|test|perf|build|ci|style)(\(.+\))?: .+`
- **Branch naming** — on pushes (not `main`), branch must match `^(feat|fix|chore|docs|refactor|test|perf)/issue[0-9]+-.*`
- **Issue reference** — at least one commit message or the PR body must contain `#[0-9]+`

### Soft warnings (print but don't fail)

- **Size limits** — if `git diff --shortstat` shows more than 10 files or 300 lines changed, a warning is printed.

## Exceptions

These changes may skip the issue requirement:

- Typo fixes in documentation
- Formatting-only changes (whitespace, lint auto-fixes)
- Dependency version bumps with no behavioral impact

Even then, use a conventional commit type (`docs`, `chore`) and keep the change in a separate PR.
