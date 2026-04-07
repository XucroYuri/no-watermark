# Release Process

## 1. Freeze Scope

- Decide the target version
- Limit the release to reviewed changes
- Move unfinished work out of the release milestone

## 2. Validate

- Run the release checklist in [release-checklist.md](./release-checklist.md)
- Confirm documentation and changelog updates are complete
- Confirm no local-only artifacts are staged
- Save the smoke-script output JSON and any filtered aggregation summaries used for the release decision

## 3. Prepare The Commit

- Review `git status`
- Stage only intended project files
- Create a release commit with a version-oriented message such as `release: v0.3.0`

## 4. Tag The Release

- Create an annotated tag such as `v0.3.0`
- Draft release notes from [CHANGELOG.md](../../CHANGELOG.md) and merged changes

## 5. Post-Release Follow-Up

- Advance `TODO.md` and `ROADMAP.md`
- Open follow-up issues for deferred items
- Start the next development iteration on top of the tagged state
