# Contributing

## Scope

This project is a reusable batch watermark removal and restoration framework. Contributions should improve the framework itself rather than embed assumptions tied to one private image set.

## Before You Start

- Read [docs/ARCHITECTURE.md](./docs/ARCHITECTURE.md)
- Read [docs/DEVELOPMENT.md](./docs/DEVELOPMENT.md)
- Check [TODO.md](./TODO.md) and [ROADMAP.md](./ROADMAP.md)
- Check [docs/releases/README.md](./docs/releases/README.md) for release-facing changes

## Contribution Rules

- Keep defaults dataset-agnostic
- Put heavyweight model integrations behind provider interfaces
- Do not commit private input images, benchmark artifacts, or runtime outputs
- Add or update tests for behavior changes
- Prefer small, reviewable pull requests

## Local Workflow

```powershell
python -m pip install -r .\requirements.txt
python -m unittest discover -s tests -v
python .\benchmark.py list-providers
```

## Pull Request Checklist

- Tests pass locally
- Docs are updated if behavior changes
- Generated outputs are not committed
- Paths and wording are generic
- New providers fail gracefully when their environments are missing
- Changelog and release notes inputs are updated for user-visible changes
