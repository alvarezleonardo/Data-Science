# Reglas del repositorio — Data Science

## Git — autoría de commits

- **Co-autor:** SIEMPRE agregar al final de cada commit
  `Co-Authored-By: leitoalvarez <225464551+leitoalvarez@users.noreply.github.com>`.
  **NUNCA** usar `Co-Authored-By: Claude` ni footers tipo "Generated with Claude Code".
- Autor principal visible: `Leonardo Alvarez <19362850+alvarezleonardo@users.noreply.github.com>`
  (configurado local al repo).

## Acceso a GitHub

- Remoto vía **SSH**: `git@github-personal:alvarezleonardo/Data-Science.git`
  (SSH key `~/.ssh/id_ed25519_github_personal`, cuenta `alvarezleonardo`).
- El `gh` CLI está logueado como otra cuenta (`leitoalvarez`), que **no es colaboradora**
  de este repo: no puede crear PRs por API. Operar por SSH (`git push/pull/merge`);
  los PRs se abren/mergean desde la web.

## Convenciones

- Carpetas y archivos nuevos en `kebab-case`, sin acentos ni espacios.
- Estructura por módulos del programa (ver `README.md`): `NN-nombre/` con
  `teoria/`, `notebooks/`, `datasets/`.
- No versionar entornos `.venv/` ni datasets pesados `*.zip` (>100 MB).
