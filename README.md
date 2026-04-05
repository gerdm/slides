# Slides

A monorepo of [Slidev](https://sli.dev) presentations managed with [slidev-workspace](https://github.com/leochiu-a/slidev-workspace) and pnpm workspaces. Each presentation lives under `slides/<name>/` as an independent package. A single GitHub Actions workflow builds and deploys all of them to GitHub Pages.

## 🚀 Local development

```bash
pnpm install
pnpm preview   # starts the workspace index on http://localhost:3000/slides
               # and a Slidev dev server per presentation on :3001, :3002, …
```

## ➕ Creating a new presentation

Use the custom `new-slidev` command **from inside the `slides/` directory**:

```bash
cd slides
new-slidev "My Talk Title"
```

This scaffolds a new package under `slides/<slugified-name>/` with the correct `package.json`, `slides.md`, `style.css`, and `setup/katex.ts`.

### ⚠️ Required step after scaffolding

The script writes a `package.json` that uses `catalog:` dependencies. You must **run `pnpm install` from the workspace root** afterwards to update `pnpm-lock.yaml` and create the correct symlinks:

```bash
cd ..          # back to workspace root
pnpm install
```

Skipping this causes the CI build to fail with `ERR_PNPM_OUTDATED_LOCKFILE`.

### Checklist after running `new-slidev`

- [ ] Run `pnpm install` from the workspace root (see above).
- [ ] The `dev` script in `package.json` is set to `slidev --base /<name>/`. Verify that `<name>` **matches the directory name exactly** — the workspace uses the directory name to construct image URLs. If they differ, background images will 404.
- [ ] Place any background or image assets in `slides/<name>/public/` and reference them as `/filename.png` in `slides.md`.
- [ ] Commit **both** the new `slides/<name>/` directory **and** the updated `pnpm-lock.yaml`.

## ✍️ Writing slides — known gotchas

### LaTeX and Vue template syntax conflict

Slidev compiles `slides.md` through Vue's template engine before rendering LaTeX. **Double curly braces `{{ }}` are intercepted by Vue** as template interpolations, causing a build error:

```
Error parsing JavaScript expression: Expecting Unicode escape sequence \uXXXX
```

**Avoid double braces in LaTeX.** Use single braces — they are semantically identical in LaTeX:

```diff
- y_t \sim p_{{\rm env}}(\cdot \mid x_t)   ❌ Vue intercepts {{ }}
+ y_t \sim p_{\rm env}(\cdot \mid x_t)     ✓ Safe
```

## 🏗️ Repository structure

```
slides/
├── slidev-workspace.yaml     # baseUrl + exclude list
├── pnpm-workspace.yaml       # workspace globs
├── pnpm-lock.yaml            # must be kept up to date
├── slides/
│   ├── <presentation-a>/
│   │   ├── package.json      # name, dev/build scripts, deps
│   │   ├── slides.md
│   │   ├── style.css
│   │   ├── setup/katex.ts
│   │   └── public/           # static assets (images, etc.)
│   └── <presentation-b>/
│       └── …
└── .github/workflows/
    └── deploy.yml
```

## 🔧 `slidev-workspace.yaml`

```yaml
baseUrl: /slides
exclude:
  - node_modules
  - .git
  - _gh-pages
  - dist
```

`exclude` prevents the workspace from trying to scan output/build directories as slide packages.

## 🚢 Deployment

Pushing to `main` triggers the **Deploy pages** GitHub Actions workflow which:

1. Installs dependencies (`pnpm install --frozen-lockfile`).
2. Builds every presentation under `slides/`.
3. Deploys the workspace index + all built presentations to GitHub Pages.

The `pnpm-lock.yaml` must be committed and up to date, or the install step will fail.
