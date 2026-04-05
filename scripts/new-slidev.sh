#!/usr/bin/env bash
# new-slidev.sh — scaffold a new Slidev presentation in this workspace.
# Usage (from workspace root): pnpm new-slidev "My Talk Title"
set -euo pipefail

SLIDES_DIR="$(cd "$(dirname "$0")/.." && pwd)/slides"

title="${1:-}"
if [ -z "$title" ]; then
  echo "Usage: pnpm new-slidev \"Title of the talk\""
  exit 1
fi

# Slugify: lowercase, spaces/slashes → hyphen, strip invalid chars
slugify() {
  local s="$1"
  s="$(printf '%s' "$s" | tr '[:upper:]' '[:lower:]')"
  s="$(printf '%s' "$s" | sed -E 's#[[:space:]/]+#-#g')"
  s="$(printf '%s' "$s" | sed -E 's/[^a-z0-9._-]//g')"
  s="$(printf '%s' "$s" | sed -E 's/-+/-/g')"
  s="$(printf '%s' "$s" | sed -E 's/^[.-]+|[.-]+$//g')"
  if [ -z "$s" ]; then s="slide"; fi
  printf '%s' "$s"
}

name="$(slugify "$title")"
dest="$SLIDES_DIR/$name"

if [ -d "$dest" ]; then
  echo "❌ '$dest' already exists. Choose a different title."
  exit 1
fi

mkdir -p "$dest"/{public,setup}

# ── package.json ───────────────────────────────────────────────────────────────
# dev base MUST match the directory name so slidev-workspace resolves image URLs correctly.
cat > "$dest/package.json" <<JSON
{
  "name": "$name",
  "type": "module",
  "private": true,
  "scripts": {
    "build": "slidev build",
    "dev": "slidev --base /$name/",
    "export": "slidev export"
  },
  "dependencies": {
    "@slidev/cli": "catalog:",
    "@slidev/theme-default": "catalog:",
    "vue": "catalog:"
  }
}
JSON

# ── slides.md ──────────────────────────────────────────────────────────────────
# NOTE: avoid double curly braces {{ }} inside LaTeX blocks — Vue's template
# compiler intercepts them before KaTeX runs, causing a build error.
# Use single braces instead: {\rm env} not {{\rm env}}.
cat > "$dest/slides.md" <<MD
---
layout: cover
mdc: true
title: "$title"
author: ""
date: "$(date +%Y-%m-%d)"
theme: default
---

# $title
MD

# ── style.css ──────────────────────────────────────────────────────────────────
cat > "$dest/style.css" <<'CSS'
.centered {
  position: fixed;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
}

.horizontal-center {
  display: block;
  margin-left: auto;
  margin-right: auto;
}

.bottom-right {
  position: absolute;
  bottom: 0;
  right: 0;
}

.float-left {
  float: left;
}

.footnotes {
  font-size: 0.6em;
}

li.footnote-item {
  margin: -2em 0;
}
CSS

# ── setup/katex.ts ─────────────────────────────────────────────────────────────
cat > "$dest/setup/katex.ts" <<'TS'
import { defineKatexSetup } from '@slidev/types'

export default defineKatexSetup(() => {
  return {
    maxExpand: 2000,
    macros: {
      "\\vA": "{\\mathbf A}", "\\vB": "{\\mathbf B}", "\\vC": "{\\mathbf C}",
      "\\vD": "{\\mathbf D}", "\\vE": "{\\mathbf E}", "\\vF": "{\\mathbf F}",
      "\\vG": "{\\mathbf G}", "\\vH": "{\\mathbf H}", "\\vI": "{\\mathbf I}",
      "\\vJ": "{\\mathbf J}", "\\vK": "{\\mathbf K}", "\\vL": "{\\mathbf L}",
      "\\vM": "{\\mathbf M}", "\\vN": "{\\mathbf N}", "\\vO": "{\\mathbf O}",
      "\\vP": "{\\mathbf P}", "\\vQ": "{\\mathbf Q}", "\\vR": "{\\mathbf R}",
      "\\vS": "{\\mathbf S}", "\\vT": "{\\mathbf T}", "\\vU": "{\\mathbf U}",
      "\\vV": "{\\mathbf V}", "\\vW": "{\\mathbf W}", "\\vX": "{\\mathbf X}",
      "\\vY": "{\\mathbf Y}", "\\vZ": "{\\mathbf Z}",
      "\\va": "{\\mathbf a}", "\\vb": "{\\mathbf b}", "\\vc": "{\\mathbf c}",
      "\\vd": "{\\mathbf d}", "\\ve": "{\\mathbf e}", "\\vf": "{\\mathbf f}",
      "\\vg": "{\\mathbf g}", "\\vh": "{\\mathbf h}", "\\vi": "{\\mathbf i}",
      "\\vj": "{\\mathbf j}", "\\vk": "{\\mathbf k}", "\\vl": "{\\mathbf l}",
      "\\vm": "{\\mathbf m}", "\\vn": "{\\mathbf n}", "\\vo": "{\\mathbf o}",
      "\\vp": "{\\mathbf p}", "\\vq": "{\\mathbf q}", "\\vr": "{\\mathbf r}",
      "\\vs": "{\\mathbf s}", "\\vt": "{\\mathbf t}", "\\vu": "{\\mathbf u}",
      "\\vv": "{\\mathbf v}", "\\vw": "{\\mathbf w}", "\\vx": "{\\mathbf x}",
      "\\vy": "{\\mathbf y}", "\\vz": "{\\mathbf z}",
      "\\vzero": "{\\mathbf 0}", "\\vone": "{\\mathbf 1}",
      "\\vtheta": "{\\bm \\theta}",
      "\\vmu": "{\\bm \\mu}",
      "\\vSigma": "{\\bm \\Sigma}",
      "\\cov": "{\\rm Cov}",
      "\\var": "{\\rm Var}",
      "\\d": "{\\rm d}",
      "\\cond": "\\,\\vert\\,",
      "\\data": "{\\cal D}",
      "\\reals": "{\\mathbb R}"
    }
  }
})
TS

echo ""
echo "✅ Created slides/$name"
echo ""
echo "Next: run 'pnpm install' from the workspace root to update pnpm-lock.yaml."
echo "Then: pnpm preview  (or cd slides/$name && pnpm dev)"
