#!/usr/bin/env bash

set -euo pipefail

script_dir="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
repo_root="$(CDPATH= cd -- "${script_dir}/.." && pwd)"

paper_dir="${repo_root}/paper"
docs_paper_dir="${repo_root}/docs/paper"
tmp_md="$(mktemp "${TMPDIR:-/tmp}/qmcpy-paper.XXXXXX.md")"
tmp_html="$(mktemp "${TMPDIR:-/tmp}/qmcpy-paper.XXXXXX.html")"

cleanup() {
  rm -f "${tmp_md}"
  rm -f "${tmp_html}"
}
trap cleanup EXIT

# Render the paper through Pandoc so citations and LaTeX math display cleanly
# in MkDocs without changing the source paper. A few LaTeX-only cross-references
# are rewritten to natural prose for the web view.
perl -0pe '
  s/^csl:.*\n//m;
  s/^# References\s*$//m;
  s/\\autoref\{fig:points\}/The figure below/g;
  s/\\eqref\{eq:mu-uniform\}/this transformed form/g;
  s/\\autoref\{fig:stopping_crit\}/The figure below/g;
' "${paper_dir}/paper.md" > "${tmp_md}"

pandoc "${tmp_md}" \
  --from markdown+raw_tex \
  --to html5 \
  --mathml \
  --citeproc \
  --shift-heading-level-by=1 \
  --resource-path="${paper_dir}:${repo_root}" \
  --output "${tmp_html}"

perl -0pi -e 's!<div id="refs"!<h2 id="references">References</h2>\n<div id="refs"!' \
  "${tmp_html}"

perl -0pi -e 's!\bsrc="\./figs/!src="../figs/!g; s!\bhref="\./figs/!href="../figs/!g' \
  "${tmp_html}"

rm -f "${docs_paper_dir}/paper.rendered.html"

{
  printf '%s\n' '---'
  printf '%s\n' 'hide:'
  printf '%s\n' '  - toc'
  printf '%s\n' '---'
  printf '\n'
  printf '%s\n' '# QMCPy: A Python Framework for (Quasi-)Monte Carlo Algorithms'
  printf '\n'
  printf '%s\n' '<div class="paper-render">'
  cat "${tmp_html}"
  printf '\n%s\n' '</div>'
} > "${docs_paper_dir}/paper.md"
