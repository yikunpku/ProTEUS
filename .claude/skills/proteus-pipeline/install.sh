#!/usr/bin/env bash
# Install this skill for Claude Code and Codex.
#
# Both read the same format -- a directory holding SKILL.md with YAML
# frontmatter -- so one copy of the file serves both. They differ only in where
# they look:
#
#   Claude Code   <repo>/.claude/skills/<name>/   (already here, nothing to do)
#                 ~/.claude/skills/<name>/        (any directory)
#   Codex         $CODEX_HOME/skills/<name>/      (default ~/.codex/skills)
#
# So inside a checkout of this repository Claude Code already sees the skill.
# This script is for using it from anywhere, and for Codex.
#
#   ./install.sh              copy into both
#   ./install.sh --link       symlink instead, so repo edits take effect
#   ./install.sh --uninstall  remove what it installed
#   ./install.sh --check      report what is installed where
#   ./install.sh --force      replace a target that is not this skill
#
# A copied install does not contain install.sh, so run --uninstall from the
# repository, not from an installed copy.

set -euo pipefail

SRC="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
NAME="$(basename "$SRC")"
CLAUDE_DIR="${CLAUDE_HOME:-$HOME/.claude}/skills/$NAME"
CODEX_DIR="${CODEX_HOME:-$HOME/.codex}/skills/$NAME"
MODE=copy
FORCE=no

# Mode flags are not last-one-wins: `--uninstall --link` is a mistake, not an
# install.
set_mode() {
  if [ "$MODE" != copy ] && [ "$MODE" != "$1" ]; then
    echo "conflicting modes: --$MODE and --$1" >&2
    exit 2
  fi
  MODE=$1
}

for arg in "$@"; do
  case "$arg" in
    --link)      set_mode link ;;
    --uninstall) set_mode uninstall ;;
    --check)     set_mode check ;;
    --force)     FORCE=yes ;;
    --help|-h)   sed -n '/^set -euo/q;2,$p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *)           echo "unknown option: $arg (try --help)" >&2; exit 2 ;;
  esac
done

report() {
  local label=$1 dir=$2
  if [ -L "$dir" ]; then
    local target
    target="$(readlink "$dir")"
    if [ -e "$dir" ]; then
      printf '  %-14s symlink -> %s\n' "$label" "$target"
    else
      printf '  %-14s BROKEN symlink -> %s (target is gone)\n' "$label" "$target"
    fi
  elif [ -d "$dir" ]; then
    if [ -f "$dir/SKILL.md" ]; then
      printf '  %-14s copy at %s\n' "$label" "$dir"
    else
      printf '  %-14s a directory, but no SKILL.md in it: %s\n' "$label" "$dir"
    fi
  else
    printf '  %-14s not installed\n' "$label"
  fi
}

if [ "$MODE" = check ]; then
  echo "skill: $NAME"
  echo "source: $SRC"
  report "claude code" "$CLAUDE_DIR"
  report "codex" "$CODEX_DIR"
  exit 0
fi

# Never operate on a target that is, contains, or is contained by the source.
# Reachable in practice: CLAUDE_HOME (or HOME) pointing at the repository that
# holds this skill makes the target identical to the source, and the rm -rf
# below would delete the thing it is about to copy.
same_tree() {
  local dir=$1 resolved
  # A symlink at the target is only a link: `rm -rf` on it, with no trailing
  # slash, removes the link and not what it points at. Resolving it would
  # report the source and refuse every mode -- which is what made a --link
  # install impossible to --uninstall or to re-link.
  [ -L "$dir" ] && return 1
  resolved="$(cd -P "$dir" 2>/dev/null && pwd -P)" || return 1
  [ "$resolved" = "$SRC" ] && return 0
  case "$resolved/" in "$SRC"/*) return 0 ;; esac   # target sits inside source
  case "$SRC/" in "$resolved"/*) return 0 ;; esac   # source sits inside target
  return 1
}

# Identify our own installs so a same-named skill from somewhere else is not
# destroyed. "Has a SKILL.md" is not enough -- a rival skill has one too.
ours() {
  local dir=$1 target
  [ ! -e "$dir" ] && [ ! -L "$dir" ] && return 0      # nothing there
  if [ -L "$dir" ]; then
    # Follow the link and check what it points at, so --uninstall cannot
    # remove a link someone else put at this path. A dangling link is fine to
    # remove: it holds no data.
    [ -e "$dir" ] || return 0
    target="$(cd -P "$dir" 2>/dev/null && pwd -P)" || return 1
    [ -f "$target/SKILL.md" ] || return 1
    grep -qx "name: $NAME" "$target/SKILL.md" 2>/dev/null
    return
  fi
  [ -f "$dir/SKILL.md" ] || return 1
  grep -qx "name: $NAME" "$dir/SKILL.md" 2>/dev/null
}

for dir in "$CLAUDE_DIR" "$CODEX_DIR"; do
  if same_tree "$dir"; then
    echo "$dir is inside this skill's own directory ($SRC)." >&2
    echo "Operating on it would delete the source. Unset CLAUDE_HOME/CODEX_HOME," >&2
    echo "or point them somewhere other than this repository." >&2
    exit 1
  fi
  if [ "$FORCE" != yes ] && ! ours "$dir"; then
    echo "$dir already holds something that is not this skill (no" >&2
    echo "'name: $NAME' in its SKILL.md). Move it aside, or pass --force to" >&2
    echo "replace it." >&2
    exit 1
  fi
done

install_one() {
  local dir=$1 label=$2
  if [ "$MODE" = uninstall ]; then
    if [ -e "$dir" ] || [ -L "$dir" ]; then
      rm -rf "$dir"
      printf '  %-14s removed\n' "$label"
    else
      printf '  %-14s was not installed\n' "$label"
    fi
    return
  fi
  rm -rf "$dir"
  mkdir -p "$(dirname "$dir")"
  if [ "$MODE" = link ]; then
    ln -s "$SRC" "$dir"
    printf '  %-14s symlinked -> %s\n' "$label" "$SRC"
  else
    cp -r "$SRC" "$dir"
    rm -f "$dir/install.sh"
    printf '  %-14s copied to %s\n' "$label" "$dir"
  fi
}

echo "skill: $NAME ($MODE)"
install_one "$CLAUDE_DIR" "claude code"
install_one "$CODEX_DIR" "codex"

if [ "$MODE" != uninstall ]; then
  cat <<MSG

Invoke it by asking for the work -- "prepare my protein", "run depth 2",
"why did featurize fail" -- or by name:

  Claude Code   /proteus-pipeline
  Codex         ask for it by name; it loads from \$CODEX_HOME/skills

Both agents read the description to decide when to load it, so a new shell or
session may be needed before it shows up.
MSG
fi
