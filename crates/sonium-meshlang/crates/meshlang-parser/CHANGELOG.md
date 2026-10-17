# Changelog

## 0.2.0

### Changed

- Bumped the package version from 0.1.5 to 0.2.0 as part of the coordinated minor release.

## 0.1.4

### Fixed

- `parse_string` on unterminated string literals silently truncated content instead of producing
  an error. Now checks for closing quote and returns a parse error for unterminated strings.
- Comparison expression span started at the operator token instead of the left operand, producing
  wrong source locations in diagnostics.
- `parse_kv_args` crashed with a confusing error when called with an empty argument list (e.g.,
  trailing comma before `)`).
- Removed the dead `is_pipe_arrow` CSG guard and parse `PipeArrow` directly according to the
  normative precedence grammar.
