# Changelog

All notable changes to `sonium-audio` are documented here.

## Unreleased

### Fixed

- Performance gate uses the median of 32 callbacks after warm-up, timed
  by thread CPU time instead of wall time, so scheduler preemption under
  parallel load no longer reads as a realtime overrun.


## 0.2.0

### Changed

- Introduced this changelog for the coordinated minor release from 0.1.0 to 0.2.0.
