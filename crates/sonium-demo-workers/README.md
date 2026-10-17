# sonium-demo-workers

Native simulation adapters for the standalone GPUI demo
applications. The binary validates a demo project, runs the
corresponding bounded native study, validates the scientific
result, and writes it as JSON for the front end.

## Usage

```bash
# Print the study catalog for one simulator
sonium-demo-workers --catalog room

# Run one project
sonium-demo-workers INPUT.json OUTPUT.json

# Validate inputs without running the study
sonium-demo-workers INPUT.json OUTPUT.json --validate
```

Catalogs cover the room, PC-fan, propeller, driver, and structural
demo studies. Each catalog declares its bounded input domain;
out-of-domain projects and invalid results fail closed.

## Development

```bash
cargo test -p sonium-demo-workers
```

See `CHANGELOG.md` for release history.
