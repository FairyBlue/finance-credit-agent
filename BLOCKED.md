# Blocked framework changes

If a required feature cannot be implemented through Agent-R1's public `ToolEnv` / `BaseTool`
plugin interfaces, add an entry below with:

- the required feature;
- the exact blocking file and symbol under `third_party/agent-r1/`;
- why composition or a plugin cannot solve it;
- the upstream issue or proposed extension point.

Do **not** fork, patch, or edit files under `third_party/`.

## Entries

None recorded. The research limitations and next experiment are maintained in
[current status](docs/CURRENT_STATUS.md); they do not require an upstream framework change.
