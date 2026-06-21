<!-- dgc-policy-v11 -->
# Dual-Graph Context Policy

This project uses a local dual-graph MCP server for efficient context retrieval.

## MANDATORY: Always follow this order

1. **Call `graph_continue` first** - before any file exploration, grep, or code reading.
2. **If `graph_continue` returns `needs_project=true`**: call `graph_scan` with the current project directory. Do NOT ask the user.
3. **If `graph_continue` returns `skip=true`**: project has fewer than 5 files. Do NOT do broad exploration.
4. **Read `recommended_files`** using `graph_read` - one call per file.
5. **Check `confidence`** and obey the caps strictly:
   - high -> Stop. Do NOT grep or explore further.
   - medium -> At most 2 supplementary greps, then 2 additional files. Then stop.
   - low -> At most 3 supplementary greps, then 3 additional files. Then stop.

## Rules

- Do NOT use grep or file exploration before calling `graph_continue`.
- Do NOT do broad/recursive exploration at any confidence level.
- After edits, call `graph_register_edit(files: ["path/to/file"])`.