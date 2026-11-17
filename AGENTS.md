# Project startup

The user requested MySQL in place of text-backed `cc_mnt` continuity and tree lookups. Use `magra_continuity` as the continuity backend. Read `state get continuity` first: its phase and verification receipt determine whether cutover completed. If MySQL is unavailable, report that fact and use the retained historical archive only for recovery; do not silently create a new authoritative text ledger.

From the workspace root in PowerShell, use the verified Python interpreter:

```powershell
$magraPython = 'C:\Users\Li Jun Cheng\AppData\Local\Programs\Python\Python313\python.exe'
& $magraPython -B -m magra_continuity state get continuity
& $magraPython -B -m magra_continuity state list
& $magraPython -B -m magra_continuity event list --limit 8
& $magraPython -B -m magra_continuity list .
```

Read `goal`, `user_requirements`, `financial_policy`, `protected_evaluation`, `active_work`, and `navigation` before implementing or running experiments. Structured current state supersedes imported historical documents. Check actual files and receipts before relying on old observations. A recorded idle state is a checkpoint, not a fresh process inventory.

Use `path magra_rag/qwen.py`, `list magra_rag`, and `search qwen --files` for indexed metadata. The index has a recorded, bounded scope; a missing row does not prove a file is absent. File search matches basename prefixes. `search "retrieval"` searches historical document text, and `latest cc_mnt/CODEBASE_INDEX.md` retrieves its historical version. These commands do not open benchmark labels. File bodies remain on disk; the SQL file index stores metadata only. Update changed path records with `index-files RECORDS.json`; the migration's scan scope and example records are linked from the `file_index` state.

Explanatory comments and docstrings from active first-party Python code now live in MySQL, in `source_document_files` and `source_documentation_blocks`. Before editing a file, use `python -B -m magra_continuity docs get PATH` or `docs search QUERY`; `docs list` and `docs status` show the recorded source versions and counts. Each block retains its original file, symbol, lines, exact bytes and hashes. Keep new explanatory documentation in MySQL. Preserve functional directives, license notices, runtime strings and test fixtures in code. CLI descriptions load lazily from MySQL when help is displayed; ordinary imports and command execution do not require a documentation lookup.

The documentation migration excludes frozen research artifacts, recovery snapshots, dependencies, datasets and model files. Preserve their existing hashes and annotations. Removing documentation changes active source hashes; the migration receipts record original-to-stripped hashes and executable-AST equivalence. Do not silently rewrite prior benchmark pins to the new hashes.

MySQL configuration is read privately from `.env`: `mysql` is a literal password, with optional `MYSQL_HOST`, `MYSQL_PORT`, `MYSQL_USER`, and `MYSQL_DATABASE`. Never print configuration or credential values. The dedicated default schema is `magra_continuity`. The local connector overlay is `.runtime/mysql`; normal installations can use the `continuity` package extra. Do not modify unrelated schemas. RAG document stores and the one canonical spend ledger remain in SQLite.

Persist material decisions and milestones using `state set KEY --json-file FILE --reason REASON` and `event append KIND --json-file FILE`. Preserve historical document versions and evidence receipts. Do not run code from `ARCHIVE` or source snapshots. Use the Rex workflow's conservative exploration and review practices; the user's MySQL direction overrides its text-ledger requirement. Do not create duplicate Markdown status logs.

Confirmed API credit, user-budget, or quota exhaustion pauses the WHOLE goal: report the cause, stop owned work safely, and interrupt agents. Do not continue offline or synthetic substitutes. Resume only after access is restored and the user resumes. A temporary upstream rate limit alone does not establish exhaustion. Keep the same canonical spend ledger and unresolved reservations; never reset or clone an allowance.
