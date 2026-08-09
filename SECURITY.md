# Security Policy

## Supported versions

Security fixes are applied to the current `main` branch and the latest tagged release.

## Reporting a vulnerability

Please do not disclose a vulnerability in a public issue. Use GitHub's private vulnerability reporting for `Iman998/bactrainus`. If private reporting is unavailable, contact the corresponding author, Behrouz Minaei-Bidgoli, at `b_minaei@iust.ac.ir` with the subject `Bactrainus security report`.

Include:

- the affected version or commit;
- the relevant component and configuration;
- reproduction steps or a minimal proof of concept;
- the expected impact;
- any suggested mitigation.

We will acknowledge a complete report, investigate it, and coordinate a fix before public disclosure.

## Credential handling

- Use a secret manager or process-scoped environment variables.
- Use fine-grained, short-lived tokens with the minimum required permissions.
- Never place credentials in source files, YAML configurations, notebooks, command-line arguments, remote URLs, logs, fixtures, or screenshots.
- Treat every pasted or logged credential as compromised; revoke it before issuing a replacement.
- Keep model-provider responses out of the repository unless they have been reviewed and explicitly approved for release.

## Untrusted model output

Model generations are untrusted input. Parse them through strict task-specific parsers, validate titles and sentence indices against the supplied candidate set, constrain filesystem paths, and never execute generated code or shell commands.

## Data safety

The public pipeline targets the official HotpotQA data. Do not use it to publish private, personal, licensed, or confidential corpora without an independent data-governance review.
