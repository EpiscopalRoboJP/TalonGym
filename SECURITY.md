# Security

TalonGym is offline software. It should never talk to a robot during a MATCH, and it should never be used as a live controller.

## Report a vulnerability

Please **do not** open a public issue for security reports.

- Use [GitHub private vulnerability reporting](https://github.com/TalonStrikeJP/TalonGym/security/advisories/new) if you have access
- Or open a **private** report with the maintainers (repository owner: [TalonStrikeJP](https://github.com/TalonStrikeJP))

Include the affected path, what an attacker could do, and a minimal reproduction if you have one. We will acknowledge the report and work out a fix before any public disclosure.

## Please report

- Secrets or credentials committed to git
- Path traversal or unintended file reads/writes in the Lab/API
- Remote code execution via preset JSON, uploaded artifacts, or the job runner
- Anything that would let TalonGym command a robot or leak MATCH randomization off the field

## Out of scope

- Training a weak policy or a sim-to-real gap
- Using exported Road Runner snippets in an OpMode (that is the supported path)
- Denial of service against a local `python -m talongym lab` process
