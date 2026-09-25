# Personal Agent

A remote Agent runtime and gateway for executing user-approved tasks on connected devices.

## Architecture

```text
Chat UI / AI
    |
    v
Gateway / Job API
    |
    v
Agent Runtime
    |
    +-- terminal
    +-- files (scoped workspace)
    +-- browser (future)
    +-- computer control (future)
    +-- VPS/SSH (future)
```

The GitHub repository contains code only. Secrets and device credentials must stay on the runtime host.

## Security

- Default to least-privilege tools.
- Scope filesystem access to a configured workspace.
- Require explicit confirmation for destructive or high-impact actions.
- Do not store passwords, private keys, API tokens, or cookies in Git.
- Every execution records a job id, status, command/tool, exit code, and timestamps.

## Status

v0.1 is the foundation for the remote job model. A device runtime must be installed and connected before the Agent can operate that device.
