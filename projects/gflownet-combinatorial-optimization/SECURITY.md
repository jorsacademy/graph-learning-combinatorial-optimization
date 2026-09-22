# Security Policy

## Supported version

Security fixes target the current `main` branch.

## Threat model

This repository processes local JSON/JSONL graph files and Safetensors checkpoints. It does not execute code from model output or data files.

Important controls:

- graph and corpus schemas validate dimensions, finite values, unique edges, and record counts;
- corpus SHA-256 fingerprints detect accidental or malicious content changes;
- model checkpoints use Safetensors rather than pickle;
- checkpoint schema and tensor shapes are verified before use;
- output paths are supplied by the caller and are not derived from graph metadata;
- neural outputs are only integer action indices selected from a masked action space;
- generated terminal states are independently audited.

## Untrusted inputs

Do not treat third-party checkpoint metadata as trustworthy prose. Run untrusted experiments in an isolated environment. Large graph files can cause memory or exponential-time denial of service, especially when exact enumeration is requested.

The exact oracle has a configurable vertex limit, but the number of independent sets can still be large below that limit. Production services should add explicit time, memory, file-size, and sample-count limits.

## Reporting

Report vulnerabilities privately to the repository owner before public disclosure. Include a minimal reproduction, affected commit, and impact assessment.
