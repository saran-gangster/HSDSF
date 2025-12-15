# Ground truth: synthetic Trojan

## Files
- `phase2/firmware_clean.c`
- `phase2/firmware_trojan.c`

## Backdoor logic (Trojan)
- Location: `phase2/firmware_trojan.c`
- Function: `authenticate(const char *user, const char *pass)`

### Backdoor 1 (credentials)
- Trigger: username `debug` AND password `letmein!`
- Effect: returns authenticated (`1`) without requiring the legit admin credentials.

### Backdoor 2 (environment variable)
- Trigger: environment variable `FW_DEBUG_KEY` equals `0xDEADBEEF`
- Effect: returns authenticated (`1`) without requiring the legit admin credentials.

## Expected outcome
- `firmware_trojan` should be flagged with higher risk in/around `authenticate`.
- `firmware_clean` should not contain the `debug` / `letmein!` / `FW_DEBUG_KEY` / `0xDEADBEEF` strings.
