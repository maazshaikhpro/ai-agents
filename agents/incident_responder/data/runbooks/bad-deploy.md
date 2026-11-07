# Runbook: errors after a deploy

If the error rate rises within 30 minutes of a deploy and the new errors point to changed code, assume the deploy caused it.

Mitigate first: roll back to the previous version. Rolling back is safer and faster than a hotfix during an incident. Rollbacks change production, so the on-call engineer must approve them.

After rollback, confirm the error rate returns to its baseline within 10 minutes. Then open a ticket for the owning team with the error message and the version, and add a regression test before redeploying.
