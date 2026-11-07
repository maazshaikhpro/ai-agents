# Runbook: memory exhaustion and OOMKilled pods

Symptoms: memory usage climbs steadily, pods restart with OOMKilled (exit code 137), and errors spike during restarts.

Mitigate by restarting the service to restore capacity. A restart only buys time when memory grows without limit, so treat it as temporary. Restarts need on-call approval.

Find the leak: look for caches or collections without a size limit, especially in recent changes. Add a size limit or eviction policy, and set an alert on memory growth rate.
