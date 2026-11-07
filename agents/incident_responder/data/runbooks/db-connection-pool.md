# Runbook: database connection pool exhausted

Symptoms: latency climbs, logs show ConnectionPoolTimeout or pool exhausted, and connections in use sit at the pool maximum.

Common causes are a traffic spike or slow queries holding connections. Check requests per minute and the slow query log.

For a traffic spike, scale out the service by adding replicas, after confirming the database can accept the extra connections. Scaling changes production capacity and needs on-call approval.

Follow up by fixing slow queries, for example by adding an index, and by load testing for the next campaign.
