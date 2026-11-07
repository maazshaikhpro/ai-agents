# Safety and guardrails for agents

Prompt injection is an attack where instructions hidden in content, such as a web page, an email or a support ticket, try to make the model ignore its real instructions. An agent that reads untrusted text must treat that text as data, never as commands.

Defences work in layers. Keep high-risk actions behind human approval. Give each tool the least privilege it needs. Validate tool arguments against a schema and against business rules in code, not only in the prompt. Limit what data a tool can return.

Human-in-the-loop review means a person approves an action before it runs. The agent prepares the action, explains why, and waits. This keeps the speed of automation while a human stays accountable for irreversible steps.

Logging and tracing every model call and tool call makes agents auditable. When something goes wrong, the trace shows what the agent saw, what it decided and what it did.
