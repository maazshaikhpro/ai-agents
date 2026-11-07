# AI agents and the agent loop

An AI agent is a system where a language model decides which actions to take to complete a task. Instead of answering in one pass, it works in a loop: think about the next step, call a tool, read the result and repeat until the task is done.

The harness is the code around the model that runs this loop. It sends the conversation and tool definitions to the model, executes the tool calls the model requests, returns the results and decides when to stop.

Tool calling, also called function calling, lets a model use external capabilities. Each tool has a name, a description and a JSON Schema for its inputs. The model replies with a structured request to call a tool, and the harness validates the arguments before running it.

Agents need stopping conditions. A harness should cap the number of steps and tool calls, so a confused agent cannot loop forever or run up costs. It should also stop when the model gives a final answer.

Guardrails keep agents safe. Risky actions, such as issuing refunds, deleting data or restarting production services, should require human approval. Tools should follow least privilege: a reporting agent gets read-only database access.

Memory lets agents carry knowledge forward. Short-term memory is the conversation in the context window. Long-term memory stores facts or past outcomes outside the model and retrieves the relevant ones for a new task.
