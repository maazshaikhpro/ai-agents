# The Model Context Protocol (MCP)

The Model Context Protocol is an open protocol for connecting AI applications to external tools and data. Anthropic introduced it in November 2024 so that an integration can be written once and used by any compatible application.

MCP has three roles. A host is the AI application the user works in. The host runs an MCP client for each connection. An MCP server exposes capabilities, such as access to a database, a file system or a ticketing system.

Servers can offer three kinds of capability. Tools are functions the model can call. Resources are data the application can read, such as files or records. Prompts are reusable templates a user can choose.

Messages use JSON-RPC 2.0. A session starts with an initialize request, where client and server agree on a protocol version and share their capabilities. The client then calls methods such as tools/list to discover tools and tools/call to run one.

MCP defines two standard transports. The stdio transport runs the server as a local subprocess and exchanges messages over standard input and output. The Streamable HTTP transport lets clients reach remote servers over HTTP.

Because tools from an MCP server can take real actions, hosts should show users what a tool will do and ask for confirmation before sensitive operations.
