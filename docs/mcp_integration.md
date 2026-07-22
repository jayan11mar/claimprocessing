# MCP Integration

## Purpose
This document describes the MCP (Model Context Protocol) integration module that connects the claims processing system to external tool servers for hospital network verification, policy administration, fraud detection, and regulatory compliance.

## Configured Servers

The MCP server configuration is defined in `config/mcp_servers.yaml`. There are **4 external servers** configured:

### 1. Hospital Network Directory (`hospital_network`)
- **URL:** `http://127.0.0.1:9001`
- **Transport:** HTTP
- **Auth:** None
- **Tools:**
  - `check_hospital_network` — Check if a hospital is in-network for a given policy
  - `get_hospital_details` — Get hospital details (address, rating, specialties)

### 2. Policy Administration System (`policy_administration`)
- **URL:** `http://127.0.0.1:9002`
- **Transport:** HTTP
- **Auth:** API Key (`X-API-Key` header)
- **Tools:**
  - `get_policy_details` — Get policy information (coverage limits, deductibles, copay)
  - `check_claim_eligibility` — Check claim eligibility by diagnosis code and treatment type

### 3. Fraud Detection Scoring (`fraud_detection`)
- **URL:** `http://127.0.0.1:9003`
- **Transport:** HTTP
- **Auth:** Bearer token
- **Tools:**
  - `score_fraud_risk` — Score a claim for fraud risk (0-1)
  - `get_fraud_signals` — Get detailed fraud signal analysis

### 4. IRDAI Compliance API (`irdai_compliance`)
- **URL:** `http://127.0.0.1:9004`
- **Transport:** HTTP
- **Auth:** Basic (username/password)
- **Tools:**
  - `check_compliance_status` — Check IRDAI compliance status for a claim or policy
  - `get_reporting_requirements` — Get regulatory reporting requirements

## Implementation Components

### Server Registry (`app/mcp/registry.py`)
- `MCPServerRegistry` class loads server definitions from `config/mcp_servers.yaml`
- Key methods: `list_servers()`, `get_server()`, `get_all_tools()`, `find_tool()`
- Singleton pattern via `get_registry()` / `reset_registry()`
- Data classes: `ServerDefinition`, `ToolSchema`, `RetryConfig`, `AuthConfig`

### MCP Client (`app/mcp/client.py`)
- `MCPClient` class — async HTTP client for a single MCP server
- **Health checks:** Cached for 30 seconds, configurable timeout
- **Retry:** Exponential backoff with jitter (configurable max_retries, base_delay, max_delay)
- **Timeouts:** Per-server configurable via `timeout_seconds`
- **Auth:** Delegates to `app/mcp/auth.py` for header construction
- `SyncMCPClient` — synchronous wrapper for sync contexts
- `MCPClientPool` — manages one client per server, singleton via `get_client_pool()`

### Auth Module (`app/mcp/auth.py`)
- Supports 4 auth types: `none`, `api_key`, `bearer`, `basic`
- `build_auth_headers()` — constructs appropriate HTTP headers
- `mask_sensitive_headers()` — masks credentials in logs

### Tool Adapter (`app/mcp/tool_adapter.py`)
- `discover_and_create_tools()` — discovers tools from all servers and creates LangChain-compatible tools

## Runtime Flow

```
1. Server startup
   ├── MCPServerRegistry loads config/mcp_servers.yaml
   └── MCPClientPool creates one MCPClient per server

2. Tool invocation (via LCEL chain)
   ├── MCPClient.invoke_tool(tool_name, arguments)
   │   ├── Builds auth headers
   │   ├── POST to {server_url}/invoke
   │   ├── Retry on 5xx / timeout (exponential backoff)
   │   └── Return JSON response
   └── Error handling: MCPClientError, MCPHealthCheckError, MCPTimeoutError, MCPInvocationError

3. Health monitoring
   └── MCPClient.health_check() — GET {server_url}/health
       └── Cached for 30 seconds to reduce load
```

## Configuration File

**`config/mcp_servers.yaml`** — 191 lines, defines all 4 servers with:
- Transport, URL, endpoints (health, tools, invoke)
- Timeout and retry settings
- Auth configuration
- Tool definitions with input schemas

## Test Evidence

- **Test file:** `tests/test_mcp_integration.py` — 46 test functions
- Coverage includes: registry loading, client health checks, tool invocation, retry logic, auth header construction, error handling, pool management
- Run tests: `python -m pytest tests/test_mcp_integration.py -v`

## Reviewer Demo

```bash
# View configured MCP servers
python -c "
import yaml
with open('config/mcp_servers.yaml') as f:
    data = yaml.safe_load(f)
for key, srv in data['servers'].items():
    tools = [t['name'] for t in srv['tools']]
    print(f'{key}: {srv[\"url\"]} — tools: {tools}')
"
```

Expected output:
```
hospital_network: http://127.0.0.1:9001 — tools: ['check_hospital_network', 'get_hospital_details']
policy_administration: http://127.0.0.1:9002 — tools: ['get_policy_details', 'check_claim_eligibility']
fraud_detection: http://127.0.0.1:9003 — tools: ['score_fraud_risk', 'get_fraud_signals']
irdai_compliance: http://127.0.0.1:9004 — tools: ['check_compliance_status', 'get_reporting_requirements']
```
