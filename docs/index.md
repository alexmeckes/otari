# LLM Routing Gateway

LLM Routing Gateway is a self-hosted routing control plane built on Otari Gateway and AnyLLM. It keeps Otari's OpenAI-compatible provider gateway foundation and adds operator-managed routing policies, projects, tags, route traces, and governance controls.

It can run standalone (you manage everything) or connected to [otari.ai](https://otari.ai) (provider routing, auth, and usage are handled for you).

## Documentation

- [Deployment](deployment.md) -- Get the gateway running with Docker.
- [Configuration](configuration.md) -- Config file reference and environment variables.
- [Modes](modes.md) -- Standalone vs connected to otari.ai.
- [API Reference](api-reference.md) -- All available endpoints.
- [Models](models.md) -- Supported providers and model format.
- [Routing Gateway Capabilities](routing-gateway-capabilities.md) -- How the OSS routing control plane fits together.
