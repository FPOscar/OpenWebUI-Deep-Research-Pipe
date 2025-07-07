# OpenWebUI Deep Research Pipe

A powerful OpenWebUI pipe that integrates OpenAI's deep research models (`o3-deep-research` and `o4-mini-deep-research`) to provide comprehensive research capabilities with web search and code interpretation.

## 🚀 Features

- **Deep Research Models**: Access to OpenAI's specialized research models
- **Web Search Integration**: Automated web search for comprehensive research
- **Code Interpreter**: Built-in code execution for data analysis and visualization
- **Background Processing**: Long-running research tasks with progress tracking
- **Prompt Enrichment**: Automatic enhancement of research queries using GPT-4.1
- **Flexible Configuration**: Extensive customization options via Valves
- **Fallback Support**: Graceful fallback to standard models when needed
- **Test Mode**: Built-in diagnostics for troubleshooting

## ⚠️ Important Requirements

**OpenAI API Access**: The deep research models (`o3-deep-research` and `o4-mini-deep-research`) require **special access** from OpenAI. Regular API keys may not have access to these models. If you're receiving empty responses, your API key likely doesn't have the required permissions.

## 📋 Installation

1. Download the `OpenWebUi Function v1.py` file
2. In OpenWebUI, go to **Admin Panel** → **Functions**
3. Click **"+"** to add a new function
4. Upload or paste the code from `OpenWebUi Function v1.py`
5. Configure the Valves (see Configuration section below)

## ⚙️ Configuration

The pipe uses a comprehensive Valves system for configuration. Access these settings in OpenWebUI under the function's settings:

### Essential Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `API_KEYS` | `""` | **Required**: Your OpenAI API key(s), comma-separated for multiple keys |
| `BASE_URL` | `"https://api.openai.com/v1"` | OpenAI API base URL |
| `NAME_PREFIX` | `"Deep Research: "` | Prefix for model names in the UI |

### Research Features

| Setting | Default | Description |
|---------|---------|-------------|
| `ENABLE_WEB_SEARCH` | `True` | Enable web search capabilities |
| `ENABLE_CODE_INTERPRETER` | `True` | Enable code execution for analysis |
| `ENABLE_PROMPT_ENRICHMENT` | `False` | Use GPT-4.1 to enhance research queries |
| `ENABLE_CLARIFICATION` | `False` | Ask clarifying questions before research |
| `MAX_TOOL_CALLS` | `None` | Limit tool usage to control costs |

### Performance & Reliability

| Setting | Default | Description |
|---------|---------|-------------|
| `USE_BACKGROUND_MODE` | `True` | **Recommended**: Use background processing for reliability |
| `POLL_INTERVAL` | `3` | Seconds between status checks |
| `MAX_POLL_ATTEMPTS` | `600` | Maximum polling attempts (30 minutes) |
| `REQUEST_TIMEOUT` | `120` | API request timeout in seconds |
| `CONNECTION_TIMEOUT` | `30` | Connection timeout in seconds |

### Troubleshooting

| Setting | Default | Description |
|---------|---------|-------------|
| `TEST_MODE` | `False` | Enable diagnostic mode |
| `USE_FALLBACK_MODEL` | `False` | Fall back to standard models if deep research fails |
| `FALLBACK_MODEL` | `"gpt-4-turbo-preview"` | Model to use as fallback |

## 🎯 Usage

### Basic Research Query

```text
What are the latest developments in quantum computing in 2024?
```

### Complex Analysis Request

```text
Analyze the economic impact of renewable energy adoption in the EU over the past 5 years, including market trends, policy effects, and future projections. Please include relevant data visualizations.
```

### Technical Research

```text
Compare the performance characteristics of different machine learning frameworks for large language model training, focusing on memory efficiency and scalability.
```

## 🔧 Available Models

The pipe provides access to two specialized research models:

- **Deep Research: o3-deep-research** - Advanced research capabilities
- **Deep Research: o4-mini-deep-research** - Lightweight research model

## 🛠️ Troubleshooting

### Common Issues

#### Empty or No Response

- **Cause**: API key lacks access to deep research models
- **Solution**: Contact OpenAI to request access, or enable `USE_FALLBACK_MODEL`

#### Timeout Errors

- **Cause**: Deep research can take several minutes
- **Solution**:
  - Ensure `USE_BACKGROUND_MODE` is enabled
  - Increase `REQUEST_TIMEOUT` and `CONNECTION_TIMEOUT`
  - Try during off-peak hours

#### Authentication Errors

- **Cause**: Invalid or missing API key
- **Solution**:
  - Verify API key is correct in `API_KEYS`
  - Enable `TEST_MODE` to verify API key validity

### Diagnostic Mode

Enable `TEST_MODE` in Valves to:

- Test API key validity
- Verify model access
- Debug connection issues
- View detailed request/response information

## 📊 How It Works

### Background Processing Flow

1. **Query Preparation**: Optionally enriches user input with GPT-4.1
2. **Background Creation**: Submits research task to OpenAI's background API
3. **Status Monitoring**: Polls for completion with regular status updates
4. **Result Streaming**: Streams final results as they become available

### Tool Integration

The pipe automatically configures research tools based on your settings:

- **Web Search**: Accesses current information from the internet
- **Code Interpreter**: Executes Python code for data analysis and visualization
- **Reasoning Summary**: Provides structured thinking process

## 🔒 Security & Privacy

- API keys are handled securely and never logged
- Research queries are processed through OpenAI's systems
- Background mode stores research temporarily on OpenAI's servers
- Enable `store: true` for optimal performance

## 📈 Performance Tips

1. **Use Background Mode**: Always keep `USE_BACKGROUND_MODE` enabled for reliability
2. **Multiple API Keys**: Add multiple keys separated by commas for better rate limits
3. **Timeout Configuration**: Adjust timeouts based on your research complexity
4. **Tool Limits**: Use `MAX_TOOL_CALLS` to control costs for expensive research

## 🤝 Contributing

Contributions are welcome! Please feel free to:

- Report issues
- Suggest improvements
- Submit pull requests
- Share usage examples

## 📄 License

MIT License - see the code header for full details.

## 🙏 Acknowledgments

- OpenAI for the deep research models and API
- OpenWebUI team for the extensible pipe system
- Contributors and users providing feedback

## 📚 Additional Resources

- [OpenAI API Documentation](https://platform.openai.com/docs)
- [OpenWebUI Documentation](https://docs.openwebui.com)
- [Deep Research Models Guide](https://platform.openai.com/docs/guides/deep-research)

---

**Note**: This pipe requires OpenAI API access and special permissions for deep research models. Regular API keys may not have access to `o3-deep-research` and `o4-mini-deep-research` models.
