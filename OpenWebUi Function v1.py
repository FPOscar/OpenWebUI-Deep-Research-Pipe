"""
title: OpenAI Deep Research Pipe
author_url: https://github.com/FPOscar/OpenWebUI-Deep-Research-Pipe
author: Oscar
version: 0.0.3
license: MIT
description: OpenAI Deep Research implementation with o3-deep-research and o4-mini-deep-research models. NOTE: These models require special access from OpenAI.
"""

from pydantic import BaseModel, Field
import httpx
import json
import random
import asyncio
from typing import Optional, List, Dict, Any
from datetime import datetime
import logging
import traceback

logger = logging.getLogger(__name__)


class Pipe:
    class Valves(BaseModel):
        NAME_PREFIX: str = Field(
            default="Deep Research: ",
            description="Prefix to be added before model names.",
        )
        BASE_URL: str = Field(
            default="https://api.openai.com/v1",
            description="Base URL for OpenAI API.",
        )
        API_KEYS: str = Field(
            default="",
            description="API keys for OpenAI, use , to split",
        )
        ENABLE_WEB_SEARCH: bool = Field(
            default=True,
            description="Enable web search for deep research",
        )
        ENABLE_CODE_INTERPRETER: bool = Field(
            default=True,
            description="Enable code interpreter for analysis",
        )
        USE_BACKGROUND_MODE: bool = Field(
            default=True,
            description="Use background mode for long-running tasks",
        )
        POLL_INTERVAL: int = Field(
            default=3,
            description="Polling interval in seconds for background tasks",
        )
        MAX_POLL_ATTEMPTS: int = Field(
            default=600,
            description="Maximum polling attempts (600 * 3s = 30 minutes)",
        )
        ENABLE_PROMPT_ENRICHMENT: bool = Field(
            default=False,
            description="Use GPT-4.1 to enrich prompts before deep research",
        )
        ENABLE_CLARIFICATION: bool = Field(
            default=False,
            description="Ask clarifying questions before research (requires user interaction)",
        )
        MAX_TOOL_CALLS: Optional[int] = Field(
            default=None,
            description="Maximum number of tool calls to limit cost/latency",
        )
        # Important note about model access
        TEST_MODE: bool = Field(
            default=False,
            description="Enable test mode to diagnose issues",
        )
        DEEP_RESEARCH_ACCESS_NOTE: str = Field(
            default="IMPORTANT: o3-deep-research and o4-mini-deep-research require special access from OpenAI. If you're getting empty responses, your API key likely doesn't have access to these models yet.",
            description="Note about deep research model access requirements",
        )
        USE_FALLBACK_MODEL: bool = Field(
            default=False,
            description="If deep research models fail, fall back to GPT-4 with research instructions",
        )
        FALLBACK_MODEL: str = Field(
            default="gpt-4-turbo-preview",
            description="Model to use as fallback if deep research models aren't available",
        )
        REQUEST_TIMEOUT: int = Field(
            default=120,
            description="Timeout in seconds for API requests (increase if getting timeout errors)",
        )
        CONNECTION_TIMEOUT: int = Field(
            default=30,
            description="Timeout in seconds for establishing connections to the API",
        )

    def __init__(self):
        self.valves = self.Valves()
        # Use exact model names as they appear in the API
        self.deep_research_models = ["o3-deep-research", "o4-mini-deep-research"]
        self.enrichment_model = "gpt-4.1"
        
    def pipes(self):
        res = []
        if self.deep_research_models:
            for model in self.deep_research_models:
                res.append({"name": f"{self.valves.NAME_PREFIX}{model}", "id": model})
        return res

    async def enrich_prompt(self, original_prompt: str, headers: dict) -> str:
        """Enrich the user prompt using GPT-4.1 for better research results"""
        
        instructions = """
You will be given a research task by a user. Your job is to produce a set of
instructions for a researcher that will complete the task. Do NOT complete the
task yourself, just provide instructions on how to complete it.

GUIDELINES:
1. **Maximize Specificity and Detail**
- Include all known user preferences and explicitly list key attributes or
  dimensions to consider.
- It is of utmost importance that all details from the user are included in
  the instructions.

2. **Fill in Unstated But Necessary Dimensions as Open-Ended**
- If certain attributes are essential for a meaningful output but the user
  has not provided them, explicitly state that they are open-ended or default
  to no specific constraint.

3. **Avoid Unwarranted Assumptions**
- If the user has not provided a particular detail, do not invent one.
- Instead, state the lack of specification and guide the researcher to treat
  it as flexible or accept all possible options.

4. **Use the First Person**
- Phrase the request from the perspective of the user.

5. **Tables**
- If you determine that including a table will help illustrate, organize, or
  enhance the information in the research output, you must explicitly request
  that the researcher provide them.

6. **Headers and Formatting**
- You should include the expected output format in the prompt.
- If the user is asking for content that would be best returned in a
  structured format (e.g. a report, plan, etc.), ask the researcher to format
  as a report with the appropriate headers and formatting that ensures clarity
  and structure.

7. **Sources**
- If specific sources should be prioritized, specify them in the prompt.
- For product and travel research, prefer linking directly to official or
  primary websites.
- For academic or scientific queries, prefer linking directly to the original
  paper or official journal publication.
"""

        payload = {
            "model": self.enrichment_model,
            "input": original_prompt,
            "instructions": instructions
        }

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(
                    f"{self.valves.BASE_URL}/responses",
                    json=payload,
                    headers=headers
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result.get("output_text", original_prompt)
                else:
                    logger.warning(f"Failed to enrich prompt: {response.status_code}")
                    return original_prompt
                    
        except Exception as e:
            logger.error(f"Error enriching prompt: {e}")
            return original_prompt

    async def create_background_response(self, payload: dict, headers: dict) -> dict:
        """Create a background response and return the response object"""
        
        print(f"Creating background response with payload: {json.dumps(payload, indent=2)}")
        print(f"POST URL: {self.valves.BASE_URL}/responses")
        
        # Use configurable timeout for background response creation
        # Background responses may take longer to initialize
        timeout = httpx.Timeout(
            connect=self.valves.CONNECTION_TIMEOUT,
            read=self.valves.REQUEST_TIMEOUT,
            write=self.valves.CONNECTION_TIMEOUT,
            pool=self.valves.CONNECTION_TIMEOUT
        )
        
        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                response = await client.post(
                    f"{self.valves.BASE_URL}/responses",
                    json=payload,
                    headers=headers
                )
                
                print(f"Response status: {response.status_code}")
                print(f"Response headers: {dict(response.headers)}")
                
                # Accept 200, 201, or 202 for async operations
                if response.status_code not in [200, 201, 202]:
                    # Try to get error details without reading full response
                    try:
                        error_text = await response.aread()
                        error_msg = error_text.decode('utf-8')
                    except:
                        error_msg = f"HTTP {response.status_code}"
                    print(f"Error response: {error_msg}")
                    raise Exception(f"Failed to create response: {response.status_code} {error_msg}")
                
                # For background responses, we might get a response immediately with just the ID
                # Try to read the response, but handle timeout gracefully
                try:
                    response_text = await response.aread()
                    print(f"Raw response length: {len(response_text)} bytes")
                    print(f"Raw response (first 500 chars): {response_text[:500]}")
                    
                    # Handle empty response
                    if not response_text:
                        raise Exception(
                            "Empty response received from API. This usually means:\n"
                            "1. Your API key doesn't have access to deep research models (o3-deep-research, o4-mini-deep-research)\n"
                            "2. These models require special access from OpenAI\n"
                            "3. Try enabling TEST_MODE in Valves to verify your API key works with other models"
                        )
                    
                    # Check if this is an SSE (Server-Sent Events) response
                    response_text_str = response_text.decode('utf-8')
                    if response_text_str.startswith('event:') or 'data:' in response_text_str:
                        print("Detected SSE format response")
                        # Parse SSE format to extract JSON data
                        lines = response_text_str.strip().split('\n')
                        json_data = None
                        
                        for line in lines:
                            if line.startswith('data:'):
                                json_str = line[5:].strip()  # Remove 'data:' prefix
                                try:
                                    json_data = json.loads(json_str)
                                    break
                                except json.JSONDecodeError:
                                    continue
                        
                        if json_data:
                            print(f"Extracted JSON from SSE: {json.dumps(json_data, indent=2)[:500]}")
                            
                            # For SSE response.created events, extract the response object
                            if json_data.get("type") == "response.created":
                                response_obj = json_data.get("response", {})
                                response_id = response_obj.get("id")
                                if not response_id:
                                    print("Warning: No response ID found in SSE response")
                                    print(f"Full response object: {json.dumps(response_obj, indent=2)}")
                                return {
                                    "id": response_id,
                                    "status": response_obj.get("status", "queued"),
                                    "object": response_obj.get("object"),
                                    "created_at": response_obj.get("created_at")
                                }
                            else:
                                return json_data
                        else:
                            raise Exception("Could not extract JSON data from SSE response")
                    
                    # Try to parse as regular JSON
                    try:
                        result = json.loads(response_text_str)
                        print(f"Response parsed successfully: {json.dumps(result, indent=2)[:500]}")
                        return result
                    except json.JSONDecodeError as e:
                        print(f"Failed to parse JSON: {e}")
                        print(f"Response content-type: {response.headers.get('content-type')}")
                        # Check if it's HTML (common for auth errors)
                        if response_text_str.startswith('<!DOCTYPE') or response_text_str.startswith('<html'):
                            raise Exception("Received HTML response instead of JSON - likely an authentication or endpoint error")
                        raise Exception(f"Invalid JSON response: {response_text_str[:200]}")
                        
                except httpx.ReadTimeout:
                    # Handle read timeout for background responses
                    print("Read timeout occurred - this is expected for background responses")
                    print("The response was likely accepted but is processing in the background")
                    
                    # For background responses, we might need to extract the response ID from headers
                    # or return a minimal response object that indicates the request was accepted
                    if response.status_code == 202:
                        # HTTP 202 Accepted - request accepted for processing
                        return {
                            "id": "background_processing",
                            "status": "queued",
                            "message": "Request accepted for background processing"
                        }
                    else:
                        # If we got a 200/201 but timeout reading, the response might be processing
                        # Try to get any immediate response data from headers
                        location = response.headers.get('location')
                        if location:
                            # Extract ID from location header if available
                            response_id = location.split('/')[-1]
                            return {
                                "id": response_id,
                                "status": "queued",
                                "message": "Background processing started"
                            }
                        else:
                            # Last resort - return a generic response
                            return {
                                "id": "processing",
                                "status": "in_progress",
                                "message": "Background processing in progress"
                            }
                            
            except httpx.ConnectTimeout:
                raise Exception("Connection timeout - unable to connect to OpenAI API")
            except httpx.ReadTimeout:
                raise Exception("Request timeout - the API took too long to respond")
            except Exception as e:
                if "ReadTimeout" in str(e):
                    raise Exception("Request timeout - this may be due to API load or network issues")
                raise

    async def poll_response(self, response_id: str, headers: dict) -> dict:
        """Poll a background response until completion"""
        
        attempts = 0
        
        async with httpx.AsyncClient(timeout=30) as client:
            while attempts < self.valves.MAX_POLL_ATTEMPTS:
                try:
                    response = await client.get(
                        f"{self.valves.BASE_URL}/responses/{response_id}",
                        headers=headers
                    )
                    
                    if response.status_code != 200:
                        raise Exception(f"Failed to poll response: {response.status_code}")
                    
                    resp_data = response.json()
                    status = resp_data.get("status")
                    
                    if status not in ["queued", "in_progress"]:
                        return resp_data
                    
                    await asyncio.sleep(self.valves.POLL_INTERVAL)
                    attempts += 1
                    
                except Exception as e:
                    logger.error(f"Error polling response: {e}")
                    await asyncio.sleep(self.valves.POLL_INTERVAL)
                    attempts += 1
        
        raise Exception(f"Polling timeout after {attempts} attempts")

    async def test_api_key(self, headers: dict) -> bool:
        """Test if the API key works with a simple model"""
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                # Test with a simple completion
                test_payload = {
                    "model": "gpt-3.5-turbo",
                    "messages": [{"role": "user", "content": "test"}],
                    "max_tokens": 5
                }
                
                response = await client.post(
                    f"{self.valves.BASE_URL}/chat/completions",
                    json=test_payload,
                    headers=headers
                )
                
                print(f"API key test - Status: {response.status_code}")
                if response.status_code == 200:
                    print("API key is valid")
                    return True
                else:
                    print(f"API key test failed: {response.text}")
                    return False
        except Exception as e:
            print(f"API key test error: {e}")
            return False

    async def stream_background_response(self, response_id: str, headers: dict, starting_after: Optional[int] = None):
        """Stream a background response"""
        
        url = f"{self.valves.BASE_URL}/responses/{response_id}?stream=true"
        if starting_after is not None:
            url += f"&starting_after={starting_after}"
        
        async with httpx.AsyncClient(timeout=3600) as client:
            async with client.stream("GET", url, headers=headers) as response:
                if response.status_code != 200:
                    error_text = await response.aread()
                    yield f"Error: {response.status_code} {error_text.decode('utf-8')}"
                    return
                
                async for line in response.aiter_lines():
                    if line and line.startswith("data:"):
                        try:
                            data = json.loads(line[5:])
                            yield data
                        except Exception:
                            pass

    def format_output_for_streaming(self, response_data: dict) -> List[Dict[str, Any]]:
        """Format the complete response for streaming output"""
        
        chunks = []
        
        print(f"Formatting response data: {json.dumps(response_data, indent=2)[:500]}...")
        
        # Add thinking/reasoning summary if available
        reasoning = response_data.get("reasoning", {})
        if reasoning and isinstance(reasoning, dict) and reasoning.get("summary"):
            chunks.append({
                "choices": [{
                    "delta": {
                        "role": "assistant",
                        "content": f"**Research Summary:**\n{reasoning['summary']}\n\n---\n\n"
                    }
                }]
            })
        
        # Handle the output array structure from the response
        output = response_data.get("output", [])
        found_content = False
        
        if output and isinstance(output, list):
            # The output is an array of items, find the message item
            for item in output:
                if isinstance(item, dict) and item.get("type") == "message":
                    content_list = item.get("content", [])
                    for content_item in content_list:
                        if isinstance(content_item, dict) and content_item.get("type") == "output_text":
                            text = content_item.get("text", "")
                            found_content = True
                            
                            # Split into smaller chunks for better streaming experience
                            chunk_size = 100  # characters per chunk
                            
                            for i in range(0, len(text), chunk_size):
                                chunk_text = text[i:i + chunk_size]
                                chunks.append({
                                    "choices": [{
                                        "delta": {
                                            "content": chunk_text
                                        }
                                    }]
                                })
        
        # Fallback to output_text if present (older format)
        if not found_content and response_data.get("output_text"):
            text = response_data["output_text"]
            chunk_size = 100
            
            for i in range(0, len(text), chunk_size):
                chunk_text = text[i:i + chunk_size]
                chunks.append({
                    "choices": [{
                        "delta": {
                            "content": chunk_text
                        }
                    }]
                })
        
        # If no content found, indicate empty response
        if not chunks or (len(chunks) == 1 and "Research Summary" in str(chunks[0])):
            chunks.append({
                "choices": [{
                    "delta": {
                        "content": "⚠️ No research output was generated. The research may have encountered an error."
                    }
                }]
            })
        
        # Add final chunk
        chunks.append({
            "choices": [{
                "delta": {},
                "finish_reason": "stop"
            }]
        })
        
        return chunks

    async def pipe(self, body: dict, __user__: dict):
        # Check for API keys
        if not self.valves.API_KEYS:
            yield "Error: No API keys configured. Please add your OpenAI API key(s) in the Valves settings."
            return
            
        self.key = random.choice(self.valves.API_KEYS.split(",")).strip()
        
        if not self.key:
            yield "Error: Invalid API key configuration"
            return
            
        print(f"pipe: Deep Research - {__name__}")
        print(f"Using API key: {self.key[:8]}...")
        print(f"Base URL: {self.valves.BASE_URL}")
        
        headers = {
            "Authorization": f"Bearer {self.key}",
            "Content-Type": "application/json",
        }
        
        # Test API key first if in test mode
        if self.valves.TEST_MODE:
            yield {
                "choices": [{
                    "delta": {
                        "role": "assistant",
                        "content": "🔍 Testing API key...\n"
                    }
                }]
            }
            
            api_key_valid = await self.test_api_key(headers)
            
            yield {
                "choices": [{
                    "delta": {
                        "content": f"API Key Test: {'✅ Valid' if api_key_valid else '❌ Invalid'}\n\n"
                    }
                }]
            }
        
        # Extract model ID - handle different formats
        model_id = body.get("model", "")
        if "." in model_id:
            model_id = model_id.split(".")[-1]  # Get everything after the last dot
        
        print(f"Full model name: {body.get('model', '')}")
        print(f"Extracted model ID: {model_id}")
        
        # Validate model
        if model_id not in self.deep_research_models:
            yield f"Error: Invalid model '{model_id}'. Available models: {', '.join(self.deep_research_models)}"
            return
        
        # --- 1️⃣   slice the incoming messages --------------------------------------
        messages: list = body.get("messages", [])

        # (a) All system messages – concatenate if the UI lets users stack them)
        system_instructions = "\n\n".join(
            m["content"] for m in messages if m.get("role") == "system"
        ).strip()

        # (b) The latest user message (unchanged logic, but fall back gracefully)
        user_input = next(
            (
                m["content"]
                if isinstance(m["content"], str)
                else "".join(p.get("text", "") for p in m["content"])
                for m in reversed(messages)
                if m.get("role") == "user"
            ),
            None,
        )
        if not user_input:
            yield "Error: No user input found"
            return
            
        # Test mode - just echo back to verify pipe is working
        if self.valves.TEST_MODE:
            yield {
                "choices": [{
                    "delta": {
                        "role": "assistant",
                        "content": f"🧪 TEST MODE ENABLED\n\n"
                    }
                }]
            }
            yield {
                "choices": [{
                    "delta": {
                        "content": f"Model: {model_id}\nInput: {user_input}\nAPI Key: {'Set' if self.key else 'Not Set'}\n"
                    }
                }]
            }
            yield {
                "choices": [{
                    "delta": {},
                    "finish_reason": "stop"
                }]
            }
            return
        
        # Enrich the prompt if enabled
        if self.valves.ENABLE_PROMPT_ENRICHMENT:
            try:
                enriched_input = await self.enrich_prompt(user_input, headers)
                yield {
                    "choices": [{
                        "delta": {
                            "role": "assistant",
                            "content": "🔍 Preparing deep research query...\n\n"
                        }
                    }]
                }
            except Exception as e:
                logger.error(f"Failed to enrich prompt: {e}")
                enriched_input = user_input
        else:
            enriched_input = user_input
        
        # Build the payload for deep research
        payload = {
            "model": model_id,
            "input": enriched_input,
            "tools": [],
            "store": True  # Required for background mode
        }

        # Only add instructions if the user actually supplied a system prompt
        if system_instructions:
            payload["instructions"] = system_instructions
        
        # Add tools based on configuration
        if self.valves.ENABLE_WEB_SEARCH:
            payload["tools"].append({"type": "web_search_preview"})
            
        if self.valves.ENABLE_CODE_INTERPRETER:
            payload["tools"].append({
                "type": "code_interpreter",
                "container": {"type": "auto"}
            })
        
        # Add max_tool_calls if specified
        if self.valves.MAX_TOOL_CALLS:
            payload["max_tool_calls"] = self.valves.MAX_TOOL_CALLS
        # Add reasoning summary
        payload["reasoning"] = {"summary": "auto"}
        
        try:
            if self.valves.USE_BACKGROUND_MODE:
                # Use background mode for reliability
                payload["background"] = True
                
                # Only set stream=true if we plan to stream immediately
                # For polling mode, we don't need streaming on creation
                if body.get("stream", True):
                    payload["stream"] = True
                
                # Create the background response
                yield {
                    "choices": [{
                        "delta": {
                            "role": "assistant",
                            "content": "🚀 Starting deep research in background mode...\n"
                        }
                    }]
                }
                
                yield {
                    "choices": [{
                        "delta": {
                            "content": f"📝 Input: {enriched_input[:100]}{'...' if len(enriched_input) > 100 else ''}\n"
                        }
                    }]
                }
                
                yield {
                    "choices": [{
                        "delta": {
                            "content": f"🛠️ Tools enabled: {', '.join([t['type'] for t in payload.get('tools', [])])}\n"
                        }
                    }]
                }
                
                yield {
                    "choices": [{
                        "delta": {
                            "content": f"🔗 API Endpoint: {self.valves.BASE_URL}/responses\n"
                        }
                    }]
                }
                
                yield {
                    "choices": [{
                        "delta": {
                            "content": f"⚙️ Timeout Settings: {self.valves.REQUEST_TIMEOUT}s read, {self.valves.CONNECTION_TIMEOUT}s connect\n\n"
                        }
                    }]
                }
                
                try:
                    response = await self.create_background_response(payload, headers)
                    response_id = response.get("id")
                    status = response.get("status")
                    
                    print(f"Background response created - ID: {response_id}, Status: {status}")
                    
                    if not response_id:
                        yield {
                            "choices": [{
                                "delta": {
                                    "content": f"Error: Failed to get response ID from response: {json.dumps(response)}\n"
                                }
                            }]
                        }
                        return
                    
                    # Handle special case where we don't have a real response ID due to timeout
                    if response_id in ["background_processing", "processing"]:
                        yield {
                            "choices": [{
                                "delta": {
                                    "content": "⚠️ Background response creation timed out. This is often due to high API load.\n"
                                }
                            }]
                        }
                        
                        yield {
                            "choices": [{
                                "delta": {
                                    "content": "� Try again in a few moments, or enable TEST_MODE to verify your API key has access to deep research models.\n"
                                }
                            }]
                        }
                        
                        yield {
                            "choices": [{
                                "delta": {},
                                "finish_reason": "stop"
                            }]
                        }
                        return
                    
                    yield {
                        "choices": [{
                            "delta": {
                                "content": f"�📋 Research ID: {response_id}\n📊 Initial Status: {status}\n"
                            }
                        }]
                    }
                    
                except Exception as e:
                    error_msg = str(e)
                    yield f"Error creating background response: {error_msg}\n"
                    
                    # Provide helpful guidance based on error type
                    if "timeout" in error_msg.lower():
                        yield "\n💡 **Timeout Solutions:**\n"
                        yield "- OpenAI's deep research models can be slow to initialize\n"
                        yield "- Try again in a few moments when API load is lower\n"
                        yield f"- Current timeout settings: {self.valves.REQUEST_TIMEOUT}s read, {self.valves.CONNECTION_TIMEOUT}s connect\n"
                        yield "- You can increase REQUEST_TIMEOUT in Valves if needed\n"
                        yield "- Enable TEST_MODE to verify your API key works with other models\n"
                        yield "- Check if your API key has access to o3-deep-research/o4-mini-deep-research\n"
                    elif "authentication" in error_msg.lower() or "unauthorized" in error_msg.lower():
                        yield "\n🔑 **Authentication Issues:**\n"
                        yield "- Verify your API key is correct\n"
                        yield "- Ensure your API key has access to deep research models\n"
                        yield "- These models require special access from OpenAI\n"
                    
                    return
                
                # For background mode, we should poll until completion
                # Streaming background responses immediately often has high latency
                yield {
                    "choices": [{
                        "delta": {
                            "content": "⏳ Research in progress. This may take several minutes...\n\n"
                        }
                    }]
                }
                
                poll_count = 0
                try:
                    while poll_count < self.valves.MAX_POLL_ATTEMPTS:
                        await asyncio.sleep(self.valves.POLL_INTERVAL)
                        poll_count += 1
                        
                        # Skip polling if we don't have a real response ID
                        if response_id in ["background_processing", "processing"]:
                            yield {
                                "choices": [{
                                    "delta": {
                                        "content": "⚠️ Cannot poll background response - no valid response ID\n"
                                    }
                                }]
                            }
                            break
                        
                        # Poll for status with timeout handling
                        try:
                            timeout = httpx.Timeout(
                                connect=self.valves.CONNECTION_TIMEOUT,
                                read=30.0,  # Polling requests should be quick
                                write=self.valves.CONNECTION_TIMEOUT,
                                pool=self.valves.CONNECTION_TIMEOUT
                            )
                            
                            async with httpx.AsyncClient(timeout=timeout) as client:
                                response = await client.get(
                                    f"{self.valves.BASE_URL}/responses/{response_id}",
                                    headers=headers
                                )
                                
                                if response.status_code != 200:
                                    yield f"Error polling response: {response.status_code} {response.text}"
                                    return
                                
                                resp_data = response.json()
                                status = resp_data.get("status")
                                
                                print(f"Poll {poll_count}: Status = {status}")
                                
                                # Provide periodic status updates
                                if poll_count % 10 == 0:  # Every 30 seconds
                                    yield {
                                        "choices": [{
                                            "delta": {
                                                "content": f"⏳ Still researching... (Status: {status}, Time: ~{poll_count * self.valves.POLL_INTERVAL}s)\n"
                                            }
                                        }]
                                    }
                                
                                if status == "completed":
                                    # Stream the complete response
                                    yield {
                                        "choices": [{
                                            "delta": {
                                                "content": "\n✅ Research completed! Here are the results:\n\n"
                                            }
                                        }]
                                    }
                                    
                                    print(f"Final response data keys: {list(resp_data.keys())}")
                                    
                                    for chunk in self.format_output_for_streaming(resp_data):
                                        yield chunk
                                    return
                                    
                                elif status == "failed":
                                    error_msg = resp_data.get("error", {}).get("message", "Unknown error")
                                    yield f"Error: Research failed - {error_msg}"
                                    return
                                    
                                elif status == "cancelled":
                                    yield "Error: Research was cancelled"
                                    return
                                    
                        except httpx.TimeoutException:
                            yield {
                                "choices": [{
                                    "delta": {
                                        "content": f"⚠️ Polling timeout (attempt {poll_count}), retrying...\n"
                                    }
                                }]
                            }
                            continue
                        except Exception as poll_error:
                            yield {
                                "choices": [{
                                    "delta": {
                                        "content": f"⚠️ Polling error (attempt {poll_count}): {str(poll_error)}\n"
                                    }
                                }]
                            }
                            continue
                    
                    yield f"Error: Polling timeout after {poll_count * self.valves.POLL_INTERVAL} seconds. The research may still be running on OpenAI's servers."
                    
                except Exception as e:
                    yield f"Error during polling: {str(e)}"
                    return
                        
            else:
                # Synchronous mode (not recommended for deep research)
                payload["stream"] = True
                
                yield {
                    "choices": [{
                        "delta": {
                            "content": "⚠️ Using synchronous mode (not recommended for deep research)...\n\n"
                        }
                    }]
                }
                
                async with httpx.AsyncClient(timeout=3600) as client:
                    async with client.stream(
                        "POST",
                        f"{self.valves.BASE_URL}/responses",
                        json=payload,
                        headers=headers,
                    ) as response:
                        if response.status_code != 200:
                            error_text = await response.aread()
                            yield f"Error: {response.status_code} {error_text.decode('utf-8')}"
                            return

                        async for line in response.aiter_lines():
                            if line:
                                try:
                                    if line.startswith("data:"):
                                        line_data = json.loads(line[5:])
                                        if "delta" in line_data:
                                            yield {
                                                "choices": [{
                                                    "delta": {
                                                        "content": line_data["delta"]
                                                    }
                                                }]
                                            }
                                except Exception as e:
                                    logger.error(f"Error processing line: {e}")
                                    
        except Exception as e:
            yield f"Error in main pipe: {str(e)}\nTraceback: {traceback.format_exc()}"
            return
