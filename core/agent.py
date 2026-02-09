"""
GaiaChat Agent Module

LLM-powered agent that translates natural language queries into
Gaia data operations and provides scientific explanations.
"""

import json
from typing import Optional, Dict, Any, List, Generator
from dataclasses import dataclass
from openai import OpenAI

from .config import config
from .gaia_service import gaia_service, QueryResult


@dataclass
class AgentResponse:
    """Container for agent response."""
    message: str
    data: Optional[Any] = None
    plot_type: Optional[str] = None
    query_used: Optional[str] = None


class GaiaChatAgent:
    """
    LLM-powered agent for natural language Gaia data exploration.
    
    Uses OpenAI function calling to translate user queries into
    appropriate Gaia data operations.
    """
    
    def __init__(self):
        """Initialize the agent."""
        config.validate()
        self.client = OpenAI(api_key=config.openai_api_key)
        self.model = config.model
        self.temperature = config.temperature
        self.conversation_history: List[Dict[str, str]] = []
        self.last_result: Optional[QueryResult] = None
        
        # Build system prompt
        self.system_prompt = self._build_system_prompt()
        
        # Define available tools
        self.tools = self._define_tools()
    
    def _build_system_prompt(self) -> str:
        """Build the system prompt for the agent."""
        return """You are GaiaChat, an expert astronomical assistant specialized in the Gaia DR3 stellar catalog.

Your role is to help scientists and students explore Gaia data using natural language.

## Your Capabilities:
1. **Search queries**: Cone searches, solar neighborhood, distance-based queries
2. **Stellar streams**: Search for Nyx, Gaia-Sausage-Enceladus (GSE), Helmi, Sequoia streams
3. **Kinematics**: Find hypervelocity stars, accreted halo stars, retrograde orbit stars
4. **Visualizations**: HR diagrams, sky plots, velocity plots, Toomre diagrams

## Scientific Background:
- **Nyx**: Prograde stream near the disk plane, discovered by Necib et al. 2020
- **Gaia-Sausage-Enceladus (GSE)**: Major merger remnant with radial orbits
- **Dark Matter**: Stellar kinematics trace gravitational potential and DM distribution

## Guidelines:
1. Always explain what the query is searching for
2. After showing results, explain what the data reveals scientifically
3. Suggest relevant follow-up analyses or visualizations
4. When discussing velocities, use Galactocentric cylindrical (V_R, V_phi, V_z)
5. If a query returns no results, explain why and suggest alternatives

## Important Notes:
- Parallax > 0 is required for distance calculations (parallax in milliarcseconds)
- Distance (pc) = 1000 / parallax (mas)
- Quality cuts: parallax_over_error > 5, ruwe < 1.4 for clean samples
"""
    
    def _define_tools(self) -> List[Dict[str, Any]]:
        """Define the function calling tools."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "search_solar_neighborhood",
                    "description": "Search for stars near the Sun within a specified distance",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "distance_pc": {
                                "type": "number",
                                "description": "Maximum distance in parsecs (default 100)"
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of results to return"
                            }
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_cone",
                    "description": "Search for stars within a cone around a sky position",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "ra": {
                                "type": "number",
                                "description": "Right Ascension in degrees"
                            },
                            "dec": {
                                "type": "number",
                                "description": "Declination in degrees"
                            },
                            "radius": {
                                "type": "number",
                                "description": "Search radius in degrees (default 1.0)"
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of results"
                            }
                        },
                        "required": ["ra", "dec"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_hypervelocity_stars",
                    "description": "Search for hypervelocity star candidates with high total velocities",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "distance_kpc": {
                                "type": "number",
                                "description": "Maximum distance in kiloparsecs (default 5)"
                            },
                            "min_velocity_kms": {
                                "type": "number",
                                "description": "Minimum total velocity in km/s (default 300)"
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of results"
                            }
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_stellar_stream",
                    "description": "Search for stars belonging to known stellar streams like Nyx, GSE, Helmi, or Sequoia",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "stream_name": {
                                "type": "string",
                                "description": "Name of the stream: 'Nyx', 'GSE', 'Gaia-Sausage-Enceladus', 'Helmi', or 'Sequoia'",
                                "enum": ["Nyx", "GSE", "Gaia-Sausage-Enceladus", "Helmi", "Sequoia"]
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of results"
                            }
                        },
                        "required": ["stream_name"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_accreted_halo",
                    "description": "Search for accreted halo stars from past dwarf galaxy mergers",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "retrograde_only": {
                                "type": "boolean",
                                "description": "If true, only return stars on retrograde orbits"
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of results"
                            }
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "execute_custom_adql",
                    "description": "Execute a custom ADQL query against Gaia DR3",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The ADQL query to execute"
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "suggest_visualization",
                    "description": "Suggest an appropriate visualization for the current data",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "plot_type": {
                                "type": "string",
                                "description": "Type of plot to generate",
                                "enum": ["hr_diagram", "sky_map", "velocity_plot", "toomre_diagram", "proper_motion"]
                            }
                        },
                        "required": ["plot_type"]
                    }
                }
            }
        ]
    
    def _execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool and return the result."""
        try:
            if tool_name == "search_solar_neighborhood":
                result = gaia_service.search_solar_neighborhood(
                    distance_pc=arguments.get("distance_pc", 100),
                    limit=arguments.get("limit")
                )
                self.last_result = result
                return {
                    "success": True,
                    "row_count": result.row_count,
                    "description": result.description,
                    "query": result.query,
                    "columns": list(result.data.columns) if len(result.data) > 0 else [],
                    "sample": result.data.head(5).to_dict() if len(result.data) > 0 else {}
                }
            
            elif tool_name == "search_cone":
                result = gaia_service.search_cone(
                    ra=arguments["ra"],
                    dec=arguments["dec"],
                    radius=arguments.get("radius", 1.0),
                    limit=arguments.get("limit")
                )
                self.last_result = result
                return {
                    "success": True,
                    "row_count": result.row_count,
                    "description": result.description,
                    "query": result.query
                }
            
            elif tool_name == "search_hypervelocity_stars":
                result = gaia_service.search_hypervelocity_stars(
                    distance_kpc=arguments.get("distance_kpc", 5.0),
                    min_velocity_kms=arguments.get("min_velocity_kms", 300),
                    limit=arguments.get("limit")
                )
                self.last_result = result
                return {
                    "success": True,
                    "row_count": result.row_count,
                    "description": result.description,
                    "columns": list(result.data.columns) if len(result.data) > 0 else [],
                    "has_velocities": "v_total" in result.data.columns if len(result.data) > 0 else False
                }
            
            elif tool_name == "search_stellar_stream":
                result = gaia_service.search_stellar_stream(
                    stream_name=arguments["stream_name"],
                    limit=arguments.get("limit")
                )
                self.last_result = result
                return {
                    "success": True,
                    "row_count": result.row_count,
                    "description": result.description,
                    "stream": arguments["stream_name"]
                }
            
            elif tool_name == "search_accreted_halo":
                result = gaia_service.search_accreted_halo(
                    retrograde_only=arguments.get("retrograde_only", False),
                    limit=arguments.get("limit")
                )
                self.last_result = result
                return {
                    "success": True,
                    "row_count": result.row_count,
                    "description": result.description
                }
            
            elif tool_name == "execute_custom_adql":
                result = gaia_service.execute_adql(arguments["query"])
                self.last_result = result
                return {
                    "success": True,
                    "row_count": result.row_count,
                    "query": result.query
                }
            
            elif tool_name == "suggest_visualization":
                return {
                    "success": True,
                    "plot_type": arguments["plot_type"],
                    "has_data": self.last_result is not None and len(self.last_result.data) > 0
                }
            
            else:
                return {"success": False, "error": f"Unknown tool: {tool_name}"}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def chat(self, user_message: str) -> AgentResponse:
        """
        Process a user message and return a response.
        
        Args:
            user_message: The user's natural language query
            
        Returns:
            AgentResponse with message and optional data/plot
        """
        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })
        
        # Build messages for API call
        messages = [
            {"role": "system", "content": self.system_prompt},
            *self.conversation_history[-10:]  # Keep last 10 turns
        ]
        
        # Call OpenAI API with tools
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=self.tools,
            tool_choice="auto",
            temperature=self.temperature
        )
        
        assistant_message = response.choices[0].message
        
        # Check if tool calls are needed
        plot_type = None
        query_used = None
        
        if assistant_message.tool_calls:
            # Execute each tool call
            tool_results = []
            for tool_call in assistant_message.tool_calls:
                tool_name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments)
                
                result = self._execute_tool(tool_name, arguments)
                tool_results.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "content": json.dumps(result)
                })
                
                # Track visualization request
                if tool_name == "suggest_visualization":
                    plot_type = arguments.get("plot_type")
                
                # Track query used
                if "query" in result:
                    query_used = result.get("query")
            
            # Add assistant message with tool calls to history
            self.conversation_history.append({
                "role": "assistant",
                "content": assistant_message.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in assistant_message.tool_calls
                ]
            })
            
            # Add tool results
            for tr in tool_results:
                self.conversation_history.append(tr)
            
            # Get final response incorporating tool results
            messages = [
                {"role": "system", "content": self.system_prompt},
                *self.conversation_history[-15:]
            ]
            
            final_response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature
            )
            
            final_content = final_response.choices[0].message.content
            
            # Add to history
            self.conversation_history.append({
                "role": "assistant",
                "content": final_content
            })
            
            return AgentResponse(
                message=final_content,
                data=self.last_result.data if self.last_result else None,
                plot_type=plot_type,
                query_used=query_used
            )
        
        else:
            # No tool calls, just return the message
            content = assistant_message.content or ""
            self.conversation_history.append({
                "role": "assistant",
                "content": content
            })
            
            return AgentResponse(message=content)
    
    def stream_chat(self, user_message: str) -> Generator[str, None, AgentResponse]:
        """
        Stream a response to a user message.
        
        Yields chunks of the response as they arrive.
        Returns the final AgentResponse.
        """
        # For now, use non-streaming and yield the full response
        # TODO: Implement proper streaming with tool calls
        response = self.chat(user_message)
        yield response.message
        return response
    
    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
        self.last_result = None
    
    def get_last_data(self):
        """Get the data from the last query."""
        if self.last_result:
            return self.last_result.data
        return None


# Global agent instance
agent = GaiaChatAgent()
