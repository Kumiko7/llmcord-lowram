import asyncio
import contextlib
import io
import json
import logging
import re
import traceback
from collections import defaultdict
from datetime import date
from typing import Any, Optional

import httpx


class ToolManager:
    """
    Manages the definition, parsing, and execution of tools for the LLM.
    This class encapsulates all logic related to tool usage, including:
    - The functions that perform the actions (e.g., searching Google).
    - The schema definitions for the tools to be passed to the LLM.
    - The logic for handling a tool request from the LLM and dispatching it
      to the correct function.
    """

    def __init__(self, config: dict[str, Any], client: httpx.AsyncClient):
        """
        Initializes the ToolManager.

        Args:
            config: The application's configuration dictionary.
            client: An httpx.AsyncClient instance for making HTTP requests.
        """
        self.config = config
        self.client = client
        self._tool_functions = {
            "google_search": self.perform_google_search,
            "search_anilist": self.search_anilist,
            "execute_python_code": self.execute_python_code,
            # VNDB tools are added dynamically below
            **{f"query_vndb_{endpoint}": getattr(self, f"query_vndb_{endpoint}") for endpoint in 
               ["vn", "release", "producer", "character", "staff", "tag", "trait", "quote"]}
        }

    def get_tool_definitions(self, is_admin: bool) -> list[dict[str, Any]]:
        """
        Returns a list of tool definitions to be passed to the LLM.
        Conditionally includes admin-only tools.

        Args:
            is_admin: A boolean indicating if the user has admin privileges.

        Returns:
            A list of tool definition dictionaries.
        """
        tools = []
        if self.config.get("serpapi_api_key"):
            tools.append({
                "type": "function",
                "function": {
                    "name": "google_search",
                    "description": "Get information from the web for recent events or specific facts.",
                    "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "The search query to use."}}, "required": ["query"]},
                },
            })
        
        # Add VNDB tool definitions
        tools.extend(self._get_vndb_tool_definitions())

        # tools.append({
            # "type": "function",
            # "function": {
                # "name": "search_anilist",
                # "description": "Search for anime or manga on AniList to get details like description, score, status, episodes, etc.",
                # "parameters": {
                    # "type": "object",
                    # "properties": {
                        # "query": {"type": "string", "description": "The title of the anime or manga to search for."},
                        # "media_type": {"type": "string", "enum": ["ANIME", "MANGA"], "description": "The type of media to search for."}
                    # },
                    # "required": ["query", "media_type"],
                # },
            # },
        # })

        if is_admin:
            tools.append({
                "type": "function",
                "function": {
                    "name": "execute_python_code",
                    "description": "Executes arbitrary Python code. The code runs within an `exec()` statement. It can be used for calculations or quick tests. Use with caution.",
                    "parameters": {
                        "type": "object",
                        "properties": {"code": {"type": "string", "description": "The Python code to execute, often provided within a ```python ... ``` block."}},
                        "required": ["code"],
                    },
                },
            })

        return tools

    def _get_vndb_tool_definitions(self) -> list[dict[str, Any]]:
        """Generates the definitions for all VNDB query tools with detailed parameter descriptions."""
        
        # Base schema for common query parameters. Descriptions will be customized per endpoint.
        common_params_base = {
            "filters": {"type": "string"},
            "fields": {"type": "string"},
            "reverse": {"type": "boolean", "description": "Set to true to sort in descending order. Defaults to false."},
            "results": {"type": "integer", "description": "Number of results per page (max 100). Defaults to 10."},
            "page": {"type": "integer", "description": "Page number to request, starting from 1. Defaults to 1."},
            "count": {"type": "boolean", "description": "Whether the response should include the total count of matching entries. Defaults to false."},
            "compact_filters": {"type": "boolean", "description": "Whether the response should include a compact string representation of the filters. Defaults to false."},
            "normalized_filters": {"type": "boolean", "description": "Whether the response should include a normalized JSON representation of the filters. Defaults to false."},
        }

        # Comprehensive data for each VNDB endpoint based on the API documentation.
        # Format: (endpoint_name, description, sort_options, filter_options, field_options)
        vndb_endpoints = [
            (
                "vn", 
                "Query visual novel entries.", 
                ["id", "title", "released", "rating", "votecount", "searchrank"],
                [
                    "id", "search", "lang", "olang", "platform", "length", "released", "rating", "votecount", 
                    "has_description", "has_anime", "has_screenshot", "has_review", "devstatus", "tag", 
                    "dtag", "anime_id", "label", "release", "character", "staff", "developer", "Basic example filter searching for the VN Steins;Gate: '[\"search\", \"=\", \"Steins;Gate\"]'. Complex example filter for a vn called 'aokana', made by 'sprite', with an anime: '[\"and\", [\"developer\", \"=\", [\"search\", \"=\", \"sprite\"]], [\"search\", \"=\", \"aokana\"], [\"has_anime\", \"=\", 1]]'. Complex example filter for a vn written by 'Watanabe Ryouichi' with the 'Science Fiction' tag: [\"and\",[\"staff\",\"=\",[\"search\",\"=\",\"Watanabe Ryouichi\"]],[\"tag\",\"=\",\"g105\"]]"
                ],
                [
                    "id", "title", "alttitle", "titles", "aliases", "olang", "devstatus", "released", 
                    "languages", "platforms", "image", "length", "length_minutes", "length_votes", 
                    "description", "average", "rating", "votecount", "screenshots", "relations", 
                    "tags", "developers", "editions", "staff", "va", "extlinks"
                ]
            ),
            (
                "release", 
                "Query visual novel release entries.", 
                ["id", "title", "released", "searchrank"],
                [
                    "id", "search", "lang", "platform", "released", "resolution", "resolution_aspect", 
                    "minage", "medium", "voiced", "engine", "rtype", "extlink", "patch", "freeware", 
                    "uncensored", "official", "has_ero", "vn", "producer"
                ],
                [
                    "id", "title", "alttitle", "languages", "platforms", "media", "vns", "producers", 
                    "images", "released", "minage", "patch", "freeware", "uncensored", "official", 
                    "has_ero", "resolution", "engine", "voiced", "notes", "gtin", "catalog", "extlinks"
                ]
            ),
            (
                "producer", 
                "Query producer (developer/publisher) entries.", 
                ["id", "name", "searchrank"],
                ["id", "search", "lang", "type", "extlink"],
                ["id", "name", "original", "aliases", "lang", "type", "description", "extlinks"]
            ),
            (
                "character", 
                "Query character entries. ", 
                ["id", "name", "searchrank"],
                [
                    "id", "search", "role", "blood_type", "sex", "sex_spoil", "gender", "gender_spoil", 
                    "height", "weight", "bust", "waist", "hips", "cup", "age", "trait", "dtrait", 
                    "birthday", "seiyuu", "vn", "Trait filter only takes a trait id and you can only query once per function call, so for character traits you should query the trait name first to get the id and then query the character, whereas for vn filters you can include the vn name search directly in a nested query. Example filter for a female character in a vn called 'aokana', with trait id 'i8': '[\"and\", [\"vn\", \"=\", [\"search\", \"=\", \"aokana\"]], [\"trait\", \"=\", \"i8\"], [\"gender\", \"=\", \"f\"]]'"
                ],
                [
                    "id", "name", "original", "aliases", "description", "image", "blood_type", "height", 
                    "weight", "bust", "waist", "hips", "cup", "age", "birthday", "sex", "gender", "vns", "traits"
                ]
            ),
            (
                "staff", 
                "Query staff (creators, voice actors) names/entries.", 
                ["id", "name", "searchrank"],
                ["id", "aid", "search", "lang", "gender", "role", "extlink", "ismain"],
                ["id", "aid", "ismain", "name", "original", "lang", "gender", "description", "extlinks", "aliases"]
            ),
            (
                "tag", 
                "Query tag entries (e.g., 'Nakige', 'Female Protagonist').", 
                ["id", "name", "vn_count", "searchrank"],
                ["id", "search", "category"],
                ["id", "name", "aliases", "description", "category", "searchable", "applicable", "vn_count"]
            ),
            (
                "trait", 
                "Query character trait entries (e.g., 'Blond', 'Younger Sister')", 
                ["id", "name", "char_count", "searchrank"],
                ["id", "search", "For 'pink hair', just search for 'pink'. For 'blue eyes', just search for 'blue'. Example filter: '[\"search\", \"=\", \"pink\"]'"],
                [
                    "id", "name", "aliases", "description", "searchable", "applicable", "sexual", 
                    "group_id", "group_name", "char_count"
                ]
            ),
            (
                "quote", 
                "Query visual novel quotes.", 
                ["id", "score"],
                ["id", "vn", "character", "random"],
                ["id", "quote", "score", "vn", "character"]
            ),
        ]
        
        definitions = []
        for endpoint, desc, sort_options, filter_options, field_options in vndb_endpoints:
            # Create a fresh copy of parameters for this endpoint
            params = {k: v.copy() for k, v in common_params_base.items()}
            
            # Customize descriptions with the specific options for this endpoint
            params["filters"]["description"] = (
                "A JSON string representing filter predicates. IMPORTANT: Filters that link to other entities ('vn', 'staff', 'character', 'producer', 'developer', 'release', 'seiyuu') MUST use a nested filter array for their value, while other filters MUST use a simple filter."
                f"Valid filter names: {', '.join(filter_options)}."
            )
            params["fields"]["description"] = (
                "A comma-separated list of fields to fetch, e.g., 'title,rating,image.url'. Plural object fields such as developers, vns, tags, traits MUST specify a subfield, e.g. 'vns.title,developers.name' "
                f"Valid fields: {', '.join(field_options)}."
            )
            
            # Add the sort parameter with its specific enum values
            params["sort"] = {
                "type": "string", 
                "enum": sort_options, 
                "description": "Field to sort the results on."
            }
            
            # 'fields' is essential for getting any data back, so it should be required.
            required = ["fields"]
            
            definitions.append({
                "type": "function",
                "function": {
                    "name": f"query_vndb_{endpoint}",
                    "description": desc,
                    "parameters": {
                        "type": "object", 
                        "properties": params, 
                        "required": required
                    }
                }
            })
            
        return definitions

    async def handle_tool_calls(self, requested_tool_calls: list[dict]) -> tuple[list[dict], list[str]]:
        """
        Processes tool calls requested by the LLM, executes them, and returns the results.
        """
        tool_messages = []
        executed_tool_calls_display = []

        # Create a list of async tasks to run in parallel
        tasks = []
        for tool_call in requested_tool_calls:
            tasks.append(self._execute_single_tool(tool_call))
        
        # Wait for all tool calls to complete
        results = await asyncio.gather(*tasks)

        # Process the results
        for tool_message, display_string in results:
            if tool_message:
                tool_messages.append(tool_message)
            if display_string:
                executed_tool_calls_display.append(display_string)

        return tool_messages, executed_tool_calls_display

    async def _execute_single_tool(self, tool_call: dict) -> tuple[Optional[dict], Optional[str]]:
        """Executes a single tool call and returns its message and display string."""
        func_name = tool_call["function"]["name"]
        tool_call_id = tool_call["id"]
        
        try:
            args = json.loads(tool_call["function"]["arguments"])
            func = self._tool_functions.get(func_name)
            
            logging.info(f"Calling tool: {func_name} with parameters {str(args)}")

            if not func:
                result = f"Error: Unknown tool '{func_name}' called."
                display = f"❌ Error: Unknown tool '{func_name}'"
            else:
                if func_name.startswith("query_vndb_"):
                    endpoint = func_name.split("_")[-1].upper()
                    filters_str = args.get("filters", "")
                    # Try to find a 'search' query in filters for a better display name
                    search_match = re.search(r'\["search",\s*"=",\s*"([^"]+)"\]', filters_str)
                    query_display = search_match.group(1) if search_match else filters_str
                    display = f"📚 VNDB/{endpoint}: '{query_display}'"
                    result = await func(**args)
                elif func_name == "google_search":
                    query = args.get("query", "")
                    display = f"🔎 Google: '{(query[:25] + '...') if len(query) > 25 else query}'"
                    result = await func(query=query)
                elif func_name == "search_anilist":
                    query = args.get("query", "")
                    media_type = args.get("media_type", "ANIME")
                    display = f"💽 AniList/{media_type.title()}: '{(query[:25] + '...') if len(query) > 25 else query}'"
                    result = await func(query=query, media_type=media_type)
                elif func_name == "execute_python_code":
                    code = args.get("code", "")
                    display = f"🐍 Python: \n'{code}'"
                    result = await func(code=code)
                else:
                    # Fallback for any unhandled but existing functions
                    display = f"⚙️ Executing: {func_name}"
                    result = await func(**args)

        except json.JSONDecodeError:
            logging.exception(f"Failed to decode arguments for tool {func_name}. Args: {tool_call['function']['arguments']}")
            result = "Error: Invalid arguments provided. Ensure arguments are valid JSON."
            display = f"❌ Error decoding args for {func_name}"
        except Exception as e:
            logging.exception(f"An error occurred while executing tool {func_name}")
            result = f"An unexpected error occurred: {e}"
            display = f"❌ Error executing {func_name}"

        message = {"role": "tool", "tool_call_id": tool_call_id, "name": func_name, "content": result}
        return message, display

    async def perform_google_search(self, query: str) -> str:
        """Performs a Google search using SerpApi and returns formatted results."""
        if not (api_key := self.config.get("serpapi_api_key")):
            return "Google Search is not configured with a SerpApi key."
        try:
            params = {"q": query, "api_key": api_key, "engine": "google"}
            response = await self.client.get("https://serpapi.com/search", params=params)
            response.raise_for_status()
            search_results = response.json()
            snippets = []
            if "answer_box" in search_results:
                answer = search_results["answer_box"].get("answer") or search_results["answer_box"].get("snippet")
                if answer: snippets.append({"answer": answer})
            if "organic_results" in search_results:
                for result in search_results.get("organic_results", [])[:5]:
                    snippets.append({"title": result.get("title"), "link": result.get("link"), "snippet": result.get("snippet")})
            return json.dumps(snippets) if snippets else "No results found."
        except httpx.HTTPStatusError as e:
            logging.exception(f"HTTP error during Google search: {e}")
            return f"An HTTP error occurred during search: {e.response.text}"
        except Exception as e:
            logging.exception(f"Error during Google search: {e}")
            return f"An error occurred during search: {e}"

    # --- VNDB API TOOLS ---
    
    async def _vndb_query(self, endpoint: str, **kwargs: Any) -> str:
        """Helper function to query the VNDB API."""
        api_url = f"https://api.vndb.org/kana/{endpoint}"
        
        # Build payload from provided arguments
        payload = {}
        # The LLM will provide filters as a JSON string, which we must parse.
        if filters_str := kwargs.get("filters"):
            try:
                payload["filters"] = json.loads(filters_str)
            except json.JSONDecodeError:
                return "Error: The 'filters' parameter must be a valid JSON string."
        
        # Copy other valid parameters
        valid_keys = ["fields", "sort", "reverse", "results", "page", "user", "count", "compact_filters", "normalized_filters"]
        for key in valid_keys:
            if key in kwargs:
                payload[key] = kwargs[key]
        
        # Ensure results are within API limits
        payload["results"] = min(int(payload.get("results", 10)), 100)

        try:
            response = await self.client.post(api_url, json=payload, timeout=15.0)
            response.raise_for_status()
            data = response.json()

            output_str = json.dumps(data, indent=2)
            # Truncate very large responses to avoid context overflow
            if len(output_str) > 4000:
                truncated_results = data.get("results", [])[:3] 
                data["results"] = truncated_results
                note = f"Response truncated to {len(truncated_results)} items to fit context window."
                if "count" in data:
                    note += f" Total items found: {data['count']}."
                data["note"] = note
                output_str = json.dumps(data, indent=2)

            return output_str
        except httpx.HTTPStatusError as e:
            return f"Error: Received status code {e.response.status_code} from VNDB API. Body: {e.response.text}"
        except Exception as e:
            logging.exception(f"An unexpected error occurred during the VNDB query for /{endpoint}.")
            return f"An unexpected error occurred during the VNDB search: {e}"

    async def query_vndb_vn(self, **kwargs: Any) -> str:
        return await self._vndb_query("vn", **kwargs)

    async def query_vndb_release(self, **kwargs: Any) -> str:
        return await self._vndb_query("release", **kwargs)

    async def query_vndb_producer(self, **kwargs: Any) -> str:
        return await self._vndb_query("producer", **kwargs)

    async def query_vndb_character(self, **kwargs: Any) -> str:
        return await self._vndb_query("character", **kwargs)
    
    async def query_vndb_staff(self, **kwargs: Any) -> str:
        return await self._vndb_query("staff", **kwargs)

    async def query_vndb_tag(self, **kwargs: Any) -> str:
        return await self._vndb_query("tag", **kwargs)

    async def query_vndb_trait(self, **kwargs: Any) -> str:
        return await self._vndb_query("trait", **kwargs)

    async def query_vndb_quote(self, **kwargs: Any) -> str:
        return await self._vndb_query("quote", **kwargs)

    # --- OTHER TOOLS ---

    async def search_anilist(self, query: str, media_type: str) -> str:
        """Searches for anime or manga on AniList and returns formatted results."""
        api_url = "https://graphql.anilist.co"
        gql_query = "query($s:String,$t:MediaType){Media(search:$s,type:$t,sort:SEARCH_MATCH){id title{romaji english}description(asHtml:!1)format status episodes chapters volumes averageScore genres studios(isMain:!0){nodes{name}}staff(sort:RELEVANCE,perPage:4){edges{role node{name{full}}}}siteUrl startDate{year month day}}}"
        variables = {"s": query, "t": media_type.upper()}
        try:
            resp = await self.client.post(api_url, json={"query": gql_query, "variables": variables}, timeout=10.0)
            resp.raise_for_status()
            media = resp.json().get("data", {}).get("Media")
            if not media: return f"No {media_type.lower()} found for '{query}' on AniList."
            desc = media.get('description', 'No desc.')
            if desc: desc = re.sub(r'<br\s*/?>|\s*<[^>]+>\s*', lambda m: '\n' if '<br' in m.group(0) else '', desc)[:400] + ("..." if len(desc) > 400 else "")
            score = f"{media.get('averageScore')}/100" if media.get('averageScore') else "Not rated"
            t_r, t_e = media.get('title', {}).get('romaji', 'N/A'), media.get('title', {}).get('english')
            title = f"{t_r}" + (f" ({t_e})" if t_e and t_e != t_r else "")
            details = [f"Title: {title}", f"Format: {str(media.get('format', 'N/A')).replace('_', ' ')}", f"Status: {str(media.get('status', 'N/A')).replace('_', ' ')}", f"Score: {score}", f"Genres: {', '.join(media.get('genres', []))}"]
            if sd := media.get('startDate'):
                if all(sd.values()):
                    try: details.append(f"Release: {date(sd['year'],sd['month'],sd['day']).strftime('%Y/%m/%d')}")
                    except ValueError: details.append("Release: Invalid date")
            if media_type == "ANIME":
                if e := media.get('episodes'): details.append(f"Episodes: {e}")
                if s := media.get('studios', {}).get('nodes'): details.append(f"Studio: {', '.join([n['name'] for n in s])}")
            elif media_type == "MANGA":
                if c := media.get('chapters'): details.append(f"Chapters: {c}")
                if v := media.get('volumes'): details.append(f"Volumes: {v}")
                if s := media.get('staff', {}).get('edges'):
                    if a := [n['node']['name']['full'] for n in s if n['role'] in ('Story & Art', 'Story')]: details.append(f"Author(s): {', '.join(a)}")
            details.extend([f"\nDesc:\n{desc}", f"\nURL: {media.get('siteUrl')}"])
            return "\n".join(details)
        except httpx.HTTPStatusError as e: return f"Error: Received status {e.response.status_code} from AniList API. Body: {e.response.text}"
        except Exception as e: return f"An unexpected error during the AniList search: {e}"

    async def execute_python_code(self, code: str) -> str:
        """Executes arbitrary Python code in a restricted environment and returns the output."""
        if code.startswith("```python"): code = code[9:]
        elif code.startswith("```"): code = code[3:]
        if code.endswith("```"): code = code[:-3]
        code = code.strip()
        if not code: return "No code provided to execute."
        logging.info(f"Code Execution: {code}")
        stdout = io.StringIO()
        try:
            with contextlib.redirect_stdout(stdout):
                exec_globals = {**globals()} # Create a copy of globals for execution
                await asyncio.wait_for(asyncio.to_thread(exec, code, exec_globals), timeout=10.0)
            output = stdout.getvalue()
            if len(output) > 1900: output = "Output truncated:\n" + output[:1900] + "..."
            return f"Output:\n```\n{output or '[No output]'}\n```"
        except asyncio.TimeoutError:
            return "Error: Code execution timed out after 10 seconds."
        except Exception:
            error_trace = traceback.format_exc()
            if len(error_trace) > 1900: error_trace = "Error truncated:\n" + error_trace[:1900] + "..."
            return f"An exception occurred:\n```\n{error_trace}\n```"