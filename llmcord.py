import asyncio
from base64 import b64encode
from dataclasses import dataclass, field
from datetime import datetime, timezone, date
import logging
from typing import Any, Literal, Optional

import discord
from discord.app_commands import Choice
from discord.ext import commands
from discord.ui import LayoutView, TextDisplay
import httpx
from openai import AsyncOpenAI
import yaml

import platform
import urllib.parse
from zoneinfo import ZoneInfo;
import schedule
from serpapi import GoogleSearch
from collections import defaultdict
import json
import re
import contextlib
import traceback
import io


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)

MEMORY_LIMIT_MB = 100
RESTART_LIMIT_MB = 85
IMAGE_LIMIT = 720

VISION_MODEL_TAGS = ("claude", "gemini", "gemma", "gpt-4", "gpt-5", "grok-4", "llama", "llava", "mistral", "o3", "o4", "vision", "vl")
PROVIDERS_SUPPORTING_USERNAMES = ("openai", "x-ai")

EMBED_COLOR_COMPLETE = discord.Color.dark_green()
EMBED_COLOR_INCOMPLETE = discord.Color.orange()

STREAMING_INDICATOR = " ⚪"
EDIT_DELAY_SECONDS = 1

MAX_MESSAGE_NODES = 50


def set_memory_limit(max_bytes):
    if platform.system() != "Windows":
        try:
            import resource
            soft, hard = resource.getrlimit(resource.RLIMIT_DATA)
            resource.setrlimit(resource.RLIMIT_DATA, (max_bytes, hard))
            logging.info(f"Memory limit set to {max_bytes / (1024*1024):.2f} MB")
        except ValueError as e:
            logging.info(f"Failed to set memory limit: {e}")
        except Exception as e:
            logging.info(f"An unexpected error occurred while setting memory limit: {e}")
    else:
        logging.info("The 'resource' module is not available on Windows for memory limits.")
set_memory_limit(MEMORY_LIMIT_MB * 1024 * 1024) # Convert MB to bytes

def get_config(filename: str = "config.yaml") -> dict[str, Any]:
    with open(filename, encoding="utf-8") as file:
        return yaml.safe_load(file)


config = get_config()
curr_model = next(iter(config["models"]))

msg_nodes = {}
last_task_time = 0

intents = discord.Intents.default()
intents.message_content = True
activity = discord.CustomActivity(name=(config["status_message"] or "github.com/jakobdylanc/llmcord")[:128])
discord_bot = commands.Bot(intents=intents, activity=activity, command_prefix=None)

httpx_client = httpx.AsyncClient()



@dataclass
class MsgNode:
    text: Optional[str] = None
    images: list[dict[str, Any]] = field(default_factory=list)

    role: Literal["user", "assistant"] = "assistant"
    user_id: Optional[int] = None

    has_bad_attachments: bool = False
    fetch_parent_failed: bool = False

    parent_msg: Optional[discord.Message] = None

    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


async def perform_google_search(query: str) -> str:
    """Performs a Google search using SerpApi and returns formatted results."""
    if not (api_key := config.get("serpapi_api_key")):
        return "Google Search is not configured with a SerpApi key."
    try:
        def search():
            params = {"q": query, "api_key": api_key, "engine": "google"}
            search_results = GoogleSearch(params).get_dict()
            
            snippets = []
            if "answer_box" in search_results:
                answer = search_results["answer_box"].get("answer") or search_results["answer_box"].get("snippet")
                if answer:
                    snippets.append({"answer": answer})
            
            if "organic_results" in search_results:
                for result in search_results["organic_results"][:5]:
                    snippet = {
                        "title": result.get("title"),
                        "link": result.get("link"),
                        "snippet": result.get("snippet"),
                    }
                    snippets.append(snippet)
            
            return json.dumps(snippets) if snippets else "No results found."

        return await asyncio.to_thread(search)
    except Exception as e:
        logging.exception(f"Error during Google search: {e}")
        return f"An error occurred during search: {e}"
        
# --- VNDB TOOL IMPLEMENTATION (WITH STAFF PRIORITIZATION) ---
async def search_vndb(query: str) -> str:
    """Searches for a visual novel on VNDB and returns formatted results including staff and characters."""
    try:
        api_url = "https://api.vndb.org/kana/vn"
        headers = {"Content-Type": "application/json"}
        payload = {
            "filters": ["search", "=", query],
            "fields": (
                "id, title, alttitle, description, released, rating, votecount, image.url, "
                "staff{role, name}, va{character{name}, staff{name}}"
            ),
            "sort": "searchrank",
            "results": 3,
        }
        
        response = await httpx_client.post(api_url, json=payload, timeout=10.0)
        response.raise_for_status()
        data = response.json()

        if not data.get("results"):
            return "No visual novels found for that query."

        formatted_results = []
        for vn in data["results"]:
            description = vn.get('description', 'No description available.')
            if description:
                description = description.split('\n[spoiler]')[0]
                description = "".join(filter(lambda c: c not in "[]", description))
                if len(description) > 400:
                    description = description[:400] + "..."

            rating = f"{vn.get('rating', 0) / 10:.1f}/10" if vn.get('rating') else "Not rated"

            result_str = (
                f"ID: {vn.get('id')}\n"
                f"Title: {vn.get('title')}\n"
                f"Alternative Title: {vn.get('alttitle')}\n"
                f"Release Date: {vn.get('released')}\n"
                f"Rating: {rating} (from {vn.get('votecount', 0)} votes)\n"
                f"Description: {description}\n"
            )

            # Process staff, grouping by role with prioritization
            if staff_list := vn.get("staff"):
                staff_by_role = defaultdict(list)
                for member in staff_list:
                    if name := member.get("name"):
                        staff_by_role[member.get("role", "Unknown")].append(name)
                
                if staff_by_role:
                    staff_str = "\nStaff:\n"
                    
                    # --- Prioritization Logic ---
                    # Define priority roles. 'art' is the official role name for artist.
                    priority_roles = ['scenario', 'art']
                    all_roles = list(staff_by_role.keys())
                    # Sort roles: priority roles come first, then the rest alphabetically
                    all_roles.sort(key=lambda role: (role not in priority_roles, role))
                    # --- End Prioritization Logic ---

                    staff_count = 0
                    limit = 10
                    for role in all_roles:
                        if staff_count >= limit:
                            staff_str += "- ...and more.\n"
                            break
                        
                        names = staff_by_role[role]
                        role_str = f"- {role.replace('_', ' ').title()}: {', '.join(sorted(list(set(names)))[:3])}"
                        if len(names) > 3:
                            role_str += ", ..."
                        staff_str += role_str + "\n"
                        staff_count += len(names)
                    result_str += staff_str

            # Process characters and voice actors
            if va_list := vn.get("va"):
                char_limit = 8
                char_list = [
                    f"{va.get('character', {}).get('name', 'Unknown Char')} "
                    f"(VA: {va.get('staff', {}).get('name', 'Unknown VA')})"
                    for va in va_list
                    if va.get('character', {}).get('name') and va.get('staff', {}).get('name')
                ]
                unique_chars = list(dict.fromkeys(char_list))
                if unique_chars:
                    result_str += "\nCharacters:\n- " + "\n- ".join(unique_chars[:char_limit])
                    if len(unique_chars) > char_limit:
                        result_str += "\n- ...and more."
            
            formatted_results.append(result_str)
        
        return "\n\n---\n\n".join(formatted_results)

    except httpx.HTTPStatusError as e:
        logging.exception(f"HTTP error during VNDB search: {e.response.status_code} - {e.response.text}")
        return f"Error: Received status code {e.response.status_code} from VNDB API."
    except Exception as e:
        logging.exception(f"Error during VNDB search: {e}")
        return f"An unexpected error occurred during the VNDB search: {e}"
        
# --- ANILIST TOOL IMPLEMENTATION ---
async def search_anilist(query: str, media_type: str) -> str:
    """Searches for anime or manga on AniList and returns formatted results."""
    api_url = "https://graphql.anilist.co"
    
    graphql_query = """
    query ($search: String, $type: MediaType) {
      Media (search: $search, type: $type, sort: SEARCH_MATCH) {
        id
        title { romaji english }
        description(asHtml: false)
        format status episodes chapters volumes averageScore
        genres
        studios(isMain: true) { nodes { name } }
        staff(sort: RELEVANCE, perPage: 4) { edges { role node { name { full } } } }
        siteUrl
        startDate { year month day }
      }
    }
    """
    variables = {"search": query, "type": media_type.upper()}

    try:
        response = await httpx_client.post(api_url, json={"query": graphql_query, "variables": variables}, timeout=10.0)
        response.raise_for_status()
        data = response.json()

        media = data.get("data", {}).get("Media")
        if not media:
            return f"No {media_type.lower()} found for '{query}' on AniList."

        description = media.get('description', 'No description available.')
        if description:
            description = re.sub(r'<br\s*/?>', '\n', description)
            description = re.sub(r'<[^>]+>', '', description)
            if len(description) > 400: description = description[:400] + "..."
        
        score = f"{media.get('averageScore')}/100" if media.get('averageScore') else "Not rated"

        title_romaji = media.get('title', {}).get('romaji', 'N/A')
        title_english = media.get('title', {}).get('english')
        title_str = f"{title_romaji}" + (f" ({title_english})" if title_english and title_english != title_romaji else "")

        details = [
            f"Title: {title_str}",
            f"Format: {str(media.get('format', 'N/A')).replace('_', ' ')}",
            f"Status: {str(media.get('status', 'N/A')).replace('_', ' ')}",
            f"Score: {score}",
            f"Genres: {', '.join(media.get('genres', []))}",
        ]
        
        start_date = media.get('startDate')
        if start_date and all(start_date.values()):
            try:
                # Prioritising Japanese date format as requested
                release_date_str = date(start_date['year'], start_date['month'], start_date['day']).strftime("%Y/%m/%d")
                details.append(f"Release Date: {release_date_str}")
            except ValueError:
                details.append("Release Date: Invalid date")

        if media_type == "ANIME":
            if episodes := media.get('episodes'): details.append(f"Episodes: {episodes}")
            if studios := media.get('studios', {}).get('nodes'): details.append(f"Studio: {', '.join([s['name'] for s in studios])}")
        elif media_type == "MANGA":
            if chapters := media.get('chapters'): details.append(f"Chapters: {chapters}")
            if volumes := media.get('volumes'): details.append(f"Volumes: {volumes}")
            if staff := media.get('staff', {}).get('edges'):
                authors = [s['node']['name']['full'] for s in staff if s['role'] in ('Story & Art', 'Story')]
                if authors: details.append(f"Author(s): {', '.join(authors)}")

        details.extend([f"\nDescription:\n{description}", f"\nURL: {media.get('siteUrl')}"])
        return "\n".join(details)

    except httpx.HTTPStatusError as e:
        logging.exception(f"HTTP error during AniList search: {e.response.status_code} - {e.response.text}")
        return f"Error: Received status code {e.response.status_code} from AniList API."
    except Exception as e:
        logging.exception(f"Error during AniList search: {e}")
        return f"An unexpected error occurred during the AniList search: {e}"
        
async def execute_python_code(code: str) -> str:
    """Executes arbitrary Python code in a restricted environment and returns the output."""
    # Clean up the code block from Markdown
    if code.startswith("```python"):
        code = code[9:]
    elif code.startswith("```"):
        code = code[3:]
    if code.endswith("```"):
        code = code[:-3]
    code = code.strip()

    if not code:
        return "No code provided to execute."
    
    logging.info(f"Code Execution: {code}")

    local_namespace = {}
    stdout_capture = io.StringIO()
    
    # Define the synchronous execution part
    def sync_exec():
        with contextlib.redirect_stdout(stdout_capture):
            exec(code, globals(), local_namespace)

    try:
        # Run the synchronous code in a separate thread with a timeout
        await asyncio.wait_for(asyncio.to_thread(sync_exec), timeout=10.0)
        
        output = stdout_capture.getvalue()
        # Truncate output to avoid Discord message limit
        if len(output) > 1900:
            output = "Output truncated:\n" + output[:1900] + "..."
        
        return f"Output:\n```\n{output or '[No output]'}\n```"

    except asyncio.TimeoutError:
        return "Error: Code execution timed out after 10 seconds."
    except Exception:
        # Capture and format the full exception traceback
        error_trace = traceback.format_exc()
        if len(error_trace) > 1900:
            error_trace = "Error truncated:\n" + error_trace[:1900] + "..."
        return f"An exception occurred:\n```\n{error_trace}\n```"

async def restart_discloud_app():
    try:
        DISCLOUD_API_BASE_URL = "https://api.discloud.app/v2"
        DISCLOUD_API_TOKEN = config["discloud_token"]
        DISCLOUD_APP_ID = config["discloud_app"]
            
        logging.info(f" Attempting to restart Discloud app: {DISCLOUD_APP_ID}...")

        headers = {
            "api-token": DISCLOUD_API_TOKEN
        }

        # Construct the API endpoint for restarting the application
        restart_url = f"{DISCLOUD_API_BASE_URL}/app/{DISCLOUD_APP_ID}/restart"
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.put(restart_url, headers=headers)
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)

        data = response.json()
        logging.info(f"Successfully requested restart for app: {DISCLOUD_APP_ID}. API response: {data.get('message', 'No message provided')}")

    except Exception as e:
        logging.exception(f"An unhandled error occurred restarting Discloud")
        
async def check_discloud_ram():
    try:
        DISCLOUD_API_BASE_URL = "https://api.discloud.app/v2"
        DISCLOUD_API_TOKEN = config["discloud_token"]
        DISCLOUD_APP_ID = config["discloud_app"]
        
        headers = {
            "api-token": DISCLOUD_API_TOKEN,
            "Content-Type": "application/json"
        }

        status_url = f"{DISCLOUD_API_BASE_URL}/app/{DISCLOUD_APP_ID}/status"
        response = await httpx_client.get(status_url, headers=headers)
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)

        data = response.json()["apps"]['memory']
        logging.info(f"RAM: {data} for app {DISCLOUD_APP_ID}")
        return float(data.split("MB/")[0])

    except Exception as e:
        logging.exception(f"An unhandled error occurred checking Discloud status")
        return 0
        
        
@discord_bot.tree.command(name="model", description="View or switch the current model")
async def model_command(interaction: discord.Interaction, model: str) -> None:
    global curr_model

    if model == curr_model:
        output = f"Current model: `{curr_model}`"
    else:
        if user_is_admin := interaction.user.id in config["permissions"]["users"]["admin_ids"]:
            curr_model = model
            output = f"Model switched to: `{model}`"
            logging.info(output)
        else:
            output = "You don't have permission to change the model."

    await interaction.response.send_message(output, ephemeral=(interaction.channel.type == discord.ChannelType.private))


@model_command.autocomplete("model")
async def model_autocomplete(interaction: discord.Interaction, curr_str: str) -> list[Choice[str]]:
    global config

    #if curr_str == "":
    #    config = await asyncio.to_thread(get_config)

    choices = [Choice(name=f"○ {model}", value=model) for model in config["models"] if model != curr_model and curr_str.lower() in model.lower()][:24]
    choices += [Choice(name=f"◉ {curr_model} (current)", value=curr_model)] if curr_str.lower() in curr_model.lower() else []

    return choices


@discord_bot.event
async def on_ready() -> None:
    if client_id := config["client_id"]:
        logging.info(f"\n\nBOT INVITE URL:\nhttps://discord.com/oauth2/authorize?client_id={client_id}&permissions=412317191168&scope=bot\n")

    await discord_bot.tree.sync()


@discord_bot.event
async def on_message(new_msg: discord.Message) -> None:
    try:
        global last_task_time

        is_dm = new_msg.channel.type == discord.ChannelType.private

        if (not is_dm and discord_bot.user not in new_msg.mentions) or new_msg.author.bot:
            return

        role_ids = set(role.id for role in getattr(new_msg.author, "roles", ()))
        channel_ids = set(filter(None, (new_msg.channel.id, getattr(new_msg.channel, "parent_id", None), getattr(new_msg.channel, "category_id", None))))

        # config = await asyncio.to_thread(get_config)

        allow_dms = config.get("allow_dms", True)

        permissions = config["permissions"]

        user_is_admin = new_msg.author.id in permissions["users"]["admin_ids"]

        (allowed_user_ids, blocked_user_ids), (allowed_role_ids, blocked_role_ids), (allowed_channel_ids, blocked_channel_ids) = (
            (perm["allowed_ids"], perm["blocked_ids"]) for perm in (permissions["users"], permissions["roles"], permissions["channels"])
        )

        allow_all_users = not allowed_user_ids if is_dm else not allowed_user_ids and not allowed_role_ids
        is_good_user = user_is_admin or allow_all_users or new_msg.author.id in allowed_user_ids or any(id in allowed_role_ids for id in role_ids)
        is_bad_user = not is_good_user or new_msg.author.id in blocked_user_ids or any(id in blocked_role_ids for id in role_ids)

        allow_all_channels = not allowed_channel_ids
        is_good_channel = user_is_admin or allow_dms if is_dm else allow_all_channels or any(id in allowed_channel_ids for id in channel_ids)
        is_bad_channel = not is_good_channel or any(id in blocked_channel_ids for id in channel_ids)

        if is_bad_user or is_bad_channel:
            return

        provider_slash_model = curr_model
        provider, model = provider_slash_model.removesuffix(":vision").split("/", 1)

        provider_config = config["providers"][provider]

        base_url = provider_config["base_url"]
        api_key = provider_config.get("api_key", "sk-no-key-required")
        openai_client = AsyncOpenAI(base_url=base_url, api_key=api_key)

        model_parameters = config["models"].get(provider_slash_model, None)

        extra_headers = provider_config.get("extra_headers", None)
        extra_query = provider_config.get("extra_query", None)
        extra_body = (provider_config.get("extra_body", None) or {}) | (model_parameters or {}) or None

        accept_images = any(x in provider_slash_model.lower() for x in VISION_MODEL_TAGS)
        accept_usernames = any(x in provider_slash_model.lower() for x in PROVIDERS_SUPPORTING_USERNAMES)

        max_text = config.get("max_text", 100000)
        max_images = config.get("max_images", 5) if accept_images else 0
        max_messages = config.get("max_messages", 25)

        # Build message chain and set user warnings
        messages = []
        user_warnings = set()
        curr_msg = new_msg

        while curr_msg != None and len(messages) < max_messages:
            curr_node = msg_nodes.setdefault(curr_msg.id, MsgNode())

            async with curr_node.lock:
                if curr_node.text == None:
                    cleaned_content = curr_msg.content.removeprefix(discord_bot.user.mention).lstrip()

                    good_attachments = [att for att in curr_msg.attachments if att.content_type and any(att.content_type.startswith(x) for x in ("text", "image"))]
                    if config.get("cloudinary_name"):
                        for att in good_attachments:
                            parsedurl = urllib.parse.quote_plus(att.url)
                            if att.content_type.startswith("image") and len(parsedurl) <= 255:
                                att.url = f"https://res.cloudinary.com/{config['cloudinary_name']}/image/fetch/c_limit,h_{IMAGE_LIMIT},w_{IMAGE_LIMIT}/{parsedurl}"

                    attachment_responses = await asyncio.gather(*[httpx_client.get(att.url) for att in good_attachments])
                    attachment_responses  = [res for res in attachment_responses if not res.is_error]

                    curr_node.text = "\n".join(
                        ([cleaned_content] if cleaned_content else [])
                        + ["\n".join(filter(None, (embed.title, embed.description, embed.footer.text))) for embed in curr_msg.embeds]
                    + [component.content for component in curr_msg.components if component.type == discord.ComponentType.text_display]
                        + [resp.text for att, resp in zip(good_attachments, attachment_responses) if att.content_type.startswith("text")]
                    )

                    curr_node.images = [
                        dict(type="image_url", image_url=dict(url=f"data:{att.content_type};base64,{b64encode(resp.content).decode('utf-8')}"))
                        for att, resp in zip(good_attachments, attachment_responses)
                        if att.content_type.startswith("image")
                    ]

                    curr_node.role = "assistant" if curr_msg.author == discord_bot.user else "user"

                    curr_node.user_id = curr_msg.author.id if curr_node.role == "user" else None

                    curr_node.has_bad_attachments = len(curr_msg.attachments) > len(good_attachments)

                    try:
                        if (
                            curr_msg.reference == None
                            and discord_bot.user.mention not in curr_msg.content
                            and (prev_msg_in_channel := ([m async for m in curr_msg.channel.history(before=curr_msg, limit=1)] or [None])[0])
                            and prev_msg_in_channel.type in (discord.MessageType.default, discord.MessageType.reply)
                            and prev_msg_in_channel.author == (discord_bot.user if curr_msg.channel.type == discord.ChannelType.private else curr_msg.author)
                        ):
                            curr_node.parent_msg = prev_msg_in_channel
                        else:
                            is_public_thread = curr_msg.channel.type == discord.ChannelType.public_thread
                            parent_is_thread_start = is_public_thread and curr_msg.reference == None and curr_msg.channel.parent.type == discord.ChannelType.text

                            if parent_msg_id := curr_msg.channel.id if parent_is_thread_start else getattr(curr_msg.reference, "message_id", None):
                                if parent_is_thread_start:
                                    curr_node.parent_msg = curr_msg.channel.starter_message or await curr_msg.channel.parent.fetch_message(parent_msg_id)
                                else:
                                    curr_node.parent_msg = curr_msg.reference.cached_message or await curr_msg.channel.fetch_message(parent_msg_id)

                    except (discord.NotFound, discord.HTTPException):
                        logging.exception("Error fetching next message in the chain")
                        curr_node.fetch_parent_failed = True

                if curr_node.images[:max_images]:
                    content = ([dict(type="text", text=curr_node.text[:max_text])] if curr_node.text[:max_text] else []) + curr_node.images[:max_images]
                else:
                    content = curr_node.text[:max_text]

                if content != "":
                    message = dict(content=content, role=curr_node.role)
                    if accept_usernames and curr_node.user_id != None:
                        message["name"] = str(curr_node.user_id)

                    messages.append(message)

                if len(curr_node.text) > max_text:
                    user_warnings.add(f"⚠️ Max {max_text:,} characters per message")
                if len(curr_node.images) > max_images:
                    user_warnings.add(f"⚠️ Max {max_images} image{'' if max_images == 1 else 's'} per message" if max_images > 0 else "⚠️ Can't see images")
                if curr_node.has_bad_attachments:
                    user_warnings.add("⚠️ Unsupported attachments")
                if curr_node.fetch_parent_failed or (curr_node.parent_msg != None and len(messages) == max_messages):
                    user_warnings.add(f"⚠️ Only using last {len(messages)} message{'' if len(messages) == 1 else 's'}")

                curr_msg = curr_node.parent_msg

        logging.info(f"Message received (user ID: {new_msg.author.id}, attachments: {len(new_msg.attachments)}, conversation length: {len(messages)}):\n{new_msg.content}")

        if system_prompt := config["system_prompt"]:
            now = datetime.now().astimezone()

            system_prompt = system_prompt.replace("{date}", now.strftime("%B %d %Y")).replace("{time}", now.strftime("%H:%M:%S %Z%z")).strip()
            system_prompt = system_prompt.replace("{userid}", str(new_msg.author.id))
            now_jst = datetime.now(timezone.utc).astimezone(ZoneInfo('Asia/Tokyo'))
            system_prompt = system_prompt.replace("{date_jst}", now_jst.strftime("%B %d %Y")).replace("{time_jst}", now_jst.strftime("%H:%M:%S %Z%z"))
            
            if accept_usernames:
                system_prompt += "\n\nUser's names are their Discord IDs and should be typed as '<@ID>'."

            messages.append(dict(role="system", content=system_prompt))

        # --- Tool Definition ---
        tools = []
        if config.get("serpapi_api_key"):
            tools.append({
                "type": "function",
                "function": {
                    "name": "google_search",
                    "description": "Get information from the web for recent events or specific facts.",
                    "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "The search query to use."}}, "required": ["query"]},
                },
            })
            
        # Add the VNDB tool to the list of available tools
        tools.append({
            "type": "function",
            "function": {
                "name": "search_vndb",
                "description": "Search for a visual novel (VN) on VNDB.org to get its title, description, rating, and other details.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string", "description": "The title or search term for the visual novel."}},
                    "required": ["query"],
                },
            },
        })
        tools.append({
            "type": "function",
            "function": {
                "name": "search_anilist",
                "description": "Search for anime or manga on AniList to get details like description, score, status, episodes, etc.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The title of the anime or manga to search for."},
                        "media_type": {"type": "string", "enum": ["ANIME", "MANGA"], "description": "The type of media to search for."}
                    },
                    "required": ["query", "media_type"],
                },
            },
        })
        
        if user_is_admin:
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


        conversation_history = messages[::-1]
        openai_kwargs = dict(model=model, messages=conversation_history, stream=True, extra_body=extra_body)
        if tools:
            openai_kwargs["tools"] = tools
            openai_kwargs["tool_choice"] = "auto"
        
        response_msgs, response_contents = [], []
        use_plain_responses = config.get("use_plain_responses", False)
        max_message_length = 4000 if use_plain_responses else 3900
        
        async def reply_helper(**kwargs):
            reply_target = new_msg if not response_msgs else response_msgs[-1]
            msg = await reply_target.reply(**kwargs)
            response_msgs.append(msg)
            msg_nodes[msg.id] = MsgNode(parent_msg=new_msg)
            await msg_nodes[msg.id].lock.acquire()
            return msg

        async def stream_and_update(stream, executed_tools: list[str] = None):
            global last_task_time
            curr_content = finish_reason = None
            tool_call_assembler = defaultdict(lambda: {"id": "", "function": {"name": "", "arguments": ""}})
            if use_plain_responses := config.get("use_plain_responses", False):
                max_message_length = 4000
            else:
                max_message_length = 4096 - len(STREAMING_INDICATOR)
                embed = discord.Embed.from_dict(dict(fields=[dict(name=warning, value="", inline=False) for warning in sorted(user_warnings)]))
            # --- Prepare footer text if tools were used ---
            footer_text = None
            if executed_tools:
                footer_text = " | ".join(executed_tools)
                if len(footer_text) > 2048:  # Discord footer limit
                    footer_text = footer_text[:2045] + "..."
                    
            async for chunk in stream:
                if finish_reason != None:
                    break
                    
                if not (choice := chunk.choices[0] if chunk.choices else None):
                    continue
                    
                # Assemble tool calls if they are in the delta
                if choice.delta.tool_calls:
                    for tc_chunk in choice.delta.tool_calls:
                        asm = tool_call_assembler[tc_chunk.index]
                        if tc_chunk.id: asm["id"] = tc_chunk.id
                        if tc_chunk.function:
                            asm["function"]["name"] += tc_chunk.function.name or ""
                            asm["function"]["arguments"] += tc_chunk.function.arguments or ""

                finish_reason = choice.finish_reason

                prev_content = curr_content or ""
                curr_content = choice.delta.content or ""

                new_content = prev_content if finish_reason == None else (prev_content + curr_content)

                if response_contents == [] and new_content == "":
                    continue

                if start_next_msg := response_contents == [] or len(response_contents[-1] + new_content) > max_message_length:
                    response_contents.append("")

                response_contents[-1] += new_content

                if not use_plain_responses:
                    time_delta = datetime.now().timestamp() - last_task_time

                    ready_to_edit = time_delta >= EDIT_DELAY_SECONDS
                    msg_split_incoming = finish_reason == None and len(response_contents[-1] + curr_content) > max_message_length
                    is_final_edit = finish_reason != None or msg_split_incoming
                    is_good_finish = finish_reason != None and finish_reason.lower() in ("stop", "end_turn")

                    if start_next_msg or ready_to_edit or is_final_edit:
                        embed.description = response_contents[-1] if is_final_edit else (response_contents[-1] + STREAMING_INDICATOR)
                        embed.color = EMBED_COLOR_COMPLETE if msg_split_incoming or is_good_finish else EMBED_COLOR_INCOMPLETE
                        if footer_text:
                            embed.set_footer(text=footer_text)

                        if start_next_msg:
                            await reply_helper(embed=embed, silent=True)
                        else:
                            await asyncio.sleep(EDIT_DELAY_SECONDS - time_delta)
                            await response_msgs[-1].edit(embed=embed)

                        last_task_time = datetime.now().timestamp()

            if use_plain_responses:
                for content in response_contents:
                    await reply_helper(view=LayoutView().add_item(TextDisplay(content=content)))
                    
            # Return assembled tool calls if the reason for finishing was a tool call
            if finish_reason == 'tool_calls':
                return [
                    {"id": asm["id"], "type": "function", "function": asm["function"]}
                    for _, asm in sorted(tool_call_assembler.items())
                ]
            return None

        try:
            async with new_msg.channel.typing():
                # --- First Streaming Attempt ---
                stream = await openai_client.chat.completions.create(**openai_kwargs)
                requested_tool_calls = await stream_and_update(stream)
                executed_tool_calls = []

                # --- If Tool Calls Were Requested ---
                if requested_tool_calls:
                    
                    
                    tool_messages = []
                    for tool_call in requested_tool_calls:
                        func_name = tool_call["function"]["name"]
                        try:
                            args = tool_call["function"]["arguments"]
                            args = json.loads(args)
                            query = args.get("query", "")
                            
                            # Truncate query for display
                            query_display = (query[:25] + '...') if len(query) > 25 else query

                            if func_name == "google_search":
                                executed_tool_calls.append(f"🔎 Google: '{query_display}'")
                                search_results = await perform_google_search(query)
                            elif func_name == "search_vndb":
                                executed_tool_calls.append(f"📚 VNDB: '{query_display}'")
                                search_results = await search_vndb(query)
                            elif func_name == "search_anilist":
                                media_type = args.get("media_type", "ANIME")
                                executed_tool_calls.append(f"💽 AniList/{media_type.title()}: '{query_display}'")
                                search_results = await search_anilist(query, media_type)
                            elif func_name == "execute_python_code":
                                code = args.get("code", "")
                                code_display = code
                                executed_tool_calls.append(f"🐍 Python: \n'{code_display}'")
                                search_results = await execute_python_code(code)
                            else:
                                search_results = "Unknown tool called."

                            tool_messages.append({"role": "tool", "tool_call_id": tool_call["id"], "name": func_name, "content": search_results})
                            
                            conversation_history.append({"role": "assistant", "content": None, "tool_calls": requested_tool_calls})
                            # Append tool results to history
                            conversation_history.extend(tool_messages)

                        except json.JSONDecodeError:
                            logging.exception(f"Failed to decode tool arguments. {tool_call}")
                            executed_tool_calls.append(f"❌ Error using tool {func_name}'")
                    
                    # --- Second Streaming Attempt with Tool Results ---
                    openai_kwargs['messages'] = conversation_history
                    final_stream = await openai_client.chat.completions.create(**openai_kwargs)
                    await stream_and_update(final_stream, executed_tools=executed_tool_calls)

        except Exception as e:
            raise e

        for response_msg in response_msgs:
            msg_nodes[response_msg.id].text = "".join(response_contents)
            msg_nodes[response_msg.id].lock.release()

        # Delete oldest MsgNodes (lowest message IDs) from the cache
        if (num_nodes := len(msg_nodes)) > MAX_MESSAGE_NODES:
            for msg_id in sorted(msg_nodes.keys())[: num_nodes - MAX_MESSAGE_NODES]:
                async with msg_nodes.setdefault(msg_id, MsgNode()).lock:
                    msg_nodes.pop(msg_id, None)
                    
        if (await check_discloud_ram()) > RESTART_LIMIT_MB:
            logging.info("RAM is getting high, restarting app")
            await restart_discloud_app()
    except Exception as e:
        logging.exception("Error during message event")
        try:
            async with new_msg.channel.typing():
                msg_nodes.clear()
                errorname = e.__class__.__name__
                content = f"⚠️ Error Generating Message: {errorname}"
                response_msg = await new_msg.reply(content=content, suppress_embeds=True)
                if isinstance(e, MemoryError):
                    await restart_discloud_app()
        except Exception:
            logging.exception("Nested Error while generating exception warning")

schedule.every(48).hours.do(lambda: asyncio.create_task(restart_discloud_app()))

async def run_schedule_continuously():
    if config.get("discloud_token"):
        while True:
            schedule.run_pending()
            await asyncio.sleep(3600) 


async def main() -> None:
    try:
        bot_task = asyncio.create_task(discord_bot.start(config["bot_token"]))
        schedule_task = asyncio.create_task(run_schedule_continuously())
        await asyncio.gather(bot_task, schedule_task)
    except Exception as e:
        logging.exception("Critical error during bot, restarting in 30 seconds")
        await discord_bot.close()
        await asyncio.sleep(30) 
        await restart_discloud_app()
    


try:
    asyncio.run(main())
except KeyboardInterrupt:
    pass
