import asyncio
from base64 import b64encode
from dataclasses import dataclass, field
from datetime import datetime, timezone 
import logging
from typing import Any, Literal, Optional

import discord
from discord.app_commands import Choice
from discord.ext import commands
import httpx
from openai import AsyncOpenAI
import yaml
import platform
import urllib.parse
from zoneinfo import ZoneInfo;
import schedule
import re
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)

def get_config(filename: str = "config.yaml") -> dict[str, Any]:
    with open(filename, encoding="utf-8") as file:
        return yaml.safe_load(file)


config = get_config()

DISCLOUD_API_BASE_URL = "https://api.discloud.app/v2"
DISCLOUD_API_TOKEN = config["discloud_token"]
DISCLOUD_APP_ID = config["discloud_app"]
            

async def discloud_app():
    try:
        print(f" Attempting to check Discloud app: {DISCLOUD_APP_ID}...")

        headers = {
            "api-token": DISCLOUD_API_TOKEN,
            "Content-Type": "application/json"
        }

        status = f"{DISCLOUD_API_BASE_URL}/app/{DISCLOUD_APP_ID}/status"
        async with httpx.AsyncClient(timeout=30.0) as client:
            status_response = await client.get(status, headers=headers)
        status_response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)

        logs = f"{DISCLOUD_API_BASE_URL}/app/{DISCLOUD_APP_ID}/logs"
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(logs, headers=headers)
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)

        data = response.json()['apps']['terminal']['big']
        print(f"Successfully requested logs for app: {DISCLOUD_APP_ID}. API response: " + "\n" + str(data).replace('\\n', '\n'))
        
        data = status_response.json()["apps"]
        try:
            started_at_jst = datetime.fromisoformat(re.sub(r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?)\d*Z?$', r'\1', data["startedAt"])).replace(tzinfo=ZoneInfo("UTC")).astimezone(ZoneInfo("Asia/Tokyo")).strftime('%Y-%m-%d %H:%M:%S JST')
        except:
            started_at_jst = data["startedAt"]
        
        print(f"App Status: Container: {data['container']}, Memory: {data['memory']}, Last Restart: {data['last_restart']}, Started At: {started_at_jst}")

    except Exception as e:
        logging.exception(f"An unhandled error occurred checking Discloud")
        
        

async def start_discloud_app():
    try:
        print(f" Attempting to start Discloud app: {DISCLOUD_APP_ID}...")

        headers = {
            "api-token": DISCLOUD_API_TOKEN
        }

        start = f"{DISCLOUD_API_BASE_URL}/app/{DISCLOUD_APP_ID}/start"
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.put(start, headers=headers)
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)

        data = response.json()
        print(f"Successfully started app: {DISCLOUD_APP_ID}. API response: " + "\n" + str(data))
        
    except Exception as e:
        logging.exception(f"An unhandled error occurred starting Discloud")
        
async def stop_discloud_app():
    try:
        print(f" Attempting to stop Discloud app: {DISCLOUD_APP_ID}...")

        headers = {
            "api-token": DISCLOUD_API_TOKEN
        }

        stop = f"{DISCLOUD_API_BASE_URL}/app/{DISCLOUD_APP_ID}/stop"
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.put(stop, headers=headers)
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)

        data = response.json()
        print(f"Successfully stopped app: {DISCLOUD_APP_ID}. API response: " + "\n" + str(data))
        
    except Exception as e:
        logging.exception(f"An unhandled error occurred stopping Discloud")

async def main() -> None:
    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == "start":
            await start_discloud_app()
        elif command == "stop":
            await stop_discloud_app()
        else:
            print(f"Unknown command: {command}. Defaulting to status.")
            await discloud_app()
    else:
        print("No command specified. Defaulting to status.")
        await discloud_app()
    


asyncio.run(main())