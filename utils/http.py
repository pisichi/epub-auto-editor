import aiohttp

async def post_request(session, url: str, data: dict, timeout: int = 5000) -> dict:
    async with session.post(url, json=data, timeout=timeout) as response:
        return await response.json()