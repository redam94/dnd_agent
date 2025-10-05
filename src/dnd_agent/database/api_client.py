from typing import Any, Dict

import httpx


class DnD5eAPIClient:
    """Client for D&D 5e API"""

    def __init__(self, base_url: str = "https://www.dnd5eapi.co"):
        self.base_url = base_url
        self.client = httpx.Client(timeout=30.0)

    def close(self):
        self.client.close()

    def get_resource(self, endpoint: str, index: str = None, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Get a resource from the D&D 5e API"""
        if index:
            url = f"{self.base_url}/api/2014/{endpoint}/{index}"
        else:
            url = f"{self.base_url}/api/2014/{endpoint}"
        
        if params is None:
            params = {}
        

        try:
            response = self.client.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            return {"error": str(e)}
