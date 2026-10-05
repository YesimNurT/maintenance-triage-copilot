"""Small HTTP client for the API, used by the Streamlit app."""

from typing import Any


class ApiClient:
    def __init__(self, base_url: str, session: Any = None) -> None:
        if session is None:
            import requests  # optional dependency of the UI

            session = requests.Session()
        self.base_url, self.session = base_url.rstrip("/"), session

    def _json(self, response: Any) -> Any:
        response.raise_for_status()
        return response.json()

    def ranking(self, limit: int) -> list[dict[str, Any]]:
        return self._json(self.session.get(f"{self.base_url}/ranking", params={"limit": limit}))

    def note(self, flight_id: int) -> dict[str, Any]:
        return self._json(self.session.post(f"{self.base_url}/flights/{flight_id}/note"))

    def review(self, flight_id: int, decision: str, comment: str = "") -> dict[str, Any]:
        body = {"decision": decision, "comment": comment}
        return self._json(
            self.session.post(f"{self.base_url}/flights/{flight_id}/review", json=body)
        )
