"""Fetch current weather for a city using wttr.in API.
---
{"type":"object","properties":{"city":{"type":"string","description":"城市名（中文或英文，如 北京 / Beijing）"}},"required":["city"]}
"""

import json
import urllib.parse
import urllib.request
import urllib.error


def run(params: dict) -> str:
    """Fetch weather data for a city from wttr.in."""
    city = params.get("city", "").strip()
    if not city:
        return json.dumps({"error": "city parameter is required"}, ensure_ascii=False)

    try:
        url = f"https://wttr.in/{urllib.parse.quote(city)}?format=j1"
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "curl/7.68.0"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        current = data.get("current_condition", [{}])[0]

        return json.dumps(
            {
                "status": "ok",
                "city": city,
                "temperature": f"{current.get('temp_C', 'N/A')}°C",
                "feels_like": f"{current.get('FeelsLikeC', 'N/A')}°C",
                "weather": current.get("weatherDesc", [{}])[0].get("value", "N/A"),
                "wind_speed": f"{current.get('windspeedKmph', 'N/A')} km/h",
                "wind_direction": current.get("winddir16Point", "N/A"),
                "humidity": f"{current.get('humidity', 'N/A')}%",
                "visibility": f"{current.get('visibility', 'N/A')} km",
                "uv_index": current.get("uvIndex", "N/A"),
                "pressure": f"{current.get('pressure', 'N/A')} hPa",
            },
            ensure_ascii=False,
        )

    except urllib.error.HTTPError as e:
        return json.dumps(
            {"status": "error", "city": city, "error": f"HTTP {e.code}: {e.reason}"},
            ensure_ascii=False,
        )
    except Exception as e:
        return json.dumps(
            {"status": "error", "city": city, "error": str(e)},
            ensure_ascii=False,
        )
