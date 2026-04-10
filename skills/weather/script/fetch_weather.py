"""Fetch current weather for a city using wttr.in API.
---
{"type":"object","properties":{"city":{"type":"string","description":"城市名（中文或英文，如 北京 / Beijing）"}},"required":["city"]}
"""

import json
import socket
import urllib.parse
import urllib.request
import urllib.error
import http.client
import time


def run(params: dict) -> str:
    """Fetch weather data for a city from wttr.in."""
    city = params.get("city", "").strip()
    if not city:
        return json.dumps({"error": "city parameter is required"}, ensure_ascii=False)

    max_retries = 2
    for attempt in range(max_retries + 1):
        try:
            url = f"https://wttr.in/{urllib.parse.quote(city)}?format=j1&lang=zh"
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "curl/7.68.0"},
            )
            # Increased timeout to handle slow network responses
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            current_condition = data.get("current_condition", [])
            if not current_condition:
                return json.dumps(
                    {"status": "error", "city": city, "error": f"Failed to fetch weather data for '{city}'. The location might be ambiguous or unsupported by the weather service. Try providing a more specific city name (e.g., '平湖市,Pinghu' or 'Pinghu,Zhejiang')."},
                    ensure_ascii=False,
                )

            current = current_condition[0]

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
            if e.code == 500:
                return json.dumps(
                    {"status": "error", "city": city, "error": f"Failed to fetch weather data for '{city}'. The location might be ambiguous or unsupported by the weather service. Try providing a more specific city name (e.g., '平湖市,Pinghu' or 'Pinghu,Zhejiang')."},
                    ensure_ascii=False,
                )
            return json.dumps(
                {"status": "error", "city": city, "error": f"HTTP {e.code}: {e.reason}"},
                ensure_ascii=False,
            )
        except (http.client.RemoteDisconnected, ConnectionResetError, BrokenPipeError) as e:
            if attempt < max_retries:
                time.sleep(1 * (attempt + 1))
                continue
            return json.dumps(
                {"status": "error", "city": city, "error": f"Network Error: The remote server closed the connection unexpectedly. Please try again later."},
                ensure_ascii=False,
            )
        except (urllib.error.URLError, socket.timeout, TimeoutError) as e:
            if attempt < max_retries:
                time.sleep(1 * (attempt + 1))
                continue
            return json.dumps(
                {"status": "error", "city": city, "error": f"Network Error: Failed to connect to the weather service due to a timeout. Please check your network connection and try again later."},
                ensure_ascii=False,
            )
        except Exception as e:
            return json.dumps(
                {"status": "error", "city": city, "error": str(e)},
                ensure_ascii=False,
            )
    
    return json.dumps(
        {"status": "error", "city": city, "error": "Network Error: The remote server closed the connection unexpectedly. Please try again later."},
        ensure_ascii=False,
    )