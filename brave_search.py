import requests
import os

BRAVE_API_KEY = os.getenv("BRAVE_API_KEY")

def brave_search(query, k=5):
    url = "https://api.search.brave.com/res/v1/web/search"
    headers = {"X-Subscription-Token": BRAVE_API_KEY}
    params = {"q": query, "count": k}

    resp = requests.get(url, headers=headers, params=params)
    data = resp.json()

    resultados = []
    for item in data.get("web", {}).get("results", []):
        resultados.append(item.get("description", ""))

    return resultados
