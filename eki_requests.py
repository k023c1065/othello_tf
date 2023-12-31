import requests

#呼び出すAPIのURL
url = "https://express.heartrails.com/api/json"

#パラメータ設定
param = {"method": "getStations", "line": "JR京浜東北線","prefecture":"東京都" }
data = requests.get(url, param).json()
for v in data["response"]["station"]:
    print(v["name"])