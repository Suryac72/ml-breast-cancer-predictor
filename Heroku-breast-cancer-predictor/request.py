import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'Mean Radius':11, 'Mean Texture':17, 'Mean Perimeter':130,'Mean Area':1020,'Mean Smoothness':1})

print(r.json())