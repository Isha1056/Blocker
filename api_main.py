from flask import Flask, render_template, jsonify, request
from flask_cors import CORS, cross_origin
import json
from bs4 import BeautifulSoup

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

def remove_tags(html):
  
    # parse html content
    soup = BeautifulSoup(html, "html.parser")
  
    for data in soup(['style', 'script']):
        # Remove tags
        data.decompose()
    
    # return data by retrieving the tag content
    return list(soup.stripped_strings)

@app.route('/', methods=['POST', 'GET'])
def get_data():
    if request.method=='POST':
        json1=request.get_json()
        # json1 = {"title": "html", "status": "recieved"}
        #pay = json.dumps(json1)
        page = json1['title']
        # print(json1['title'])
        page = remove_tags(json1['title'])
        print(page)
        response = jsonify(
            response=page,
            mimetype='application/json'
        )
        return response

if __name__ == "__main__":
    app.run(debug=True)