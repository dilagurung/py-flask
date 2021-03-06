#https://www.youtube.com/watch?v=s_ht4AKnWZg https://kirankoduru.github.io/python/python_if_name_main.html
#https://pythonhow.com/using-virtualenv-with-your-flask-app/
#https://www.youtube.com/watch?v=Us9DuF4KWUE
#https://flask-cors.readthedocs.io/en/latest/  py charm https://www.youtube.com/watch?v=ZVGwqnjOKjk
#https://stackoverflow.com/questions/19135867/what-is-pips-equivalent-of-npm-install-package-save-dev
from flask import Flask,request,jsonify
from flask_restful import Resource,Api
from flask_cors import CORS,cross_origin
from custom_cors import crossdomain
app=Flask(__name__)
CORS(app, support_credentials=True)
#api=Api(app)
#@crossdomain(origin='*',headers=['Access-Control-Allow-Methods: DELETE, GET, HEAD, OPTIONS, PATCH, POST, PUT','Access-Control-Allow-Headers: accept, content-type','Access-Control-Allow-Origin:http://localhost:4200','Content-Type:application/json'])
@app.route('/api/test', methods=['POST', 'GET','OPTIONS'])
@cross_origin(supports_credentials=True)
def index():
    if(request.method=='POST'):
     some_json=request.get_json()
     return jsonify({"key":some_json})
    else:
        return jsonify({"GET":"GET"})

@app.route('/multi/<int:num>',methods=['GET'])
def getMultiplication(num):
    return jsonify({'resule':num*10})
"""
...
api=Api(app)

class ClassOne(Resource):
    @crossdomain(origin='*')
    def get(self):
        return {"key":"one"}

    @crossdomain(origin='*')
    def post(self):
        some_json=request.get_json()
        return {'key':'modified '+some_json},201

class ClassTwo(Resource):
    def get(self,num):
        return {'result':num*20}

api.add_resource(ClassOne,'/api/test')
api.add_resource(ClassTwo,'/multi/<int:num>')

...
"""


if __name__=="__main__":
    app.run(host='0.0.0.0', port=5000)
