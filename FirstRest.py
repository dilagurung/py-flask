#https://www.youtube.com/watch?v=s_ht4AKnWZg https://kirankoduru.github.io/python/python_if_name_main.html
#https://pythonhow.com/using-virtualenv-with-your-flask-app/
#https://www.youtube.com/watch?v=Us9DuF4KWUE
#https://flask-cors.readthedocs.io/en/latest/  py charm https://www.youtube.com/watch?v=ZVGwqnjOKjk
from flask import Flask,request,jsonify
from flask_restful import Resource,Api
from flask_cors import CORS,cross_origin

app=Flask(__name__)
app.config['SECRET_KEY'] = 'the quick brown fox jumps over the lazy   dog'
app.config['CORS_HEADERS'] = 'Content-Type'
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})

"""
@app.route('/',methods=['GET','POST'])
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

#cors = CORS(app, resources={r"/api/*": {"origins": "*"}})
api=Api(app)

class ClassOne(Resource):
    @cross_origin(origin='*', headers=['Content- Type', 'Authorization','Access-Control-Allow-Origin'])
    def get(self):
        return {"key":"one"}

    @cross_origin(origin='*', headers=['Content- Type', 'Authorization','Access-Control-Allow-Origin'])
    def post(self):
        some_json=request.get_json()
        return {'key':'modified '+some_json},201

class ClassTwo(Resource):
    def get(self,num):
        return {'result':num*20}


api.add_resource(ClassOne,'/api/test')
api.add_resource(ClassTwo,'/multi/<int:num>')




if __name__=="__main__":
    app.run(debug=True)
