from flask import Flask,request,jsonify
import pickle
import pandas as pd

with open("wine_decisiontree_trained_model.pkl","rb") as file_reading_obj:
    train_model=pickle.load(file_reading_obj)


app_name=Flask(__name__) #__name__ is an attribute or magic method

@app_name.route("/decisiontree_model_predict",methods=["POST"])
def decisiontree_prediction():
    data=request.get_json()#getting the data from request in the form of JSON
    user_input=data.get('my_input')
    print(user_input)
    print("##################################")
#checking if the user input is valid or not and checks if the key is appropriate from the dictionary by means of list
    if not user_input or not isinstance(user_input,list):
     return jsonify({'error msg':'please validate your input'}),400


    data_input = pd.DataFrame(user_input)
    print(data_input.head())
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    predicted_output=train_model.predict(data_input)
    print(predicted_output)
    return jsonify({'data': predicted_output.tolist()})
    #except Exception as e:
    #return jsonify({'error': str(e)}), 500

@app_name.route("/")
def landing():
    return "welcome to Uptor"

@app_name.route("/login")
def login():
    return "welcome to my login"

#since python runs the programme in a sequential order we have to ask python to start from main
if __name__ =="__main__":
    app_name.run(debug=True)