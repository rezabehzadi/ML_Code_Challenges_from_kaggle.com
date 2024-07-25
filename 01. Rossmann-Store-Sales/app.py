from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the trained model 
with open("random_forest_model.pkl", "rb") as f:
    model = pickle.load(f)

# Route for the root URL ("/")
@app.route("/")
def index():
    return render_template("index.html")  # Render the index.html template

# Route for predictions ("/predict") with POST method
@app.route("/predict", methods=["POST"])
def predict(): 
    data= [int(x) for x in request.form.values()]
    prediction = model.predict([data])[0] 
    return render_template('index.html', prediction_text=f"prediction Sales: {prediction: 0.2f}")

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)