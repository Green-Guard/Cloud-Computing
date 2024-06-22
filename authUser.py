from flask import Flask, request, jsonify
import pyrebase

# Firebase configuration
config = {
    "apiKey": "AIzaSyDaOYJwzFdSz312lxDlSMoaB8P5447ulGk",
    "authDomain": "greenguard1.firebaseapp.com",
    "databaseURL": "https://greenguard1-default-rtdb.asia-southeast1.firebasedatabase.app/",
    "projectId": "greenguard1",
    "storageBucket": "greenguard1.appspot.com",
    "messagingSenderId": "885257974291",
    "appId": "1:885257974291:android:52a111c20c628032c885e6"
}

# Initialize Firebase
firebase = pyrebase.initialize_app(config)
auth = firebase.auth()
db = firebase.database()

app = Flask(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True

@app.route('/register', methods=['POST'])
def register():
    try:
        email = request.json['email']
        password = request.json['password']
        user = auth.create_user_with_email_and_password(email, password)
        return jsonify({"message": "User registered successfully", "user": user}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/login', methods=['POST'])
def login():
    try:
        email = request.json['email']
        password = request.json['password']
        user = auth.sign_in_with_email_and_password(email, password)
        return jsonify({"message": "User logged in successfully", "user": user}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/logout', methods=['POST'])
def logout():
    try:
        # Here you would handle the token invalidation logic if needed
        return jsonify({"message": "User logged out successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/add_data', methods=['POST'])
def add_data():
    try:
        data = request.json
        db.push(data)
        return jsonify({"message": "Data added successfully"}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
