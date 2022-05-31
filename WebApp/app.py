from distutils.log import info
from msilib.schema import Class
from time import perf_counter
from flask import Flask, redirect, url_for, render_template, request, session, flash, get_flashed_messages, send_file
import secrets
from datetime import timedelta
from flask_sqlalchemy import SQLAlchemy
import os
import hashlib
import re
from werkzeug.utils import secure_filename
from architecture import *
from predict import *
import sys




app = Flask(__name__)
app.config["SECRET_KEY"] = secrets.token_hex(16)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///accounts.db"
app.config["SQLALCHEMY_TRACK_MODIFICATION"] = False
app.config["UPLOAD_PATH"] = "./Liver_Lesions_Segmentaiton/files/uploaded"
app.config["SAVE_PATH"] = "./Liver_Lesions_Segmentaiton/files/predicted"
app.config["DOWNLOAD_PATH"] = "files/predicted"
app.permanent_session_lifetime = timedelta(minutes= 20)

# liver_model_path = "./Liver_Lesions_Segmentaiton/models/liver_weights_best.hdf5"
# tumor_model_path = "./Liver_Lesions_Segmentaiton/models/tumor_weights_best.hdf5"

# input_shape = (512, 512, 3)
# liver_model = build_resnet50_unet(input_shape)
# liver_model.load_weights(liver_model_path)

# input_shape = (320, 320, 3)
# tumor_model = build_resnet50_unet1(input_shape)
# tumor_model.load_weights(tumor_model_path)


liver_model_path = "./Liver_Lesions_Segmentaiton/models/liver_weights_best.h5"
tumor_model_path = "./Liver_Lesions_Segmentaiton/models/tumor-liver-crops-512x512_weights_best.h5"
liver_model = load_model(liver_model_path)
tumor_model = load_model(tumor_model_path)


db = SQLAlchemy(app)

class User(db.Model):
    user_id = db.Column(db.Integer, primary_key = True)
    username = db.Column(db.String(100))
    email = db.Column(db.String(100))
    password = db.Column(db.String(100))

    def __init__(self, username, email, password):
        self.username = username
        self.email = email
        self.password = password

# username condition:

# 1) Username must be 6-30 characters long
# 2) Username may only contain:

# Uppercase and lowercase letters
# Numbers from 0-9 and
# Special characters _ - .
# 3) Username may not:

# Begin or finish with characters _ - .

# Have more than one sequential character _ - . inside

# if re.match(r'^(?![-._])(?!.*[_.-]{2})[\w.-]{6,30}(?<![-._])$',username) is not None:

username_reg = r'^(?![-._])(?!.*[_.-]{2})[\w.-]{6,30}(?<![-._])$'

# Conditions for a valid password are:

# 1. Should have at least one number.
# 2. Should have at least one uppercase and one lowercase character.
# 3. Should have at least one special symbol.
# 4. Should be between 6 to 20 characters long.

pwd_reg = r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*#?&])[A-Za-z\d@$!#%*?&]{6,20}$"

email_reg = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'

@app.route("/")
@app.route("/home")
def home():
    return render_template("home.html")

@app.route("/login", methods=["POST", "GET"])
def login():
    if request.method == "POST":
        username_email = request.form["username_email"]
        password = request.form["password"]
        encoded_password = password.encode()
        hashed_obj = hashlib.sha256(encoded_password)
        hashed_password = hashed_obj.hexdigest()
        print(hashed_password)
        session.permanent = True
        if username_email:
            # email = None
            # username = None
            # password = None
            if ("@gmail" in username_email):
                email = username_email
                found_user = User.query.filter_by(email=email).first() 
            else:
                username = username_email
                found_user = User.query.filter_by(username=username).first()
            if found_user:
                if hashed_password == found_user.password:
                    session["email"] = found_user.email
                    session["username"] = found_user.username

                else:
                    flash("Username or password is incorrect", category=info)
            else:
                flash("Username or password is incorrect", category=info)
    if ("username" in session):
        username = session["username"]
        email = session["email"]
        return redirect(url_for("user"))
    return render_template("login.html")

@app.route("/logout", methods=["POST", "GET"])
def logout():
    session.pop("email", None)
    session.pop("username", None)
    return redirect(url_for("login"))

@app.route("/signup", methods=["POST", "GET"])
def signup():
    if request.method == "POST":
        username = request.form["username"]
        email = request.form["email"]
        password = request.form["password"]
        confirm_pwd = request.form["confirm_pwd"]
        if re.match(username_reg, username):
            if re.match(email_reg, email):
                if re.match(pwd_reg, password):
                    if password == confirm_pwd:
                        encoded_password = password.encode()
                        hashed_obj = hashlib.sha256(encoded_password)
                        hashed_password = hashed_obj.hexdigest()
                        user = User(username=username, password=hashed_password, email=email)
                        db.session.add(user)
                        db.session.commit()
                        render_template("login.html")
                    else:
                        flash("Password and confirm password not match", category=info)
                else:
                    flash("Password must contain 6-20 characters, at least one lowercase, uppercase, number and special character", category=info)
            else:
                flash("Invalid email")
        else:
            flash("Username contains 6-30 characters, including uppercase and lowercase letters, numbers from 0-9 and special characters _ - .", category=info)


    return render_template("signup.html")

@app.route("/user", methods=["POST", "GET"])
def user():
    if request.method == "POST":
        f = request.files["file_name"]
        f.save(os.path.join(app.config["UPLOAD_PATH"], f.filename))
        session["curr_file"] = f.filename
        print("Upload file successfuly")
        flash("Upload file successfuly")
    return render_template("user.html")

@app.route("/predict", methods=["POST", "GET"])
def predict():
    if request.method == "POST":
        if "curr_file" in session:
            nii_path = os.path.join(app.config["UPLOAD_PATH"], session["curr_file"])
            liver_tumor_predict(nii_path=nii_path, liver_model=liver_model, tumor_model=tumor_model,
                            mode = "None",# or "tumor_seg", "liver_seg"
                            save_path =app.config["SAVE_PATH"], batch_size = 4, 
                            input_channel_mode="channel last",# or "channel first" 
                            output_channel_mode = "channel last", # or "channel first"
                            save = True,
                            liver_model_input_shape = (512,512,3), 
                            tumor_model_input_shape = (512,512,3))
            session["curr_download_file"] = session["curr_file"].replace("volume", "segmentation")
            flash("Predicted file successfuly")
        return redirect(url_for("user"))  
        # return render_template("user.html") 


@app.route("/download", methods=["POST", "GET"])
def download_file():
    if request.method == "POST":
        if "curr_download_file" in session:
            p = os.path.join(app.config["DOWNLOAD_PATH"], session["curr_download_file"])
            return send_file(p, as_attachment=True)
        else:
            flash("Prediction is in progress", category=info)

    return redirect(url_for("user"))
    # return render_template("user.html")
        

if __name__ == "__main__":
    if not os.path.exists("accounts.db"):
        db.create_all(app=app)
    app.run(debug=True)