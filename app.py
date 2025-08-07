from flask import Flask, request, jsonify, render_template, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from recommenders import ContextAwareRecommender
import pandas as pd
import json
from datetime import datetime
import random
import re

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Configure PostgreSQL database
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:project@localhost:5432/songs'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)

# Load dataset
music_data = pd.read_csv('Music.csv')

# Ensure the 'context' column exists
if 'context' not in music_data.columns:
    music_data['mood'] = music_data['valence'].apply(lambda x: 'happy' if x > 0.5 else 'sad')
    music_data['activity'] = music_data['danceability'].apply(lambda x: 'active' if x > 0.5 else 'relaxed')
    music_data['context'] = music_data['mood'] + " " + music_data['activity']

# Initialize the recommender system and train it
recommender = ContextAwareRecommender()
recommender.train(music_data)

# Define the User model
class User(UserMixin, db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    recommendations = db.Column(db.PickleType, nullable=True)  # Store recommendations as a serialized object

    def __repr__(self):
        return f'<User {self.username}>'

# Create tables
with app.app_context():
    db.create_all()

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/')
def welcome():
    return render_template('welcome.html')
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Validate the username
        if not re.match(r"^[A-Za-z\s]+$", username):
            flash('Username should contain only letters and spaces.', 'danger')
            return redirect(url_for('signup'))

        # Check if the username already exists
        if User.query.filter_by(username=username).first():
            flash('Username already exists. Please log in.', 'danger')
            return redirect(url_for('login'))

        # Create a new user with plain text password
        new_user = User(username=username, password=password)
        db.session.add(new_user)
        db.session.commit()

        flash('Signup successful! Please log in.', 'success')
        return redirect(url_for('login'))

    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        user = User.query.filter_by(username=username).first()
        if user and user.password == password:  # Add password hashing for security
            login_user(user)
            return redirect(url_for('index'))

        flash('Invalid username or password.')
        return redirect(url_for('login'))

    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.')
    return redirect(url_for('welcome'))

@app.route('/index')
@login_required
def index():
    moods = ['Happy', 'Sad', 'Relaxed', 'Energetic']
    activities = ['Working', 'Exercising', 'Driving', 'Studying']
    return render_template('index.html', moods=moods, activities=activities)

@app.route('/results', methods=['GET'])
@login_required
def results():
    # Load a random subset of songs from the dataset
    random_songs = music_data.sample(n=10).to_dict(orient='records')  # 6 random songs

    # Render the results page with the random songs
    return render_template('results.html', random_songs=random_songs)

@app.route('/search', methods=['GET', 'POST'])
def search():
    if request.method == 'POST':
        query = request.form.get('query')
        if not query:
            return render_template(
                'search_results.html', 
                query=query, 
                results=[], 
                error="Please enter a search term."
            )
        
        try:
            # Filter the dataset by the query
            results = music_data[music_data['name'].str.contains(query, case=False, na=False)].to_dict(orient='records')
        except KeyError:
            return render_template(
                'search_results.html', 
                query=query, 
                results=[], 
                error="Invalid column name in the dataset."
            )
        
        # Render search results
        return render_template('search_results.html', query=query, results=results)
    
    # Render the search form for GET requests
    return render_template('search.html')


@app.route('/recommend', methods=['POST'])
@login_required
def recommend():
    # Get user preferences
    mood = request.form.get('mood')
    activity = request.form.get('activity')
    current_hour = datetime.now().hour

    # Determine the time of day
    if 5 <= current_hour < 12:
        time_of_day = 'Morning'
    elif 12 <= current_hour < 18:
        time_of_day = 'Afternoon'
    elif 18 <= current_hour < 22:
        time_of_day = 'Evening'
    else:
        time_of_day = 'Night'

    # Create the user context
    user_context = f"{mood} {activity} {time_of_day}"

    # Generate recommendations
    try:
        recommendations = recommender.recommend(user_context)
        current_user.recommendations = recommendations
        db.session.commit()
    except ValueError as e:
        flash("Error generating recommendations: " + str(e))
        return redirect(url_for('results'))

    # Redirect to the results page
    return redirect(url_for('results'))

@app.route('/recommendations', methods=['GET'])
@login_required
def recommendations():
    print("Recommendations route accessed.")  # Debugging log
    print("Current User Recommendations:", current_user.recommendations)  # Debugging log

    recommendations = current_user.recommendations or []
    return render_template('recommendations.html', recommendations=recommendations)

if __name__ == "__main__":
    app.run(debug=True)