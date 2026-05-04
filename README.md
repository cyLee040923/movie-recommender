Instructions of running the recommender:

## Create an environment

# Movie Recommender — Project README

This repository contains a simple movie recommender web app built with Flask.

## Quick start

1. Create and activate a Python environment

```bash
conda create -n projectdemo python=3.11 -y
conda activate projectdemo
```

2. Install dependencies

```bash
pip install --upgrade setuptools wheel pyquery
conda install -c conda-forge scikit-surprise -y
pip install -r requirements.txt
```

3. Run the app

```bash
flask --app flaskr run
```

4. Open http://127.0.0.1:5000 in your browser.

## UI images 


<section class="section">
	<div>
		<h2 class="title is-4">App UI</h2>
        <h3>UI Imagese<h3>
		<figure>
			<img src="{{ url_for('static', filename='img/select-genre.png') }}" alt="select genre img">
		</figure>
		<h3>Rate movies<h3>
		<figure>
			<img src="{{ url_for('static', filename='img/ratings.png') }}" 
            alt="Rate movies img">
		</figure>
        <h3>Home Page<h3>
		<figure>
			<img src="{{ url_for('static', filename='img/home-page.png') }}" 
            alt="Home page img">
		</figure>
        <h3>Because you like<h3>
		<figure>
			<img src="{{ url_for('static', filename='img/Because-you-like.png') }}" 
            alt="Because-you-like img">
		</figure>
        <h3>Details Page<h3>
		<figure>
			<img src="{{ url_for('static', filename='img/details-page.png') }}" 
            alt="Details page img">
		</figure>
        <h3>Sequential Recommendation<h3>
		<figure>
			<img src="{{ url_for('static', filename='img/sequential-recommendation.png') }}" 
            alt="Sequential recommendation img">
		</figure>
        <h3>Rated Page<h3>
		<figure>
			<img src="{{ url_for('static', filename='img/rated-page.png') }}" 
            alt="Rated page img">
		</figure>
	</div>
</section>

