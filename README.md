
# CMPM 118 Chess Engine Project

This repo contains the techinal implementation of a chess engine for CMPM 118 at UC Santa Cruz. The engine is built using a combination of a neural network value function and a minimax search algorithm. The project is structured into a backend engine that handles the chess logic and a frontend UI for user interaction.

## Project Structure

```bash
.
├── backend_engine
│   ├── api
│   │   └── routes_engine_access.py
│   ├── engines
│   │   └── random_engine.py
│   ├── data_processing
│   │   ├── pgn_parser.py
│   │   └── tensor_converter.py
│   ├── ml
│   │   ├── model.py
│   │   └── train.py
│   ├── main.py
│   └── requirements.txt
├── chess-ui
│   ├── index.html
│   ├── package-lock.json
│   ├── package.json
│   ├── public
│   │   └── vite.svg
│   └── src
│       ├── javascript.svg
│       ├── main.js
│       └── style.css
├── data
│   ├── raw
│   └── processed
└── README.md
```

## Running the Project

### 1. Clone the repository

```bash
git clone https://github.com/abbycakes02/cmpm118_chessengine.git
```

### 2. Setting up the frontend

Install all the frontend dependencies.

```bash
cd chess-ui
npm install
```

### 3. Setting up the backend

Create a python venv or conda env and install all the requirements.

```bash
cd backend-engine
python3 -m venv chessbot
```

or using conda:

```bash
conda create -n chessbot python=3.10
conda activate chessbot
```

Start the python venv & install reqs:

```bash
source chessbot/bin/activate
# or for conda
conda activate chessbot
pip install -r requirements.txt
```

### 4. Starting it all up

To start the frontend run:

```bash
cd chess-ui
npm run dev
```

Now the front end should be accessible at `http://localhost:5173`.

The Backend Engine server is run by simply running the `main.py` script which starts a uvicorn server at `http://localhost:8000`.

```bash
cd backend_engine
uvicorn main:app --reload
```

## Training the Neural Network

### Data and PGN Parsing

The training data is sourced from the Lichess Elite Database, which contains high-quality chess games in PGN format.

[Lichess Elite Database](https://database.nikonoel.fr/)

To parse these PGN files into a format suitable for training, use the `pgn_parser.py` script located in `backend_engine/data_processing/`.

download the pgn files into the `data/raw/` directory.

Then run the following command to parse the PGN data:

```bash
# Run from inside the 'backend_engine' folder
cd backend_engine

# run the parser
python data_processing/pgn_parser.py
```

This will read the PGN data from the `data/raw/` directory, parse each game, and extract board positions along with the game outcomes. The parsed data will be saved in chunks as Parquet files in the `data/processed/` directory.
