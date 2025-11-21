
# Setup Guide for CMPM 118 Chess Engine Project

```bash
.
├── backend_engine
│   ├── api
│   │   └── routes_engine_access.py
│   ├── engines
│   │   └── random_engine.py
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
└── README.md
```

## 1. Clone the repository

```bash
git clone https://github.com/abbycakes02/cmpm118_chessengine.git
```

## 2. Setting up the frontend

Install all the frontend dependencies.

```bash
cd chess-ui
npm install
```

## 3. Setting up the backend

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
