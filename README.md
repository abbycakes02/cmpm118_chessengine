
# Setup Guide:

After cloning this repo.


### 1. Clone the repository
```bash
git clone https://github.com/abbycakes02/cmpm118_chessengine.git
```

### 2. Setting up the frontend:
Install all the frontend dependencies.
```
cd chess-ui
npm install
```



### 3. Setting up the backend:
Create a python venv and install all the requirements.
```
cd backend-engine
python3 -m venv pychess-venv
```

Start the python venv & install reqs:
```
source pychess-venv/bin/activate
pip install -r requirements.txt
```


### 4. Starting it all up!
To start the frontend run:
```
npm run dev
```
Now the front end should be accessible at `http://localhost:5173`.

The Backend Engine server is run by simply running the `engine.py` script which starts a uvicorn server at `http://localhost:8000`. 

```
source pychess-venv/bin/activate
python3 engine.py
```

