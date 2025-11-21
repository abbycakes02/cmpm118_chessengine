# Project Roadmap & TODOs

## Phase 1: Infrastructure & Data Pipeline

*Goal: Prepare the training data from the Lichess database.*

- [ ] **Data Acquisition**
  - [ ] Download a segment of the Lichess Elite Database (PGN format).
  - [ ] Create a scripts in `backend_engine/ml/data_processing` to parse PGN files into the CSVs we need, and to build the tensors for training.

- [ ] **Tensor Representation Logic**
  - [ ] Implement the `fen_to_tensor(fen)` function in `backend_engine/data_processing/tensor_converter.py`.
  - [ ] **Requirement:** Transform `python-chess` board into a $21 \times 8 \times 8$ tensor.
    - [ ] 12 Channels: Piece Planes (White P/N/B/R/Q/K, Black P/N/B/R/Q/K).
    - [ ] 1 Channel: Side to move.
    - [ ] 4 Channels: Castling rights (White/Black Kingside/Queenside).
    - [ ] 1 Channel: En Passant target.
    - [ ] 1 Channel: 50-move rule counter.
    - [ ] 1 Channel: Repetition count.
    - [ ] 1 Channel: Move number (Game phase proxy).

- [ ] **Dataset Creation**
  - [ ] Create a PyTorch `Dataset` class that loads processed FENs and outcomes.
  - [ ] Split data into Training (80%) and Validation (20%) sets.

## Phase 2: Neural Network Architecture (The "Brain")

*Goal: Build and train the Value Network to evaluate positions.*

- [ ] **Model Definition**
  - [ ] Create `backend_engine/ml/model.py`.
  - [ ] Implement `ResidualBlock` class.
  - [ ] **Value Head:**
    - [ ] Convolutional body (Shared).
    - [ ] Fully connected layers ending in `Tanh` activation.
    - [ ] Output: Scalar float between -1 (Black Win) and 1 (White Win).
  - [ ] *(Stretch)* **Policy Head:** Add branch for move probability vector.

- [ ] **Training Loop**
  - [ ] Create `backend_engine/ml/train.py`.
  - [ ] Loss Function: Mean Squared Error (MSE) for Value Head.
  - [ ] Optimizer: Adam or SGD.
  - [ ] Implement checkpoint saving (`.pth` files) to `backend_engine/models/`.

## Phase 3: The Engine Logic (The "Muscle")

*Goal: Replace the `random_engine` with a Minimax engine.*

- [ ] **Search Algorithm Core**
  - [ ] Create `backend_engine/engines/minimax_engine.py`.
  - [ ] Implement `evaluate_board(fen)`:
    - [ ] Load the trained PyTorch model.
    - [ ] Convert FEN -> Tensor -> Model Inference -> Scalar Score.
  - [ ] Implement `minimax(board, depth, alpha, beta, maximizing_player)`:

- [ ] **Optimization (Critical for Performance)**
  - [ ] **Move Ordering:**
    - [ ] Prioritize captures and checks to improve pruning.
    - [ ] *(Stretch)* Use Policy Head probabilities to sort moves.
  - [ ] **Iterative Deepening:**
    - [ ] Implement a loop that searches depth 1, then 2, then 3 within a time limit.
  - [ ] **Transposition Table:**
    - [ ] Implement Zobrist Hashing to map Board State -> Score.
    - [ ] Store search results to avoid re-calculating the same positions.

## Phase 4: Integration & API

*Goal: Connect the smart engine to the Frontend.*

- [ ] **Refactor Router**
  - [ ] Ensure `api/routes_engine_access.py` uses the `ENGINE_MAP` pattern to switch between "random", "minimax_shallow", and "minimax_deep".
  - [ ] Handle `time_limit` parameters from the frontend request.

- [ ] **Model Deployment**
  - [ ] Load the trained model *once* at server startup (global variable), do not load it per request (too slow).
  - [ ] (Optional) Convert model to ONNX for faster CPU inference.

## Phase 5: Frontend Refinements

*Goal: Polish the UI for the final presentation.*

- [ ] **Game State Feedback**
  - [ ] Visual indicator for "Engine Thinking..." while waiting for API response.
  - [ ] Highlight the "Best Move" arrow based on engine response.
- [ ] **Settings Panel**
  - [ ] Add a dropdown to select difficulty (Depth 1 vs Depth 3 vs Adaptive).

## Phase 6: Testing & Validation

*Goal: Prove the engine works for the report.*

- [ ] **Self-Play Simulations**
  - [ ] Write a script `benchmarks/self_play.py`.
  - [ ] Pit `Minimax` vs `Random` to verify it wins 100% of the time.
  - [ ] Pit `Minimax (Depth 1)` vs `Minimax (Depth 3)` to measure improvement.
- [ ] **ELO Estimation**
  - [ ] Play against other open-source engines (e.g., Stockfish at Level 1/2) to estimate rating.
- [ ] **Report Data**
  - [ ] Generate graphs of Training Loss vs. Epochs.
  - [ ] Record average time per move.

## Feature Wishlist (If Time Permits)

- [ ] Implement Polyglot Opening Book (don't calculate moves for the first 10 turns, just look them up).
- [ ] Implement the **Policy Head** (Stretch Goal).
- [ ] Re-write move generation in `NumPy` or optimize `python-chess` usage.
