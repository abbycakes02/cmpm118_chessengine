import './style.css';
import { Chessground } from '@lichess-org/chessground';
import { Chess, SQUARES } from 'chess.js';

// Initialize Game State
const game = new Chess();
let playerColor = 'white'; 
const boardElement = document.getElementById('board');
const statusElement = document.getElementById('status');

// Initialize Overlay Elements
const overlay = document.getElementById('game-overlay');
const winnerText = document.getElementById('winner-text');
const rematchBtn = document.getElementById('overlay-rematch');

// helper function to compute possible destinations for each square
function computeDests() {
    const dests = new Map();
    SQUARES.forEach(s => {
        const ms = game.moves({ square: s, verbose: true });
        if (ms.length) dests.set(s, ms.map(m => m.to));
    });
    return dests;
}

// helper function to update game status, check for game over, and show overlays
function updateStatus() {
    // Check Game Over
    if (game.isGameOver()) {
        let title = '';
        
        if (game.isCheckmate()) {
            const winner = game.turn() === 'w' ? 'Black' : 'White';
            title = `${winner} Wins!`;
        } else if (game.isDraw()) {
            title = 'Draw';
        } else {
            title = 'Game Over';
        }

        // Show Overlay
        winnerText.innerText = title;
        overlay.classList.remove('hidden');
        statusElement.innerText = "Game Over";
    } else {
        // Hide Overlay
        overlay.classList.add('hidden');
        
        // Update Text Status
        let text = (game.turn() === 'w' ? "White's" : "Black's") + " turn";
        if (game.isCheck()) text += " (Check!)";
        statusElement.innerText = text;
    }
}

// Board Init
const ground = Chessground(boardElement, {
    fen: game.fen(),
    orientation: playerColor,
    coordinates: true,
    movable: {
        color: 'white',
        free: false,
        dests: computeDests(),
        events: { after: onPlayerMove }, 
    },
});

updateStatus();

// movement logic

async function onPlayerMove(orig, dest) {
    game.move({ from: orig, to: dest, promotion: 'q' }); 
    updateStatus(); // after Player moves check for Game Over

    if (game.isGameOver()) {
        ground.set({ movable: { color: null } }); 
        return;
    }

    // Lock board
    ground.set({ movable: { color: null } });

    // Trigger Engine Move
    await makeEngineMove();
}

async function makeEngineMove() {
    try {
        const data = await sendMoveToBackend(game.fen());// Send current FEN to backend
        const engineMove = data.move;  // get UCI move string from response expecting format like 'e2e4'
        
        if (!engineMove || game.isGameOver()) { // Safety check
            updateStatus();
            return;
        }

        // Apply Engine Move to Game
        const fromSquare = engineMove.substring(0, 2);
        const toSquare = engineMove.substring(2, 4);
        game.move({ from: fromSquare, to: toSquare, promotion: 'q' });

        ground.set({
            fen: game.fen(),
            turnColor: playerColor,
            movable: {
                color: playerColor,
                dests: computeDests()
            }
        });
        
        updateStatus(); // Check for Engine win

    } catch (err) {
        console.error("Engine failed:", err);
        // Unlock on error
        ground.set({ movable: { color: playerColor, dests: computeDests() } });
    }
}

async function sendMoveToBackend(FEN) {
    const response = await fetch('http://localhost:8000/move', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ FEN: FEN, engine: "random" }), 
    });
    return await response.json();
}

// button handlers

function resetGame(color) {
    game.reset();
    playerColor = color;
    
    // Reset Board
    ground.set({
        fen: game.fen(),
        orientation: color,
        movable: { 
            color: color === 'white' ? 'white' : null, 
            dests: computeDests() 
        }
    });
    
    updateStatus(); // Hides overlay

    // If playing Black, trigger Engine to move first
    if (color === 'black') {
        makeEngineMove();
    }
}

document.getElementById('play-white').addEventListener('click', () => resetGame('white'));
document.getElementById('play-black').addEventListener('click', () => resetGame('black'));
rematchBtn.addEventListener('click', () => resetGame(playerColor));
