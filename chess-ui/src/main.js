import './style.css';

import { Chessground } from '@lichess-org/chessground';  
import { Chess, SQUARES } from 'chess.js';

let playerColor = 'white';
const board = document.getElementById('board');
const ground = Chessground(board, {
    fen: 'start',
    orientation: playerColor,
    coordinates: true,
    movable: {
        free: false,
        color:playerColor
    },
});
// Get Buttons
const btnBlack = document.getElementById('play-black');
const btnWhite = document.getElementById('play-white');
const btnAIvsAI = document.getElementById('ai-vs-ai');

// ========== Initialize Chessground Board ==========
const game = new Chess();
window.game = game; // expose game for debugging

function computeDests() {
  // compute all possible destinations for each piece
  const dests = new Map();
  for (const s of SQUARES) {
    const moves = game.moves({ square: s, verbose: true });
    if (moves.length) dests.set(s, moves.map((m) => m.to));
  }
  return dests;
}

const ground = Chessground(board, {
  fen: game.fen(),
  orientation: playerColor,
  coordinates: true,
  movable: {
    free: false, // disable free moves
    color: playerColor,
    dests: computeDests(),
    events: {
      after: handleMove,
    },
  },
});
window.ground = ground; // expose ground for debugging


// ========== Click Handlers for buttons ==========
btnBlack.addEventListener('click', () => {
    console.log('Play as Black');
    playerColor = 'black';
    ground.set({
        fen: 'start',
        orientation: 'black',
        movable: {
            color: 'black',
        }
    });
});

btnWhite.addEventListener('click', () => {
    console.log('Play as White');
    playerColor = 'white';
    ground.set({
        fen: 'start',
        orientation: 'white',
        movable: {
            color: 'white'    
        }
    });
});

btnAIvsAI.addEventListener('click', () => {
    console.log('Engine vs Engine mode!');
})

// ========== Connecting the UI to the backend engine ==========

// async function that calls the backend engine endpoint
async function sendMove(FEN, move) {
    const response = await fetch('http://localhost:8000/move', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ FEN, move }),
    });
    const data = await response.json();
    console.log('Move Returned', data);
    return data;
}

ground.set({
    movable:{
        color: playerColor, // piece color player can move
        free: false, // restrict moves to legal moves
        events: {
            after: async (orig, dest) => {
                const move = orig + dest;
                const fen = ground.getFen();
                sendMove(fen, move);
            }
        }
    }
});



// Initialize the board with starting position  
/*const ground = Chessground(document.getElementById('board'), {  
fen: 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR',  
orientation: 'black',  
coordinates: true,  
movable: {  
free: true,  // Allow all moves (no validation)  
color: 'both'  // Both sides can move  
}  
});
*/



