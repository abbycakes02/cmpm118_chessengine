import './style.css';

import { Chessground } from '@lichess-org/chessground';  
import { Chess, SQUARES } from 'chess.js';

let playerColor = 'white';
const board = document.getElementById('board');
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
async function sendMove(fen, move) {
  console.log(`Sending move ${move} for position ${fen} to backend engine`);
  const response = await fetch('http://localhost:8000/move', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ fen, move }),
  });
  const data = await response.json();
  console.log(data);
  return data;
}

async function handleMove(orig, dest) {
  console.log(`Handling move from ${orig} to ${dest}`);

  const fenBefore = game.fen();

  const moveObj = game.move({ from: orig, to: dest, promotion: 'q' });

  if (moveObj === null) {
    console.log('Illegal move attempted');
    refreshBoard();
    return;
  }


  const move = `${orig}${dest}`;
  console.log(`send move ${move}`);

  const response = await sendMove(fenBefore, move);

  if (response.engine_move) {
    console.log(`Engine move received: ${response.engine_move}`);
    if (response?.fen) {
      game.load(response.fen);
      ground.set({ fen: response.fen });
    }
    requestAnimationFrame(() => refreshBoard());
  }
}

function refreshBoard() {
  console.log('Refreshing board');
  const destinations = computeDests();
  console.log('Computed destinations:', destinations);
  ground.set({
    movable: {
      free: false,
      color: game.turn() === 'w' ? 'white' : 'black', // piece color player can move
      dests: destinations,
      events: {
        after: handleMove,
      }
    }
  });
}



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



