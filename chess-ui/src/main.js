import './style.css';

import { Chessground } from '@lichess-org/chessground';  

let playerColor = 'white';
const board = document.getElementById('board');

// ========== Initialize Chessground Board ==========
const ground = Chessground(board, {
  fen: 'start',
  orientation: playerColor,
  coordinates: true,
  movable: {
    free: true, // allow ALL moves
    color: playerColor,
    events: {
      after: handleMove,
    },
  },
});


// Get Buttons
const btnBlack = document.getElementById('play-black');
const btnWhite = document.getElementById('play-white');
const btnAIvsAI = document.getElementById('ai-vs-ai');

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
  const move = orig + dest;
  const fen = ground.getFen();
  console.log(`Player moved: ${move} in position ${fen}`);
  await sendMove(fen, move);

  ground.set({
    movable:{
      color: playerColor, // piece color player can move
      free: true, // allow all moves (no validation)
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



