import './style.css';

import { Chessground } from '@lichess-org/chessground';  
  
// Initialize the board with starting position  
const ground = Chessground(document.getElementById('board'), {  
  fen: 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR',  
  orientation: 'black',  
  coordinates: true,  
  movable: {  
    free: true,  // Allow all moves (no validation)  
    color: 'both'  // Both sides can move  
  }  
});




