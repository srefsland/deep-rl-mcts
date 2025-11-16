import Move from "./move";
import Player from "./player";

interface HexGameState {
    board: BoardCell[][];
    playerToMove: Player;
    switched: boolean;
    winner: Player | null;
    lastMove: Move | null;  
}

type BoardCell = 1 | -1 | 0;

const createDefaultHexGameState = (boardSize: number): HexGameState => {
    const board: BoardCell[][] = [];
    for (let i = 0; i < boardSize; i++) {
        const row: BoardCell[] = [];
        for (let j = 0; j < boardSize; j++) {
            row.push(0);
        }
        board.push(row);
    }

    return {
        board,
        playerToMove: 1,
        switched: false,
        winner: null,
        lastMove: null
    };
}

export type { HexGameState };
export { createDefaultHexGameState };