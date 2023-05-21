use std::collections::VecDeque;
use pgn_reader::Nag;
use shakmaty::Chess;
use shakmaty::san::San;
use shakmaty::Position;
use crate::record::Record;
use crate::score::Score;

pub struct State {
    pub file_index: u32,
    pub game_index: u32,
    pub white_elo: u16,
    pub black_elo: u16,
    pub valid: bool,
    pub have_scores: bool,
    pub game: Chess,
    pub nags: VecDeque<Option<Nag>>,
    pub sans: VecDeque<San>,
    pub scores: VecDeque<Option<Score>>,
}


impl Iterator for State {
    type Item = Record;

    fn next(&mut self) -> Option<Self::Item> {
        if !self.valid || !self.have_scores {
            return None;
        }

        while !self.sans.is_empty() {
            // Need to pop all of the stuff from state
            let san = self.sans.pop_front().unwrap();
            let nag = self.nags.pop_front().unwrap();
            let maybe_score = self.scores.pop_front().unwrap();

            // Check if move is valid
            let m = match san.to_move(&self.game) {
                Ok(m) => m,
                Err(_) => {
                    self.valid = false;
                    return None;
                }
            };

            // Make the move
            self.game.play_unchecked(&m);

            // Move to next if no score.
            if maybe_score.is_none() {
                continue;
            }

            let score = maybe_score.unwrap();
            let board = self.game.board();
            return Some(Record {
                white: board.white().0,
                pawns: board.pawns().0,
                knights: board.knights().0,
                bishops: board.bishops().0,
                rooks: board.rooks().0,
                queens: board.queens().0,
                kings: board.kings().0,
                turn: self.game.turn() as u8,
                moves: self.game.fullmoves().get() as u16,
                half_moves: self.game.halfmoves() as u16,
                ep_square: self.game.maybe_ep_square().map_or(255, |m| m as u8),
                promoted: self.game.promoted().0,
                castling_rights: self.game.castles().castling_rights().0,
                score: score.raw,
                mate: score.mate as u8,
                white_elo: self.white_elo,
                black_elo: self.black_elo,
                file_index: self.file_index as u8,
                game_index: self.game_index,
            });
        }

        None
    }
}
