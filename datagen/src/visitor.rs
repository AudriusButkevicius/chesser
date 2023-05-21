use std::collections::VecDeque;
use std::mem;
use pgn_reader::{Nag, RawComment, RawHeader, Visitor};
use shakmaty::{CastlingMode, Chess};
use shakmaty::fen::Fen;
use shakmaty::san::SanPlus;
use crate::score::parse_from_comment;
use crate::state::State;

pub struct StateGenerator {
    pub file_index: u32,
    pub games: u32,
    pub state: State,
}

impl StateGenerator {
    pub fn new(file_index: u32) -> StateGenerator {
        StateGenerator {
            file_index,
            games: 0,
            state: State {
                file_index,
                game_index: 0,
                white_elo: 0,
                black_elo: 0,
                valid: true,
                have_scores: false,
                game: Chess::default(),
                nags: VecDeque::with_capacity(80),
                sans: VecDeque::with_capacity(80),
                scores: VecDeque::with_capacity(80),
            },
        }
    }
}


impl Visitor for StateGenerator {
    type Result = State;

    fn begin_game(&mut self) {
        //println!("Begin game {}", self.games);
    }

    fn header(&mut self, key: &[u8], value: RawHeader<'_>) {
        // Support games from a non-standard starting position.
        let key_str = std::str::from_utf8(key).unwrap();
        let value_str = std::str::from_utf8(value.as_bytes()).unwrap();
        //println!("Key: {key_str:?} = {value_str:?}");
        if key == b"WhiteElo" {
            if let Ok(white_elo) = value_str.parse::<u16>() {
                self.state.white_elo = white_elo;
            }
        }
        if key == b"BlackElo" {
            if let Ok(black_elo) = value_str.parse::<u16>() {
                self.state.black_elo = black_elo;
            }
        }
        if key == b"FEN" {
            let fen = match Fen::from_ascii(value.as_bytes()) {
                Ok(fen) => fen,
                Err(err) => {
                    println!("Going invalid in fen 1");
                    self.state.valid = false;
                    return;
                }
            };

            self.state.game = match fen.into_position(CastlingMode::Chess960) {
                Ok(game) => game,
                Err(err) => {
                    // eprintln!(
                    //     "illegal fen header in game {}: {} ({:?})",
                    //     self.games, err, value
                    // );
                    println!("Going invalid in fen 2");
                    self.state.valid = false;
                    return;
                }
            };
        }
    }

    fn san(&mut self, san_plus: SanPlus) {
        //println!("SanPlus {}", san_plus.clone());
        self.state.sans.push_back(san_plus.san);
        self.state.nags.push_back(None);
        self.state.scores.push_back(None);
    }

    fn nag(&mut self, _nag: Nag) {
        self.state.nags.pop_back();
        self.state.nags.push_back(Some(_nag.clone()));
        //self.state.last_nag = Some(_nag)
        //println!("Nag {}", _nag.to_pretty());
    }

    fn comment(&mut self, _comment: RawComment<'_>) {
        //let comment = std::str::from_utf8(_comment.0).unwrap();
        //println!("Comment: {comment:?}");
        if let Some(score) = parse_from_comment(_comment) {
            self.state.scores.pop_back();
            self.state.scores.push_back(Some(score));
            self.state.have_scores = true;
        }
    }

    fn end_game(&mut self) -> Self::Result {
        let idx = self.state.file_index;
        let result = mem::replace(
            &mut self.state,
            State {
                file_index: idx,
                game_index: self.games,
                white_elo: 0,
                black_elo: 0,
                valid: true,
                have_scores: false,
                game: Chess::default(),
                nags: VecDeque::with_capacity(80),
                sans: VecDeque::with_capacity(80),
                scores: VecDeque::with_capacity(80),
            },
        );
        self.games += 1;
        return result;
    }
}
