use std::cmp::min;
use std::cmp::max;
use std::fmt::Debug;
use regex::Regex;
use lazy_static::lazy_static;
use crate::RawComment;

fn lichess_raw_wins(cp: i32) -> i32 {
    return (1000.0 / (1.0 + (-0.004 * cp as f32).exp())).round() as i32;
}

fn sf12_wins(cp: i32, ply: i32) -> i32 {
    // https://github.com/official-stockfish/Stockfish/blob/sf_12/src/uci.cpp#L198-L218
    let m = (min(240, max(ply, 0)) / 64) as f32;
    let a = (((-8.24404295 * m + 64.23892342) * m + -95.73056462) * m) + 153.86478679;
    let b = (((-3.37154371 * m + 28.44489198) * m + -56.67657741) * m) + 72.05858751;
    let x = min(1000, max(cp, -1000)) as f32;
    return (0.5 + 1000.0 / (1.0 + ((a - x) / b).exp())) as i32;
}

fn sf14_wins(cp: i32, ply: i32) -> i32 {
    // https://github.com/official-stockfish/Stockfish/blob/sf_14/src/uci.cpp#L200-L220
    let m = (min(240, max(ply, 0)) / 64) as f32;
    let a = (((-3.68389304 * m + 30.07065921) * m + -60.52878723) * m) + 149.53378557;
    let b = (((-2.01818570 * m + 15.85685038) * m + -29.83452023) * m) + 47.59078827;
    let x = min(2000, max(cp, -2000)) as f32;
    return (0.5 + 1000.0 / (1.0 + ((a - x) / b).exp())) as i32;
}

fn sf15_wins(cp: i32, ply: i32) -> i32 {
    // https://github.com/official-stockfish/Stockfish/blob/sf_15/src/uci.cpp#L200-L220
    let m = (min(240, max(ply, 0)) / 64) as f32;
    let a = (((-1.17202460e-1 * m + 5.94729104e-1) * m + 1.12065546e+1) * m) + 1.22606222e+2;
    let b = (((-1.79066759 * m + 11.30759193) * m + -17.43677612) * m) + 36.47147479;
    let x = min(2000, max(cp, -2000)) as f32;
    return (0.5 + 1000.0 / (1.0 + ((a - x) / b).exp())) as i32;
}

fn sf15_1_wins(cp: i32, ply: i32) -> i32 {
    // https://github.com/official-stockfish/Stockfish/blob/sf_15.1/src/uci.cpp#L200-L224
    // https://github.com/official-stockfish/Stockfish/blob/sf_15.1/src/uci.h#L38
    const NORMALIZE_TO_PAWN_VALUE: i32 = 361;
    let m = (min(240, max(ply, 0)) / 64) as f32;
    let a = (((-0.58270499 * m + 2.68512549) * m + 15.24638015) * m) + 344.49745382;
    let b = (((-2.65734562 * m + 15.96509799) * m + -20.69040836) * m) + 73.61029937;
    let x = min(4000, max(cp * NORMALIZE_TO_PAWN_VALUE / 100, -4000)) as f32;
    return (0.5 + 1000.0 / (1.0 + ((a - x) / b).exp())) as i32;
}

#[derive(Debug)]
pub struct WDL {
    pub wins: i32,
    pub draws: i32,
    pub losses: i32,
}

#[derive(Clone, Eq, PartialEq, Debug)]
pub enum WDLModel {
    Lichess,
    SF12,
    SF14,
    SF15,
    SF,
}

#[derive(Clone, Eq, PartialEq, Debug)]
pub enum ScoreType {
    CentiPawn,
    Mate,
}

#[derive(Clone, Debug)]
pub struct Score {
    pub val: i32,
    pub score_type: ScoreType,
    pub raw: f32,
    pub mate: bool,
}

impl Score {
    pub fn wdl(&self, model: WDLModel, ply: i32) -> WDL {
        if self.score_type == ScoreType::CentiPawn {
            self.cp_wdl(model, ply)
        } else {
            self.mate_wdl(model)
        }
    }

    fn cp_wdl(&self, model: WDLModel, ply: i32) -> WDL {
        let wins;
        let losses;
        match model {
            WDLModel::Lichess => {
                wins = lichess_raw_wins(max(-1000, min(self.val, 1000)));
                losses = 1000 - wins;
            }
            WDLModel::SF12 => {
                wins = sf12_wins(self.val, ply);
                losses = sf12_wins(-self.val, ply);
            }
            WDLModel::SF14 => {
                wins = sf14_wins(self.val, ply);
                losses = sf14_wins(-self.val, ply);
            }
            WDLModel::SF15 => {
                wins = sf15_wins(self.val, ply);
                losses = sf15_wins(-self.val, ply);
            }
            WDLModel::SF => {
                wins = sf15_1_wins(self.val, ply);
                losses = sf15_1_wins(-self.val, ply);
            }
        }
        let draws = 1000 - wins - losses;
        return WDL {
            wins,
            draws,
            losses,
        };
    }

    fn mate_wdl(&self, model: WDLModel) -> WDL {
        if model == WDLModel::Lichess {
            let cp = (21 - min(10, self.val.abs())) * 100;
            let wins = lichess_raw_wins(cp);
            if self.val > 0 {
                WDL {
                    wins,
                    draws: 0,
                    losses: 1000 - wins,
                }
            } else {
                WDL {
                    wins: 1000 - wins,
                    draws: 0,
                    losses: wins,
                }
            }
        } else {
            if self.val > 0 {
                WDL {
                    wins: 1000,
                    draws: 0,
                    losses: 0,
                }
            } else {
                WDL {
                    wins: 0,
                    draws: 0,
                    losses: 1000,
                }
            }
        }
    }
}

lazy_static! {
    static ref RE: Regex = Regex::new(r"(?P<prefix>\s?)\[%eval\s(?:\#(?P<mate>[+-]?\d+)|(?P<cp>[+-]?(?:\d{0,10}\.\d{1,2}|\d{1,10}\.?)))(?:,(?P<depth>\d+))?\](?P<suffix>\s?)").unwrap();
}

pub fn parse_from_comment(comment: RawComment<'_>) -> Option<Score> {
    //return Some(Score { val: 10, score_type: ScoreType::CentiPawn })
    let comment = std::str::from_utf8(comment.0).unwrap();
    if !comment.contains("%eval") {
        return None
    }
    let maybe_captures = RE.captures(comment);
    if maybe_captures.is_none() {
        return None;
    }
    let captures = maybe_captures.unwrap();
    if let Some(mate) = captures.name("mate") {
        let mate: i32 = mate.as_str().parse().unwrap();
        return Some(Score { val: mate, score_type: ScoreType::Mate, raw: mate as f32, mate: true });
    }
    let cp: f32 = captures.name("cp").unwrap().as_str().parse().unwrap();
    return Some(Score { val: (cp * 100.0) as i32, score_type: ScoreType::CentiPawn, raw: cp, mate: false });
}
