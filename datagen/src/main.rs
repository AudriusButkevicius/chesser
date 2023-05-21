#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_imports)]

mod score;
mod visitor;
mod state;
mod record;

use std::{env, fs, io, mem};
use std::collections::VecDeque;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Error, ErrorKind, Seek, Write};
use std::io::ErrorKind::InvalidData;
use shakmaty::{Board, CastlingMode, Chess, Color, Position};
use shakmaty::fen::Fen;
use shakmaty::san::{San, SanPlus};
use pgn_reader::{BufferedReader, Nag, RawComment, RawHeader, Skip, Visitor};
use regex::Regex;
use std::mem::transmute;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime};
use byte_unit::Byte;
use hhmmss::Hhmmss;
use crate::record::Record;
use crate::visitor::StateGenerator;


unsafe fn any_as_u8_slice<T: Sized>(p: &T) -> &[u8] {
    ::core::slice::from_raw_parts(
        (p as *const T) as *const u8,
        ::core::mem::size_of::<T>(),
    )
}

fn main() -> Result<(), io::Error> {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.len() < 2 {
        println!("Needs at least two arguments");
        return Err(io::Error::from(ErrorKind::InvalidData));
    }
    println!("Args: {}", args.join(" "));
    println!("Record size {}", mem::size_of::<Record>());

    let (output_file_path, files) = args.split_last().unwrap();

    if fs::metadata(output_file_path).is_ok() {
        println!("Destination file already exists");
        return Err(Error::from(io::ErrorKind::AlreadyExists));
    }

    let mut total_size = 0;
    let mut bytes_done_across_files = 0;
    for file in files {
        println!("Checking {}", file);
        total_size += fs::metadata(file)?.len();
    }
    let total_adjusted = Byte::from_bytes(total_size as u128).get_appropriate_unit(true);

    let start = SystemTime::now();
    let output_bytes = Arc::new(AtomicU64::new(0));

    let output_file = OpenOptions::new().create(true).write(true).truncate(true).open(output_file_path)?;

    for (i, arg) in files.iter().enumerate() {
        println!("Reading {arg}");
        let file = File::open(&arg).expect("fopen");
        let mut file_copy = file.try_clone()?;
        let file_size = file.metadata()?.len();

        let uncompressed: Box<dyn io::Read + Send> = if arg.ends_with(".zst") {
            Box::new(zstd::Decoder::new(file)?)
        } else if arg.ends_with(".bz2") {
            Box::new(bzip2::read::MultiBzDecoder::new(file))
        } else if arg.ends_with(".xz") {
            Box::new(xz2::read::XzDecoder::new(file))
        } else if arg.ends_with(".gz") {
            Box::new(flate2::read::GzDecoder::new(file))
        } else if arg.ends_with(".lz4") {
            Box::new(lz4::Decoder::new(file)?)
        } else {
            Box::new(file)
        };

        let output_bytes = output_bytes.clone();
        let output_bytes_copy = output_bytes.clone();
        let mut output_writer = BufWriter::new(output_file.try_clone()?);

        crossbeam::scope(|scope| {
            let (send_state, recv_state) = crossbeam::channel::bounded(128);
            let (send_record, recv_record) = crossbeam::channel::bounded::<Record>(128);
            scope.spawn(move |_| {
                let mut validator = StateGenerator::new(i.try_into().unwrap());
                let mut last_print = SystemTime::now();
                let mut num_with_evals = 0;
                let mut num = -1;
                for state_result in BufferedReader::new(uncompressed).into_iter(&mut validator) {
                    num += 1;
                    if state_result.is_err() {
                        println!("Skipping over bad record at {}: {}", num, state_result.err().unwrap());
                        continue;
                    }
                    let state = state_result.unwrap();
                    let idx = state.game_index;
                    if state.valid && state.have_scores {
                        num_with_evals += 1;
                    }
                    send_state.send(state).expect("io");
                    let now = SystemTime::now();
                    if now.duration_since(last_print).unwrap().as_secs() > 2 {
                        let duration_so_far = now.duration_since(start).unwrap().as_secs();
                        let bytes_done = bytes_done_across_files + file_copy.stream_position().unwrap();
                        let rate_bps = bytes_done / duration_so_far;
                        let left_duration = Duration::from_secs((total_size - bytes_done) / rate_bps);
                        let rate = Byte::from_bytes(rate_bps as u128).get_appropriate_unit(true);
                        let stream_pos_adjusted = Byte::from_bytes(bytes_done as u128).get_appropriate_unit(true);
                        let output = Byte::from_bytes(output_bytes_copy.load(Ordering::Relaxed) as u128).get_appropriate_unit(true);
                        println!("Current: {}/{} ({}%) ({} / {} @ {}/s eta: {}) = Output: {}", num_with_evals, idx, (num_with_evals * 100 / idx), stream_pos_adjusted, total_adjusted, rate, left_duration.hhmmss(), output);
                        last_print = now;
                    }
                }
                println!("Producer done");
            });

            scope.spawn(move |_| {
                let mut bytes = 0;
                let record_size = std::mem::size_of::<Record>() as u64;
                for record in recv_record {
                    bytes += record_size;
                    let record_bytes = unsafe { any_as_u8_slice(&record) };
                    output_writer.write(record_bytes).expect("io");
                    output_bytes.store(bytes, Ordering::Relaxed);
                }
                println!("writer done");
                output_writer.flush().expect("output flush");
                println!("writer done2");
            });

            for _ in 0..32 {
                let recv_state = recv_state.clone();
                let send_record = send_record.clone();
                scope.spawn(move |_| {
                    for state in recv_state {
                        for record in state {
                            send_record.send(record).expect("io");
                        }
                    }
                    println!("Consumer done");
                });
            }
        })
            .unwrap();

        bytes_done_across_files += file_size
    }

    Ok(())
}
