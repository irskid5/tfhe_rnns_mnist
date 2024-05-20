#![allow(non_snake_case)]
#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_mut)]
#![allow(unused_imports)]

use colored::Colorize;
use core::time;
use std::error::Error;
use time_graph;

pub mod mnist_rnn;
pub mod utils;

use crate::utils::keys::*;
use mnist_rnn::*;

fn run_mnist_rnn() -> Result<(), Box<dyn Error>> {
    println!("");
    println!("{}", "Beginning MNIST RNN run.".bold());
    print_rnn_banner!(mnist_rnn, true, true, &*SET2);
    Ok(())
}

fn main() {
    // Do we enable timing collection?
    time_graph::enable_data_collection(true);

    match run_mnist_rnn() {
        Ok(()) => println!("{}\n", "MNIST RNN run completed.".bold()),
        Err(e) => {
            println!("");
            println!("{}", "ERROR!".red().bold());
            println!("{:?}", e);
            println!("{}", e);
            println!("");
        }
    };

    // Get timings logged by time_graph
    let timings = time_graph::get_full_graph();
    println!("\n{}\n", timings.as_table());

    println!("End.");
}
