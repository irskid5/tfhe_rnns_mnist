#![allow(non_snake_case)]
#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_mut)]
#![allow(unused_imports)]

use concrete_core::prelude::*;

use std::error::Error;

pub fn init_engines(secret: u128) -> Result<(DefaultEngine, DefaultSerializationEngine, DefaultParallelEngine, CudaEngine, AmortizedCudaEngine), Box<dyn Error>> {
    
    let mut default_engine = DefaultEngine::new(Box::new(UnixSeeder::new(secret)))?;
    let mut serial_engine = DefaultSerializationEngine::new(())?;
    let mut parallel_engine = DefaultParallelEngine::new(Box::new(UnixSeeder::new(secret)))?;
    let mut cuda_engine = CudaEngine::new(())?;
    let mut amortized_cuda_engine = AmortizedCudaEngine::new(())?;
    
    println!("Constructed Engines.");
    println!("Running on {} GPUs.", cuda_engine.get_number_of_gpus().0);
    
    Ok((default_engine, serial_engine, parallel_engine, cuda_engine, amortized_cuda_engine))
}