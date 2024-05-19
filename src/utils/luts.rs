#![allow(non_snake_case)]
#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_mut)]
#![allow(unused_imports)]

use concrete_core::prelude::*;

use crate::utils::keys::Parameters;

use std::error::Error;

/**
 * Creates a lookup table that encodes the sign function.
 */
pub fn sign_lut(
    log_p: i32,
    log_q: i32,
    num_cts: usize,
    config: &Parameters,
    default_engine: &mut DefaultEngine,
) -> Result<GlweCiphertextVector64, Box<dyn Error>> 
{
    // Create raw sign lut
    let chunk_size = config.N.0 >> (log_p-1); // chunk_size = N/(P/2) for P elements since sign is negacyclic
    let half_chunk_size = chunk_size >> 1;
    let start = config.N.0 - half_chunk_size;
    let mut luts: Vec<u64> = vec![];
    for i in 0..num_cts {
        let mut sign_lut = vec![1u64 << (log_q - log_p); config.N.0]; // Create all ones
        for i in start..config.N.0 {
            sign_lut[i] = u64::MAX << (log_q - log_p); // Last N/(2P) elements need to be -1 for correctness 
        }
        luts.append(&mut sign_lut);
    }    

    // Create GLWE trivial ciphertext
    let lut_pts = default_engine.create_plaintext_vector_from(&luts)?;
    let result = default_engine
        .trivially_encrypt_glwe_ciphertext_vector(
            config.k.to_glwe_size(),
            GlweCiphertextCount(num_cts),
            &lut_pts,
        )?;
        
    Ok(result)
}

pub fn identity_lut(
    log_p: i32,
    log_q: i32,
    num_cts: usize,
    config: &Parameters,
    default_engine: &mut DefaultEngine,
) -> Result<GlweCiphertextVector64, Box<dyn Error>> 
{
    // if fn is not negacyclic, you encode all 2^log_p elements)
    // for identity lut, fn is not negacyclic.
    let chunk_size = config.N.0 >> (log_p-1); // chunk_size = N/P for P elements
    let half_chunk_size = chunk_size >> 1;
    let end = config.N.0 - half_chunk_size;
    let mut luts: Vec<u64> = vec![];
    for i in 0..num_cts {
        let mut identity_lut = vec![0u64; config.N.0]; // Create all zeros
        for j in half_chunk_size..end {
            identity_lut[j] = (((j + half_chunk_size) / chunk_size) as u64) << (log_q - log_p);
        }
        luts.append(&mut identity_lut);
    }    

    // DEBUG
    // println!("identity lut = {:?}", &luts[0..config.N.0]);

    // Create GLWE trivial ciphertext
    let lut_pts = default_engine.create_plaintext_vector_from(&luts)?;
    let result = default_engine
        .trivially_encrypt_glwe_ciphertext_vector(
            config.k.to_glwe_size(),
            GlweCiphertextCount(num_cts),
            &lut_pts,
        )?;
        
    Ok(result)
}

pub fn absolute_lut(
    log_p: i32,
    log_q: i32,
    num_cts: usize,
    config: &Parameters,
    default_engine: &mut DefaultEngine,
) -> Result<GlweCiphertextVector64, Box<dyn Error>> 
{
    // Create raw sign lut (only encoding 2^(log_p-1) elements, negacyclic fn.
    // if fn is not negacyclic, you encode all 2^log_p elements)
    let chunk_size = config.N.0 >> (log_p - 1);
    let half_chunk_size = chunk_size >> 1;
    let end = config.N.0 - half_chunk_size;
    let mut luts: Vec<u64> = vec![];
    for i in 0..num_cts {
        let mut identity_lut = vec![0u64; config.N.0]; // Create all zeros
        for j in half_chunk_size..end {
            identity_lut[j] = (((j + half_chunk_size) / chunk_size) as u64) << (log_q - log_p);
        }
        luts.append(&mut identity_lut);
    }    

    // DEBUG
    // println!("identity lut = {:?}", &luts[0..config.N.0]);

    // Create GLWE trivial ciphertext
    let lut_pts = default_engine.create_plaintext_vector_from(&luts)?;
    let result = default_engine
        .trivially_encrypt_glwe_ciphertext_vector(
            config.k.to_glwe_size(),
            GlweCiphertextCount(num_cts),
            &lut_pts,
        )?;
        
    Ok(result)
}