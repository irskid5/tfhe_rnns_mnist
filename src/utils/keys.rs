#![allow(non_snake_case)]
#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_mut)]
#![allow(unused_imports)]

use bincode::{deserialize_from, serialize_into};
use concrete_core::prelude::*;
use std::error::Error;

use std::fs;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::mem::*;
use lazy_static::lazy_static;

#[derive(Debug)]
pub struct Parameters {
    pub n: LweDimension,
    pub lwe_var: Variance,
    pub N: PolynomialSize,
    pub k: GlweDimension,
    pub rlwe_var: Variance,
    pub l_pbs: DecompositionLevelCount,
    pub Bg_bit_pbs: DecompositionBaseLog,
    pub l_ks: DecompositionLevelCount,
    pub base_bit_ks: DecompositionBaseLog,
    pub set_id: String,
}

#[derive(Debug, Clone)]
pub struct Keys {
    pub lwe: LweSecretKey64,
    pub glwe: GlweSecretKey64,
    pub extracted: LweSecretKey64,
    pub ksk_extracted_lwe: LweKeyswitchKey64,
    pub bsk: LweBootstrapKey64,
}

pub struct CudaKeys {
    pub ksk_extracted_lwe: CudaLweKeyswitchKey64,
    pub bsk: CudaFourierLweBootstrapKey64,
}

// Statically define pararmeter sets
lazy_static! {
    pub static ref SET1: Parameters = Parameters {
        n: LweDimension(732),
        lwe_var: Variance(3.870875046122880e-11),
        N: PolynomialSize(2048),
        k: GlweDimension(1),
        rlwe_var: Variance(4.905643852600860e-32),
        l_pbs: DecompositionLevelCount(3),
        Bg_bit_pbs: DecompositionBaseLog(14),
        l_ks: DecompositionLevelCount(2),
        base_bit_ks: DecompositionBaseLog(8),
        set_id: 8.to_string()
    };

    pub static ref SET2: Parameters = Parameters {
        n: LweDimension(585),
        lwe_var: Variance(8.357206851101390e-09),
        N: PolynomialSize(1024),
        k: GlweDimension(1),
        rlwe_var: Variance(8.934364862023360e-16),
        l_pbs: DecompositionLevelCount(5),
        Bg_bit_pbs: DecompositionBaseLog(5),
        l_ks: DecompositionLevelCount(2),
        base_bit_ks: DecompositionBaseLog(8),
        set_id: 10.to_string()
    };
}

pub fn create_keys(
    config: &Parameters,
    default_engine: &mut DefaultEngine,
    parallel_engine: &mut DefaultParallelEngine,
) -> Result<Keys, Box<dyn Error>> {
    // Create the keys
    println!("Creating keys...");
    let lwe: LweSecretKey64 = default_engine.generate_new_lwe_secret_key(config.n)?;
    let glwe: GlweSecretKey64 = default_engine.generate_new_glwe_secret_key(config.k, config.N)?;
    let extracted: LweSecretKey64 =
        default_engine.transform_glwe_secret_key_to_lwe_secret_key(glwe.clone())?;
    let ksk_extracted_lwe: LweKeyswitchKey64 = default_engine.generate_new_lwe_keyswitch_key(
        &extracted,
        &lwe,
        config.l_ks,
        config.base_bit_ks,
        config.rlwe_var,
    )?;
    let bsk: LweBootstrapKey64 = parallel_engine.generate_new_lwe_bootstrap_key(
        &lwe,
        &glwe,
        config.Bg_bit_pbs,
        config.l_pbs,
        config.rlwe_var,
    )?;
    println!("Keys created.");

    // Return keys struct with new keys
    Ok(Keys {
        lwe,
        glwe,
        extracted,
        ksk_extracted_lwe,
        bsk,
    })
}

pub fn save_keys(
    filename: &str,
    filepath: &str,
    keys: &Keys,
    serial_engine: &mut DefaultSerializationEngine,
) -> Result<(), Box<dyn Error>> {
    println!("Saving keys...");

    fs::create_dir_all(filepath)?;

    // Serialize the keys
    let lwe_s = serial_engine.serialize(&keys.lwe)?;
    let glwe_s = serial_engine.serialize(&keys.glwe)?;
    let lwe_extracted_s = serial_engine.serialize(&keys.extracted)?;
    let lwe_extracted_to_lwe_ksk_s = serial_engine.serialize(&keys.ksk_extracted_lwe)?;
    let lwe_bsk_s = serial_engine.serialize(&keys.bsk)?;

    // Save the serialized keys into the file (ORDER MATTERS)
    let mut f = BufWriter::new(File::create(filename)?);
    serialize_into(&mut f, &lwe_s)?;
    serialize_into(&mut f, &glwe_s)?;
    serialize_into(&mut f, &lwe_extracted_s)?;
    serialize_into(&mut f, &lwe_extracted_to_lwe_ksk_s)?;
    serialize_into(&mut f, &lwe_bsk_s)?;

    println!("Keys saved.");

    Ok(())
}

pub fn load_keys(
    filename: &str,
    serial_engine: &mut DefaultSerializationEngine,
) -> Result<Keys, Box<dyn Error>> {
    println!("Loading keys...");

    // Read into vectors (which are owned) (ORDER MATTERS)
    let mut f = BufReader::new(File::open(filename)?);
    let lwe_s: Vec<u8> = deserialize_from(&mut f)?;
    let glwe_s: Vec<u8> = deserialize_from(&mut f)?;
    let lwe_extracted_s: Vec<u8> = deserialize_from(&mut f)?;
    let lwe_extracted_to_lwe_ksk_s: Vec<u8> = deserialize_from(&mut f)?;
    let lwe_bsk_s: Vec<u8> = deserialize_from(&mut f)?;

    // Deserialize into keys
    let lwe: LweSecretKey64 = serial_engine.deserialize(&lwe_s[..])?;
    let glwe: GlweSecretKey64 = serial_engine.deserialize(&glwe_s[..])?;
    let extracted: LweSecretKey64 = serial_engine.deserialize(&lwe_extracted_s[..])?;
    let ksk_extracted_lwe: LweKeyswitchKey64 =
        serial_engine.deserialize(&lwe_extracted_to_lwe_ksk_s[..])?;
    let bsk: LweBootstrapKey64 = serial_engine.deserialize(&lwe_bsk_s[..])?;

    println!("Keys loaded.");

    Ok(Keys {
        lwe,
        glwe,
        extracted,
        ksk_extracted_lwe,
        bsk,
    })
}

pub fn get_cuda_keys(h_keys: &Keys, cuda_engine: &mut CudaEngine) -> Result<CudaKeys, Box<dyn Error>> 
{
    let ksk = cuda_engine.convert_lwe_keyswitch_key(&h_keys.ksk_extracted_lwe)?;
    let bsk = cuda_engine.convert_lwe_bootstrap_key(&h_keys.bsk)?;
    Ok(CudaKeys {
        ksk_extracted_lwe: ksk,
        bsk
    })
}

fn print_key_info(keys: &Keys) {
    println!("The useful size of `lwe` is {}", size_of_val(&keys.lwe));
    println!(
        "The useful size of `glwe_key` is {}",
        size_of_val(&keys.glwe)
    );
    println!("The useful size of `bsk` is {}", size_of_val(&keys.bsk));
    println!(
        "The useful size of `extracted` is {}",
        size_of_val(&keys.extracted)
    );
    println!(
        "The useful size of `ksk_extracted_lwe` is {}",
        size_of_val(&keys.ksk_extracted_lwe)
    );
}
