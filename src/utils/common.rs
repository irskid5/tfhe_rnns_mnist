#![allow(non_snake_case)]
#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_mut)]
#![allow(unused_imports)]
#![allow(arithmetic_overflow)]

use concrete_core::backends::cuda;
use concrete_core::commons::crypto::lwe::LweList as ImplLweList;
use concrete_core::commons::numeric::SignedInteger;
use concrete_core::commons::numeric::UnsignedInteger;
use concrete_core::prelude::*;
use concrete_core::*;
use hdf5::*;
use ndarray::prelude::*;
use ndarray::*;
use ndarray_stats::DeviationExt;
use ndarray_stats::QuantileExt;
use num::traits::RefNum;
use time_graph::{instrument, spanned};

use crate::utils::keys::*;
use crate::utils::luts::*;

use num::*;
use std::borrow::Borrow;
use std::error::Error;
use std::fmt::Display;
use std::ops::AddAssign;
use std::ops::SubAssign;
use std::result;

fn convert_to_u64(input: impl Into<u64>) -> u64 {
    input.into()
}

#[instrument]
pub fn encrypt_lwe<T: Integer + NumCast>(
    input: &T,
    log_p: i32,
    log_q: i32,
    key: &LweSecretKey64,
    config: &Parameters,
    default_engine: &mut DefaultEngine,
) -> Result<LweCiphertext64, Box<dyn Error>> {
    let result = (input.to_i64().unwrap() as u64) << (log_q - log_p);
    let result = default_engine.create_plaintext_from(&result)?;
    let result = default_engine.encrypt_lwe_ciphertext(key, &result, config.lwe_var)?;
    Ok(result)
}

#[instrument]
pub fn encrypt_lwe_array<T: Integer + NumCast, D: ndarray::Dimension>(
    input: &ArrayView<T, D>,
    log_p: i32,
    log_q: i32,
    key: &LweSecretKey64,
    config: &Parameters,
    default_engine: &mut DefaultEngine,
) -> Result<Array<LweCiphertext64, D>, Box<dyn Error>> {
    let result = input.map(|x| encrypt_lwe(x, log_p, log_q, key, config, default_engine).unwrap());
    Ok(result)
}

#[instrument]
pub fn decrypt_lwe<T: Integer + NumCast>(
    input: &LweCiphertext64,
    log_p: i32,
    log_q: i32,
    key: &LweSecretKey64,
    default_engine: &mut DefaultEngine,
) -> Result<T, Box<dyn Error>> {
    let round_off = 1u64 << (log_q - log_p - 1);
    let pt = default_engine.decrypt_lwe_ciphertext(key, input)?;
    let raw = default_engine.retrieve_plaintext(&pt)?;
    let res = (raw.wrapping_add(round_off)) >> (log_q - log_p);
    let res = T::from(res).unwrap();
    Ok(res)
}

#[instrument]
pub fn decrypt_lwe_array<T: Integer + NumCast, D: ndarray::Dimension>(
    input: &ArrayView<LweCiphertext64, D>,
    log_p: i32,
    log_q: i32,
    key: &LweSecretKey64,
    default_engine: &mut DefaultEngine,
) -> Result<Array<T, D>, Box<dyn Error>> {
    let result = input.map(|x| decrypt_lwe::<T>(x, log_p, log_q, key, default_engine).unwrap());
    Ok(result)
}

pub fn count_different_elts<T: PartialEq, D: ndarray::Dimension>(
    a: &ArrayView<T, D>,
    b: &ArrayView<T, D>,
) -> usize {
    let mut count = 0;
    Zip::from(a).and(b).for_each(|a_i, b_i| {
        if *a_i != *b_i {
            count += 1;
        }
    });
    count
}

/**
 * Performs matrix multiplication between binary ciphertexts and ternary
 * plaintexts by adding ciphertext if plaintext is 1, subtracting if
 * plaintext is -1, and no operation when plaintext is 0.
 */
#[instrument]
pub fn matmul_custom_1D(
    ct: &ArrayView1<LweCiphertext64>,
    pt: &ArrayView2<i8>,
    num_accs: usize, // needs to be a factor of ct dimension
    config: &Parameters,
    default_engine: &mut DefaultEngine,
) -> Result<Array2<LweCiphertext64>, Box<dyn Error>> {
    // Init acc
    let zero_pt = default_engine.create_plaintext_from(&0u64)?;
    let acc_to_clone: LweCiphertext64 = default_engine
        .trivially_encrypt_lwe_ciphertext(ct[[0]].lwe_dimension().to_lwe_size(), &zero_pt)?;
    let chunk_size = ct.dim() / num_accs;

    // For each column
    let mut output = Array2::<LweCiphertext64>::from_elem((num_accs, pt.dim().1), acc_to_clone);
    for (c, column) in pt.columns().into_iter().enumerate() {
        // IMPORTANT: Using N for LweSize since we are using AP type 1 ^^^^^^^^^^^^^^
        // Multiply c_ij * p_jk and accumulate in acc
        let zipped = ct.iter().zip(column.iter()).collect::<Vec<_>>();
        for (chunk_i, chunk) in zipped.chunks(chunk_size).enumerate() {
            for (ci, pi) in chunk.iter() {
                if **pi < 0 {
                    default_engine.fuse_sub_lwe_ciphertext(&mut output[[chunk_i, c]], *ci)?;
                }
                if **pi > 0 {
                    default_engine.fuse_add_lwe_ciphertext(&mut output[[chunk_i, c]], *ci)?;
                }
            }
        }
    }
    Ok(output)
}

/**
 * Performs in the same was as matmul_custom_1D, but for clear values.
 */
#[instrument]
pub fn plaintext_matmul_custom_1D<T: Integer + AddAssign + SubAssign + Clone + Copy, S: Integer>(
    ct: &ArrayView1<T>,
    pt: &ArrayView2<S>,
) -> Result<Array1<T>, Box<dyn Error>> {
    // For each column
    let mut output: Vec<T> = vec![];
    for column in pt.columns().into_iter() {
        let mut acc = T::zero();
        // Multiply c_ij * p_jk and accumulate in acc
        for (ci, pi) in ct.iter().zip(column.iter()) {
            if *pi < S::zero() {
                acc -= *ci;
            }
            if *pi > S::zero() {
                acc += *ci;
            }
        }
        output.push(acc);
    }
    Ok(arr1(&output))
}

#[instrument]
pub fn decrypt_lwe_ciphertext_vector(
    input: &LweCiphertextVector64,
    log_p: i32,
    log_q: i32,
    keys: &Keys,
    default_engine: &mut DefaultEngine,
) -> Result<Vec<u64>, Box<dyn Error>> {
    // Rounding after decryption
    let round_off: u64 = 1u64 << (log_q - log_p - 1);

    // Decrypt into ptxts
    let result_pts = default_engine.decrypt_lwe_ciphertext_vector(&keys.lwe, input)?;

    // Get raw u64s
    let result_raw = default_engine.retrieve_plaintext_vector(&result_pts)?;

    // Perform rounding
    let result: Vec<u64> = result_raw
        .iter()
        .map(|x| (x + round_off) >> (log_q - log_p))
        .collect();

    Ok(result)
}

#[instrument]
pub fn decrypt_lwe_ciphertexts(
    input: &Vec<LweCiphertext64>,
    log_p: i32,
    log_q: i32,
    keys: &Keys,
    default_engine: &mut DefaultEngine,
) -> Result<Vec<u64>, Box<dyn Error>> {
    // Rounding after decryption
    let round_off: u64 = 1u64 << (log_q - log_p - 1);

    // Decrypt into ptxts
    let result: Vec<u64> = input
        .iter()
        .map(|x| {
            let pt = default_engine.decrypt_lwe_ciphertext(&keys.lwe, x).unwrap();
            let clrt = default_engine.retrieve_plaintext(&pt).unwrap();
            (clrt + round_off) >> (log_q - log_p)
        })
        .collect();

    Ok(result)
}

#[instrument]
pub fn populate_lwe_vector(
    ct_arr: Array1<LweCiphertext64>, // Note: deconstructs ct_arr after this function
    default_engine: &mut DefaultEngine,
) -> Result<LweCiphertextVector64, Box<dyn Error>> {
    let lwe_size = ct_arr[[0]].lwe_dimension().to_lwe_size();
    let mut lwes_raw: Vec<u64> = vec![];
    for lwe in ct_arr.iter() {
        let lwe_raw = default_engine.consume_retrieve_lwe_ciphertext(lwe.clone())?;
        lwes_raw.extend(lwe_raw);
    }
    let result = default_engine.create_lwe_ciphertext_vector_from(lwes_raw, lwe_size)?;
    Ok(result)
}

#[instrument]
pub fn depopulate_lwe_vector(
    ct_vec: LweCiphertextVector64,
    default_engine: &mut DefaultEngine,
) -> Result<Array1<LweCiphertext64>, Box<dyn Error>> {
    let chunk_size = ct_vec.lwe_dimension().to_lwe_size().0;
    let mut result_vec: Vec<LweCiphertext64> = vec![];
    let ct_vec_raw = default_engine.consume_retrieve_lwe_ciphertext_vector(ct_vec)?;
    for chunk in ct_vec_raw.chunks(chunk_size) {
        let lwe = default_engine.create_lwe_ciphertext_from(chunk.to_vec())?;
        result_vec.push(lwe);
    }
    Ok(Array::from(result_vec))
}

/**
 * Given a LUT, performs the PBS operation following AP'1 from 
 * https://eprint.iacr.org/2022/704 since it has less cost (DP->KS->PBS).
 * That means we perform a keyswitch (KS) then programmable bootstrap (PBS).
 */
#[instrument]
pub fn amortized_ks_pbs(
    output: &mut CudaLweCiphertextVector64,
    temp: &mut CudaLweCiphertextVector64,
    input: &CudaLweCiphertextVector64,
    luts: &CudaGlweCiphertextVector64,
    d_keys: &CudaKeys,
    cuda_engine: &mut CudaEngine,
    amortized_cuda_engine: &mut AmortizedCudaEngine,
) -> Result<(), Box<dyn Error>> {
    // Using AP'1 from https://eprint.iacr.org/2022/704 since it has less cost (DP->KS->PBS)
    spanned!("cuda_ks", {
        cuda_engine.discard_keyswitch_lwe_ciphertext_vector(temp, input, &d_keys.ksk_extracted_lwe)?;
    });
    spanned!("cuda_am_pbs", {
        amortized_cuda_engine.discard_bootstrap_lwe_ciphertext_vector(output, temp, luts, &d_keys.bsk)?;
    });
    Ok(())
}

/**
 * Given the sign function LUTs, performs the sign activation function by first moving the
 * relevant data to the GPU, performing the amortized ks and pbs operations, and then moving
 * the data back to the CPU for further operations.
 */
#[instrument]
pub fn sign_activation(
    input: Array1<LweCiphertext64>,
    d_temp: &mut CudaLweCiphertextVector64, // Can be persistant (per layer or per equal sized array)
    d_output: &mut CudaLweCiphertextVector64,
    d_sign_luts: &CudaGlweCiphertextVector64, // Can be persistant (per layer or per equal sized array)
    d_keys: &CudaKeys,
    cuda_engine: &mut CudaEngine,
    amortized_cuda_engine: &mut AmortizedCudaEngine,
    default_engine: &mut DefaultEngine,
) -> Result<Array1<LweCiphertext64>, Box<dyn Error>> {
    // Create and copy input vector
    let input_vec = populate_lwe_vector(input, default_engine)?;
    let d_input_vec = cuda_engine.convert_lwe_ciphertext_vector(&input_vec)?;

    // Perform sign fn
    amortized_ks_pbs(d_output, d_temp, &d_input_vec, d_sign_luts, d_keys, cuda_engine, amortized_cuda_engine)?;

    // Repopulate output array
    let h_output = cuda_engine.convert_lwe_ciphertext_vector(d_output)?;
    let result = depopulate_lwe_vector(h_output, default_engine)?;

    Ok(result)
}

/**
 * Performs the sign function where (x < 0) -> -1, (x >= 9) -> +1
 */
pub fn sgn_zero_is_one<T: Signed + AddAssign>(x: T) -> T {
    let mut s = x.signum();
    if s == T::zero() {
        s += T::one()
    };
    s
}

/**
 * The signed function. Performs the modulus operation on a signed integer for a given modulus (log_p), 
 * converting it from Z -> Z_(2^log_p).
 */
pub fn signed<T: Signed + AddAssign + std::ops::Shl<i32, Output=T> + std::cmp::PartialOrd>(x: T, log_p: i32) -> T {
    let base = T::one() << log_p;
    let half_base = T::one() << (log_p-1);
    let modded = x % base;
    let mut s: T;
    if modded >= half_base {
        s = modded - half_base;
    } else {
        s = modded;
    }
    s
}

/**
 * The ModSign function. Performs the sign function on an integer after converting it to Z_(2^log_p).
 */
pub fn mod_sign<T: Signed + AddAssign + std::ops::Shl<i32, Output=T> + std::cmp::PartialOrd>(x: T, log_p: i32) -> T {
    let s = signed(x, log_p);
    sgn_zero_is_one(s)
}

/**
 * Takes arrays of LweCt64 (which it decrypts) and a pt to measure against. It converts both plaintext arrays
 * to i32 for comparison (since the pt version might be accumulator). The decrypted pt is converted to i32
 * by preserving its precision (log_p).
 */
pub fn check_pt_ct_difference<T: Integer + NumCast + Clone, D: ndarray::Dimension>(
    ct: &ArrayView<LweCiphertext64, D>,
    pt: &ArrayView<T, D>,
    check_msg: &str,
    debug: bool,
    log_p: i32,
    log_q: i32,
    h_keys: &Keys,
    default_engine: &mut DefaultEngine,
) -> Result<(f32, f32), Box<dyn Error>> {
    let ct_decrypted: Array<u64, D> = decrypt_lwe_array(ct, log_p, log_q, &h_keys.extracted, default_engine)?;
    let ct_decrypted_i32 = ct_decrypted.mapv(|x| iP_to_iT::<i32>(x, log_p));
    if debug {
        println!("ct_raw = {:?}", ct_decrypted_i32);
    }
    check_pt_pt_difference(&ct_decrypted_i32.view(), &pt.view(), check_msg, debug)
}

pub fn check_pt_pt_difference<T1: Integer + NumCast + Clone, T2: Integer + NumCast + Clone, D: ndarray::Dimension>(
    pt1: &ArrayView<T1, D>,
    pt2: &ArrayView<T2, D>,
    check_msg: &str,
    debug: bool,
) -> Result<(f32, f32), Box<dyn Error>> {
    let pt1_i32 = pt1.mapv(|x| x.to_i32().unwrap());
    let pt2_i32 = pt2.mapv(|x| x.to_i32().unwrap());
    if debug {
        println!("pt1    = {:?}", pt1_i32);
        println!("pt2    = {:?}", pt2_i32);
    }
    
    // Calculate stats
    let num_dif_ele = count_different_elts(&pt1_i32.view(), &pt2_i32.view());
    let dif_ele_percent = 100_f32 * num_dif_ele as f32 / pt1_i32.len() as f32;
    let mae_pt_ct = pt1_i32.mean_abs_err(&pt2_i32)? as f32;

    println!(
        "[{}]: Number of different elements = {}, percentage {:.2}%, mae_pt_ct = {:.2}",
        check_msg, num_dif_ele, dif_ele_percent, mae_pt_ct,
    );

    // For printing out in csv style
    // println!(
    //     "{},{},{:.2}",
    //     num_dif_ele, dif_ele_percent, mae_pt_ct,
    // );

    Ok((dif_ele_percent, mae_pt_ct))
}

#[instrument]
pub fn softmax<T: Integer + NumCast + Clone>(a: &Array1<T>) -> Result<Array1<f64>, Box<dyn Error>> {
    let a_f64 = a.mapv(|x| x.to_f64().unwrap());
    let exp_a = a_f64.mapv(f64::exp);
    let sum_exp_a = exp_a.sum();
    Ok(exp_a / sum_exp_a)
}

pub fn compute_softmax_then_argmax<T: Integer + NumCast + Clone>(
    a: &Array1<T>,
) -> Result<usize, Box<dyn Error>> {
    let softmax_output = softmax(a)?;
    let argmax = a.argmax()?;
    Ok(argmax)
}

pub fn return_top_n<T: SignedInteger + NumCast + Clone>(
    a: &Array1<T>,
    n: usize
) -> Result<Array1<usize>, Box<dyn Error>> {
    let mut tmp = a.clone();
    let mut topn = arr1(&vec![0; n]);
    for i in 0..n {
        let top1 = tmp.argmax()?;
        tmp[top1] = -T::MAX;
        topn[[i]] = top1;
    }
    Ok(topn)
}

// Converts a value of signed precision p stored in a u64 to any signed integer you want.
pub fn iP_to_iT<T: SignedInteger + NumCast>(value: u64, p: i32) -> T {
    // let value = value % (1 << (p - 1));
    if (value & (1u64 << (p - 1))) != 0 {
        // MSB is 1, so value is negative
        return T::from(value).unwrap() - (T::ONE << (p) as usize);
    } else {
        // MSB is 0, so value is positive
        return T::from(value).unwrap();
    }
}

// Returns a vector of num_elements random i32s in a range between low and high
pub fn random_int64_vector(low: i64, high: i64, num_elements: usize) -> Vec<i64> {
    use rand::{distributions::Uniform, Rng};
    let mut rng = rand::thread_rng();
    let range = Uniform::new(low, high);
    return (0..num_elements).map(|_| rng.sample(&range)).collect()
}