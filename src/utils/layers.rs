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
use indicatif::*;

use crate::utils::keys::*;
use crate::utils::luts::*;
use crate::utils::common::*;

use num::*;
use std::borrow::Borrow;
use std::error::Error;
use std::ops::AddAssign;
use std::ops::SubAssign;
use std::result;

#[instrument]
pub fn prepare_layer(
    log_p: i32,
    log_q: i32,
    units: usize,
    config: &Parameters,
    cuda_engine: &mut CudaEngine,
    default_engine: &mut DefaultEngine,
) -> Result<
    (
        CudaLweCiphertextVector64,
        CudaLweCiphertextVector64,
        CudaGlweCiphertextVector64,
    ),
    Box<dyn Error>,
> {
    // Create placeholder vector
    let placeholder_pt_vector = default_engine.create_plaintext_vector_from(&vec![0u64; units])?;
    let placeholder_ct_vector_output = default_engine.trivially_encrypt_lwe_ciphertext_vector(LweSize(config.N.0 + 1), &placeholder_pt_vector)?;
    let placeholder_ct_vector_temp = default_engine.trivially_encrypt_lwe_ciphertext_vector(config.n.to_lwe_size(), &placeholder_pt_vector)?;
    // IMPORTANT: We use N for output (after pbs n->N) and n for temp (after keyswitch N->n) since we do AP type 1 ^^^^^^

    // Create device placeholders
    let mut d_output = cuda_engine.convert_lwe_ciphertext_vector(&placeholder_ct_vector_output)?;
    let mut d_temp = cuda_engine.convert_lwe_ciphertext_vector(&placeholder_ct_vector_temp)?;

    // Create device luts
    let h_luts = sign_lut(log_p, log_q, d_output.lwe_ciphertext_count().0, &config, default_engine)?;
    let mut d_luts = cuda_engine.convert_glwe_ciphertext_vector(&h_luts)?;

    Ok((d_output, d_temp, d_luts))
}

#[instrument]
pub fn encrypted_rnn_block(
    encrypted_input: &ArrayView2<LweCiphertext64>,
    pt_input: &ArrayView2<i32>, // For debugging
    kernel: &ArrayView2<i8>,
    recurrent_kernel: &ArrayView2<i8>,
    layer_name: &str,
    log_p: i32, log_q: i32,
    d_keys: &CudaKeys, h_keys: &Keys, config: &Parameters,
    cuda_engine: &mut CudaEngine, amortized_cuda_engine: &mut AmortizedCudaEngine, default_engine: &mut DefaultEngine, 
// ) -> Result<Array2<LweCiphertext64>, Box<dyn Error>>
) -> Result<(Array2<LweCiphertext64>, Array2<i32>), Box<dyn Error>> // For debugging
{
    let num_units = kernel.dim().1;
    let num_ts = encrypted_input.dim().0;
    let (mut d_output, mut d_temp, d_luts) = prepare_layer(log_p, log_q, num_units, config, cuda_engine, default_engine)?;
    let mut states: Array2<LweCiphertext64> = Array2::from_shape_vec((num_ts, num_units), vec![encrypted_input[[0,0]].clone(); num_ts*num_units])?;
    let mut pt_states: Array2<i32> = Array2::from_shape_vec((num_ts, num_units), vec![0i32; num_ts*num_units])?; // For debugging

    // let encrypted_input = encrypt_lwe_array(pt_input, log_p, log_q, &h_keys.extracted, config, default_engine)?; // REFRESH CTXT FOR DEBUG
    // for (t, ct_t) in encrypted_input.rows().into_iter().enumerate() {
    for (t, (ct_t, pt_t)) in encrypted_input.rows().into_iter().zip(pt_input.rows().into_iter()).enumerate().progress_count(encrypted_input.dim().0 as u64) { // For debugging
        // W_x * x
        let mut output = matmul_custom_1D(&ct_t, kernel, 1, &config, default_engine)?.row(0).to_owned();
        let mut pt_output = plaintext_matmul_custom_1D(&pt_t, kernel)?; // For debugging

        // DEBUG
        // check_pt_ct_difference(&output.view(), &pt_output.view(), format!("{}: ts {}: kernel matmul", layer_name, t).as_str(), false, log_p, log_q, h_keys, default_engine)?;

        if t != 0 {
            // First state is zeros but I don't want to initialize zeros
            // W_h * h
            let mut hidden = matmul_custom_1D(&states.row(t - 1), recurrent_kernel, 1, &config, default_engine)?.row(0).to_owned();
            let mut pt_hidden = plaintext_matmul_custom_1D(&pt_states.row(t - 1), recurrent_kernel)?; // For debugging

            // DEBUG
            // check_pt_ct_difference(&hidden.view(), &pt_hidden.view(), format!("{}: ts {}: recurrent matmul", layer_name, t).as_str(), false, log_p, log_q, h_keys, default_engine)?;

            // W_x * x + W_h * h
            for (x_i, h_i) in output.iter_mut().zip(hidden.iter()) { default_engine.fuse_add_lwe_ciphertext(x_i, h_i)?; }
            pt_output = pt_output + pt_hidden; // For debugging

            // DEBUG
            // check_pt_ct_difference(&output.view(), &pt_output.view(), format!("{}: ts {}: addition", layer_name, t).as_str(), false, log_p, log_q, h_keys, default_engine)?;
        }

        // sign(W_x * x + W_h * h)
        output = sign_activation(output, &mut d_temp, &mut d_output, &d_luts, &d_keys,cuda_engine, amortized_cuda_engine, default_engine)?;
        pt_output = pt_output.mapv(|x| mod_sign(x, log_p-1)); // For debugging

        // DEBUG
        // check_pt_ct_difference(&output.view(), &pt_output.view(), format!("{}: ts {}: activation", layer_name, t).as_str(), false, log_p, log_q, h_keys, default_engine)?;

        states.row_mut(t).assign(&output);
        pt_states.row_mut(t).assign(&pt_output); // For debugging
    }

    // DEBUG
    // println!("\nDone {}", layer_name);
    check_pt_ct_difference(&states.view(), &pt_states.view(), format!("{}: output", layer_name).as_str(), false, log_p, log_q, h_keys, default_engine)?;
    // println!("Done {}\n", layer_name);
    // Ok(states)
    Ok((states, pt_states)) // For debugging
}

#[instrument]
pub fn encrypted_dense_block(
    encrypted_input: &ArrayView2<LweCiphertext64>,
    pt_input: &ArrayView2<i32>, // For debugging
    kernel: &ArrayView2<i8>,
    layer_name: &str,
    compute_activation: bool,
    num_accs: usize,
    log_p: i32, log_q: i32,
    d_keys: &CudaKeys, h_keys: &Keys, config: &Parameters,
    cuda_engine: &mut CudaEngine, amortized_cuda_engine: &mut AmortizedCudaEngine, default_engine: &mut DefaultEngine,
// ) -> Result<Array2<LweCiphertext64>, Box<dyn Error>>
) -> Result<(Array2<LweCiphertext64>, Array2<i32>), Box<dyn Error>> // For debuggin
{
    let num_units = kernel.dim().1;
    let num_ts = encrypted_input.dim().0;
    let (mut d_output, mut d_temp, d_luts) = prepare_layer(log_p, log_q, num_units, config, cuda_engine, default_engine)?;
    let mut states: Array2<LweCiphertext64> = Array2::from_shape_vec((num_ts, num_units), vec![encrypted_input[[0,0]].clone(); num_ts*num_units])?;
    let mut pt_states: Array2<i32> = Array2::from_shape_vec((num_ts, num_units), vec![0i32; num_ts*num_units])?; // For debugging

    // let encrypted_input = encrypt_lwe_array(pt_input, log_p, log_q, &h_keys.extracted, config, default_engine)?; // REFRESH CTXT FOR DEBUG
    // for (t, ct_t) in encrypted_input.rows().into_iter().enumerate() {
    if num_accs == 1 {
        for (t, (ct_t, pt_t)) in encrypted_input.rows().into_iter().zip(pt_input.rows().into_iter()).enumerate().progress_count(encrypted_input.dim().0 as u64) { // For debugging
            // W_x * x
            let mut output = matmul_custom_1D(&ct_t, kernel, 1, &config, default_engine)?.row(0).to_owned();
            let mut pt_output = plaintext_matmul_custom_1D(&pt_t, kernel)?; // For debugging
                
            // DEBUG
            // check_pt_ct_difference(&output.view(), &pt_output.view(), format!("{}: ts {}: kernel matmul", layer_name, t).as_str(), false, log_p, log_q, h_keys, default_engine)?; 
            // pt_output = pt_output.mapv(|x| mod_to_precision(x, log_p)); // Correct pt precision

            // sign(W_x * x)
            if compute_activation {
                output = sign_activation(output, &mut d_temp, &mut d_output, &d_luts, &d_keys, cuda_engine, amortized_cuda_engine, default_engine)?;
                pt_output = pt_output.mapv(|x| mod_sign(x, log_p-1)); // For debugging

                // DEBUG
                // check_pt_ct_difference(&output.view(), &pt_output.view(), format!("{}: ts {}: activation", layer_name, t).as_str(), false, log_p, log_q, h_keys, default_engine)?; 
            }
            states.row_mut(t).assign(&output);
            pt_states.row_mut(t).assign(&pt_output); // For debugging
        }

        // DEBUG
        // println!("\nDone {}", layer_name);
        check_pt_ct_difference(&states.view(), &pt_states.view(), format!("{}: output", layer_name).as_str(), false, log_p, log_q, h_keys, default_engine)?;
        // println!("Done {}\n", layer_name);
    } else {
        // Very hacky
        let mut output = matmul_custom_1D(&encrypted_input.row(0), kernel, num_accs, &config, default_engine)?;
        let mut pt_output = plaintext_matmul_custom_1D(&pt_input.row(0), kernel)?; // For debugging
        
        states = output;
        pt_states.row_mut(0).assign(&pt_output); // For debugging
    }

    // Ok(states)
    Ok((states, pt_states)) // For debugging
}

#[instrument]
pub fn plaintext_rnn_block(
    input: &ArrayView2<i32>,
    kernel: &ArrayView2<i32>,
    recurrent_kernel: &ArrayView2<i32>,
) -> Result<Array2<i32>, Box<dyn Error>> 
{
    let num_units = kernel.dim().1;
    let mut states: Array2<i32> = Array2::from_shape_vec((0, num_units), vec![])?;
    for (t, ct_t) in input.rows().into_iter().enumerate() {
        // W_x * x
        // let mut output = ct_t.dot(kernel);
        let mut output = plaintext_matmul_custom_1D(&ct_t, kernel)?;

        if t != 0 {
            // First state is zeros but I don't want to initialize zeros
            // W_h * h
            // let mut hidden = states.row(t-1).dot(recurrent_kernel);
            let mut hidden = plaintext_matmul_custom_1D(&states.row(t - 1), recurrent_kernel)?;

            // W_x * x + W_h * h
            for (x_i, h_i) in output.iter_mut().zip(hidden.iter()) {
                *x_i += h_i;
            }
        }

        // sign(W_x * x + W_h * h) where sign(0) = 1
        output = output.mapv(|x| sgn_zero_is_one(x));

        states.append(Axis(0), output.view().into_shape((1, num_units))?)?;
    }
    Ok(states)
}

#[instrument]
pub fn plaintext_dense_block(
    input: &ArrayView2<i32>,
    kernel: &ArrayView2<i32>,
    compute_activation: bool,
) -> Result<Array2<i32>, Box<dyn Error>> {
    let num_units = kernel.dim().1;
    let mut states: Array2<i32> = Array2::from_shape_vec((0, num_units), vec![])?;
    for (t, ct_t) in input.rows().into_iter().enumerate() {
        // W_x * x
        // let mut output = ct_t.dot(kernel);
        let mut output = plaintext_matmul_custom_1D(&ct_t, kernel)?;

        if compute_activation {
            // sign(W_x * x) where sign(0) = 1
            output = output.mapv(|x| sgn_zero_is_one(x));
        }

        states.append(Axis(0), output.view().into_shape((1, num_units))?)?;
    }
    Ok(states)
}

#[instrument]
pub fn time_reduction<T>(input: ArrayView2<T>) -> Result<ArrayView2<T>, Box<dyn Error>> {
    let (dim_0, dim_1) = input.dim();
    let output = input.into_shape((dim_0 / 2, dim_1 * 2))?;
    Ok(output)
}

#[instrument]
pub fn flatten_2D<T>(input: ArrayView2<T>) -> Result<ArrayView2<T>, Box<dyn Error>> {
    let (dim_0, dim_1) = input.dim();
    let output = input.into_shape((1, dim_0 * dim_1))?;
    Ok(output)
}