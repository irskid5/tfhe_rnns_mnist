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
    run_pt: bool,
    encrypted_input: &ArrayView2<LweCiphertext64>,
    pt_input: &ArrayView2<i32>,
    kernel: &ArrayView2<i8>,
    recurrent_kernel: &ArrayView2<i8>,
    layer_name: &str,
    log_p: i32, log_q: i32,
    d_keys: &CudaKeys, h_keys: &Keys, config: &Parameters,
    cuda_engine: &mut CudaEngine, amortized_cuda_engine: &mut AmortizedCudaEngine, default_engine: &mut DefaultEngine, 
) -> Result<(Array2<LweCiphertext64>, Array2<i32>), Box<dyn Error>>
{
    let num_units: usize = kernel.dim().1;
    let num_ts: usize = encrypted_input.dim().0;
    let (mut d_output, mut d_temp, d_luts) = prepare_layer(log_p, log_q, num_units, config, cuda_engine, default_engine)?;
    let mut states: Array2<LweCiphertext64> = Array2::from_shape_vec((num_ts, num_units), vec![encrypted_input[[0,0]].clone(); num_ts*num_units])?;
    let mut pt_states: Array2<i32> = Array2::from_shape_vec((num_ts, num_units), vec![0i32; num_ts*num_units])?;

    for (t, ct_t) in encrypted_input.rows().into_iter().enumerate() {
        // W_x * x
        let mut output = matmul_custom_1D(&ct_t, kernel, 1, &config, default_engine)?.row(0).to_owned();
        
        if t != 0 {
            // W_h * h
            let mut hidden = matmul_custom_1D(&states.row(t - 1), recurrent_kernel, 1, &config, default_engine)?.row(0).to_owned();

            // W_x * x + W_h * h
            for (x_i, h_i) in output.iter_mut().zip(hidden.iter()) { default_engine.fuse_add_lwe_ciphertext(x_i, h_i)?; }
        }

        // sign(W_x * x + W_h * h)
        output = sign_activation(output, &mut d_temp, &mut d_output, &d_luts, &d_keys,cuda_engine, amortized_cuda_engine, default_engine)?;
        states.row_mut(t).assign(&output);
    }

    if run_pt {
        for (t, pt_t) in pt_input.rows().into_iter().enumerate() { 
            // W_x * x
            let mut pt_output = plaintext_matmul_custom_1D(&pt_t, kernel)?; 

            if t != 0 {
                // W_h * h
                let mut pt_hidden = plaintext_matmul_custom_1D(&pt_states.row(t - 1), recurrent_kernel)?;

                // W_x * x + W_h * h
                pt_output = pt_output + pt_hidden;
            }

            // sign(W_x * x + W_h * h)
            pt_output = pt_output.mapv(|x| mod_sign(x, log_p-1)); 
            
            pt_states.row_mut(t).assign(&pt_output); 
        }
    }
    
    Ok((states, pt_states))
}

#[instrument]
pub fn encrypted_dense_block(
    run_pt: bool,
    encrypted_input: &ArrayView2<LweCiphertext64>,
    pt_input: &ArrayView2<i32>,
    kernel: &ArrayView2<i8>,
    layer_name: &str,
    compute_activation: bool,
    num_accs: usize,
    log_p: i32, log_q: i32,
    d_keys: &CudaKeys, h_keys: &Keys, config: &Parameters,
    cuda_engine: &mut CudaEngine, amortized_cuda_engine: &mut AmortizedCudaEngine, default_engine: &mut DefaultEngine,
) -> Result<(Array2<LweCiphertext64>, Array2<i32>), Box<dyn Error>>
{
    let num_units = kernel.dim().1;
    let num_ts = encrypted_input.dim().0;
    let (mut d_output, mut d_temp, d_luts) = prepare_layer(log_p, log_q, num_units, config, cuda_engine, default_engine)?;
    let mut states: Array2<LweCiphertext64> = Array2::from_shape_vec((num_ts, num_units), vec![encrypted_input[[0,0]].clone(); num_ts*num_units])?;
    let mut pt_states: Array2<i32> = Array2::from_shape_vec((num_ts, num_units), vec![0i32; num_ts*num_units])?; // For debugging

    if num_accs == 1 {
        for (t, ct_t) in encrypted_input.rows().into_iter().enumerate() {
            // W_x * x
            let mut output = matmul_custom_1D(&ct_t, kernel, 1, &config, default_engine)?.row(0).to_owned();

            // sign(W_x * x)
            if compute_activation {
                output = sign_activation(output, &mut d_temp, &mut d_output, &d_luts, &d_keys, cuda_engine, amortized_cuda_engine, default_engine)?;
            }
            states.row_mut(t).assign(&output);
        }

        if run_pt {
            for (t, pt_t) in pt_input.rows().into_iter().enumerate() { 
                let mut pt_output = plaintext_matmul_custom_1D(&pt_t, kernel)?;
    
                // sign(W_x * x)
                if compute_activation {
                    pt_output = pt_output.mapv(|x| mod_sign(x, log_p-1)); 
                }
                pt_states.row_mut(t).assign(&pt_output); 
            }
        }
    } else {
        // Very hacky
        let mut output = matmul_custom_1D(&encrypted_input.row(0), kernel, num_accs, &config, default_engine)?;       
        states = output;

        if run_pt {
            let mut pt_output = plaintext_matmul_custom_1D(&pt_input.row(0), kernel)?; // For debugging
            pt_states.row_mut(0).assign(&pt_output); 
        }
    }

    Ok((states, pt_states)) 
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