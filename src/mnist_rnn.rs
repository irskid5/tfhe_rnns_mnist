#![allow(non_snake_case)]
#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_mut)]
#![allow(unused_imports)]

use concrete_core::prelude::*;
use hdf5::{Dataset, File};
use ndarray::*;
use time_graph::{instrument, spanned};

use std::collections::HashMap;
use std::time::{Duration, Instant};

use std::error::Error;

use crate::utils::common::*;
use crate::utils::datasets::*;
use crate::utils::init::*;
use crate::utils::keys::*;
use crate::utils::layers::*;
use crate::utils::luts::*;

#[macro_export]
macro_rules! print_rnn_banner {
    ($func_name:ident, $($args:expr),*) => {
        println!("\n--------------------------------------------------");
        println!("Beginning {}", stringify!($func_name).to_uppercase());
        println!("--------------------------------------------------\n");
        $func_name($($args),*)?;
        println!("\n--------------------------------------------------");
        println!("Ending {}", stringify!($func_name).to_uppercase());
        println!("--------------------------------------------------\n");
    };
}

fn mnist_weights_import_hashmap(
    filename: &str,
    default_engine: &mut DefaultEngine,
) -> Result<HashMap<String, Array2<i8>>, Box<dyn Error>> {
    // Open the HDF5 file and get the datasets for the weight matrices
    println!("Loading MNIST RNN weights into hashmap<name, array>.");
    println!("Loading from {}", filename);
    let file = hdf5::File::open(filename)?;

    // Create a HashMap to store the weight matrices
    let mut weight_matrices: HashMap<String, Array2<i8>> = HashMap::new();

    println!("Opening HDF5 file, is_empty = {:?}", file.is_empty());

    let mut datasets: Vec<Dataset> = Vec::new();
    datasets.push(file.dataset("QRNN_0/QRNN_0/QRNN_0/quantized_kernel:0")?);
    datasets.push(file.dataset("QRNN_0/QRNN_0/QRNN_0/quantized_recurrent_kernel:0")?);
    datasets.push(file.dataset("QRNN_1/QRNN_1/QRNN_1/quantized_kernel:0")?);
    datasets.push(file.dataset("QRNN_1/QRNN_1/QRNN_1/quantized_recurrent_kernel:0")?);
    datasets.push(file.dataset("DENSE_0/DENSE_0/quantized_kernel:0")?);
    datasets.push(file.dataset("DENSE_OUT/DENSE_OUT/quantized_kernel:0")?);

    for dataset in datasets {
        let name = dataset.name();
        // println!("{:?}", name);
        if name.contains("quantized") {
            let parts: Vec<&str> = name.split("/").collect();
            let last_two = format!("{}/{}", parts[parts.len() - 2], parts[parts.len() - 1]);
            let mut data: Vec<i8> = dataset.read_raw()?;
            // let mut data: Vec<Cleartext64> = data.iter().map(|x| {
            //     default_engine.create_cleartext_from(x).unwrap()
            // }).collect();
            let shape = dataset.shape();
            let array: Array2<i8> =
                Array::from_shape_vec((shape[0] as usize, shape[1] as usize), data)?;
            weight_matrices.insert(last_two, array);
        }
    }

    // Print the weight matrices in the HashMap
    println!("Loaded weights.");
    // for (name, matrix) in weight_matrices.iter() {
    //     println!("{}:\n{:?}", name, matrix);
    // }

    Ok(weight_matrices)
}

/**
 * Evaluates the MNIST RNN. If run_pt is true, then plaintext run will occur concurrently
 * with encrypted run, and error metrics will be calculated and displayed as well. If run_pt
 * is false, only encrypted run will occur and error metrics are not calculated.
 */
#[instrument]
pub fn mnist_rnn(
    run_pt: bool,
    model_type: bool, // 'true' for regular, 'false' for enlarged
    config: &Parameters,
) -> Result<(), Box<dyn Error>> {
    // Create the necessary engines
    const UNSAFE_SECRET: u128 = 1997; // For experimental reproducibility
    let (
        mut default_engine,
        mut serial_engine,
        mut parallel_engine,
        mut cuda_engine,
        mut amortized_cuda_engine,
    ) = init_engines(UNSAFE_SECRET)?;

    // COMMENT BOTH LINES OUT TO LOAD KEYS, UNCOMMENT BOTH LINES TO CREATE AND SAVE KEYS
    let h_keys: Keys = create_keys(config, &mut default_engine, &mut parallel_engine)?;
    save_keys("./keys/keys.bin", "./keys/", &h_keys, &mut serial_engine)?;

    // COMMENT OUT TO SAVE KEYS, UNCOMMENT TO LOAD KEYS
    // let h_keys: Keys = load_keys("./keys/keys.bin", &mut serial_engine)?;

    let d_keys: CudaKeys = get_cuda_keys(&h_keys, &mut cuda_engine)?;
    println!("{:?}", config);

    // Establish precision
    let log_q: i32 = 64; // Using 64-bit LWE ciphertexts
    let log_p: i32 = 6;
    let round_off: u64 = 1u64 << (log_q - log_p - 1);

    // Import dataset (MUST BE TERNARIZED)
    let mnist_config: MNISTConfig = if model_type {
        MNISTConfig {
            mnist_images_file: "mnist_preprocessed/mnist_images_norm_tern.npy",
            mnist_labels_file: "mnist_preprocessed/mnist_labels.npy",
        }
    } else {
        MNISTConfig {
            mnist_images_file: "mnist_preprocessed/mnist_images_norm_tern_128x128.npy",
            mnist_labels_file: "mnist_preprocessed/mnist_labels_128x128.npy",
        }
    };
    let (mut x, mut y): (ndarray::ArrayD<i8>, ndarray::ArrayD<i8>) = import_mnist(&mnist_config)?;
    let mut x = x.into_dimensionality::<Ix3>()?;
    let mut y = y.into_dimensionality::<Ix1>()?;

    // Import weights (MUST BE BINARIZED)
    let weights: HashMap<String, Array2<i8>> = if model_type {
        mnist_weights_import_hashmap(
            "models/regular/20230205-190604-parameters.hdf5",
            &mut default_engine,
        )?
    } else {
        mnist_weights_import_hashmap(
            "models/enlarged/20230402-145048-parameters.hdf5",
            &mut default_engine,
        )?
    };

    if model_type {
        println!("\nRunning REGULAR MNIST RNN (1,914,368 parameters)")
    } else {
        println!("\nRunning ENLARGED MNIST RNN (8,480,768 parameters)")
    }

    println!("\n==================================================\n");

    // Loop through dataset (one epoch)
    let mut correct_preds = 0;
    let mut pt_correct_preds = 0;
    let mut correct_top5_preds = 0;
    let mut pt_correct_top5_preds = 0;
    let mut dense_out_dif_percent: Vec<f32> = vec![];
    let mut dense_out_mae: Vec<f32> = vec![];
    let dense_out_num_accs = 4; // SET THE NUMBER OF ACCUMULATION CIPHERTEXTS FOR OUTPUT LAYER
    let num_test_images = 10000; // SET THE NUMBER OF TEST IMAGES YOU WANT TO RUN
    for (i, img) in x.axis_iter(ndarray::Axis(0)).enumerate() {
        spanned!("encrypted_run", {
            let start = Instant::now();

            // Stopping condition
            if i == num_test_images {
                break;
            }

            println!("Running sample {} ------------------------------------------------------------------", i+1);

            // Encrypt inputs
            let pt = img.clone().to_owned().mapv(|x| x as i32);
            let ct = encrypt_lwe_array(
                &img,
                log_p,
                log_q,
                &h_keys.extracted,
                &config,
                &mut default_engine,
            )?;

            // -------------------------- START FWD STEP ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

            println!("Beginning encrypted run.");

            // FIRST RNN(128) --------------------------------------------------------------------------------------------------------
            let (qrnn_0, pt_qrnn_0) = spanned!("FIRST RNN(128)", {
                encrypted_rnn_block(
                    run_pt,
                    &ct.view(),
                    &pt.view(),
                    &weights["QRNN_0/quantized_kernel:0"].view(),
                    &weights["QRNN_0/quantized_recurrent_kernel:0"].view(),
                    "FIRST RNN(128)",
                    log_p,
                    log_q,
                    &d_keys,
                    &h_keys,
                    config,
                    &mut cuda_engine,
                    &mut amortized_cuda_engine,
                    &mut default_engine,
                )?
            });
            if run_pt {
                check_pt_ct_difference(
                    &qrnn_0.view(),
                    &pt_qrnn_0.view(),
                    format!("{}: output", "FIRST RNN(128)").as_str(),
                    false,
                    log_p,
                    log_q,
                    &h_keys,
                    &mut default_engine,
                )?;
            }

            // TIME REDUCTION LAYER ------------------------------------------------------------------------------------------------
            let tr = time_reduction(qrnn_0.view())?;
            let pt_tr = time_reduction(pt_qrnn_0.view())?;

            // SECOND RNN(128) ---------------------------------------------------------------------------------------------------------
            let (qrnn_1, pt_qrnn_1) = spanned!("SECOND RNN(128)", {
                encrypted_rnn_block(
                    run_pt,
                    &tr,
                    &pt_tr,
                    &weights["QRNN_1/quantized_kernel:0"].view(),
                    &weights["QRNN_1/quantized_recurrent_kernel:0"].view(),
                    "SECOND RNN(128)",
                    log_p,
                    log_q,
                    &d_keys,
                    &h_keys,
                    config,
                    &mut cuda_engine,
                    &mut amortized_cuda_engine,
                    &mut default_engine,
                )?
            });
            if run_pt {
                check_pt_ct_difference(
                    &qrnn_1.view(),
                    &pt_qrnn_1.view(),
                    format!("{}: output", "SECOND RNN(128)").as_str(),
                    false,
                    log_p,
                    log_q,
                    &h_keys,
                    &mut default_engine,
                )?;
            }
            // ---------------------------------------------------------------------------------------------------------------------

            // FLATTEN -------------------------------------------------------------------------------------------------------------
            let flattened = flatten_2D(qrnn_1.view())?;
            let pt_flattened = flatten_2D(pt_qrnn_1.view())?;

            // FF(1024) -------------------------------------------------------------------------------------------------------
            let (dense_0, pt_dense_0) = spanned!("FF(1024)", {
                encrypted_dense_block(
                    run_pt,
                    &flattened,
                    &pt_flattened,
                    &weights["DENSE_0/quantized_kernel:0"].view(),
                    "FF(1024)",
                    true,
                    1,
                    log_p,
                    log_q,
                    &d_keys,
                    &h_keys,
                    config,
                    &mut cuda_engine,
                    &mut amortized_cuda_engine,
                    &mut default_engine,
                )?
            });
            if run_pt {
                check_pt_ct_difference(
                    &dense_0.view(),
                    &pt_dense_0.view(),
                    format!("{}: output", "FF(1024)").as_str(),
                    false,
                    log_p,
                    log_q,
                    &h_keys,
                    &mut default_engine,
                )?;
            }

            // OUT(10) -------------------------------------------------------------------------------------------------------
            let (dense_out, pt_dense_out) = spanned!("dense_out", {
                encrypted_dense_block(
                    run_pt,
                    &dense_0.view(),
                    &pt_dense_0.view(),
                    &weights["DENSE_OUT/quantized_kernel:0"].view(),
                    "OUT(10)",
                    false,
                    dense_out_num_accs,
                    log_p,
                    log_q,
                    &d_keys,
                    &h_keys,
                    config,
                    &mut cuda_engine,
                    &mut amortized_cuda_engine,
                    &mut default_engine,
                )?
            });

            // Decrypt, convert to signed, compute softmax, and get argmax
            let mut ct_logits: Array2<u64> = decrypt_lwe_array(
                &dense_out.view(),
                log_p,
                log_q,
                &h_keys.extracted,
                &mut default_engine,
            )?;
            let mut ct_logits = ct_logits.mapv(|x| iP_to_iT::<i32>(x, log_p));
            let mut ct_logits = ct_logits.sum_axis(Axis(0));
            let ct_result = compute_softmax_then_argmax(&ct_logits)?;
            let ct_top_5 = return_top_n(&ct_logits, 5)?;

            let pt_dense_out = pt_dense_out.into_shape(ct_logits.dim())?;
            if run_pt {
                // Calculate stats
                let dense_out_stats = check_pt_pt_difference(
                    &ct_logits.view(),
                    &pt_dense_out.view(),
                    format!("{}: output", "OUT(10)").as_str(),
                    false,
                )?;
                dense_out_dif_percent.push(dense_out_stats.0);
                dense_out_mae.push(dense_out_stats.1);
            }

            println!("Completed encrypted run.\n");

            // -------------------------- END ENCRYPTED FWD STEP ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

            println!("Encrypted MNIST RNN result:      {}", ct_result);
            println!(
                "Encrypted Top 5 (decreasing):    {}\n",
                ct_top_5.to_string()
            );

            if run_pt {
                let pt_result = compute_softmax_then_argmax(&pt_dense_out)?;
                let pt_top_5 = return_top_n(&pt_dense_out, 5)?;
                println!("Plaintext MNIST RNN result:      {}", pt_result);
                println!(
                    "Plaintext Top 5 (decreasing):    {}\n",
                    pt_top_5.to_string()
                );

                if pt_result as i8 == y[[i]] {
                    pt_correct_preds += 1;
                }
                if pt_top_5.to_vec().contains(&(y[[i]] as usize)) {
                    pt_correct_top5_preds += 1;
                }
            }

            println!("True result:                     {}\n", y[[i]]);

            // Metric calculations
            if ct_result as i8 == y[[i]] {
                correct_preds += 1;
            }
            if ct_top_5.to_vec().contains(&(y[[i]] as usize)) {
                correct_top5_preds += 1;
            }
            println!("Correct CT predictions (top-1) = {}", correct_preds);
            println!("Correct CT predictions (top-5) = {}\n", correct_top5_preds);

            if run_pt {
                println!("Correct PT predictions (top-1) = {}", pt_correct_preds);
                println!(
                    "Correct PT predictions (top-5) = {}\n",
                    pt_correct_top5_preds
                );
            }

            let duration = start.elapsed();
            println!(
                "Time elapsed: {:.4} s\n",
                duration.as_millis() as f32 / 1000f32
            );
        });
    }

    // Stat calculations
    let acc = 100_f32 * correct_preds as f32 / num_test_images as f32;
    let top5_acc = 100_f32 * correct_top5_preds as f32 / num_test_images as f32;
    println!("\nCompleted {} predictions!", num_test_images);
    println!("\nAccuracy Statistics:");
    println!("  Encrypted Top-1 Accuracy = {:.2}%", acc);
    println!("  Encrypted Top-5 Accuracy = {:.2}%", top5_acc);
    if run_pt {
        let pt_acc = 100_f32 * pt_correct_preds as f32 / num_test_images as f32;
        let pt_top5_acc = 100_f32 * pt_correct_top5_preds as f32 / num_test_images as f32;
        let dense_out_dif_percent = arr1(&dense_out_dif_percent);
        let dense_out_mae = arr1(&dense_out_mae);
        println!("  Plaintext Top-1 Accuracy = {:.2}%", pt_acc);
        println!("  Plaintext Top-5 Accuracy = {:.2}%", pt_top5_acc);
        println!("\nOUT(10) Statistics:");
        println!(
            "  Number of output layer accumulator CTs = {}",
            dense_out_num_accs
        );
        println!(
            "  Percent different elements (mean, std) = ({:.2}%, {:.2}%)",
            dense_out_dif_percent.mean().unwrap(),
            dense_out_dif_percent.std(0f32)
        );
        println!(
            "  MAE (mean, std)                        = ({:.2}, {:.2})",
            dense_out_mae.mean().unwrap(),
            dense_out_mae.std(0f32)
        );
    }

    Ok(())
}
