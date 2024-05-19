#![allow(non_snake_case)]
#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_mut)]
#![allow(unused_imports)]

use ndarray::{prelude::*, parallel::prelude::IntoParallelIterator};
use npyz::{NpyFile, WriterBuilder, npz, Deserialize};
use indicatif::*;
use rayon::prelude::*;

use std::{error::Error, fs::File, io};

pub struct MNISTConfig {
    pub mnist_images_file: &'static str,
    pub mnist_labels_file: &'static str,
}

pub struct VoxCelebConfig {
    pub voxceleb_file: &'static str,
    pub voxceleb_labels_file: &'static str,
}

fn import_npz_into_vector(filename: &str) -> Result<Vec<Array2<i8>>, Box<dyn Error>> {
    let mut file = npz::NpzArchive::open(filename)?;
    let num_samples = file.zip_archive().len();
    println!("Number of files in {} = {}", filename, num_samples);

    // let num_samples = 100;

    println!("Importing (parallel) from {}", filename);
    let empty_element = Array2::from_elem((2,2), 1 as i8);
    let mut output = vec![empty_element; num_samples];
    output.par_iter_mut().enumerate().progress_count(num_samples as u64).for_each(|(i, mut arr)| {
        let mut file = npz::NpzArchive::open(filename).unwrap();
        let array_name = format!("arr_{}", i);
        let reader = file.by_name(array_name.as_str()).unwrap().unwrap();
        let shape = reader.shape().to_vec();
        let order = reader.order();
        let data: Vec<i8> = reader.into_vec::<i8>().unwrap();
        let array = to_array_d(data.clone(), shape.clone(), order);
        let array = array.into_dimensionality::<Ix2>().unwrap();

        *arr = array;
    });
    println!("Importing from {} completed!", filename);

    Ok(output)
}

fn import_npy_into_array<T: Clone+Deserialize>(filename: &str) -> Result<ndarray::ArrayD<T>, Box<dyn Error>> {
    let bytes = std::fs::read(filename)?;
    let reader = npyz::NpyFile::new(&bytes[..])?;
    let shape = reader.shape().to_vec();
    let order = reader.order();
    // let dtype = reader.dtype();
    // println!("{}", dtype.descr());
    let data: Vec<T> = reader.into_vec::<T>()?;

    Ok(to_array_d::<T>(data.clone(), shape.clone(), order))
}

// Example of parsing to an array with fixed NDIM.
fn to_array_3<T>(data: Vec<T>, shape: Vec<u64>, order: npyz::Order) -> ndarray::Array3<T> {
    use ndarray::ShapeBuilder;

    let shape = match shape[..] {
        [i1, i2, i3] => [i1 as usize, i2 as usize, i3 as usize],
        _ => panic!("expected 3D array"),
    };
    let true_shape = shape.set_f(order == npyz::Order::Fortran);

    ndarray::Array3::from_shape_vec(true_shape, data)
        .unwrap_or_else(|e| panic!("shape error: {}", e))
}

// Example of parsing to an array with dynamic NDIM.
fn to_array_d<T>(data: Vec<T>, shape: Vec<u64>, order: npyz::Order) -> ndarray::ArrayD<T> {
    use ndarray::ShapeBuilder;

    let shape = shape.into_iter().map(|x| x as usize).collect::<Vec<_>>();
    let true_shape = shape.set_f(order == npyz::Order::Fortran);

    ndarray::ArrayD::from_shape_vec(true_shape, data)
        .unwrap_or_else(|e| panic!("shape error: {}", e))
}

pub fn import_mnist(mnist_config: &MNISTConfig) -> Result<(ndarray::ArrayD<i8>, ndarray::ArrayD<i8>), Box<dyn Error>> {
    let x: ndarray::ArrayD<i8> = import_npy_into_array(&mnist_config.mnist_images_file)?;
    let y: ndarray::ArrayD<i8> = import_npy_into_array(&mnist_config.mnist_labels_file)?;
    // assert_eq!(dim, &(10000, 28, 28));
    // assert_eq!(img_data.order(), npyz::Order::C);

    // RESHAPE OP
    // let x = x.into_shape(IxDyn(&[10000, 28, 28, 1]))?;

    // DOCS: https://docs.rs/ndarray/latest/ndarray/struct.ArrayBase.html#conversions

    // TEST ARRAY
    // for b in 0..10 {
    //     println!("\n");
    //     for i in 0..28 {
    //         println!("");
    //         for j in 0..28 {
    //             print!("{}", x[[b, i, j]])
    //         }
    //     }
    //     print!("         {}", y[[b]])
    // }
    // Seems like this is working, giving me images and labels

    Ok((x, y))
}

pub fn import_voxceleb(voxceleb_config: &VoxCelebConfig) -> Result<(Vec<Array2<i8>>, ndarray::Array1<i32>), Box<dyn Error>> {
    let x: Vec<Array2<i8>> = import_npz_into_vector(&voxceleb_config.voxceleb_file)?;
    let y: ndarray::ArrayD<i32> = import_npy_into_array::<i32>(&voxceleb_config.voxceleb_labels_file)?;
    let y = y.into_dimensionality::<Ix1>()?;
    // assert_eq!(dim, &(10000, 28, 28));
    // assert_eq!(img_data.order(), npyz::Order::C);

    // NOTE: for voxceleb, 132 <= 1st dim <= 2548, mean=261

    // println!("{:?}", x.dim());
    // println!("{:?}", y.dim());

    // for i in 0..10 {
    //     println!("{:?}", x[[i]].dim());
    // }
    
    // RESHAPE OP
    // let x = x.into_shape(IxDyn(&[10000, 28, 28, 1]))?;

    // DOCS: https://docs.rs/ndarray/latest/ndarray/struct.ArrayBase.html#conversions

    // TEST ARRAY
    // for b in 0..10 {
    //     println!("\n");
    //     for i in 0..28 {
    //         println!("");
    //         for j in 0..28 {
    //             print!("{}", x[b][[i,j]])
    //         }
    //     }
    //     print!("         {}", y[[b]])
    // }
    // Seems like this is working, giving me images and labels

    Ok((x, y))
}
