#![allow(dead_code)]

use ndarray::{s, Array, Array1, Array2, ArrayView1, ArrayView2, Axis};
use nalgebra::{Matrix3, RowVector3};

//////////////////////////////////////////////////
// Fortran routines
//////////////////////////////////////////////////

/// Returns tsource if mask is true, elsewise fsource.
/// Both sources should be isize. (MERGE)
pub fn mergei(tsource: isize, fsource: isize, mask: bool) -> isize {
    if mask {
        tsource
    } else {
        fsource
    }
}


/// Replicates a source array copies times along a specified dimension axis. (SPREAD)
pub fn spread(source: &Array1<f64>, axis: u8, copies: usize) -> Array2<f64> {
    let mut a: Array2<f64> = Array::zeros((copies, source.len()));
    for i in 0..copies {
        a.slice_mut(s![i, ..]).assign(source);
    }
    if axis == 0 {
        a.reversed_axes()
    } else {
        a
    }
}

/// Returns the index of minimum value and its value
/// of an 1D-Array. (MINLOC)
pub fn argminf(a: ArrayView1<f64>) -> (Option<usize>, Option<f64>) {
    return if a.dim() == 0 {
        (None, None)
    } else {
        let (min_idx, min_val) = a.iter().enumerate()
            .fold((0, a[0]), |(idx_min, val_min), (idx, val)| {
                if &val_min < val {
                    (idx_min, val_min)
                } else {
                    (idx, *val)
                }
            });

        (Some(min_idx), Some(min_val))
    }
}

/// Returns the minimum value of an 1D-Array.
pub fn minf(a: ArrayView1<f64>) -> f64 {
    let min_val = a.iter()
        .fold(a[0], |val_min, val| {
            if &val_min < val {
                val_min
            } else {
                *val
            }
        });

    min_val
}

/// Returns the minimum value of an 2D-Array along one axis.
pub fn minf_2d(a: ArrayView2<f64>, axis: Axis) -> Array1<f64> {
    let mut min_vals: Array1<f64> = Array::zeros(a.index_axis(axis, 0).raw_dim());
    for i in 0..min_vals.len() {
        let min_val = a.index_axis(axis, i).iter()
            .fold(a.index_axis(axis, i)[0], |val_min, val| {
                if &val_min < val {
                    val_min
                } else {
                    *val
                }
            });
        min_vals[i] = min_val;
    }

    min_vals
}

//////////////////////////////////////////////////
// Linear algebra routines
//////////////////////////////////////////////////

/// Perform cross product of two 3D vectors.
pub fn crossproduct(
    a: &[f64; 3],
    b: &[f64; 3]
) -> [f64; 3] {
    let mut c: [f64; 3] = [0.0; 3];
    c[0] = a[1]*b[2] - b[1]*a[2];
    c[1] = a[2]*b[0] - b[2]*a[0];
    c[2] = a[0]*b[1] - b[0]*a[1];
    c
}

/// Return the dot product of two rust arrays.
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(a, b)| a * b).sum()
}

/// Return the 2 norm of a rust array.
pub fn norm2(a: &[f64]) -> f64 {
    a.iter().map(|a| a.powi(2)).sum::<f64>().sqrt()
}




//////////////////////////////////////////////////
// Rust specific routines
//////////////////////////////////////////////////
/// Convert a lattice as a list of lists into Array2
pub fn lat_to_array(
    lat: &[[f64; 3]; 3],
) -> Matrix3<f64> {
    Matrix3::from_rows(&[
        RowVector3::from_row_slice(&lat[0].to_vec()),
        RowVector3::from_row_slice(&lat[1].to_vec()),
        RowVector3::from_row_slice(&lat[2].to_vec())
    ])
}
