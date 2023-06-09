#![allow(dead_code)]

use chemfiles::{Frame, Atom};
use crate::model::Molecule;
use crate::data::Element;

pub fn get_uracil() -> Molecule {
    let atomic_numbers: Vec<u8> = vec![6, 6, 6, 7, 6, 7, 8, 8, 1, 1, 1, 1,
                                       1, 6, 8, 6, 6, 1, 7, 7, 6, 1, 1, 8];
    let positions: [[f64; 3]; 24] = [
        [-0.90911,    2.07844,   -1.26358],
        [-1.34926,    0.86684,   -1.79758],
        [-0.51039,    0.17171,   -2.66740],
        [ 0.71266,    0.67198,   -2.98479],
        [ 1.14119,    1.86431,   -2.45693],
        [ 0.33001,    2.56345,   -1.59862],
        [-1.62412,    2.70297,   -0.49951],
        [ 2.24150,    2.30048,   -2.74842],
        [ 1.32909,    0.13684,   -3.63745],
        [ 0.65904,    3.47098,   -1.19597],
        [-2.32350,    0.46720,   -1.54376],
        [-0.83573,   -0.77059,   -3.08958],
        [-0.29367,    0.86590,    2.21647],
        [ 0.38281,    0.38511,    1.52043],
        [ 1.90650,    2.05645,    1.65060],
        [ 1.59553,    0.98018,    1.17101],
        [ 0.05054,   -0.84513,    0.95500],
        [-0.88728,   -1.31994,    1.21400],
        [ 2.43524,    0.35241,    0.28579],
        [ 0.89807,   -1.44767,    0.08007],
        [ 2.09020,   -0.85860,   -0.26005],
        [ 3.34328,    0.80061,    0.02387],
        [ 0.63225,   -2.36913,   -0.33536],
        [ 2.84038,   -1.40928,   -1.04736]
    ];

    let mut frame = Frame::new();
    for (an, xyz) in atomic_numbers.iter().zip(positions.iter()) {
        let atom: Atom = Atom::new(Element::from(*an).symbol());
        frame.add_atom(&atom, *xyz, None);
    }

    let mol: Molecule = Molecule::from(frame);
    mol
}