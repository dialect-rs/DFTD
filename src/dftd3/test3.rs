#![allow(dead_code)]

use nalgebra::Matrix3xX;
use crate::cutoff::{RealspaceCutoff, RealspaceCutoffBuilder, get_lattice_points_cutoff};
use crate::dftd3::model3::D3Model;
use crate::dftd3::ncoord3::get_coordination_number3;
use crate::model::Molecule;

pub fn set_uracil_properties3(mol: &mut Molecule) -> () {
    let cutoff:RealspaceCutoff = RealspaceCutoffBuilder::new().build();
    let lattr: Matrix3xX<f64> = get_lattice_points_cutoff(&mol.periodic, &mol.lattice, cutoff.cn);
    let disp: D3Model = D3Model::from_molecule(&mol, None);

    get_coordination_number3(mol, &lattr, cutoff.cn, disp.rcov.view());
}