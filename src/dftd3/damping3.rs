use std::ops::AddAssign;
use crate::auxliary_functions::spread;
use crate::model::Molecule;
use crate::data::{get_vdw_rad_pair, Element};
use crate::damping::triple_scale;
use crate::consts::EPSILON;
use nalgebra::{Matrix3xX, Vector3};
use soa_derive::soa_zip;
use ndarray::{s, Array, Array1, Array2, ArrayView2, ArrayViewMut1, ArrayViewMut2};


pub struct AtmDispersion3GradientsResult {
    pub energy: Array1<f64>, // dispersion energy input
    pub dedcn: Option<Array1<f64>>, // derivative of the energy w.r.t the coordination number
    pub gradient: Option<Array2<f64>>, // dispersion gradient
    pub sigma: Option<Array2<f64>>, // dispersion virial
}

fn get_atm3_dispersion(
    mol: &Molecule, // molecular structure data
    trans: &Matrix3xX<f64>, // lattice points
    cutoff: f64, // real space cutoff
    s9: f64, // scaling for dispersion coefficients
    rs9: f64, // Scaling for van-der-Waals radii in damping function
    alp: f64, // exponent of zero damping function
    rvdw: ArrayView2<f64>, // Van-der-Waals radii for all element pairs
    c6: ArrayView2<f64>, // C6 coefficients for all atom pairs
    mut energy: ArrayViewMut1<f64>, // dispersion energy input
) -> () {
    if s9.abs() < EPSILON { return; }

    let cutoff2 = cutoff * cutoff;

    // parallelise this loop
    for (iat, ((izp, xyz_i), c6_i)) in soa_zip!(&mol.atomlist, [identifier, xyz]).zip(c6.outer_iter()).enumerate() {
        for (jat, ((jzp, xyz_j), c6_j)) in soa_zip!(&mol.atomlist.slice(0..iat + 1), [identifier, xyz]).zip(c6.outer_iter()).enumerate() {
            let c6ij = c6_i[jat];
            let r0ij = rs9*rvdw[[*izp, *jzp]];
            for trans_jtr in trans.column_iter() {
                let vij: Vector3<f64> = (xyz_j - xyz_i) + trans_jtr;
                let r2ij = vij.norm_squared();
                if r2ij > cutoff2 || r2ij < EPSILON {
                    continue;
                }
                for (kat, ((kzp, xyz_k), (c6ik, c6jk))) in soa_zip!(&mol.atomlist.slice(0..jat + 1), [identifier, xyz]).zip(c6_i.iter().zip(c6_j.iter())).enumerate() {
                    let c9 = -s9 * ((c6ij*c6ik*c6jk).abs()).sqrt();
                    let r0ik = rs9*rvdw[[*izp, *kzp]];
                    let r0jk = rs9*rvdw[[*jzp, *kzp]];
                    let r0 = r0ij * r0ik * r0jk;
                    let triple = triple_scale(iat, jat, kat);
                    for trans_ktr in trans.column_iter() {
                        let vik: Vector3<f64> = (xyz_k - xyz_i) + trans_ktr;
                        let r2ik = vik.norm_squared();
                        if r2ik > cutoff2 || r2ik < EPSILON {
                            continue;
                        }
                        let vjk: Vector3<f64> = (xyz_k - xyz_j) + trans_ktr - trans_jtr;
                        let r2jk = vjk.norm_squared();
                        if r2jk > cutoff2 || r2jk < EPSILON {
                            continue;
                        }
                        let r2 = r2ij*r2ik*r2jk;
                        let r1 = r2.sqrt();
                        let r3 = r2 * r1;
                        let r5 = r3 * r2;

                        let fdmp = 1.0 / (1.0 + 6.0 * (r0 / r1).powf(alp / 3.0));
                        let ang = 0.375 * (r2ij + r2jk - r2ik) * (r2ij - r2jk + r2ik)
                            * (-r2ij + r2jk + r2ik) / r5 + (1.0 / r3);

                        let rr = ang * fdmp;

                        let de = rr * c9 * triple;
                        energy[iat] -= de/3.0;
                        energy[jat] -= de/3.0;
                        energy[kat] -= de/3.0;
                    }
                }
            }
        }
    }
}

fn get_atm3_dispersion_grad(
    mol: &Molecule, // molecular structure data
    trans: &Matrix3xX<f64>, // lattice points
    cutoff: f64, // real space cutoff
    s9: f64, // scaling for dispersion coefficients
    rs9: f64, // Scaling for van-der-Waals radii in damping function
    alp: f64, // exponent of zero damping function
    rvdw: ArrayView2<f64>, // Van-der-Waals radii for all element pairs
    c6: ArrayView2<f64>, // C6 coefficients for all atom pairs
    dc6dcn: ArrayView2<f64>, // derivative of the C6 w.r.t. the coordination number
    mut energy: ArrayViewMut1<f64>, // dispersion energy input
    mut dedcn: ArrayViewMut1<f64>, // derivative of the energy w.r.t the coordination number
    mut gradient: ArrayViewMut2<f64>, // dispersion gradient
    mut sigma: ArrayViewMut2<f64>, // dispersion virial
) -> () {
    if s9.abs() < EPSILON { return; }

    let cutoff2 = cutoff * cutoff;

    // parallelise this loop
    for (iat, ((izp, xyz_i), c6_i)) in soa_zip!(&mol.atomlist, [identifier, xyz]).zip(c6.outer_iter()).enumerate() {
        for (jat, ((jzp, xyz_j), c6_j)) in soa_zip!(&mol.atomlist.slice(0..iat + 1), [identifier, xyz]).zip(c6.outer_iter()).enumerate() {
            let c6ij = c6_i[jat];
            let r0ij = rs9 * rvdw[[*izp, *jzp]];
            for trans_jtr in trans.column_iter() {
                let vij: Vector3<f64> = (xyz_j - xyz_i) + trans_jtr;
                let r2ij = vij.norm_squared();
                if r2ij > cutoff2 || r2ij < EPSILON {
                    continue;
                }
                let vij: Array1<f64> = Array::from_iter(vij.iter().map(|x| *x));
                for (kat, ((kzp, xyz_k), (c6ik, c6jk))) in soa_zip!(&mol.atomlist.slice(0..jat + 1), [identifier, xyz]).zip(c6_i.iter().zip(c6_j.iter())).enumerate() {
                    let c9 = -s9 * ((c6ij*c6ik*c6jk).abs()).sqrt();
                    let r0ik = rs9 * rvdw[[*izp, *kzp]];
                    let r0jk = rs9 * rvdw[[*jzp, *kzp]];
                    let r0 = r0ij * r0ik * r0jk;
                    let triple = triple_scale(iat, jat, kat);
                    for trans_ktr in trans.column_iter() {
                        let vik: Vector3<f64> = (xyz_k - xyz_i) + trans_ktr;
                        let r2ik = vik.norm_squared();
                        if r2ik > cutoff2 || r2ik < EPSILON {
                            continue;
                        }
                        let vik: Array1<f64> = Array::from_iter(vik.iter().map(|x| *x));

                        let vjk: Vector3<f64> = (xyz_k - xyz_j) + trans_ktr - trans_jtr;
                        let r2jk = vjk.norm_squared();
                        if r2jk > cutoff2 || r2jk < EPSILON {
                            continue;
                        }
                        let vjk: Array1<f64> = Array::from_iter(vjk.iter().map(|x| *x));

                        let r2 = r2ij*r2ik*r2jk;
                        let r1 = r2.sqrt();
                        let r3 = r2 * r1;
                        let r5 = r3 * r2;

                        let fdmp = 1.0 / (1.0 + 6.0 * (r0 / r1).powf(alp / 3.0));
                        let ang = 0.375 * (r2ij + r2jk - r2ik) * (r2ij - r2jk + r2ik)
                            * (-r2ij + r2jk + r2ik) / r5 + (1.0 / r3);

                        let rr = ang * fdmp;

                        let dfdmp = -2.0 * alp * (r0 / r1).powf(alp / 3.0) * fdmp.powi(2);

                        // d/drij
                        let dang = -0.375 * (r2ij.powi(3) + r2ij.powi(2) * (r2jk + r2ik)
                            + r2ij * (3.0 * r2jk.powi(2) + 2.0 * r2jk * r2ik + 3.0 * r2ik.powi(2))
                            - 5.0 * (r2jk - r2ik).powi(2) * (r2jk + r2ik)) / r5;
                        let dgij: Array1<f64> = c9 * (-dang*fdmp + ang*dfdmp) / r2ij * &vij;

                        // d/drik
                        let dang = -0.375 * (r2ik.powi(3) + r2ik.powi(2) * (r2jk + r2ij)
                            + r2ik * (3.0 * r2jk.powi(2) + 2.0 * r2jk * r2ij + 3.0 * r2ij.powi(2))
                            - 5.0 * (r2jk - r2ij).powi(2) * (r2jk + r2ij)) / r5;
                        let dgik: Array1<f64> = c9 * (-dang * fdmp + ang * dfdmp) / r2ik * &vik;

                        // d/drjk
                        let dang = -0.375 * (r2jk.powi(3) + r2jk.powi(2) * (r2ik + r2ij)
                            + r2jk * (3.0 * r2ik.powi(2) + 2.0 * r2ik * r2ij + 3.0 * r2ij.powi(2))
                            - 5.0 * (r2ik - r2ij).powi(2) * (r2ik + r2ij)) / r5;
                        let dgjk: Array1<f64> = c9 * (-dang * fdmp + ang * dfdmp) / r2jk * &vjk;

                        let de = rr * c9 * triple;
                        energy[iat] -= de/3.0;
                        energy[jat] -= de/3.0;
                        energy[kat] -= de/3.0;

                        gradient.slice_mut(s![iat, ..]).add_assign(&(-(&dgij + &dgik)));
                        gradient.slice_mut(s![jat, ..]).add_assign(&(&dgij - &dgjk));
                        gradient.slice_mut(s![kat, ..]).add_assign(&(&dgik + &dgjk));

                        let ds: Array2<f64> = spread(&dgij, 1, 3) * spread(&vij, 0, 3)
                            + spread(&dgik, 1, 3) * spread(&vik, 0, 3)
                            + spread(&dgjk, 1, 3) * spread(&vjk, 0, 3);
                        sigma += &(ds * triple);

                        dedcn[iat] -= de * 0.5 * (dc6dcn[[jat, iat]] / c6ij + dc6dcn[[kat, iat]] / c6ik);
                        dedcn[jat] -= de * 0.5 * (dc6dcn[[iat, jat]] / c6ij + dc6dcn[[kat, jat]] / c6jk);
                        dedcn[kat] -= de * 0.5 * (dc6dcn[[iat, kat]] / c6ik + dc6dcn[[jat, kat]] / c6jk);
                    }
                }
            }
        }
    }
}


pub struct D3ParamBuilder {
    pub s6: f64,
    pub s8: f64,
    pub s9: f64,
    pub rs6: f64,
    pub rs8: f64,
    pub a1: f64,
    pub a2: f64,
    pub alp: f64,
    pub bet: f64,
}

impl D3ParamBuilder {
    pub fn new() -> D3ParamBuilder {
        D3ParamBuilder {
            s6: 1.0,
            s8: 1.0,
            s9: 0.0,
            rs6: 1.0,
            rs8: 1.0,
            a1: 0.4,
            a2: 5.0,
            alp: 14.0,
            bet: 0.0,
        }
    }

    pub fn set_a1(&mut self, a1: f64) -> &mut Self {
        self.a1 = a1;
        self
    }

    pub fn set_a2(&mut self, a2: f64) -> &mut Self {
        self.a2 = a2;
        self
    }

    pub fn set_s6(&mut self, s6: f64) -> &mut Self {
        self.s6 = s6;
        self
    }

    pub fn set_s8(&mut self, s8: f64) -> &mut Self {
        self.s8 = s8;
        self
    }

    pub fn set_s9(&mut self, s9: f64) -> &mut Self {
        self.s9 = s9;
        self
    }

    pub fn build(&self) -> D3Param {
        D3Param {
            s6: self.s6,
            s8: self.s8,
            s9: self.s9,
            rs6: self.rs6,
            rs8: self.rs8,
            a1: self.a1,
            a2: self.a2,
            alp: self.alp,
            bet: self.bet,
        }
    }
}

pub struct D3Param {
    pub s6: f64,
    pub s8: f64,
    pub s9: f64,
    pub rs6: f64,
    pub rs8: f64,
    pub a1: f64,
    pub a2: f64,
    pub alp: f64,
    pub bet: f64,
}

pub struct RationalDamping3Param {
    pub s6: f64,
    pub s8: f64,
    pub s9: f64,
    pub a1: f64,
    pub a2: f64,
    pub alp: f64,
    pub r4r2: Array1<f64>,
    pub rvdw: Array2<f64>,
}

impl From<(D3Param, &Vec<u8>)> for RationalDamping3Param {
    fn from((param, num): (D3Param, &Vec<u8>)) -> RationalDamping3Param {
        let mut r4r2: Array1<f64> = Array::zeros(num.len());
        for (num_i, r4r2_i) in num.iter().zip(r4r2.iter_mut()) {
            *r4r2_i = Element::from(*num_i).get_r4r2_val();
        }

        let mut rvdw: Array2<f64> = Array::zeros((num.len(), num.len()));
        for (isp, izp) in num.iter().enumerate() {
            for (jsp, jzp) in num[..=isp].iter().enumerate() {
                rvdw[[isp, jsp]] = get_vdw_rad_pair(*jzp, *izp);
                rvdw[[jsp, isp]] = rvdw[[isp, jsp]];
            }
        }

        RationalDamping3Param{
            s6: param.s6,
            s8: param.s8,
            s9: param.s9,
            a1: param.a1,
            a2: param.a2,
            alp: param.alp,
            r4r2,
            rvdw,
        }
    }
}

impl RationalDamping3Param {
    pub fn get_dispersion2(
        &self,
        mol: &Molecule, // molecular structure data
        trans: &Matrix3xX<f64>, // lattice points
        cutoff: f64, // real space cutoff
        c6: ArrayView2<f64>, // C6 coefficients for all atom pairs
        mut energy: ArrayViewMut1<f64>, // dispersion energy input
    ) -> () {
        if self.s6.abs() < EPSILON && self.s8.abs() < EPSILON { return; }

        let cutoff2 = cutoff * cutoff;

        // parallelise this loop
        for (iat, ((izp, xyz_i), c6_i)) in soa_zip!(&mol.atomlist, [identifier, xyz]).zip(c6.outer_iter()).enumerate() {
            for (jat, ((jzp, xyz_j), c6ij)) in soa_zip!(&mol.atomlist.slice(0..iat + 1), [identifier, xyz]).zip(c6_i.iter()).enumerate() {
                let rrij = 3.0*self.r4r2[*izp]*self.r4r2[*jzp];
                let r0ij = self.a1 * rrij.sqrt() + self.a2;
                for trans_jtr in trans.column_iter() {
                    let vec: Vector3<f64> = (xyz_i - xyz_j) - trans_jtr;
                    let r2 = vec.norm_squared();
                    if r2 > cutoff2 || r2 < EPSILON {
                        continue;
                    }

                    let t6 = 1.0/(r2.powi(3) + r0ij.powi(6));
                    let t8 = 1.0/(r2.powi(4) + r0ij.powi(8));

                    let edisp = self.s6*t6 + self.s8*rrij*t8;

                    let de = -c6ij*edisp*0.5;

                    energy[iat] += de;
                    if iat != jat {
                        energy[jat] += de;
                    }
                }
            }
        }
    }

    pub fn get_dispersion2_grad(
        &self,
        mol: &Molecule, // molecular structure data
        trans: &Matrix3xX<f64>, // lattice points
        cutoff: f64, // real space cutoff
        c6: ArrayView2<f64>, // C6 coefficients for all atom pairs
        dc6dcn: ArrayView2<f64>, // derivative of the C6 w.r.t. the coordination number
        mut energy: ArrayViewMut1<f64>, // dispersion energy input
        mut dedcn: ArrayViewMut1<f64>, // derivative of the energy w.r.t the coordination number
        mut gradient: ArrayViewMut2<f64>, // dispersion gradient
        mut sigma: ArrayViewMut2<f64>, // dispersion virial
    ) -> () {
        if self.s6.abs() < EPSILON && self.s8.abs() < EPSILON { return; }

        let cutoff2 = cutoff * cutoff;

        // parallelise this loop
        for (iat, ((izp, xyz_i), c6_i)) in soa_zip!(&mol.atomlist, [identifier, xyz]).zip(c6.outer_iter()).enumerate() {
            for (jat, ((jzp, xyz_j), c6ij)) in soa_zip!(&mol.atomlist.slice(0..iat + 1), [identifier, xyz]).zip(c6_i.iter()).enumerate() {
                let rrij = 3.0*self.r4r2[*izp]*self.r4r2[*jzp];
                let r0ij = self.a1 * rrij.sqrt() + self.a2;
                for trans_jtr in trans.column_iter() {
                    let vec: Vector3<f64> = (xyz_i - xyz_j) - trans_jtr;
                    let r2 = vec.norm_squared();
                    if r2 > cutoff2 || r2 < EPSILON {
                        continue;
                    }
                    let vec: Array1<f64> = Array::from_iter(vec.iter().map(|x| *x));

                    let t6 = 1.0/(r2.powi(3) + r0ij.powi(6));
                    let t8 = 1.0/(r2.powi(4) + r0ij.powi(8));

                    let d6 = -6.0 * (r2.powi(2)) * (t6.powi(2));
                    let d8 = -8.0 * (r2.powi(3)) * (t8.powi(2));

                    let edisp = self.s6*t6 + self.s8*rrij*t8;
                    let gdisp = self.s6*d6 + self.s8*rrij*d8;

                    let de = -c6ij*edisp*0.5;
                    let dg: Array1<f64> = -c6ij * gdisp * &vec;
                    let ds: Array2<f64> = spread(&dg, 1, 3) * spread(&vec, 0, 3) * 0.5;

                    energy[iat] += de;
                    dedcn[iat] -= dc6dcn[[jat, iat]] * edisp;
                    sigma += &ds;
                    if iat != jat {
                        energy[jat] += de;
                        dedcn[jat] -= dc6dcn[[iat, jat]] * edisp;
                        gradient.slice_mut(s![iat, ..]).add_assign(&dg);
                        gradient.slice_mut(s![jat, ..]).add_assign(&(-dg));
                        sigma += &ds;
                    }
                }
            }
        }
    }

    pub fn get_dispersion3(
        &self,
        mol: &Molecule, // molecular structure data
        trans: &Matrix3xX<f64>, // lattice points
        cutoff: f64, // real space cutoff
        c6: ArrayView2<f64>, // C6 coefficients for all atom pairs
        energy: ArrayViewMut1<f64>, // dispersion energy input
    )-> () {
        get_atm3_dispersion(
            mol,
            trans,
            cutoff,
            self.s9,
            4.0/3.0,
            self.alp + 2.0,
            self.rvdw.view(),
            c6,
            energy,
        )
    }

    pub fn get_dispersion3_grad(
        &self,
        mol: &Molecule, // molecular structure data
        trans: &Matrix3xX<f64>, // lattice points
        cutoff: f64, // real space cutoff
        c6: ArrayView2<f64>, // C6 coefficients for all atom pairs
        dc6dcn: ArrayView2<f64>, // derivative of the C6 w.r.t. the coordination number
        energy: ArrayViewMut1<f64>, // dispersion energy input
        dedcn: ArrayViewMut1<f64>, // derivative of the energy w.r.t the coordination number
        gradient: ArrayViewMut2<f64>, // dispersion gradient
        sigma: ArrayViewMut2<f64>, // dispersion virial
    )-> () {
        get_atm3_dispersion_grad(
            mol,
            trans,
            cutoff,
            self.s9,
            4.0/3.0,
            self.alp + 2.0,
            self.rvdw.view(),
            c6,
            dc6dcn,
            energy,
            dedcn,
            gradient,
            sigma,
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::model::Molecule;
    use crate::test::get_uracil;
    use crate::cutoff::{RealspaceCutoff, RealspaceCutoffBuilder, get_lattice_points_cutoff};
    use crate::dftd3::model3::D3Model;
    use crate::dftd3::damping3::{RationalDamping3Param, D3Param, D3ParamBuilder};
    use ndarray::{array, Array, Array1, Array2};
    use nalgebra::Matrix3xX;
    use crate::dftd3::ncoord3::{get_coordination_number3, get_coordination_number3_grad};

    #[test]
    fn damping3_dispersion2() -> () {
        let mut mol: Molecule = get_uracil();
        let cutoff: RealspaceCutoff = RealspaceCutoffBuilder::new().build();
        let disp: D3Model = D3Model::from_molecule(&mol, None);
        let d3param: D3Param = D3ParamBuilder::new()
            .set_s6(1.0000).set_s8(1.2177).set_s9(1.0000)
            .set_a1(0.4145).set_a2(4.8593).build();
        let param: RationalDamping3Param = RationalDamping3Param::from((d3param, &mol.num));

        let lattr: Matrix3xX<f64> = get_lattice_points_cutoff(&mol.periodic, &mol.lattice, cutoff.cn);
        get_coordination_number3(&mut mol, &lattr,cutoff.cn, disp.rcov.view());
        disp.weight_references(&mut mol);
        let c6: Array2<f64> = disp.get_atomic_c6(&mol);

        let mut energies: Array1<f64> = Array::zeros(mol.n_atoms);

        let lattr: Matrix3xX<f64> = get_lattice_points_cutoff(&mol.periodic, &mol.lattice, cutoff.disp2);
        param.get_dispersion2(&mol, &lattr, cutoff.disp3, c6.view(), energies.view_mut());

        let energies_ref = array![
            -0.0017284064768964, -0.0016866387030240, -0.0016772599133637, -0.0014316016628353,
            -0.0017327782595624, -0.0014830308612946, -0.0009520690613547, -0.0009504236920883,
            -0.0005607566234968, -0.0005922548877982, -0.0005416467901288, -0.0005264947139213,
            -0.0005424322299178, -0.0016822935601525, -0.0009494409508394, -0.0017289271380881,
            -0.0016614665114619, -0.0005205601916777, -0.0014954540840850, -0.0014189107720442,
            -0.0017455004291204, -0.0006007421525099, -0.0005514879852627, -0.0009661274909330,
        ];

        println!("{:?}", energies);

        assert!(energies.abs_diff_eq(&energies_ref, 1e-15));
    }

    #[test]
    fn damping3_dispersion2_derivative() -> () {
        let mut mol: Molecule = get_uracil();
        let cutoff: RealspaceCutoff = RealspaceCutoffBuilder::new().build();
        let disp: D3Model = D3Model::from_molecule(&mol, None);
        let d3param: D3Param = D3ParamBuilder::new()
            .set_s6(1.0000).set_s8(1.2177).set_s9(1.0000)
            .set_a1(0.4145).set_a2(4.8593).build();
        let param: RationalDamping3Param = RationalDamping3Param::from((d3param, &mol.num));

        let lattr: Matrix3xX<f64> = get_lattice_points_cutoff(&mol.periodic, &mol.lattice, cutoff.cn);
        let (_dcndr, _dcndl) = get_coordination_number3_grad(&mut mol, &lattr,cutoff.cn, disp.rcov.view());
        disp.weight_references_grad(&mut mol);
        let (c6, dc6dcn): (Array2<f64>, Array2<f64>) = disp.get_atomic_c6_grad(&mol);

        let mut energies: Array1<f64> = Array::zeros(mol.n_atoms);
        let mut dedcn: Array1<f64> = Array::zeros(mol.n_atoms);
        let mut gradient: Array2<f64> = Array::zeros((mol.n_atoms, 3));
        let mut sigma: Array2<f64> = Array::zeros((3, 3));

        let lattr: Matrix3xX<f64> = get_lattice_points_cutoff(&mol.periodic, &mol.lattice, cutoff.disp2);
        param.get_dispersion2_grad(&mol, &lattr, cutoff.disp3, c6.view(), dc6dcn.view(),
                                   energies.view_mut(), dedcn.view_mut(), gradient.view_mut(), sigma.view_mut());

        let energies_ref: Array1<f64> = array![
            -0.0017284064768964, -0.0016866387030240, -0.0016772599133637, -0.0014316016628353,
            -0.0017327782595624, -0.0014830308612946, -0.0009520690613547, -0.0009504236920884,
            -0.0005607566234968, -0.0005922548877982, -0.0005416467901288, -0.0005264947139213,
            -0.0005424322299178, -0.0016822935601525, -0.0009494409508394, -0.0017289271380881,
            -0.0016614665114619, -0.0005205601916777, -0.0014954540840850, -0.0014189107720442,
            -0.0017455004291204, -0.0006007421525099, -0.0005514879852627, -0.0009661274909330,
        ];

        let dedcn_ref: Array1<f64> = array![
            0.0002824056586004,  0.0002749376023473,  0.0002714338754724,  0.0000206749160026,
            0.0003083659714629,  0.0000168871035333,  0.0000762054901389,  0.0000721001231234,
            0.0000772243180363,  0.0000813052033905,  0.0000769889341374,  0.0000753714497199,
            0.0000768274733023,  0.0002736530126198,  0.0000756965972614,  0.0002816113351074,
            0.0002692499872174,  0.0000746897368659,  0.0000169860279323,  0.0000206122914426,
            0.0003114258591835,  0.0000823723045720,  0.0000763525973492,  0.0000734429040826,
        ];

        let gradient_ref_array: [[f64; 24]; 3] = [
            [ -0.0004095606330941, -0.0004887784717562, -0.0003022664551288, -0.0000090953304692,
                0.0000932422851044, -0.0000905523525509, -0.0003298574647824,  0.0002453645102031,
                0.0000596816693206,  0.0000064267614169, -0.0002294728016681, -0.0001151672527915,
                -0.0000742253886886, -0.0000131602372064,  0.0001915307423702,  0.0002640731765892,
                -0.0001046134165684, -0.0001270830932837,  0.0003924171943004,  0.0000790864030875,
                0.0003844471715300,  0.0002438361534317, -0.0000028076839004,  0.0003365345145349,],
            [  0.0003742874147197,  0.0000937364151767, -0.0000757848442471,  0.0000398498330309,
                0.0003285001253214,  0.0004129795992204,  0.0002979991298414,  0.0002518809040764,
                -0.0000430361772206,  0.0002486471672127, -0.0000198373201407, -0.0001211084889138,
                0.0000220495727566, -0.0000869301418558,  0.0002126745827432,  0.0000511558294883,
                -0.0003704142473060, -0.0001475164949399, -0.0000911371776748, -0.0004311340285862,
                -0.0004079076672984,  0.0000156635134424, -0.0002416409330975, -0.0003129765657493,],
            [ -0.0001870640280332, -0.0002984527078552, -0.0004981532047619, -0.0004763575846635,
                -0.0004706136011639, -0.0002274911685981,  0.0000340356961252, -0.0002893644991771,
                -0.0002320482841697, -0.0000388825757782, -0.0000560254364519, -0.0001769164215257,
                0.0002222082023865,  0.0005631413758007,  0.0003435692434019,  0.0005209357669401,
                0.0004320745470167,  0.0001393157833741,  0.0002682358787858,  0.0001982920448213,
                0.0001816448588098,  0.0000686800147099,  0.0000233081957061, -0.0000440720956997,],
        ];

        let mut gradient_ref: Array2<f64> = Array::zeros((3, 24));
        for i in 0..3 {
            for j in 0..24 {
                gradient_ref[[i, j]] = gradient_ref_array[i][j];
            }
        }
        let gradient_ref: Array2<f64> = gradient_ref.reversed_axes();

        let sigma_ref: Array2<f64> = array![
            [  0.0142959848028122, -0.0027316671873628,  0.0037423475257577,],
            [ -0.0027316671873628,  0.0147765107118366, -0.0044468326689623,],
            [  0.0037423475257577, -0.0044468326689623,  0.0199143869142232,],
        ];

        println!("{:?}", gradient);

        assert!(energies.abs_diff_eq(&energies_ref, 1e-15));
        assert!(dedcn.abs_diff_eq(&dedcn_ref, 1e-15));
        assert!(gradient.abs_diff_eq(&gradient_ref, 1e-15));
        assert!(sigma.abs_diff_eq(&sigma_ref, 1e-15));
    }

    #[test]
    fn damping3_dispersion3() -> () {
        let mut mol: Molecule = get_uracil();
        let cutoff: RealspaceCutoff = RealspaceCutoffBuilder::new().build();
        let disp: D3Model = D3Model::from_molecule(&mol, None);
        let d3param: D3Param = D3ParamBuilder::new()
            .set_s6(1.0000).set_s8(1.2177).set_s9(1.0000)
            .set_a1(0.4145).set_a2(4.8593).build();
        let param: RationalDamping3Param = RationalDamping3Param::from((d3param, &mol.num));

        let lattr: Matrix3xX<f64> = get_lattice_points_cutoff(&mol.periodic, &mol.lattice, cutoff.cn);
        get_coordination_number3(&mut mol, &lattr,cutoff.cn, disp.rcov.view());
        disp.weight_references(&mut mol);
        let c6: Array2<f64> = disp.get_atomic_c6(&mol);

        let mut energies: Array1<f64> = Array::zeros(mol.n_atoms);

        let lattr: Matrix3xX<f64> = get_lattice_points_cutoff(&mol.periodic, &mol.lattice, cutoff.disp3);
        param.get_dispersion3(&mol, &lattr, cutoff.disp3, c6.view(), energies.view_mut());

        let energies_ref = array![
            0.0000165903171369,  0.0000174078249408,  0.0000171535714757,  0.0000172212714627,
            0.0000157308551228,  0.0000171465585488,  0.0000193553366830,  0.0000190586140515,
            0.0000122533038534,  0.0000136237299001,  0.0000121549837288,  0.0000121624067898,
            0.0000119022331316,  0.0000169463600022,  0.0000191132024779,  0.0000162400681563,
            0.0000173066236948,  0.0000117751812161,  0.0000170516748203,  0.0000176932584303,
            0.0000159318195972,  0.0000138949081558,  0.0000121820297421,  0.0000197361630639,
        ];

        println!("{:?}", energies);

        assert!(energies.abs_diff_eq(&energies_ref, 1e-15));
    }

    #[test]
    fn damping3_dispersion3_derivative() -> () {
        let mut mol: Molecule = get_uracil();
        let cutoff: RealspaceCutoff = RealspaceCutoffBuilder::new().build();
        let disp: D3Model = D3Model::from_molecule(&mol, None);
        let d3param: D3Param = D3ParamBuilder::new()
            .set_s6(1.0000).set_s8(1.2177).set_s9(1.0000)
            .set_a1(0.4145).set_a2(4.8593).build();
        let param: RationalDamping3Param = RationalDamping3Param::from((d3param, &mol.num));

        let lattr: Matrix3xX<f64> = get_lattice_points_cutoff(&mol.periodic, &mol.lattice, cutoff.cn);
        let (_dcndr, _dcndl) = get_coordination_number3_grad(&mut mol, &lattr,cutoff.cn, disp.rcov.view());
        disp.weight_references_grad(&mut mol);
        let (c6, dc6dcn): (Array2<f64>, Array2<f64>) = disp.get_atomic_c6_grad(&mol);

        let mut energies: Array1<f64> = Array::zeros(mol.n_atoms);
        let mut dedcn: Array1<f64> = Array::zeros(mol.n_atoms);
        let mut gradient: Array2<f64> = Array::zeros((mol.n_atoms, 3));
        let mut sigma: Array2<f64> = Array::zeros((3, 3));

        let lattr: Matrix3xX<f64> = get_lattice_points_cutoff(&mol.periodic, &mol.lattice, cutoff.disp3);
        param.get_dispersion3_grad(&mol, &lattr, cutoff.disp3, c6.view(), dc6dcn.view(),
                                   energies.view_mut(), dedcn.view_mut(), gradient.view_mut(), sigma.view_mut());

        let energies_ref: Array1<f64> = array![
            0.0000165903171369,  0.0000174078249408,  0.0000171535714757,  0.0000172212714627,
            0.0000157308551228,  0.0000171465585488,  0.0000193553366830,  0.0000190586140515,
            0.0000122533038534,  0.0000136237299001,  0.0000121549837288,  0.0000121624067898,
            0.0000119022331316,  0.0000169463600022,  0.0000191132024779,  0.0000162400681563,
            0.0000173066236948,  0.0000117751812161,  0.0000170516748203,  0.0000176932584303,
            0.0000159318195972,  0.0000138949081558,  0.0000121820297421,  0.0000197361630639,
        ];

        let dedcn_ref: Array1<f64> = array![
            -0.0000040544280048, -0.0000042079956604, -0.0000041165382906, -0.0000003717883032,
            -0.0000042106185601, -0.0000002929449841, -0.0000023122262811, -0.0000021658297045,
            -0.0000025339553531, -0.0000028107082343, -0.0000025790818218, -0.0000025898432016,
            -0.0000025150619186, -0.0000040995045980, -0.0000022709004030, -0.0000039649983134,
            -0.0000041473513162, -0.0000025094977252, -0.0000002910818353, -0.0000003827274814,
            -0.0000042713263029, -0.0000028646851216, -0.0000025323195930, -0.0000022565590900,
        ];

        let gradient_ref_array: [[f64; 24]; 3] = [
            [ -0.0000051018741773, -0.0000055385632618, -0.0000038138597850, -0.0000009383342109,
                0.0000007137993137, -0.0000016245067480,  0.0000091534709900, -0.0000050580491404,
                -0.0000016451644071,  0.0000004295012996,  0.0000092942278744,  0.0000048264506835,
                0.0000020147678909, -0.0000006342637840, -0.0000054439501355,  0.0000035844004952,
                -0.0000014887413062,  0.0000041161258147,  0.0000068925053135,  0.0000001577678180,
                0.0000052563005859, -0.0000059110727106, -0.0000007593171176, -0.0000084816212953,],
            [  0.0000044917267778,  0.0000002992431307, -0.0000010915689812,  0.0000002172946088,
                0.0000047279425926,  0.0000067482826112, -0.0000092840497164, -0.0000065228849601,
                0.0000007179600664, -0.0000066357250754, -0.0000001925226516,  0.0000037755084423,
                -0.0000002160346304, -0.0000012876698481, -0.0000046296001811,  0.0000003819948332,
                -0.0000034312158401,  0.0000068125396767, -0.0000015465858120, -0.0000047697791371,
                -0.0000052467491948,  0.0000001638975930,  0.0000090307346477,  0.0000074872610478,],
            [ -0.0000022381548003, -0.0000032436222444, -0.0000049585695006, -0.0000059253542614,
                -0.0000061777636190, -0.0000031924877098,  0.0000001225450022,  0.0000083632555113,
                0.0000085001141637,  0.0000018263481388,  0.0000032885128744,  0.0000080169648097,
                -0.0000095162231944,  0.0000057929010152, -0.0000099847864086,  0.0000068491720981,
                0.0000039457329790, -0.0000071183401659,  0.0000044562923515,  0.0000025131174807,
                0.0000026048143416, -0.0000021164809157, -0.0000020720266745,  0.0000002640387283,],
        ];

        let mut gradient_ref: Array2<f64> = Array::zeros((3, 24));
        for i in 0..3 {
            for j in 0..24 {
                gradient_ref[[i, j]] = gradient_ref_array[i][j];
            }
        }
        let gradient_ref: Array2<f64> = gradient_ref.reversed_axes();

        let sigma_ref: Array2<f64> = array![
            [ -0.0001241751656843, -0.0000154562411687,  0.0000522705030870,],
            [ -0.0000154562411687, -0.0001270627573189, -0.0000468573040282,],
            [  0.0000522705030870, -0.0000468573040282, -0.0000956125996532,],
        ];

        println!("{:?}", gradient);

        assert!(energies.abs_diff_eq(&energies_ref, 1e-15));
        assert!(dedcn.abs_diff_eq(&dedcn_ref, 1e-15));
        assert!(gradient.abs_diff_eq(&gradient_ref, 1e-15));
        assert!(sigma.abs_diff_eq(&sigma_ref, 1e-15));
    }
}