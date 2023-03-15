use crate::auxliary_functions::{crossproduct, dot, norm2, mergei, lat_to_array};
use nalgebra::{Matrix3, Matrix3xX};
use crate::defaults::{CN_CUTOFF_D3_DEFAULT, DISP2_CUTOFF_DEFAULT, DISP3_CUTOFF_DEFAULT};


// Usage: let rsc = RealspaceCutoffBuilder::new().[set_xx].build();
pub struct RealspaceCutoff {
    pub cn: f64,
    pub disp2: f64,
    pub disp3: f64,
}


pub struct RealspaceCutoffBuilder {
    cn: f64,
    disp2: f64,
    disp3: f64,
}

impl RealspaceCutoffBuilder {
    pub fn new() -> RealspaceCutoffBuilder {
        RealspaceCutoffBuilder {
            cn: CN_CUTOFF_D3_DEFAULT,
            disp2: DISP2_CUTOFF_DEFAULT,
            disp3: DISP3_CUTOFF_DEFAULT,
        }
    }

    pub fn set_cn(&mut self, cn: f64) -> &mut Self {
        self.cn = cn;
        self
    }

    pub fn set_disp2(&mut self, disp2: f64) -> &mut Self {
        self.disp2 = disp2;
        self
    }

    pub fn set_disp3(&mut self, disp3: f64) -> &mut Self {
        self.disp3 = disp3;
        self
    }

    pub fn build(&self) -> RealspaceCutoff {
        RealspaceCutoff {
            cn: self.cn,
            disp2: self.disp2,
            disp3: self.disp3,
        }
    }
}

/// Generate lattice points from repetitions
pub fn get_lattice_points_rep_3d(
    lat: &Matrix3<f64>,
    rep: &[usize; 3],
    origin: bool,
) -> Matrix3xX<f64> {
    let mut itr: usize = 0;
    if origin {
        let mut trans: Matrix3xX<f64> = Matrix3xX::zeros(
            rep.iter().map(|x| 2*x + 1).product::<usize>()
        );
        for ix in 0..rep[0] {
            for iy in 0..rep[1] {
                for iz in 0..rep[2] {
                    for jx in (mergei(-1, 1, ix > 0)..1+1).step_by(2).rev() {
                        for jy in (mergei(-1, 1, iy > 0)..1+1).step_by(2).rev() {
                            for jz in (mergei(-1, 1, iz > 0)..1+1).step_by(2).rev() {
                                trans.set_column(itr,
                                        &(lat.row(0) * (ix as f64) * (jx as f64)
                                        + lat.row(1) * (iy as f64) * (jy as f64)
                                        + lat.row(2) * (iz as f64) * (jz as f64)).transpose()
                                );
                                itr += 1;
                            }
                        }
                    }
                }
            }
        }
        trans
    } else {
        let mut trans: Matrix3xX<f64> = Matrix3xX::zeros(
            rep.iter().map(|x| 2*x + 1).product::<usize>() - 1
        );
        for ix in 0..rep[0] {
            for iy in 0..rep[1] {
                for iz in 0..rep[2] {
                    if ix == 0 && iy == 0 && iz == 0 {
                        continue;
                    }
                    for jx in (mergei(-1, 1, ix > 0)..1+1).step_by(2).rev() {
                        for jy in (mergei(-1, 1, iy > 0)..1+1).step_by(2).rev() {
                            for jz in (mergei(-1, 1, iz > 0)..1+1).step_by(2).rev() {
                                trans.set_column(itr,
                                        &(lat.row(0) * (ix as f64) * (jx as f64)
                                        + lat.row(1) * (iy as f64) * (jy as f64)
                                        + lat.row(2) * (iz as f64) * (jz as f64)).transpose()
                                );
                                itr += 1;
                            }
                        }
                    }
                }
            }
        }
        trans
    }
}

/// Create lattice points within a given cutoff
/// returns variable trans
pub fn get_lattice_points_cutoff(
    periodic: &[bool; 3],
    lat: &[[f64; 3]; 3],
    rthr: f64,
) -> Matrix3xX<f64> {
    if !periodic.iter().any(|x| *x) {
        Matrix3xX::zeros(1)
    } else {
        let rep = get_translations(lat, rthr);
        let lat_a: Matrix3<f64> = lat_to_array(lat);
        get_lattice_points_rep_3d(&lat_a, &rep, true)
    }
}

/// Generate a supercell based on a realspace cutoff, this subroutine
/// doesn't know anything about the convergence behaviour of the
/// associated property.
fn get_translations(
    lat: &[[f64; 3]; 3],
    rthr: f64,
) -> [usize; 3] {
    // find normal to the plane...
    let normx: [f64; 3] = crossproduct(&lat[1], &lat[2]);
    let normy: [f64; 3] = crossproduct(&lat[2], &lat[0]);
    let normz: [f64; 3] = crossproduct(&lat[0], &lat[1]);
    // ...normalize it...
    let normx: [f64; 3] = new_3d_vec_from(normx.iter().map(|x| x/norm2(&normx)));
    let normy: [f64; 3] = new_3d_vec_from(normy.iter().map(|x| x/norm2(&normy)));
    let normz: [f64; 3] = new_3d_vec_from(normz.iter().map(|x| x/norm2(&normz)));
    // cos angles between normals and lattice vectors
    let cos10 = dot(&normx, &lat[0]);
    let cos21 = dot(&normy, &lat[1]);
    let cos32 = dot(&normz, &lat[2]);

    let rep: [usize; 3] = [
        (rthr/cos10).abs().ceil() as usize, // rep[0]
        (rthr/cos21).abs().ceil() as usize, // rep[1]
        (rthr/cos32).abs().ceil() as usize, // rep[2]
    ];

    rep
}

fn new_3d_vec_from<F: Iterator<Item=f64>>(src: F) -> [f64; 3] {
    let mut result = [0.0; 3];
    for (rref, val) in result.iter_mut().zip(src) {
        *rref = val;
    }
    result
}
#[cfg(test)]
mod tests {
    use crate::test::get_uracil;
    use crate::cutoff::{RealspaceCutoffBuilder, get_lattice_points_cutoff};
    use nalgebra::{Matrix3xX, Vector3};
    use approx::AbsDiffEq;

    #[test]
    fn get_lattice() -> () {
        let mol = get_uracil();
        let cutoff = RealspaceCutoffBuilder::new().build();
        let lattr: Matrix3xX<f64> = get_lattice_points_cutoff(&mol.periodic, &mol.lattice, cutoff.cn);

        let lattr_ref: Matrix3xX<f64> = Matrix3xX::from_columns(&[
            Vector3::new(0.0000000000000000,
                         0.0000000000000000,
                         0.0000000000000000)
        ]);

        println!("{:?}", lattr);

        assert!(lattr.abs_diff_eq(&lattr_ref, 1e-15));
    }
}