use crate::consts::BOHR_TO_ANGS;
use chemfiles::{Frame, Trajectory};
use nalgebra::Point3;
use ndarray::{Array, Array1};
use soa_derive::StructOfArray;

#[derive(StructOfArray)]
#[soa_derive = "Clone"]
pub struct Atom {
    // Atomic number starting from 1
    pub atomic_number: u8,

    // Coordinate of the atom in bohr
    pub xyz: Point3<f64>,

    // The molecular attribute num contains a list of unique elements, e.g. [6, 1].
    // The identifier maps the atom to elements in that list. The list of all identifiers
    // for ethene could look like [0, 1, 1, 0, 1, 1] (C, H, H, C, H, H).
    pub identifier: usize,

    // Fractional coordination number for computing C6 coefficients.
    pub coord_number: f64,

    // Gaussian weighting factor for computing C6 coefficients.
    pub gaussian_weight: Array1<f64>,

    // Derivative of Gaussian weighting factor w.r.t coordination number.
    pub gaussian_weight_dcn: Array1<f64>,
}

impl From<(u8, Point3<f64>)> for Atom {
    fn from((atomic_number, xyz): (u8, Point3<f64>)) -> Self {
        Atom {
            atomic_number,
            xyz,
            identifier: 0,
            coord_number: 0.0,
            gaussian_weight: Array::zeros(0),
            gaussian_weight_dcn: Array::zeros(0),
        }
    }
}

#[derive(Clone)]
pub struct Molecule {
    pub atomlist: AtomVec,
    pub charge: i8,
    pub periodic: [bool; 3],
    pub lattice: [[f64; 3]; 3],
    pub n_atoms: usize,
    pub nid: usize,   // number of unique atoms
    pub num: Vec<u8>, // list of unique atomic numbers
}

pub trait DispersionInterface {
    fn get_positions(&self) -> Vec<Point3<f64>>;

    fn get_atomic_numbers(&self) -> Vec<u8>;
}

impl Molecule {
    pub fn new(interface: impl DispersionInterface) -> Self {
        let positions: Vec<Point3<f64>> = interface.get_positions();

        let atomic_numbers: Vec<u8> = interface.get_atomic_numbers();

        Molecule::from((&positions, &atomic_numbers))
    }
}

impl From<Frame> for Molecule {
    fn from(frame: Frame) -> Self {
        // Read atomic numbers and coordinates from frame.
        let positions: Vec<Point3<f64>> = frame
            .positions()
            .iter()
            .map(|xyz| {
                Point3::new(
                    xyz[0] / BOHR_TO_ANGS,
                    xyz[1] / BOHR_TO_ANGS,
                    xyz[2] / BOHR_TO_ANGS,
                )
            })
            .collect();

        // read the atomic number of each coordinate
        let atomic_numbers: Vec<u8> = (0..frame.size())
            .map(|i| frame.atom(i as usize).atomic_number() as u8)
            .collect();

        Molecule::from((&positions, &atomic_numbers))
    }
}

impl From<(&Vec<Point3<f64>>, &Vec<u8>)> for Molecule {
    fn from(pos_an: (&Vec<Point3<f64>>, &Vec<u8>)) -> Self {
        let (positions, atomic_numbers) = pos_an;

        // Obtain number of atoms.
        let n_atoms: usize = atomic_numbers.len();

        // Generate the AtomVec
        let mut atomlist = AtomVec::with_capacity(n_atoms);
        for (an, xyz) in atomic_numbers.iter().zip(positions) {
            // transform the coordinates from angstrom to bohr
            atomlist.push(Atom::from((*an, xyz.to_owned())));
        }

        // Obtain nid and id
        let mut nid: usize = 0;
        let mut num: Vec<u8> = Vec::with_capacity(n_atoms);

        for (an, id) in atomic_numbers.iter().zip(&mut atomlist.identifier) {
            if !num.iter().any(|&i| i == *an) {
                nid += 1;
                num.push(*an);
            }
            *id = num.iter().position(|&x| x == *an).unwrap();
        }

        Molecule {
            atomlist,
            charge: 0,
            periodic: [false; 3],
            lattice: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            n_atoms,
            nid,
            num,
        }
    }
}

impl Molecule {
    pub fn set_charge(&mut self, charge: i8) -> () {
        self.charge = charge;
    }

    pub fn set_periodic(&mut self, periodic: [bool; 3]) -> () {
        self.periodic = periodic;
    }

    pub fn set_lattice(&mut self, lattice: [[f64; 3]; 3]) -> () {
        self.lattice = lattice;
    }
}

pub fn get_molecule_frame(filename: &str) -> Frame {
    // read the geometry file
    let mut trajectory = Trajectory::open(filename, 'r').unwrap();
    let mut frame = Frame::new();

    // if multiple geometries are contained in the file, we will only use the first one
    trajectory.read(&mut frame).unwrap();

    frame
}

#[cfg(test)]
mod tests {
    use crate::model::Molecule;
    use crate::test::get_uracil;
    use ndarray::{array, Array, Array1};

    #[test]
    fn read_molecule() -> () {
        let mol: Molecule = get_uracil();

        let natoms = mol.n_atoms;
        let nid = mol.nid;
        let id = Array::from(mol.atomlist.identifier);
        let num: Array1<u8> = Array::from(mol.num);
        let periodic = mol.periodic;
        let lattice = mol.lattice;

        let natoms_ref: usize = 24;
        let nid_ref: usize = 4;
        let id_ref: Array1<usize> =
            array![0, 0, 0, 1, 0, 1, 2, 2, 3, 3, 3, 3, 3, 0, 2, 0, 0, 3, 1, 1, 0, 3, 3, 2];
        let num_ref: Array1<u8> = array![6, 7, 8, 1];

        assert_eq!(natoms, natoms_ref);
        assert_eq!(nid, nid_ref);
        assert_eq!(id, id_ref);
        assert_eq!(num, num_ref);

        println!("{:?}", periodic);
        println!("{:?}", lattice);
        println!("{:?}", mol.atomlist.xyz);
    }
}
