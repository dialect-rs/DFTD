use ndarray::{Array1, Array2};

pub struct DispersionResult {
    pub energy: f64,
    pub gradient: Option<Array2<f64>>,
    pub sigma: Option<Array2<f64>>,
}

pub struct PropertyResult {
    pub cn: Array1<f64>,
    pub q: Array1<f64>,
    pub c6: Array2<f64>,
    pub alpha: Array1<f64>,
}