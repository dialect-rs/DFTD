use criterion::{black_box, criterion_group, criterion_main, Criterion};
use nalgebra::{
    dimension::U1,
    storage::{Storage, StorageMut},
    Dim, Matrix, Scalar, SliceStorage, SliceStorageMut, Vector,
    Vector3,
};
use ndarray::{Array, Array1, ArrayView1, ShapeBuilder};
use rand::{thread_rng, Rng};


fn nalgebra_to_ndarray(vec: Vector3<f64>) -> Array1<f64> {
    let vec_array: Array1<f64> = Array::from_iter(vec.iter().map(|x| *x));
    vec_array
}

trait ToNdarray1 {
    type Out;

    fn into_ndarray1(self) -> Self::Out;
}

impl<'a, N: Scalar, R: Dim, RStride: Dim, CStride: Dim> ToNdarray1
for Vector<N, R, SliceStorage<'a, N, R, U1, RStride, CStride>>
{
    type Out = ArrayView1<'a, N>;

    fn into_ndarray1(self) -> Self::Out {
        unsafe {
            ArrayView1::from_shape_ptr(
                (self.shape().0,).strides((self.strides().0,)),
                self.as_ptr(),
            )
        }
    }
}

fn nshare(vec: Vector3<f64>) -> Array1<f64> {
    let arr = vec.rows(0, 3);
    unsafe {
        ArrayView1::from_shape_ptr(
            (arr.shape().0,).strides((arr.strides().0,)),
            arr.as_ptr(),
        ).to_owned()
    }
}

fn criterion_benchmark(c: &mut Criterion) {
    let mut rng = thread_rng();
    let rnd_vec = Vector3::new(rng.gen::<f64>(), rng.gen::<f64>(), rng.gen::<f64>());

    let mut group = c.benchmark_group("convert_to_ndarray");

    group.bench_function("iterator", |b| b.iter(|| nalgebra_to_ndarray(black_box(rnd_vec))));
    group.bench_function("nshare", |b| b.iter(|| nshare(black_box(rnd_vec))));

    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);