use criterion::{black_box, criterion_group, criterion_main, Criterion};
use std::collections::BTreeSet;

pub fn triple_scale(ii: usize, jj: usize, kk: usize) -> f64 {
    // atom indices: ii, jj, kk
    let triple = match ii == jj {
        true => {
            match ii == kk {
                true => 1.0/6.0,
                false => 0.5,
            }
        },
        false => {
            match ii != kk && jj != kk {
                true => 1.0,
                false => 0.5,
            }
        }
    };
    triple
}

// pub fn triple_scale(ii: usize, jj: usize, kk: usize) -> f64 {
//     // atom indices: ii, jj, kk
//     // fraction of energy: triple
//     if ii == jj {
//         if ii == kk {
//             // i,i',i'' -> 1/6
//             1.0/6.0
//         } else {
//             // i,i',j -> 1/2
//             0.5
//         }
//     } else if ii != kk && jj != kk {
//         // i,j,k -> 1 (full)
//         1.0
//     } else {
//         // i,j,j' and i,j,i' -> 1/2
//         0.5
//     }
// }

// pub fn triple_scale(ii: usize, jj: usize, kk: usize) -> f64 {
//     // atom indices: ii, jj, kk
//     let set: BTreeSet<usize> = [ii, jj, kk].iter().cloned().collect();
//     let triple = match set.len() {
//         3 => 1.0,
//         2 => 0.5,
//         _ => 1.0/6.0,
//     };
//     triple
// }

// pub fn triple_scale(ii: usize, jj: usize, kk: usize) -> f64 {
//     // atom indices: ii, jj, kk
//     let equals: u8 = ((ii == jj) as u8) + ((ii == kk) as u8) + ((jj == kk) as u8);
//     let triple = match equals {
//         0 => 1.0,
//         1 => 0.5,
//         _ => 1.0/6.0,
//     };
//     triple
// }

fn test_scale(max_range: usize) -> f64 {
    let mut a = 0.0;
    for ii in 0..max_range {
        for jj in 0..max_range {
            for kk in 0..max_range {
                a += triple_scale(ii, jj, kk);
            }
        }
    }

    return a;
}

fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("scale 50", |b| b.iter(|| test_scale(black_box(50))));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);