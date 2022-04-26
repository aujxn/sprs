use criterion::{
    criterion_group, criterion_main, measurement::WallTime, BenchmarkGroup,
    BenchmarkId, Criterion,
};
use ndarray::Array2;
use rand::Rng;
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256Plus;
use sprs::CsMat;

fn generate_random_array(
    shape: sprs::Shape,
    seed: u64,
) -> ndarray::Array2<f64> {
    // reproducible fast generator for f64s
    let mut rng = Xoshiro256Plus::seed_from_u64(seed);
    let data = (0..shape.0 * shape.1)
        .map(|_| rng.gen())
        .collect::<Vec<f64>>();
    ndarray::Array2::from_shape_vec(shape, data).unwrap()
}

fn generate_random_csr(shape: sprs::Shape, seed: u64) -> sprs::CsMat<f64> {
    let mut rng = Xoshiro256Plus::seed_from_u64(seed);
    let (rows, cols) = shape;
    let nnz = rng.gen_range(1..rows * cols / 2);

    let mut mat = sprs::TriMat::<f64>::new(shape);
    for _ in 0..nnz {
        let r = rng.gen_range(0..rows);
        let c = rng.gen_range(0..cols);
        let v = rng.gen::<f64>();
        mat.add_triplet(r, c, v);
    }

    mat.to_csr()
}

// CSR-dense colmaj mulitplication
fn csr_dense_colmaj_mullacc(c: &mut Criterion) {
    let mut group = c.benchmark_group("csr * dense (column major)");
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(5));

    // Testing on a finite element matrices
    let mat = sprs::io::read_matrix_market::<f64, usize, _>(
        "./data/matrix_market/bench/poisson4009.mtx",
    )
    .unwrap()
    .to_csr::<usize>();

    let shape = mat.shape();
    let rhs = generate_random_array((shape.1, 200), 85);
    let out = ndarray::Array2::<f64>::zeros((shape.0, 200));
    run_benches(&mut group, &mat, &rhs, out, "FEM matrix");

    // Testing on random generated matrices
    const N_LOOP: usize = 10;
    const MAX_LEFT_DIM: usize = 200;
    const MAX_RIGHT_DIM: usize = 200;
    const MAX_INNER_DIM: usize = 5000;

    let mut rng = Xoshiro256Plus::seed_from_u64(150);

    for _ in 0..N_LOOP {
        let left_shape: sprs::Shape = (
            rng.gen_range(10..=MAX_LEFT_DIM),
            rng.gen_range(10..=MAX_INNER_DIM),
        );
        let right_shape: sprs::Shape =
            (left_shape.1, rng.gen_range(1..=MAX_RIGHT_DIM));
        let mat = generate_random_csr(left_shape, 24);
        let rhs = generate_random_array(right_shape, 42);
        let out = ndarray::Array2::<f64>::zeros((left_shape.0, right_shape.1));

        let size = format!(
            "{}x{} times {}x{}",
            left_shape.0, left_shape.1, right_shape.0, right_shape.1
        );

        run_benches(&mut group, &mat, &rhs, out, &size);
    }

    group.finish();
}

fn run_benches(
    group: &mut BenchmarkGroup<WallTime>,
    mat: &CsMat<f64>,
    rhs: &Array2<f64>,
    out: Array2<f64>,
    label: &str,
) {
    group.bench_with_input(
        BenchmarkId::new("heap access", label),
        &(mat, rhs, out.clone()),
        |b, (mat, rhs, out)| {
            b.iter(|| {
                sprs::prod::csr_mulacc_dense_colmaj(
                    mat.view(),
                    rhs.view(),
                    out.clone().view_mut(),
                )
            })
        },
    );

    /*
    group.bench_with_input(
        BenchmarkId::new("parallel", label),
        &(mat, rhs, out.clone()),
        |b, (mat, rhs, out)| {
            b.iter(|| {
                sprs::prod::par_csr_mulacc_dense_colmaj(
                    mat.view(),
                    rhs.view(),
                    out.clone().view_mut(),
                )
            })
        },
    );
    */

    group.bench_with_input(
        BenchmarkId::new("cloned", label),
        &(mat, rhs, out.clone()),
        |b, (mat, rhs, out)| {
            b.iter(|| {
                sprs::prod::csr_mulacc_dense_colmaj_clone(
                    mat.view(),
                    rhs.view(),
                    out.clone().view_mut(),
                )
            })
        },
    );

    group.bench_with_input(
        BenchmarkId::new("copied", label),
        &(mat, rhs, out.clone()),
        |b, (mat, rhs, out)| {
            b.iter(|| {
                sprs::prod::csr_mulacc_dense_colmaj_copy(
                    mat.view(),
                    rhs.view(),
                    out.clone().view_mut(),
                )
            })
        },
    );

    group.bench_with_input(
        BenchmarkId::new("heap access no mul_add", label),
        &(mat, rhs, out.clone()),
        |b, (mat, rhs, out)| {
            b.iter(|| {
                sprs::prod::csr_dense_colmaj(
                    mat.view(),
                    rhs.view(),
                    out.clone().view_mut(),
                )
            })
        },
    );

    group.bench_with_input(
        BenchmarkId::new("cloned no mul_add", label),
        &(mat, rhs, out.clone()),
        |b, (mat, rhs, out)| {
            b.iter(|| {
                sprs::prod::csr_dense_colmaj_clone(
                    mat.view(),
                    rhs.view(),
                    out.clone().view_mut(),
                )
            })
        },
    );

    group.bench_with_input(
        BenchmarkId::new("copied no mul_add", label),
        &(mat, rhs, out.clone()),
        |b, (mat, rhs, out)| {
            b.iter(|| {
                sprs::prod::csr_dense_colmaj_copy(
                    mat.view(),
                    rhs.view(),
                    out.clone().view_mut(),
                )
            })
        },
    );
}

criterion_group!(benches, csr_dense_colmaj_mullacc);
criterion_main!(benches);
