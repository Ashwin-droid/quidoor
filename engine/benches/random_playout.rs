use criterion::{Criterion, criterion_group, criterion_main};
use quoridor_engine::state::GameState;
use std::hint::black_box;

fn next_u64(seed: &mut u64) -> u64 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *seed
}

fn random_playout_ops(seed: &mut u64) -> u64 {
    let mut ops = 0u64;
    for _ in 0..64 {
        let mut state = GameState::new();
        for _ in 0..200 {
            if state.winner().is_some() {
                break;
            }
            let actions = state.legal_actions();
            if actions.is_empty() {
                break;
            }
            let idx = (next_u64(seed) as usize) % actions.len();
            state
                .apply_action(actions[idx])
                .expect("selected legal action");
            ops += 1;
        }
    }
    ops
}

fn random_playout_benchmark(c: &mut Criterion) {
    c.bench_function("random_playout_legal_gen_apply", |b| {
        let mut seed = 0x5eed_1234_9876_abcd;
        b.iter(|| black_box(random_playout_ops(&mut seed)));
    });
}

criterion_group!(benches, random_playout_benchmark);
criterion_main!(benches);
