// batched_qs_ode_rk4_f64.wgsl — Full-GPU RK4 parameter sweep for QS/c-di-GMP ODE
//
// Local copy for wetSpring (Write → Absorb → Lean).
// Uses pow_f64() polyfill instead of native pow() for drivers that lack
// native f64 transcendental support. ShaderTemplate::for_driver_auto
// auto-detects capability and injects pow_f64, exp_f64, log_f64 polyfills.

const N_VARS:   u32 = 5u;
const N_PARAMS: u32 = 17u;

struct QsOdeConfig {
    n_batches:      u32,
    n_steps:        u32,
    _pad0:          u32,
    _pad1:          u32,
    h:              f64,
    t0:             f64,
    clamp_max:      f64,
    clamp_min:      f64,
}

@group(0) @binding(0) var<uniform>             config:         QsOdeConfig;
@group(0) @binding(1) var<storage, read>       initial_states: array<f64>;
@group(0) @binding(2) var<storage, read>       batch_params:   array<f64>;
@group(0) @binding(3) var<storage, read_write> output_states:  array<f64>;

fn hill(x: f64, K: f64, n: f64) -> f64 {
    let zero = f64(0.0);
    let xc = max(x, zero);
    let xn = pow_f64(xc, n);
    let Kn = pow_f64(max(K, f64(1e-30)), n);
    return xn / (Kn + xn);
}

fn qs_deriv(y0: f64, y1: f64, y2: f64, y3: f64, y4: f64,
            p_base: u32) -> array<f64, 5> {
    let mu    = batch_params[p_base + 0u];
    let K_cap = batch_params[p_base + 1u];
    let d_n   = batch_params[p_base + 2u];
    let k_ai  = batch_params[p_base + 3u];
    let d_ai  = batch_params[p_base + 4u];
    let k_h   = batch_params[p_base + 5u];
    let K_h   = batch_params[p_base + 6u];
    let n_h   = batch_params[p_base + 7u];
    let d_h   = batch_params[p_base + 8u];
    let k_dgc = batch_params[p_base + 9u];
    let k_rep = batch_params[p_base + 10u];
    let k_pde = batch_params[p_base + 11u];
    let k_act = batch_params[p_base + 12u];
    let k_bio = batch_params[p_base + 13u];
    let K_bio = batch_params[p_base + 14u];
    let n_bio = batch_params[p_base + 15u];
    let d_bio = batch_params[p_base + 16u];

    let one  = f64(1.0);
    let zero = f64(0.0);

    var dy: array<f64, 5>;
    dy[0] = mu * y0 * (one - y0 / max(K_cap, f64(1e-30))) - d_n * y0;
    dy[1] = k_ai * y0 - d_ai * y1;
    dy[2] = k_h * hill(y1, K_h, n_h) - d_h * y2;
    let repress = max(one - k_rep * y2, zero);
    dy[3] = k_dgc * repress - (k_pde + k_act * y2) * y3;
    dy[4] = k_bio * hill(y3, K_bio, n_bio) * (one - y4) - d_bio * y4;
    return dy;
}

fn rk4_step(y0: f64, y1: f64, y2: f64, y3: f64, y4: f64,
            p_base: u32, h: f64) -> array<f64, 5> {
    let h2 = h * f64(0.5);

    let k1 = qs_deriv(y0, y1, y2, y3, y4, p_base);

    let k2 = qs_deriv(
        y0 + h2 * k1[0], y1 + h2 * k1[1], y2 + h2 * k1[2],
        y3 + h2 * k1[3], y4 + h2 * k1[4], p_base);

    let k3 = qs_deriv(
        y0 + h2 * k2[0], y1 + h2 * k2[1], y2 + h2 * k2[2],
        y3 + h2 * k2[3], y4 + h2 * k2[4], p_base);

    let k4 = qs_deriv(
        y0 + h * k3[0], y1 + h * k3[1], y2 + h * k3[2],
        y3 + h * k3[3], y4 + h * k3[4], p_base);

    var y_new: array<f64, 5>;
    let sixth = f64(1.0) / f64(6.0);
    let two   = f64(2.0);
    y_new[0] = y0 + h * sixth * (k1[0] + two * k2[0] + two * k3[0] + k4[0]);
    y_new[1] = y1 + h * sixth * (k1[1] + two * k2[1] + two * k3[1] + k4[1]);
    y_new[2] = y2 + h * sixth * (k1[2] + two * k2[2] + two * k3[2] + k4[2]);
    y_new[3] = y3 + h * sixth * (k1[3] + two * k2[3] + two * k3[3] + k4[3]);
    y_new[4] = y4 + h * sixth * (k1[4] + two * k2[4] + two * k3[4] + k4[4]);
    return y_new;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let b = global_id.x;
    if (b >= config.n_batches) { return; }

    let s_base = b * N_VARS;
    let p_base = b * N_PARAMS;

    var y0 = initial_states[s_base + 0u];
    var y1 = initial_states[s_base + 1u];
    var y2 = initial_states[s_base + 2u];
    var y3 = initial_states[s_base + 3u];
    var y4 = initial_states[s_base + 4u];

    let cmax = config.clamp_max;
    let cmin = config.clamp_min;
    let h    = config.h;
    let one  = f64(1.0);

    for (var step = 0u; step < config.n_steps; step = step + 1u) {
        let yn = rk4_step(y0, y1, y2, y3, y4, p_base, h);

        y0 = clamp(yn[0], cmin, cmax);
        y1 = clamp(yn[1], cmin, cmax);
        y2 = clamp(yn[2], cmin, cmax);
        y3 = clamp(yn[3], cmin, cmax);
        y4 = clamp(yn[4], cmin, one);
    }

    output_states[s_base + 0u] = y0;
    output_states[s_base + 1u] = y1;
    output_states[s_base + 2u] = y2;
    output_states[s_base + 3u] = y3;
    output_states[s_base + 4u] = y4;
}
