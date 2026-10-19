// NVRTC-compiled f64 compressible-DG kernels.
//
// Device counterpart of the CPU element-block operator in sonium-fem
// (`physics::fluid::compressible_flow::dg`): one thread block owns one
// element, and every kernel replays the CPU accumulation order for that
// element (quadrature points ascending, local nodes ascending, facet
// records in the host-built CSR order), so device and CPU results agree
// to round-off (bit-exactly in most lanes; the host compiles with
// `--fmad=false`). State stays resident: the host uploads the problem
// tables and the initial state once, then only scalars (time step,
// retry flags, step-report reductions) cross the bus per step.
//
// Host contract: all index ranges, CSR monotonicity, and table lengths
// are validated when the problem is staged; kernels trust them.

struct DgParams {
    int dim;
    int n;
    int flux_kind;      // 0 = Rusanov, 1 = HLLC with Rusanov fallback
    int visc_kind;      // 0 = constant, 1 = Sutherland
    int sensor_mode;    // 0 = Persson-Peraire, 1 = jump/deviation, 2 = always
    int persson_highest;
    double gamma;
    double gas_constant;
    double prandtl;
    double visc_a;      // constant mu, or Sutherland reference viscosity
    double visc_b;      // Sutherland reference temperature
    double visc_c;      // Sutherland temperature
    double density_floor;
    double pressure_floor;
    double sipg_penalty;
    double cfl_scaled;  // cfl * 3 / (2p + 1), folded on the host
    double relaxation;
    double threshold;
    double persson_threshold;
};

#define DG_MAX_N 10
#define DG_INF __longlong_as_double(0x7ff0000000000000LL)

__device__ __forceinline__ double dg_pressure(const double* s, double gamma) {
    double m2 = s[1] * s[1] + s[2] * s[2] + s[3] * s[3];
    return (gamma - 1.0) * (s[4] - 0.5 * m2 / s[0]);
}

__device__ __forceinline__ bool dg_is_physical(const double* s, const DgParams* p) {
    if (!isfinite(s[0]) || !isfinite(s[4])) return false;
    if (!isfinite(s[1]) || !isfinite(s[2]) || !isfinite(s[3])) return false;
    if (s[0] < p->density_floor) return false;
    double pressure = dg_pressure(s, p->gamma);
    return isfinite(pressure) && pressure >= p->pressure_floor;
}

__device__ __forceinline__ double dg_sound(double pressure, double density, double gamma) {
    return sqrt(gamma * pressure / density);
}

__device__ __forceinline__ double dg_temperature(double pressure, double density, const DgParams* p) {
    return pressure / (density * p->gas_constant);
}

__device__ __forceinline__ double dg_viscosity(double temperature, const DgParams* p) {
    if (p->visc_kind == 0) return p->visc_a;
    double t = temperature > 2.2250738585072014e-308 ? temperature : 2.2250738585072014e-308;
    return p->visc_a * pow(t / p->visc_b, 1.5) * (p->visc_b + p->visc_c) / (t + p->visc_c);
}

__device__ __forceinline__ double dg_conductivity(double temperature, const DgParams* p) {
    double cp = p->gamma * p->gas_constant / (p->gamma - 1.0);
    return dg_viscosity(temperature, p) * cp / p->prandtl;
}

__device__ __forceinline__ void dg_euler_flux(const double* s, const DgParams* p,
                                              const double* normal, double* flux) {
    double u0 = s[1] / s[0], u1 = s[2] / s[0], u2 = s[3] / s[0];
    double un = u0 * normal[0] + u1 * normal[1] + u2 * normal[2];
    double pressure = dg_pressure(s, p->gamma);
    flux[0] = s[0] * un;
    flux[1] = s[1] * un + pressure * normal[0];
    flux[2] = s[2] * un + pressure * normal[1];
    flux[3] = s[3] * un + pressure * normal[2];
    flux[4] = (s[4] + pressure) * un;
}

__device__ __forceinline__ double dg_max_wave_speed(const double* s, const DgParams* p,
                                                    const double* normal) {
    double u0 = s[1] / s[0], u1 = s[2] / s[0], u2 = s[3] / s[0];
    double un = u0 * normal[0] + u1 * normal[1] + u2 * normal[2];
    double c = dg_sound(dg_pressure(s, p->gamma), s[0], p->gamma);
    return fabs(un) + c;
}

// Primitive gradients from the centroid tabulation: velocity gradient
// [component][direction] and temperature gradient [direction].
__device__ __forceinline__ void dg_primitive_gradients(
    const double* cell, const double* cent_grad, const DgParams* p,
    double* grad_velocity, double* grad_temperature) {
    for (int i = 0; i < 9; ++i) grad_velocity[i] = 0.0;
    for (int i = 0; i < 3; ++i) grad_temperature[i] = 0.0;
    int n = p->n;
    for (int local = 0; local < n; ++local) {
        const double* s = cell + local * 5;
        double vel[3] = {s[1] / s[0], s[2] / s[0], s[3] / s[0]};
        double temperature = dg_temperature(dg_pressure(s, p->gamma), s[0], p);
        for (int dir = 0; dir < p->dim; ++dir) {
            double g = cent_grad[local * 3 + dir];
            for (int comp = 0; comp < 3; ++comp)
                grad_velocity[comp * 3 + dir] += vel[comp] * g;
            grad_temperature[dir] += temperature * g;
        }
    }
}

// Viscous flux tensor: out[direction][equation], direction-major 3x5.
__device__ __forceinline__ void dg_viscous_tensor(
    const double* s, const double* grad_velocity, const double* grad_temperature,
    const DgParams* p, double* out) {
    double pressure = dg_pressure(s, p->gamma);
    double temperature = dg_temperature(pressure, s[0], p);
    double mu = dg_viscosity(temperature, p);
    double conductivity = dg_conductivity(temperature, p);
    double vel[3] = {s[1] / s[0], s[2] / s[0], s[3] / s[0]};
    double divergence = grad_velocity[0] + grad_velocity[4] + grad_velocity[8];
    double stress[9];
    for (int comp = 0; comp < 3; ++comp)
        for (int dir = 0; dir < 3; ++dir)
            stress[comp * 3 + dir] = mu * (grad_velocity[comp * 3 + dir]
                + grad_velocity[dir * 3 + comp]
                - (comp == dir ? 2.0 * divergence / 3.0 : 0.0));
    for (int dir = 0; dir < 3; ++dir) {
        out[dir * 5 + 0] = 0.0;
        out[dir * 5 + 1] = stress[0 * 3 + dir];
        out[dir * 5 + 2] = stress[1 * 3 + dir];
        out[dir * 5 + 3] = stress[2 * 3 + dir];
        double work = 0.0;
        for (int comp = 0; comp < 3; ++comp) work += vel[comp] * stress[comp * 3 + dir];
        out[dir * 5 + 4] = work + conductivity * grad_temperature[dir];
    }
}

__device__ __forceinline__ void dg_rusanov(const double* left, const double* right,
                                           const DgParams* p, const double* normal,
                                           double* flux) {
    double fl[5], fr[5];
    dg_euler_flux(left, p, normal, fl);
    dg_euler_flux(right, p, normal, fr);
    double smax = dg_max_wave_speed(left, p, normal);
    double sr = dg_max_wave_speed(right, p, normal);
    if (sr > smax) smax = sr;
    for (int i = 0; i < 5; ++i)
        flux[i] = 0.5 * (fl[i] + fr[i]) - 0.5 * smax * (right[i] - left[i]);
}

// HLLC star state; returns false when the star region degenerates
// (the caller then falls back to Rusanov, as the CPU does).
__device__ __forceinline__ bool dg_hllc_star(const double* base, double rho,
                                             const double* velocity, double un,
                                             double pressure, double s_k, double s_m,
                                             const DgParams* p, double* star) {
    double factor = rho * (s_k - un) / (s_k - s_m);
    if (!isfinite(factor) || factor <= 0.0) return false;
    double p_star = pressure + rho * (s_k - un) * (s_m - un);
    if (!isfinite(p_star) || p_star <= 0.0) return false;
    (void)velocity;
    double energy_star = factor * (base[4] / rho
        + (s_m - un) * (s_m + pressure / (rho * (s_k - un))));
    star[0] = factor;
    star[4] = energy_star;
    // Momentum slots filled by the caller, which owns the normal.
    (void)p;
    return true;
}

__device__ __forceinline__ bool dg_hllc(const double* left, const double* right,
                                        const DgParams* p, const double* normal,
                                        double* flux) {
    double gamma = p->gamma;
    double ul[3] = {left[1] / left[0], left[2] / left[0], left[3] / left[0]};
    double ur[3] = {right[1] / right[0], right[2] / right[0], right[3] / right[0]};
    double pl = dg_pressure(left, gamma), pr = dg_pressure(right, gamma);
    double unl = ul[0] * normal[0] + ul[1] * normal[1] + ul[2] * normal[2];
    double unr = ur[0] * normal[0] + ur[1] * normal[1] + ur[2] * normal[2];
    double cl = dg_sound(pl, left[0], gamma), cr = dg_sound(pr, right[0], gamma);
    double hl = (left[4] + pl) / left[0], hr = (right[4] + pr) / right[0];
    if (!isfinite(left[0]) || !isfinite(right[0]) || left[0] <= 0.0 || right[0] <= 0.0)
        return false;
    double sqrt_l = sqrt(left[0]), sqrt_r = sqrt(right[0]);
    double denom = sqrt_l + sqrt_r;
    double u_tilde[3], kinetic = 0.0;
    for (int i = 0; i < 3; ++i) {
        u_tilde[i] = (sqrt_l * ul[i] + sqrt_r * ur[i]) / denom;
        kinetic += u_tilde[i] * u_tilde[i];
    }
    double h_tilde = (sqrt_l * hl + sqrt_r * hr) / denom;
    double c_tilde_sq = (gamma - 1.0) * (h_tilde - 0.5 * kinetic);
    if (!isfinite(c_tilde_sq) || c_tilde_sq <= 0.0) return false;
    double c_tilde = sqrt(c_tilde_sq);
    double un_tilde = u_tilde[0] * normal[0] + u_tilde[1] * normal[1] + u_tilde[2] * normal[2];
    double s_l = fmin(unl - cl, un_tilde - c_tilde);
    double s_r = fmax(unr + cr, un_tilde + c_tilde);
    double numerator = pr - pl + left[0] * unl * (s_l - unl) - right[0] * unr * (s_r - unr);
    double denominator = left[0] * (s_l - unl) - right[0] * (s_r - unr);
    if (fabs(denominator) < 1.0e-300) return false;
    double s_m = numerator / denominator;
    double fl[5], fr[5];
    dg_euler_flux(left, p, normal, fl);
    dg_euler_flux(right, p, normal, fr);
    if (s_l >= 0.0) {
        for (int i = 0; i < 5; ++i) flux[i] = fl[i];
        return true;
    }
    if (s_m >= 0.0 || s_r >= 0.0) {
        bool from_left = s_m >= 0.0;
        const double* base = from_left ? left : right;
        const double* base_flux = from_left ? fl : fr;
        double rho = base[0];
        const double* vel = from_left ? ul : ur;
        double un = from_left ? unl : unr;
        double pressure = from_left ? pl : pr;
        double s_k = from_left ? s_l : s_r;
        double star[5];
        if (!dg_hllc_star(base, rho, vel, un, pressure, s_k, s_m, p, star)) return false;
        for (int i = 0; i < 3; ++i) {
            double v = vel[i] + (s_m - un) * normal[i];
            star[1 + i] = star[0] * v;
        }
        // CPU validates the star state against tight floors before use.
        if (!(star[0] >= 1.0e-14)) return false;
        double p_star_check = dg_pressure(star, gamma);
        if (!isfinite(p_star_check) || p_star_check < 1.0e-14) return false;
        if (!isfinite(star[4]) || !isfinite(star[1]) || !isfinite(star[2]) || !isfinite(star[3]))
            return false;
        for (int i = 0; i < 5; ++i)
            flux[i] = base_flux[i] + s_k * (star[i] - base[i]);
        return true;
    }
    for (int i = 0; i < 5; ++i) flux[i] = fr[i];
    return true;
}

// Full numerical flux: inviscid + averaged viscous + SIPG penalty.
// Returns false when either trace is nonphysical (CPU error path).
__device__ __forceinline__ bool dg_numerical_flux(
    const double* left, const double* right,
    const double* grad_l, const double* grad_tl,
    const double* grad_r, const double* grad_tr,
    const double* normal, double h, const DgParams* p, double* flux) {
    if (!dg_is_physical(left, p) || !dg_is_physical(right, p)) return false;
    double inviscid[5];
    if (p->flux_kind == 1) {
        if (!dg_hllc(left, right, p, normal, inviscid))
            dg_rusanov(left, right, p, normal, inviscid);
    } else {
        dg_rusanov(left, right, p, normal, inviscid);
    }
    double visc_l[15], visc_r[15];
    dg_viscous_tensor(left, grad_l, grad_tl, p, visc_l);
    dg_viscous_tensor(right, grad_r, grad_tr, p, visc_r);
    double temp_l = dg_temperature(dg_pressure(left, p->gamma), left[0], p);
    double temp_r = dg_temperature(dg_pressure(right, p->gamma), right[0], p);
    double diff = dg_viscosity(temp_l, p) / left[0];
    double diff_r = dg_viscosity(temp_r, p) / right[0];
    if (diff_r > diff) diff = diff_r;
    double h_safe = h > 2.2250738585072014e-308 ? h : 2.2250738585072014e-308;
    double penalty = p->sipg_penalty * diff / h_safe;
    for (int eq = 0; eq < 5; ++eq) {
        double viscous_normal = 0.0;
        for (int dir = 0; dir < p->dim; ++dir)
            viscous_normal += 0.5 * (visc_l[dir * 5 + eq] + visc_r[dir * 5 + eq]) * normal[dir];
        flux[eq] = inviscid[eq] - viscous_normal + penalty * (left[eq] - right[eq]);
    }
    return true;
}

__device__ __forceinline__ void dg_atomic_min_double(double* addr, double value) {
    unsigned long long* p = (unsigned long long*)addr;
    unsigned long long old = *p;
    while (true) {
        double current = __longlong_as_double((long long)old);
        if (!(value < current)) return;
        unsigned long long prev = atomicCAS(p, old, (unsigned long long)__double_as_longlong(value));
        if (prev == old) return;
        old = prev;
    }
}

__device__ __forceinline__ void dg_load_cell(const double* state, int element, int n,
                                             double* cell) {
    const double* src = state + (long long)element * n * 5;
    for (int i = 0; i < n * 5; ++i) cell[i] = src[i];
}

// Cooperative (block-wide, coalesced) cell load into shared memory.
__device__ __forceinline__ void dg_load_cell_par(const double* state, int element, int n,
                                                 double* cell, int tid) {
    const double* src = state + (long long)element * n * 5;
    for (int i = tid; i < n * 5; i += 32) cell[i] = src[i];
}

__device__ __forceinline__ void dg_cell_average(const double* cell, const double* avg_w,
                                                int n, double* average) {
    for (int eq = 0; eq < 5; ++eq) average[eq] = 0.0;
    for (int local = 0; local < n; ++local)
        for (int eq = 0; eq < 5; ++eq)
            average[eq] += avg_w[local] * cell[local * 5 + eq];
}

__device__ __forceinline__ void dg_from_primitive(double density, const double* velocity,
                                                  double pressure, const DgParams* p,
                                                  double* out) {
    double kinetic = 0.5 * density
        * (velocity[0] * velocity[0] + velocity[1] * velocity[1] + velocity[2] * velocity[2]);
    out[0] = density;
    out[1] = density * velocity[0];
    out[2] = density * velocity[1];
    out[3] = density * velocity[2];
    out[4] = pressure / (p->gamma - 1.0) + kinetic;
}

// Characteristic far-field ghost state (CPU characteristic_far_field_state).
__device__ __forceinline__ void dg_far_field(const double* interior, const double* far,
                                             const double* normal, const DgParams* p,
                                             double* out) {
    double gamma = p->gamma;
    double ui[3] = {interior[1] / interior[0], interior[2] / interior[0], interior[3] / interior[0]};
    double uf[3] = {far[1] / far[0], far[2] / far[0], far[3] / far[0]};
    double pi = dg_pressure(interior, gamma), pf = dg_pressure(far, gamma);
    double ci = dg_sound(pi, interior[0], gamma), cf = dg_sound(pf, far[0], gamma);
    double uni = ui[0] * normal[0] + ui[1] * normal[1] + ui[2] * normal[2];
    if (uni >= ci) {
        for (int i = 0; i < 5; ++i) out[i] = interior[i];
        return;
    }
    if (uni <= -ci) {
        for (int i = 0; i < 5; ++i) out[i] = far[i];
        return;
    }
    double unf = uf[0] * normal[0] + uf[1] * normal[1] + uf[2] * normal[2];
    double r_plus = uni + 2.0 * ci / (gamma - 1.0);
    double r_minus = unf - 2.0 * cf / (gamma - 1.0);
    double normal_velocity = 0.5 * (r_plus + r_minus);
    double sound = 0.25 * (gamma - 1.0) * (r_plus - r_minus);
    const double* upstream = normal_velocity < 0.0 ? far : interior;
    double up[3] = {upstream[1] / upstream[0], upstream[2] / upstream[0], upstream[3] / upstream[0]};
    double entropy = dg_pressure(upstream, gamma) / pow(upstream[0], gamma);
    double density = pow(fmax(sound * sound / (gamma * entropy), 2.2250738585072014e-308),
                         1.0 / (gamma - 1.0));
    double pressure = entropy * pow(density, gamma);
    double un_up = up[0] * normal[0] + up[1] * normal[1] + up[2] * normal[2];
    double velocity[3];
    for (int i = 0; i < 3; ++i)
        velocity[i] = (up[i] - un_up * normal[i]) + normal_velocity * normal[i];
    dg_from_primitive(density, velocity, pressure, p, out);
}

// Boundary ghost state. kind: 0 slip, 1 no-slip adiabatic, 2 no-slip
// isothermal, 3 far-field, 4 prescribed, 5 total inlet, 6 static outlet.
// params layout mirrors the host builder (see sonium-compute dg module).
__device__ __forceinline__ void dg_boundary_state(int kind, const double* params,
                                                  const double* interior,
                                                  const double* normal, const DgParams* p,
                                                  double* out) {
    double gamma = p->gamma;
    double pressure = dg_pressure(interior, gamma);
    double velocity[3] = {interior[1] / interior[0], interior[2] / interior[0],
                          interior[3] / interior[0]};
    if (kind == 0) {
        double un = velocity[0] * normal[0] + velocity[1] * normal[1] + velocity[2] * normal[2];
        double reflected[3];
        for (int i = 0; i < 3; ++i) reflected[i] = velocity[i] - 2.0 * un * normal[i];
        dg_from_primitive(interior[0], reflected, pressure, p, out);
    } else if (kind == 1) {
        double reflected[3];
        for (int i = 0; i < 3; ++i) reflected[i] = 2.0 * params[i] - velocity[i];
        dg_from_primitive(interior[0], reflected, pressure, p, out);
    } else if (kind == 2) {
        double interior_temperature = dg_temperature(pressure, interior[0], p);
        double ghost_temperature = fmax(2.0 * params[3] - interior_temperature, 1.0e-12);
        double density = pressure / (p->gas_constant * ghost_temperature);
        double reflected[3];
        for (int i = 0; i < 3; ++i) reflected[i] = 2.0 * params[i] - velocity[i];
        dg_from_primitive(density, reflected, pressure, p, out);
    } else if (kind == 3) {
        dg_far_field(interior, params, normal, p, out);
    } else if (kind == 4) {
        for (int i = 0; i < 5; ++i) out[i] = params[i];
    } else if (kind == 5) {
        double direction[3] = {params[2], params[3], params[4]};
        double norm = sqrt(direction[0] * direction[0] + direction[1] * direction[1]
                           + direction[2] * direction[2]);
        double speed = sqrt(velocity[0] * velocity[0] + velocity[1] * velocity[1]
                            + velocity[2] * velocity[2]);
        double sound = dg_sound(pressure, interior[0], gamma);
        double mach = speed / sound;
        if (mach > 0.999) mach = 0.999;
        if (mach < 0.0) mach = 0.0;
        double factor = 1.0 + 0.5 * (gamma - 1.0) * mach * mach;
        double static_temperature = params[1] / factor;
        double static_pressure = params[0] / pow(factor, gamma / (gamma - 1.0));
        double density = static_pressure / (p->gas_constant * static_temperature);
        double inlet_speed = mach * sqrt(gamma * p->gas_constant * static_temperature);
        double inlet_velocity[3];
        for (int i = 0; i < 3; ++i) inlet_velocity[i] = direction[i] / norm * inlet_speed;
        dg_from_primitive(density, inlet_velocity, static_pressure, p, out);
    } else {
        dg_from_primitive(interior[0], velocity, params[0], p, out);
    }
}

extern "C" __global__ void sonium_dg_reset_scalars(double* dt_out, int* flag,
                                                   double* min_thermo, double* totals) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    *dt_out = DG_INF;
    *flag = 0;
    min_thermo[0] = DG_INF;
    min_thermo[1] = DG_INF;
    for (int i = 0; i < 5; ++i) totals[i] = 0.0;
}


// Warp-per-element mapping (v3): each warp owns one element, phases
// synchronize with __syncwarp, and four warps share a 128-thread
// block. `element` is uniform across the warp, so early return and
// loop trip counts are warp-uniform.
extern "C" __global__ void sonium_dg_averages(const double* state, const double* avg_w,
                                              double* averages, DgParams params,
                                              int num_elements) {
    int element = blockIdx.x * 4 + (threadIdx.x >> 5);
    if (element >= num_elements) return;
    if ((threadIdx.x & 31) != 0) return;
    double cell[DG_MAX_N * 5];
    dg_load_cell(state, element, params.n, cell);
    double average[5];
    dg_cell_average(cell, avg_w, params.n, average);
    for (int eq = 0; eq < 5; ++eq) averages[element * 5 + eq] = average[eq];
}


extern "C" __global__ void sonium_dg_dt(const double* averages, const double* min_altitude,
                                        double* dt_out, int* flag, DgParams params,
                                        int num_elements) {
    int element = blockIdx.x * 4 + (threadIdx.x >> 5);
    if (element >= num_elements) return;
    if ((threadIdx.x & 31) != 0) return;
    const double* average = averages + element * 5;
    if (!dg_is_physical(average, &params)) {
        atomicOr(flag, 2);
        return;
    }
    double h = min_altitude[element];
    double vel[3] = {average[1] / average[0], average[2] / average[0], average[3] / average[0]};
    double pressure = dg_pressure(average, params.gamma);
    double speed = sqrt(vel[0] * vel[0] + vel[1] * vel[1] + vel[2] * vel[2])
        + dg_sound(pressure, average[0], params.gamma);
    double temperature = dg_temperature(pressure, average[0], &params);
    double mu = dg_viscosity(temperature, &params);
    double conductivity = dg_conductivity(temperature, &params);
    double cp = params.gamma * params.gas_constant / (params.gamma - 1.0);
    double diffusivity = mu / average[0];
    double thermal = conductivity / (average[0] * cp);
    if (thermal > diffusivity) diffusivity = thermal;
    double convective = h / fmax(speed, 2.2250738585072014e-308);
    double viscous = 0.5 * h * h / fmax(diffusivity, 2.2250738585072014e-308);
    double local = params.cfl_scaled * fmin(convective, viscous);
    dg_atomic_min_double(dt_out, local);
}


// Fused right-hand side (v3): volume + interior-facet + boundary
// contributions accumulated per element in the CPU merge order, then
// the inverse-mass product, in a single launch. One warp owns one
// element; quadrature-point lanes evaluate fluxes into shared memory
// and the local-node lanes accumulate. No integrated-residual buffer
// round-trips through global memory.
extern "C" __global__ void sonium_dg_rhs(
    const double* state, const double* vol_basis, const double* vol_grad,
    const double* vol_wdet, const double* cent_grad,
    const int* facet_offsets, const int* facet_neighbor, const int* facet_is_plus,
    const double* facet_normal, const double* facet_h, const int* facet_quad_offsets,
    const double* facet_basis_own, const double* facet_basis_other,
    const double* facet_weights,
    const int* bnd_offsets, const int* bnd_kind, const double* bnd_params,
    const double* bnd_normal, const double* bnd_h, const int* bnd_quad_offsets,
    const double* bnd_basis, const double* bnd_weights,
    const double* mass_inv, double* rate, int* flag, DgParams params,
    int nq_vol, int num_elements) {
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int element = blockIdx.x * 4 + warp;
    if (element >= num_elements) return;
    __shared__ double cell[4][DG_MAX_N * 5];
    __shared__ double other_cell[4][DG_MAX_N * 5];
    __shared__ double grad_velocity[4][9];
    __shared__ double grad_temperature[4][3];
    __shared__ double other_gv[4][9];
    __shared__ double other_gt[4][3];
    __shared__ double point_flux[4][32 * 3 * 5];
    __shared__ double facet_flux[4][7 * 5];
    __shared__ double res_shared[4][DG_MAX_N * 5];
    int n = params.n;
    double* my_cell = cell[warp];
    dg_load_cell_par(state, element, n, my_cell, lane);
    __syncwarp();
    if (lane == 0)
        dg_primitive_gradients(my_cell, cent_grad + (long long)element * n * 3,
                               &params, grad_velocity[warp], grad_temperature[warp]);
    __syncwarp();
    double res[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
    // Volume terms.
    const double* grads = vol_grad + ((long long)element * nq_vol * n) * 3;
    for (int base = 0; base < nq_vol; base += 32) {
        int count = nq_vol - base < 32 ? nq_vol - base : 32;
        if (lane < count) {
            int q = base + lane;
            double value[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
            for (int j = 0; j < n; ++j) {
                double basis = vol_basis[q * n + j];
                for (int eq = 0; eq < 5; ++eq) value[eq] += basis * my_cell[j * 5 + eq];
            }
            if (!dg_is_physical(value, &params)) atomicOr(flag, 1);
            double viscous[15];
            dg_viscous_tensor(value, grad_velocity[warp], grad_temperature[warp],
                              &params, viscous);
            for (int dir = 0; dir < params.dim; ++dir) {
                double normal[3] = {0.0, 0.0, 0.0};
                normal[dir] = 1.0;
                double inviscid[5];
                dg_euler_flux(value, &params, normal, inviscid);
                for (int eq = 0; eq < 5; ++eq)
                    point_flux[warp][(lane * 3 + dir) * 5 + eq] = inviscid[eq] - viscous[dir * 5 + eq];
            }
        }
        __syncwarp();
        if (lane < n) {
            for (int i = 0; i < count; ++i) {
                int q = base + i;
                double weight = vol_wdet[(long long)element * nq_vol + q];
                const double* g = grads + ((long long)q * n + lane) * 3;
                for (int dir = 0; dir < params.dim; ++dir) {
                    const double* f = point_flux[warp] + (i * 3 + dir) * 5;
                    for (int eq = 0; eq < 5; ++eq)
                        res[eq] += g[dir] * f[eq] * weight;
                }
            }
        }
        __syncwarp();
    }
    // Interior-facet terms.
    for (int r = facet_offsets[element]; r < facet_offsets[element + 1]; ++r) {
        bool plus = facet_is_plus[r] != 0;
        double* my_other = other_cell[warp];
        dg_load_cell_par(state, facet_neighbor[r], n, my_other, lane);
        __syncwarp();
        if (lane == 0)
            dg_primitive_gradients(my_other, cent_grad + (long long)facet_neighbor[r] * n * 3,
                                   &params, other_gv[warp], other_gt[warp]);
        __syncwarp();
        const double* normal = facet_normal + r * 3;
        const double* left_gv = plus ? grad_velocity[warp] : other_gv[warp];
        const double* left_gt = plus ? grad_temperature[warp] : other_gt[warp];
        const double* right_gv = plus ? other_gv[warp] : grad_velocity[warp];
        const double* right_gt = plus ? other_gt[warp] : grad_temperature[warp];
        double sign = plus ? -1.0 : 1.0;
        int q0 = facet_quad_offsets[r], q1 = facet_quad_offsets[r + 1];
        if (lane < q1 - q0) {
            int q = q0 + lane;
            double left[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
            double right[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
            for (int j = 0; j < n; ++j) {
                double b_own = facet_basis_own[q * n + j];
                double b_other = facet_basis_other[q * n + j];
                double* own_state = plus ? left : right;
                double* other_state = plus ? right : left;
                for (int eq = 0; eq < 5; ++eq) {
                    own_state[eq] += b_own * my_cell[j * 5 + eq];
                    other_state[eq] += b_other * my_other[j * 5 + eq];
                }
            }
            if (!dg_numerical_flux(left, right, left_gv, left_gt, right_gv, right_gt,
                                   normal, facet_h[r], &params, facet_flux[warp] + lane * 5))
                atomicOr(flag, 1);
        }
        __syncwarp();
        if (lane < n) {
            for (int q = q0; q < q1; ++q) {
                double basis = facet_basis_own[q * n + lane];
                double w = facet_weights[q];
                const double* flux = facet_flux[warp] + (q - q0) * 5;
                for (int eq = 0; eq < 5; ++eq)
                    res[eq] += sign * basis * flux[eq] * w;
            }
        }
        __syncwarp();
    }
    // Boundary terms.
    for (int r = bnd_offsets[element]; r < bnd_offsets[element + 1]; ++r) {
        const double* normal = bnd_normal + r * 3;
        int q0 = bnd_quad_offsets[r], q1 = bnd_quad_offsets[r + 1];
        if (lane < q1 - q0) {
            int q = q0 + lane;
            double interior[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
            for (int j = 0; j < n; ++j) {
                double b = bnd_basis[q * n + j];
                for (int eq = 0; eq < 5; ++eq) interior[eq] += b * my_cell[j * 5 + eq];
            }
            double exterior[5];
            dg_boundary_state(bnd_kind[r], bnd_params + r * 8, interior, normal, &params,
                              exterior);
            if (!dg_numerical_flux(interior, exterior, grad_velocity[warp],
                                   grad_temperature[warp], grad_velocity[warp],
                                   grad_temperature[warp], normal, bnd_h[r], &params,
                                   facet_flux[warp] + lane * 5))
                atomicOr(flag, 1);
        }
        __syncwarp();
        if (lane < n) {
            for (int q = q0; q < q1; ++q) {
                double b = bnd_basis[q * n + lane];
                double w = bnd_weights[q];
                const double* flux = facet_flux[warp] + (q - q0) * 5;
                for (int eq = 0; eq < 5; ++eq)
                    res[eq] -= b * flux[eq] * w;
            }
        }
        __syncwarp();
    }
    // Inverse mass product.
    if (lane < n) {
        double* slot = res_shared[warp] + lane * 5;
        for (int eq = 0; eq < 5; ++eq) slot[eq] = res[eq];
    }
    __syncwarp();
    if (lane < n) {
        const double* inv = mass_inv + (long long)element * n * n;
        const double* src = res_shared[warp];
        double* out = rate + ((long long)element * n + lane) * 5;
        for (int eq = 0; eq < 5; ++eq) {
            double sum = 0.0;
            for (int j = 0; j < n; ++j) sum += inv[lane * n + j] * src[j * 5 + eq];
            out[eq] = sum;
        }
    }
}

// Rust `f64::min` semantics: a NaN operand yields the other operand.
__device__ __forceinline__ double dg_min_rust(double a, double b) {
    if (isnan(a)) return b;
    if (isnan(b)) return a;
    return fmin(a, b);
}

__device__ __forceinline__ void dg_characteristic(const double* s, const double* normal,
                                                  const DgParams* p, double* out) {
    double vel[3] = {s[1] / s[0], s[2] / s[0], s[3] / s[0]};
    double pressure = dg_pressure(s, p->gamma);
    double sound = dg_sound(pressure, s[0], p->gamma);
    double un = vel[0] * normal[0] + vel[1] * normal[1] + vel[2] * normal[2];
    double seed[3] = {1.0, 0.0, 0.0};
    if (fabs(normal[0]) >= 0.9) { seed[0] = 0.0; seed[1] = 1.0; }
    double tangent1[3] = {
        normal[1] * seed[2] - normal[2] * seed[1],
        normal[2] * seed[0] - normal[0] * seed[2],
        normal[0] * seed[1] - normal[1] * seed[0],
    };
    double tn = sqrt(tangent1[0] * tangent1[0] + tangent1[1] * tangent1[1]
                     + tangent1[2] * tangent1[2]);
    for (int i = 0; i < 3; ++i) tangent1[i] /= tn;
    double tangent2[3] = {
        normal[1] * tangent1[2] - normal[2] * tangent1[1],
        normal[2] * tangent1[0] - normal[0] * tangent1[2],
        normal[0] * tangent1[1] - normal[1] * tangent1[0],
    };
    out[0] = pressure - s[0] * sound * un;
    out[1] = s[0] - pressure / (sound * sound);
    out[2] = pressure + s[0] * sound * un;
    out[3] = vel[0] * tangent1[0] + vel[1] * tangent1[1] + vel[2] * tangent1[2];
    out[4] = vel[0] * tangent2[0] + vel[1] * tangent2[1] + vel[2] * tangent2[2];
}

__device__ __forceinline__ double dg_persson_indicator(const double* density,
                                                       const double* vinv,
                                                       const double* gram,
                                                       const DgParams* p) {
    int n = p->n;
    double coefficients[DG_MAX_N];
    for (int mode = 0; mode < n; ++mode) {
        double sum = 0.0;
        for (int node = 0; node < n; ++node) sum += vinv[mode * n + node] * density[node];
        coefficients[mode] = sum;
    }
    double total = 0.0;
    for (int row = 0; row < n; ++row)
        for (int col = 0; col < n; ++col)
            total += coefficients[row] * gram[row * n + col] * coefficients[col];
    if (total <= 0.0) return -DG_INF;
    double highest = 0.0;
    for (int row = p->persson_highest; row < n; ++row)
        for (int col = p->persson_highest; col < n; ++col)
            highest += coefficients[row] * gram[row * n + col] * coefficients[col];
    if (highest < 0.0) highest = 0.0;
    if (highest <= 0.0) return -DG_INF;
    return log10(highest / total);
}


// Warp-per-element: lane 0 of the warp makes the troubled-cell
// decision; the characteristic theta is a min-reduction over
// (neighbour, nodal) pairs spread across the warp's lanes. Min is
// order-independent, so the reduced theta is the value the serial
// CPU loop computes.
extern "C" __global__ void sonium_dg_limiter(
    double* state, const double* averages, const double* avg_w,
    const int* lim_offsets, const int* lim_neighbor, const double* lim_normal,
    const double* persson_vinv, const double* persson_gram, DgParams params,
    int num_elements) {
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int element = blockIdx.x * 4 + warp;
    if (element >= num_elements) return;
    __shared__ double cell[4][DG_MAX_N * 5];
    __shared__ double theta_shared[4];
    __shared__ int active[4];
    __shared__ int range_begin[4];
    __shared__ double partials[4][32];
    double* my_cell = cell[warp];
    dg_load_cell_par(state, element, params.n, my_cell, lane);
    __syncwarp();
    if (lane == 0) {
        active[warp] = 0;
        int begin = lim_offsets[element], end = lim_offsets[element + 1];
        range_begin[warp] = begin;
        if (begin < end) {
            int n = params.n;
            const double* average = averages + element * 5;
            bool troubled;
            if (params.sensor_mode == 0) {
                double density[DG_MAX_N];
                for (int j = 0; j < n; ++j) density[j] = my_cell[j * 5];
                bool smooth = dg_persson_indicator(density, persson_vinv, persson_gram, &params)
                    <= params.persson_threshold;
                bool strong_jump = false;
                double p_avg = dg_pressure(average, params.gamma);
                for (int r = begin; r < end && !strong_jump; ++r) {
                    const double* other = averages + lim_neighbor[r] * 5;
                    double drho = fabs(other[0] - average[0])
                        / fmax(fabs(average[0]), 1.0e-300);
                    double dp = fabs(dg_pressure(other, params.gamma) - p_avg)
                        / fmax(fabs(p_avg), 1.0e-300);
                    if (drho > 0.5 || dp > 0.5) strong_jump = true;
                }
                troubled = !smooth || strong_jump;
            } else if (params.sensor_mode == 1) {
                double pressure = dg_pressure(average, params.gamma);
                double density_scale = fmax(fabs(average[0]), 1.0e-300);
                double pressure_scale = fmax(fabs(pressure), 1.0e-300);
                troubled = false;
                for (int j = 0; j < n && !troubled; ++j) {
                    const double* nodal = my_cell + j * 5;
                    if (fabs(nodal[0] - average[0]) / density_scale > params.threshold
                        || fabs(dg_pressure(nodal, params.gamma) - pressure) / pressure_scale
                            > params.threshold)
                        troubled = true;
                }
                for (int r = begin; r < end && !troubled; ++r) {
                    const double* other = averages + lim_neighbor[r] * 5;
                    if (fabs(other[0] - average[0]) / density_scale > params.threshold
                        || fabs(dg_pressure(other, params.gamma) - pressure) / pressure_scale
                            > params.threshold)
                        troubled = true;
                }
            } else {
                troubled = true;
            }
            if (troubled) active[warp] = 1;
        }
    }
    __syncwarp();
    if (!active[warp]) return;
    int n = params.n;
    int begin = range_begin[warp];
    int neighbors = lim_offsets[element + 1] - begin;
    const double* average = averages + element * 5;
    double partial = 1.0;
    int pairs = neighbors * n;
    for (int pair = lane; pair < pairs; pair += 32) {
        int r = begin + pair / n;
        int j = pair % n;
        const double* normal = lim_normal + r * 3;
        double center[5], neighbor_char[5], nodal_char[5];
        dg_characteristic(average, normal, &params, center);
        dg_characteristic(averages + lim_neighbor[r] * 5, normal, &params, neighbor_char);
        dg_characteristic(my_cell + j * 5, normal, &params, nodal_char);
        for (int wave = 0; wave < 5; ++wave) {
            double delta_neighbor = neighbor_char[wave] - center[wave];
            double margin = params.relaxation * fmax(fabs(delta_neighbor), 1.0e-12);
            double lower = fmin(center[wave], neighbor_char[wave]) - margin;
            double upper = fmax(center[wave], neighbor_char[wave]) + margin;
            double delta = nodal_char[wave] - center[wave];
            if (delta > 0.0) {
                partial = dg_min_rust(partial, (upper - center[wave]) / delta);
            } else if (delta < 0.0) {
                partial = dg_min_rust(partial, (lower - center[wave]) / delta);
            }
        }
    }
    partials[warp][lane] = partial;
    __syncwarp();
    if (lane == 0) {
        double theta = 1.0;
        for (int i = 0; i < 32; ++i) theta = dg_min_rust(theta, partials[warp][i]);
        theta_shared[warp] = theta;
    }
    __syncwarp();
    if (lane >= n) return;
    double theta = theta_shared[warp];
    if (theta < 0.0) theta = 0.0;
    if (theta > 1.0) theta = 1.0;
    double* nodal = state + ((long long)element * n + lane) * 5;
    for (int eq = 0; eq < 5; ++eq)
        nodal[eq] = average[eq] + theta * (nodal[eq] - average[eq]);
}


// Warp-per-element: lane 0 computes the positivity theta (density
// bound, then the sequential pressure bisection, as on CPU); the
// warp's local-node lanes apply the scaling.
extern "C" __global__ void sonium_dg_zhangshu(double* state, const double* avg_w,
                                              int* flag, DgParams params,
                                              int num_elements) {
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int element = blockIdx.x * 4 + warp;
    if (element >= num_elements) return;
    __shared__ double theta_shared[4];
    __shared__ double average_shared[4][5];
    __shared__ int active[4];
    if (lane == 0) {
        active[warp] = 0;
        int n = params.n;
        double cell[DG_MAX_N * 5];
        dg_load_cell(state, element, n, cell);
        double average[5];
        dg_cell_average(cell, avg_w, n, average);
        for (int eq = 0; eq < 5; ++eq) average_shared[warp][eq] = average[eq];
        if (!dg_is_physical(average, &params)) {
            atomicOr(flag, 2);
        } else {
            double theta = 1.0;
            for (int j = 0; j < n; ++j) {
                double density = cell[j * 5];
                if (density < params.density_floor) {
                    double candidate = (average[0] - params.density_floor)
                        / (average[0] - density);
                    if (candidate < theta) theta = candidate;
                }
            }
            for (int j = 0; j < n; ++j) {
                const double* nodal = cell + j * 5;
                double candidate_state[5];
                for (int eq = 0; eq < 5; ++eq)
                    candidate_state[eq] = average[eq] + theta * (nodal[eq] - average[eq]);
                if (dg_pressure(candidate_state, params.gamma) < params.pressure_floor) {
                    double lower = 0.0, upper = theta;
                    for (int it = 0; it < 60; ++it) {
                        double midpoint = 0.5 * (lower + upper);
                        for (int eq = 0; eq < 5; ++eq)
                            candidate_state[eq] = average[eq]
                                + midpoint * (nodal[eq] - average[eq]);
                        if (dg_pressure(candidate_state, params.gamma) >= params.pressure_floor)
                            lower = midpoint;
                        else
                            upper = midpoint;
                    }
                    if (lower < theta) theta = lower;
                }
            }
            theta_shared[warp] = theta;
            active[warp] = 1;
        }
    }
    __syncwarp();
    if (!active[warp] || lane >= params.n) return;
    double theta = theta_shared[warp];
    if (theta < 0.0) theta = 0.0;
    if (theta > 1.0) theta = 1.0;
    double* nodal = state + ((long long)element * params.n + lane) * 5;
    for (int eq = 0; eq < 5; ++eq)
        nodal[eq] = average_shared[warp][eq] + theta * (nodal[eq] - average_shared[warp][eq]);
}

extern "C" __global__ void sonium_dg_add_scaled(double* out, const double* x,
                                                const double* rate, double dt,
                                                int dim, int total_slots) {
    int slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= total_slots) return;
    double* o = out + (long long)slot * 5;
    const double* a = x + (long long)slot * 5;
    const double* r = rate + (long long)slot * 5;
    for (int eq = 0; eq < 5; ++eq) o[eq] = a[eq] + dt * r[eq];
    if (dim == 2) o[3] = 0.0;
}

extern "C" __global__ void sonium_dg_convex(double* out, double a, const double* x,
                                            double b, const double* y, int dim,
                                            int total_slots) {
    int slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= total_slots) return;
    double* o = out + (long long)slot * 5;
    const double* l = x + (long long)slot * 5;
    const double* r = y + (long long)slot * 5;
    for (int eq = 0; eq < 5; ++eq) o[eq] = a * l[eq] + b * r[eq];
    if (dim == 2) o[3] = 0.0;
}

extern "C" __global__ void sonium_dg_copy(double* out, const double* x, int total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    out[i] = x[i];
}

extern "C" __global__ void sonium_dg_min_thermo(const double* state, double* min_thermo,
                                                DgParams params, int total_slots) {
    int slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= total_slots) return;
    const double* s = state + (long long)slot * 5;
    dg_atomic_min_double(min_thermo, s[0]);
    dg_atomic_min_double(min_thermo + 1, dg_pressure(s, params.gamma));
}

extern "C" __global__ void sonium_dg_totals(const double* averages,
                                            const double* elem_volume, double* totals,
                                            int elements) {
    int element = blockIdx.x;
    if (threadIdx.x != 0 || element >= elements) return;
    double volume = elem_volume[element];
    const double* average = averages + element * 5;
    for (int eq = 0; eq < 5; ++eq) atomicAdd(totals + eq, volume * average[eq]);
}
