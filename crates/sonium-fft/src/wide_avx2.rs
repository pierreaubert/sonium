//! AVX2-specialised wide-butterfly stages. x86_64-only.
//!
//! Processes four consecutive `Complex<f64>` butterflies per loop iteration.
//! Complex input is interleaved in memory, so each stream uses two unaligned
//! 256-bit loads plus a lane shuffle to form planar real/imaginary vectors.
//! Stage twiddles are stride-1, which keeps twiddle loads contiguous too.

use std::arch::x86_64::*;

use num_complex::Complex;

use crate::twiddles::StageTwiddles;

#[inline]
unsafe fn load_complex2(ptr: *const Complex<f64>) -> (__m128d, __m128d) {
    unsafe {
        let v0 = _mm_loadu_pd(ptr as *const f64);
        let v1 = _mm_loadu_pd((ptr as *const f64).add(2));
        let re = _mm_shuffle_pd(v0, v1, 0b00);
        let im = _mm_shuffle_pd(v0, v1, 0b11);
        (re, im)
    }
}

#[inline]
unsafe fn store_complex2(ptr: *mut Complex<f64>, re: __m128d, im: __m128d) {
    unsafe {
        let v0 = _mm_shuffle_pd(re, im, 0b00);
        let v1 = _mm_shuffle_pd(re, im, 0b11);
        _mm_storeu_pd(ptr as *mut f64, v0);
        _mm_storeu_pd((ptr as *mut f64).add(2), v1);
    }
}

#[inline]
unsafe fn cmul2(a_re: __m128d, a_im: __m128d, b_re: __m128d, b_im: __m128d) -> (__m128d, __m128d) {
    unsafe {
        let re = _mm_fmsub_pd(a_re, b_re, _mm_mul_pd(a_im, b_im));
        let im = _mm_fmadd_pd(a_re, b_im, _mm_mul_pd(a_im, b_re));
        (re, im)
    }
}

#[inline]
unsafe fn load_complex2_f32(ptr: *const Complex<f32>) -> (__m128, __m128) {
    unsafe {
        let values = _mm_loadu_ps(ptr as *const f32);
        // The physical layout is [re0, im0, re1, im1]. Keep the first two
        // logical lanes duplicated so one pair of SIMD arithmetic operations
        // handles both independent transforms.
        let re = _mm_shuffle_ps(values, values, 0x88);
        let im = _mm_shuffle_ps(values, values, 0xdd);
        (re, im)
    }
}

#[inline]
unsafe fn store_complex2_f32(ptr: *mut Complex<f32>, re: __m128, im: __m128) {
    unsafe {
        let values = _mm_unpacklo_ps(re, im);
        _mm_storeu_ps(ptr as *mut f32, values);
    }
}

#[inline]
unsafe fn cmul_f32(a_re: __m128, a_im: __m128, b_re: __m128, b_im: __m128) -> (__m128, __m128) {
    unsafe {
        let re = _mm_fmsub_ps(a_re, b_re, _mm_mul_ps(a_im, b_im));
        let im = _mm_fmadd_ps(a_re, b_im, _mm_mul_ps(a_im, b_re));
        (re, im)
    }
}

#[inline]
unsafe fn cmul_aos_f32(a: __m256, b: __m256) -> __m256 {
    unsafe {
        let a_re = _mm256_shuffle_ps(a, a, 0xA0);
        let a_im = _mm256_shuffle_ps(a, a, 0xF5);
        let b_swap = _mm256_shuffle_ps(b, b, 0xB1);
        let p1 = _mm256_mul_ps(a_re, b);
        let p2 = _mm256_mul_ps(a_im, b_swap);
        _mm256_addsub_ps(p1, p2)
    }
}

/// AVX2/FMA radix-2 stage for one `Complex<f32>` FFT in interleaved (AoS)
/// layout: raw loads/stores, no deinterleave shuffles.
///
/// # Safety
///
/// The caller must provide AVX2 and FMA at runtime. `stage_tw` must
/// describe a radix-2 stage with `src.len() == dst.len() == 2 *
/// stage_tw.m * stage_tw.s`, `stride == src.len() / 2`, and disjoint
/// source/destination slices.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn stage_radix2_f32_aos(
    stride: usize,
    src: &[Complex<f32>],
    dst: &mut [Complex<f32>],
    stage_tw: &StageTwiddles<f32>,
) {
    unsafe {
        stage_tw.debug_assert_stage_shape(2, stride, src.len(), dst.len());
        let m = stage_tw.m;
        let s = stage_tw.s;
        let m_new = m * 2;
        let tw1 = stage_tw.k_slice(1);

        for g in 0..s {
            let base_in = g * m;
            let base_out = g * m_new;

            let mut l = 0usize;
            while l + 4 <= m {
                let a = _mm256_loadu_ps(src.as_ptr().add(base_in + l) as *const f32);
                let b = _mm256_loadu_ps(src.as_ptr().add(base_in + l + stride) as *const f32);
                let tw = _mm256_loadu_ps(tw1.as_ptr().add(l) as *const f32);
                let b = cmul_aos_f32(b, tw);
                _mm256_storeu_ps(
                    dst.as_mut_ptr().add(base_out + l) as *mut f32,
                    _mm256_add_ps(a, b),
                );
                _mm256_storeu_ps(
                    dst.as_mut_ptr().add(base_out + m + l) as *mut f32,
                    _mm256_sub_ps(a, b),
                );
                l += 4;
            }
            while l < m {
                let a = src[base_in + l];
                let b = if l == 0 {
                    src[base_in + l + stride]
                } else {
                    src[base_in + l + stride] * tw1[l]
                };
                dst[base_out + l] = a + b;
                dst[base_out + m + l] = a - b;
                l += 1;
            }
        }
    }
}

/// AVX2/FMA radix-4 stage for one `Complex<f32>` FFT in interleaved (AoS)
/// layout: raw loads/stores, no deinterleave shuffles.
///
/// # Safety
///
/// The caller must provide AVX2 and FMA at runtime. `stage_tw` must
/// describe a radix-4 stage with `src.len() == dst.len() == 4 *
/// stage_tw.m * stage_tw.s`, `stride == src.len() / 4`, and disjoint
/// source/destination slices.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn stage_radix4_f32_aos(
    stride: usize,
    src: &[Complex<f32>],
    dst: &mut [Complex<f32>],
    stage_tw: &StageTwiddles<f32>,
    dir_sign: f32,
) {
    unsafe {
        stage_tw.debug_assert_stage_shape(4, stride, src.len(), dst.len());
        let m = stage_tw.m;
        let s = stage_tw.s;
        let m_new = m * 4;
        // Turns (d.re, d.im) into (-d.im * dir, d.re * dir) after a swap.
        let id_sign = _mm256_set_ps(
            dir_sign, -dir_sign, dir_sign, -dir_sign, dir_sign, -dir_sign, dir_sign, -dir_sign,
        );
        let tw1 = stage_tw.k_slice(1);
        let tw2 = stage_tw.k_slice(2);
        let tw3 = stage_tw.k_slice(3);

        for g in 0..s {
            let base_in = g * m;
            let base_out = g * m_new;

            let mut l = 0usize;
            while l + 4 <= m {
                let x0 = _mm256_loadu_ps(src.as_ptr().add(base_in + l) as *const f32);
                let x1 = _mm256_loadu_ps(src.as_ptr().add(base_in + l + stride) as *const f32);
                let x2 = _mm256_loadu_ps(src.as_ptr().add(base_in + l + 2 * stride) as *const f32);
                let x3 = _mm256_loadu_ps(src.as_ptr().add(base_in + l + 3 * stride) as *const f32);
                let x1 = cmul_aos_f32(x1, _mm256_loadu_ps(tw1.as_ptr().add(l) as *const f32));
                let x2 = cmul_aos_f32(x2, _mm256_loadu_ps(tw2.as_ptr().add(l) as *const f32));
                let x3 = cmul_aos_f32(x3, _mm256_loadu_ps(tw3.as_ptr().add(l) as *const f32));

                let a = _mm256_add_ps(x0, x2);
                let b = _mm256_sub_ps(x0, x2);
                let c = _mm256_add_ps(x1, x3);
                let d = _mm256_sub_ps(x1, x3);
                let id = _mm256_mul_ps(_mm256_shuffle_ps(d, d, 0xB1), id_sign);

                _mm256_storeu_ps(
                    dst.as_mut_ptr().add(base_out + l) as *mut f32,
                    _mm256_add_ps(a, c),
                );
                _mm256_storeu_ps(
                    dst.as_mut_ptr().add(base_out + m + l) as *mut f32,
                    _mm256_add_ps(b, id),
                );
                _mm256_storeu_ps(
                    dst.as_mut_ptr().add(base_out + 2 * m + l) as *mut f32,
                    _mm256_sub_ps(a, c),
                );
                _mm256_storeu_ps(
                    dst.as_mut_ptr().add(base_out + 3 * m + l) as *mut f32,
                    _mm256_sub_ps(b, id),
                );
                l += 4;
            }
            while l < m {
                let x0 = src[base_in + l];
                let (x1, x2, x3) = if l == 0 {
                    (
                        src[base_in + l + stride],
                        src[base_in + l + 2 * stride],
                        src[base_in + l + 3 * stride],
                    )
                } else {
                    (
                        src[base_in + l + stride] * tw1[l],
                        src[base_in + l + 2 * stride] * tw2[l],
                        src[base_in + l + 3 * stride] * tw3[l],
                    )
                };
                let a = x0 + x2;
                let b = x0 - x2;
                let c = x1 + x3;
                let d = x1 - x3;
                let id = Complex::new(-d.im * dir_sign, d.re * dir_sign);
                dst[base_out + l] = a + c;
                dst[base_out + m + l] = b + id;
                dst[base_out + 2 * m + l] = a - c;
                dst[base_out + 3 * m + l] = b - id;
                l += 1;
            }
        }
    }
}




/// AoS complex multiply for two interleaved `Complex<f64>` lanes.
#[inline]
unsafe fn cmul_aos_f64(a: __m256d, b: __m256d) -> __m256d {
    unsafe {
        let a_re = _mm256_shuffle_pd(a, a, 0x00);
        let a_im = _mm256_shuffle_pd(a, a, 0x0F);
        let b_swap = _mm256_shuffle_pd(b, b, 0x05);
        let p1 = _mm256_mul_pd(a_re, b);
        let p2 = _mm256_mul_pd(a_im, b_swap);
        _mm256_addsub_pd(p1, p2)
    }
}

/// AVX2/FMA radix-2 stage for one `Complex<f64>` FFT in interleaved (AoS)
/// layout: raw loads/stores, no deinterleave shuffles.
///
/// # Safety
///
/// The caller must provide AVX2 and FMA at runtime. `stage_tw` must
/// describe a radix-2 stage with `src.len() == dst.len() == 2 *
/// stage_tw.m * stage_tw.s`, `stride == src.len() / 2`, and disjoint
/// source/destination slices.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn stage_radix2_f64_aos(
    stride: usize,
    src: &[Complex<f64>],
    dst: &mut [Complex<f64>],
    stage_tw: &StageTwiddles<f64>,
) {
    unsafe {
        stage_tw.debug_assert_stage_shape(2, stride, src.len(), dst.len());
        let m = stage_tw.m;
        let s = stage_tw.s;
        let m_new = m * 2;
        let tw1 = stage_tw.k_slice(1);

        for g in 0..s {
            let base_in = g * m;
            let base_out = g * m_new;

            let mut l = 0usize;
            while l + 2 <= m {
                let a = _mm256_loadu_pd(src.as_ptr().add(base_in + l) as *const f64);
                let b = _mm256_loadu_pd(src.as_ptr().add(base_in + l + stride) as *const f64);
                let tw = _mm256_loadu_pd(tw1.as_ptr().add(l) as *const f64);
                let b = cmul_aos_f64(b, tw);
                _mm256_storeu_pd(
                    dst.as_mut_ptr().add(base_out + l) as *mut f64,
                    _mm256_add_pd(a, b),
                );
                _mm256_storeu_pd(
                    dst.as_mut_ptr().add(base_out + m + l) as *mut f64,
                    _mm256_sub_pd(a, b),
                );
                l += 2;
            }
            while l < m {
                let a = src[base_in + l];
                let b = if l == 0 {
                    src[base_in + l + stride]
                } else {
                    src[base_in + l + stride] * tw1[l]
                };
                dst[base_out + l] = a + b;
                dst[base_out + m + l] = a - b;
                l += 1;
            }
        }
    }
}

/// AVX2/FMA radix-4 stage for one `Complex<f64>` FFT in interleaved (AoS)
/// layout: raw loads/stores, no deinterleave shuffles.
///
/// # Safety
///
/// The caller must provide AVX2 and FMA at runtime. `stage_tw` must
/// describe a radix-4 stage with `src.len() == dst.len() == 4 *
/// stage_tw.m * stage_tw.s`, `stride == src.len() / 4`, and disjoint
/// source/destination slices.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn stage_radix4_f64_aos(
    stride: usize,
    src: &[Complex<f64>],
    dst: &mut [Complex<f64>],
    stage_tw: &StageTwiddles<f64>,
    dir_sign: f64,
) {
    unsafe {
        stage_tw.debug_assert_stage_shape(4, stride, src.len(), dst.len());
        let m = stage_tw.m;
        let s = stage_tw.s;
        let m_new = m * 4;
        let id_sign = _mm256_set_pd(dir_sign, -dir_sign, dir_sign, -dir_sign);
        let tw1 = stage_tw.k_slice(1);
        let tw2 = stage_tw.k_slice(2);
        let tw3 = stage_tw.k_slice(3);

        for g in 0..s {
            let base_in = g * m;
            let base_out = g * m_new;

            let mut l = 0usize;
            while l + 2 <= m {
                let x0 = _mm256_loadu_pd(src.as_ptr().add(base_in + l) as *const f64);
                let x1 = _mm256_loadu_pd(src.as_ptr().add(base_in + l + stride) as *const f64);
                let x2 = _mm256_loadu_pd(src.as_ptr().add(base_in + l + 2 * stride) as *const f64);
                let x3 = _mm256_loadu_pd(src.as_ptr().add(base_in + l + 3 * stride) as *const f64);
                let x1 = cmul_aos_f64(x1, _mm256_loadu_pd(tw1.as_ptr().add(l) as *const f64));
                let x2 = cmul_aos_f64(x2, _mm256_loadu_pd(tw2.as_ptr().add(l) as *const f64));
                let x3 = cmul_aos_f64(x3, _mm256_loadu_pd(tw3.as_ptr().add(l) as *const f64));

                let a = _mm256_add_pd(x0, x2);
                let b = _mm256_sub_pd(x0, x2);
                let c = _mm256_add_pd(x1, x3);
                let d = _mm256_sub_pd(x1, x3);
                let id = _mm256_mul_pd(_mm256_shuffle_pd(d, d, 0x05), id_sign);

                _mm256_storeu_pd(
                    dst.as_mut_ptr().add(base_out + l) as *mut f64,
                    _mm256_add_pd(a, c),
                );
                _mm256_storeu_pd(
                    dst.as_mut_ptr().add(base_out + m + l) as *mut f64,
                    _mm256_add_pd(b, id),
                );
                _mm256_storeu_pd(
                    dst.as_mut_ptr().add(base_out + 2 * m + l) as *mut f64,
                    _mm256_sub_pd(a, c),
                );
                _mm256_storeu_pd(
                    dst.as_mut_ptr().add(base_out + 3 * m + l) as *mut f64,
                    _mm256_sub_pd(b, id),
                );
                l += 2;
            }
            while l < m {
                let x0 = src[base_in + l];
                let (x1, x2, x3) = if l == 0 {
                    (
                        src[base_in + l + stride],
                        src[base_in + l + 2 * stride],
                        src[base_in + l + 3 * stride],
                    )
                } else {
                    (
                        src[base_in + l + stride] * tw1[l],
                        src[base_in + l + 2 * stride] * tw2[l],
                        src[base_in + l + 3 * stride] * tw3[l],
                    )
                };
                let a = x0 + x2;
                let b = x0 - x2;
                let c = x1 + x3;
                let d = x1 - x3;
                let id = Complex::new(-d.im * dir_sign, d.re * dir_sign);
                dst[base_out + l] = a + c;
                dst[base_out + m + l] = b + id;
                dst[base_out + 2 * m + l] = a - c;
                dst[base_out + 3 * m + l] = b - id;
                l += 1;
            }
        }
    }
}




/// AVX2/FMA radix-4 last-stage (`s == 1`) for one `Complex<f64>` FFT in
/// interleaved (AoS) layout.
///
/// # Safety
///
/// The caller must provide AVX2 and FMA at runtime. `stage_tw.s` must equal
/// one, `src.len() == dst.len() == 4 * stage_tw.m`, and source and
/// destination slices must be disjoint.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn stage_radix4_f64_s1_aos(
    src: &[Complex<f64>],
    dst: &mut [Complex<f64>],
    stage_tw: &StageTwiddles<f64>,
    dir_sign: f64,
) {
    unsafe {
        let m = stage_tw.m;
        debug_assert_eq!(stage_tw.s, 1);
        let id_sign = _mm256_set_pd(dir_sign, -dir_sign, dir_sign, -dir_sign);
        let tw1 = stage_tw.k_slice(1);
        let tw2 = stage_tw.k_slice(2);
        let tw3 = stage_tw.k_slice(3);

        let mut l = 0usize;
        while l + 2 <= m {
            let x0 = _mm256_loadu_pd(src.as_ptr().add(l) as *const f64);
            let x1 = _mm256_loadu_pd(src.as_ptr().add(l + m) as *const f64);
            let x2 = _mm256_loadu_pd(src.as_ptr().add(l + 2 * m) as *const f64);
            let x3 = _mm256_loadu_pd(src.as_ptr().add(l + 3 * m) as *const f64);
            let x1 = cmul_aos_f64(x1, _mm256_loadu_pd(tw1.as_ptr().add(l) as *const f64));
            let x2 = cmul_aos_f64(x2, _mm256_loadu_pd(tw2.as_ptr().add(l) as *const f64));
            let x3 = cmul_aos_f64(x3, _mm256_loadu_pd(tw3.as_ptr().add(l) as *const f64));

            let a = _mm256_add_pd(x0, x2);
            let b = _mm256_sub_pd(x0, x2);
            let c = _mm256_add_pd(x1, x3);
            let d = _mm256_sub_pd(x1, x3);
            let id = _mm256_mul_pd(_mm256_shuffle_pd(d, d, 0x05), id_sign);

            _mm256_storeu_pd(dst.as_mut_ptr().add(l) as *mut f64, _mm256_add_pd(a, c));
            _mm256_storeu_pd(
                dst.as_mut_ptr().add(m + l) as *mut f64,
                _mm256_add_pd(b, id),
            );
            _mm256_storeu_pd(
                dst.as_mut_ptr().add(2 * m + l) as *mut f64,
                _mm256_sub_pd(a, c),
            );
            _mm256_storeu_pd(
                dst.as_mut_ptr().add(3 * m + l) as *mut f64,
                _mm256_sub_pd(b, id),
            );
            l += 2;
        }
        while l < m {
            let x0 = src[l];
            let (x1, x2, x3) = if l == 0 {
                (src[l + m], src[l + 2 * m], src[l + 3 * m])
            } else {
                (
                    src[l + m] * tw1[l],
                    src[l + 2 * m] * tw2[l],
                    src[l + 3 * m] * tw3[l],
                )
            };
            let a = x0 + x2;
            let b = x0 - x2;
            let c = x1 + x3;
            let d = x1 - x3;
            let id = Complex::new(-d.im * dir_sign, d.re * dir_sign);
            dst[l] = a + c;
            dst[m + l] = b + id;
            dst[2 * m + l] = a - c;
            dst[3 * m + l] = b - id;
            l += 1;
        }
    }
}

/// AoS radix-4 butterfly over two interleaved `Complex<f64>` lanes.
#[inline]
unsafe fn radix4_aos_f64(
    x0: __m256d,
    x1: __m256d,
    x2: __m256d,
    x3: __m256d,
    id_sign: __m256d,
) -> (__m256d, __m256d, __m256d, __m256d) {
    unsafe {
        let a = _mm256_add_pd(x0, x2);
        let b = _mm256_sub_pd(x0, x2);
        let c = _mm256_add_pd(x1, x3);
        let d = _mm256_sub_pd(x1, x3);
        let id = _mm256_mul_pd(_mm256_shuffle_pd(d, d, 0x05), id_sign);
        (
            _mm256_add_pd(a, c),
            _mm256_add_pd(b, id),
            _mm256_sub_pd(a, c),
            _mm256_sub_pd(b, id),
        )
    }
}

/// AVX2/FMA fused final pair: a radix-4 stage (`stage_a`, two or four
/// groups) followed by the closing radix-2/radix-4 stage (`stage_b`,
/// `s == 1`), executed in place on `buffer`. Mathematically identical to
/// running the two Stockham stages back to back; the intermediate values
/// never leave registers.
///
/// # Safety
///
/// The caller must provide AVX2 and FMA at runtime. `buffer.len()` must
/// equal `4 * stage_a.m * stage_a.s`, `stage_b` must be the Stockham
/// successor of `stage_a`, and `stage_a.radix == 4`.
#[allow(clippy::needless_range_loop)]
#[target_feature(enable = "avx2,fma")]
unsafe fn fused_final_pair_f64_core(
    src: *const Complex<f64>,
    dst: *mut Complex<f64>,
    stage_a: &StageTwiddles<f64>,
    stage_b: &StageTwiddles<f64>,
    dir_sign: f64,
) {
    unsafe {
        let m_a = stage_a.m;
        let s_a = stage_a.s;
        let n = 4 * m_a * s_a;
        debug_assert_eq!(stage_a.radix, 4);
        debug_assert_eq!(n, 4 * m_a * s_a);
        debug_assert_eq!(stage_b.s, 1);
        let stride = n / 4;
        let id_sign = _mm256_set_pd(dir_sign, -dir_sign, dir_sign, -dir_sign);
        let tw_a1 = stage_a.k_slice(1);
        let tw_a2 = stage_a.k_slice(2);
        let tw_a3 = stage_a.k_slice(3);

        let mut l = 0usize;
        while l + 2 <= m_a {
            // Stage-A points for each group a, vectorised over (l, l+1).
            let mut p = [[_mm256_setzero_pd(); 4]; 4];
            for (a, point) in p.iter_mut().enumerate().take(s_a) {
                let off = a * m_a + l;
                let x0 = _mm256_loadu_pd(src.add(off) as *const f64);
                let x1 = _mm256_loadu_pd(src.add(off + stride) as *const f64);
                let x2 = _mm256_loadu_pd(src.add(off + 2 * stride) as *const f64);
                let x3 = _mm256_loadu_pd(src.add(off + 3 * stride) as *const f64);
                let x1 = cmul_aos_f64(x1, _mm256_loadu_pd(tw_a1.as_ptr().add(l) as *const f64));
                let x2 = cmul_aos_f64(x2, _mm256_loadu_pd(tw_a2.as_ptr().add(l) as *const f64));
                let x3 = cmul_aos_f64(x3, _mm256_loadu_pd(tw_a3.as_ptr().add(l) as *const f64));
                let (y0, y1, y2, y3) = radix4_aos_f64(x0, x1, x2, x3, id_sign);
                *point = [y0, y1, y2, y3];
            }

            if stage_b.radix == 4 {
                let tw_b1 = stage_b.k_slice(1);
                let tw_b2 = stage_b.k_slice(2);
                let tw_b3 = stage_b.k_slice(3);
                for c in 0..4 {
                    let off = c * m_a + l;
                    let x0 = p[0][c];
                    let x1 = cmul_aos_f64(
                        p[1][c],
                        _mm256_loadu_pd(tw_b1.as_ptr().add(off) as *const f64),
                    );
                    let x2 = cmul_aos_f64(
                        p[2][c],
                        _mm256_loadu_pd(tw_b2.as_ptr().add(off) as *const f64),
                    );
                    let x3 = cmul_aos_f64(
                        p[3][c],
                        _mm256_loadu_pd(tw_b3.as_ptr().add(off) as *const f64),
                    );
                    let (y0, y1, y2, y3) = radix4_aos_f64(x0, x1, x2, x3, id_sign);
                    _mm256_storeu_pd(dst.add(off) as *mut f64, y0);
                    _mm256_storeu_pd(dst.add(off + 4 * m_a) as *mut f64, y1);
                    _mm256_storeu_pd(dst.add(off + 8 * m_a) as *mut f64, y2);
                    _mm256_storeu_pd(dst.add(off + 12 * m_a) as *mut f64, y3);
                }
            } else {
                debug_assert_eq!(stage_b.radix, 2);
                debug_assert_eq!(s_a, 2);
                let tw_b = stage_b.k_slice(1);
                for c in 0..4 {
                    let off = c * m_a + l;
                    let b = cmul_aos_f64(
                        p[1][c],
                        _mm256_loadu_pd(tw_b.as_ptr().add(off) as *const f64),
                    );
                    let a = p[0][c];
                    _mm256_storeu_pd(dst.add(off) as *mut f64, _mm256_add_pd(a, b));
                    _mm256_storeu_pd(dst.add(off + n / 2) as *mut f64, _mm256_sub_pd(a, b));
                }
            }
            l += 2;
        }
        while l < m_a {
            // Scalar tail: identical math on one lane.
            let mut p = [[Complex::new(0.0, 0.0); 4]; 4];
            for (a, point) in p.iter_mut().enumerate().take(s_a) {
                let off = a * m_a + l;
                let x0 = *src.add(off);
                let x1 = *src.add(off + stride) * tw_a1[l];
                let x2 = *src.add(off + 2 * stride) * tw_a2[l];
                let x3 = *src.add(off + 3 * stride) * tw_a3[l];
                let a_ = x0 + x2;
                let b_ = x0 - x2;
                let c_ = x1 + x3;
                let d_ = x1 - x3;
                let id = Complex::new(-d_.im * dir_sign, d_.re * dir_sign);
                *point = [a_ + c_, b_ + id, a_ - c_, b_ - id];
            }
            if stage_b.radix == 4 {
                let tw_b1 = stage_b.k_slice(1);
                let tw_b2 = stage_b.k_slice(2);
                let tw_b3 = stage_b.k_slice(3);
                for c in 0..4 {
                    let off = c * m_a + l;
                    let x0 = p[0][c];
                    let x1 = p[1][c] * tw_b1[off];
                    let x2 = p[2][c] * tw_b2[off];
                    let x3 = p[3][c] * tw_b3[off];
                    let a_ = x0 + x2;
                    let b_ = x0 - x2;
                    let c_ = x1 + x3;
                    let d_ = x1 - x3;
                    let id = Complex::new(-d_.im * dir_sign, d_.re * dir_sign);
                    *dst.add(off) = a_ + c_;
                    *dst.add(off + 4 * m_a) = b_ + id;
                    *dst.add(off + 8 * m_a) = a_ - c_;
                    *dst.add(off + 12 * m_a) = b_ - id;
                }
            } else {
                let tw_b = stage_b.k_slice(1);
                for c in 0..4 {
                    let off = c * m_a + l;
                    let b = p[1][c] * tw_b[off];
                    *dst.add(off) = p[0][c] + b;
                    *dst.add(off + n / 2) = p[0][c] - b;
                }
            }
            l += 1;
        }
    }
}

#[target_feature(enable = "avx2,fma")]
pub unsafe fn fused_final_pair_f64(
    buffer: &mut [Complex<f64>],
    stage_a: &StageTwiddles<f64>,
    stage_b: &StageTwiddles<f64>,
    dir_sign: f64,
) {
    unsafe {
        fused_final_pair_f64_core(
            buffer.as_ptr(),
            buffer.as_mut_ptr(),
            stage_a,
            stage_b,
            dir_sign,
        );
    }
}


/// AVX2/FMA fused first pair: the `m == 1` radix-4 stage plus its radix-4
/// successor (`m == 4`), out of place. Mathematically identical to running
/// the two Stockham stages back to back.
///
/// # Safety
///
/// The caller must provide AVX2 and FMA at runtime. `stage0` must describe
/// a radix-4 stage with `m == 1`, `stage1` its radix-4 successor with
/// `m == 4`, and source/destination slices must be disjoint with equal
/// length `4 * stage0.s`.
#[allow(clippy::needless_range_loop)]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn fused_first2_f64(
    src: &[Complex<f64>],
    dst: &mut [Complex<f64>],
    stage0: &StageTwiddles<f64>,
    stage1: &StageTwiddles<f64>,
    dir_sign: f64,
) {
    unsafe {
        debug_assert_eq!(stage0.radix, 4);
        debug_assert_eq!(stage0.m, 1);
        debug_assert_eq!(stage1.radix, 4);
        debug_assert_eq!(stage1.m, 4);
        debug_assert_eq!(stage0.s, stage1.s * 4);
        let groups = stage1.s;
        let stride = stage0.s;
        let id_sign = _mm256_set_pd(dir_sign, -dir_sign, dir_sign, -dir_sign);
        let tw1 = stage1.k_slice(1);
        let tw2 = stage1.k_slice(2);
        let tw3 = stage1.k_slice(3);
        // Stage-1 twiddles are scalars per (k, l); broadcast once.
        let mut w1 = [_mm256_setzero_pd(); 4];
        let mut w2 = [_mm256_setzero_pd(); 4];
        let mut w3 = [_mm256_setzero_pd(); 4];
        for l in 0..4 {
            w1[l] = _mm256_set_pd(tw1[l].im, tw1[l].re, tw1[l].im, tw1[l].re);
            w2[l] = _mm256_set_pd(tw2[l].im, tw2[l].re, tw2[l].im, tw2[l].re);
            w3[l] = _mm256_set_pd(tw3[l].im, tw3[l].re, tw3[l].im, tw3[l].re);
        }
        let w = [w1, w2, w3];

        let mut g = 0usize;
        while g + 2 <= groups {
            // Stage 0 (twiddle-free) for the four A-groups, two groups per
            // register over g.
            let mut a = [[_mm256_setzero_pd(); 4]; 4];
            for k in 0..4 {
                let off = g + k * groups;
                let x0 = _mm256_loadu_pd(src.as_ptr().add(off) as *const f64);
                let x1 = _mm256_loadu_pd(src.as_ptr().add(off + stride) as *const f64);
                let x2 = _mm256_loadu_pd(src.as_ptr().add(off + 2 * stride) as *const f64);
                let x3 = _mm256_loadu_pd(src.as_ptr().add(off + 3 * stride) as *const f64);
                let (y0, y1, y2, y3) = radix4_aos_f64(x0, x1, x2, x3, id_sign);
                a[k] = [y0, y1, y2, y3];
            }
            // Stage 1: butterfly over k per component l. Each output
            // register holds groups (g, g + 1) for one (b, l); the two
            // groups land sixteen elements apart in `dst`.
            for l in 0..4 {
                let x0 = a[0][l];
                let x1 = cmul_aos_f64(a[1][l], w[0][l]);
                let x2 = cmul_aos_f64(a[2][l], w[1][l]);
                let x3 = cmul_aos_f64(a[3][l], w[2][l]);
                let ys = radix4_aos_f64(x0, x1, x2, x3, id_sign);
                for (b, y) in [ys.0, ys.1, ys.2, ys.3].iter().enumerate() {
                    let out0 = dst.as_mut_ptr().add(g * 16 + b * 4 + l) as *mut f64;
                    let out1 = dst.as_mut_ptr().add((g + 1) * 16 + b * 4 + l) as *mut f64;
                    _mm_storeu_pd(out0, _mm256_castpd256_pd128(*y));
                    _mm_storeu_pd(out1, _mm256_extractf128_pd(*y, 1));
                }
            }
            g += 2;
        }
        while g < groups {
            let h0 = g;
            let h1 = g + groups;
            let h2 = g + 2 * groups;
            let h3 = g + 3 * groups;
            let mut a = [[Complex::new(0.0, 0.0); 4]; 4];
            for (k, h) in [h0, h1, h2, h3].iter().enumerate() {
                let x0 = src[*h];
                let x1 = src[*h + stride];
                let x2 = src[*h + 2 * stride];
                let x3 = src[*h + 3 * stride];
                let a_ = x0 + x2;
                let b_ = x0 - x2;
                let c_ = x1 + x3;
                let d_ = x1 - x3;
                let id = Complex::new(-d_.im * dir_sign, d_.re * dir_sign);
                a[k] = [a_ + c_, b_ + id, a_ - c_, b_ - id];
            }
            let base = g * 16;
            for l in 0..4 {
                let x0 = a[0][l];
                let x1 = a[1][l] * tw1[l];
                let x2 = a[2][l] * tw2[l];
                let x3 = a[3][l] * tw3[l];
                let a_ = x0 + x2;
                let b_ = x0 - x2;
                let c_ = x1 + x3;
                let d_ = x1 - x3;
                let id = Complex::new(-d_.im * dir_sign, d_.re * dir_sign);
                dst[base + l] = a_ + c_;
                dst[base + 4 + l] = b_ + id;
                dst[base + 8 + l] = a_ - c_;
                dst[base + 12 + l] = b_ - id;
            }
            g += 1;
        }
    }
}

/// AVX2/FMA fused first pair (f32): the `m == 1` radix-4 stage plus its
/// radix-4 successor (`m == 4`), out of place. f32 counterpart of
/// [`fused_first2_f64`], vectorised over four groups per register.
///
/// # Safety
///
/// The caller must provide AVX2 and FMA at runtime. `stage0` must describe
/// a radix-4 stage with `m == 1`, `stage1` its radix-4 successor with
/// `m == 4`, and source/destination slices must be disjoint with equal
/// length `4 * stage0.s`.
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::needless_range_loop)]
pub unsafe fn fused_first2_f32(
    src: &[Complex<f32>],
    dst: &mut [Complex<f32>],
    stage0: &StageTwiddles<f32>,
    stage1: &StageTwiddles<f32>,
    dir_sign: f32,
) {
    unsafe {
        debug_assert_eq!(stage0.radix, 4);
        debug_assert_eq!(stage0.m, 1);
        debug_assert_eq!(stage1.radix, 4);
        debug_assert_eq!(stage1.m, 4);
        debug_assert_eq!(stage0.s, stage1.s * 4);
        let groups = stage1.s;
        let stride = stage0.s;
        let id_sign = _mm256_set_ps(
            dir_sign, -dir_sign, dir_sign, -dir_sign, dir_sign, -dir_sign, dir_sign, -dir_sign,
        );
        let tw1 = stage1.k_slice(1);
        let tw2 = stage1.k_slice(2);
        let tw3 = stage1.k_slice(3);
        // Stage-1 twiddles are scalars per (k, l); broadcast once.
        let mut w1 = [_mm256_setzero_ps(); 4];
        let mut w2 = [_mm256_setzero_ps(); 4];
        let mut w3 = [_mm256_setzero_ps(); 4];
        for l in 0..4 {
            w1[l] = _mm256_set_ps(tw1[l].im, tw1[l].re, tw1[l].im, tw1[l].re, tw1[l].im, tw1[l].re, tw1[l].im, tw1[l].re);
            w2[l] = _mm256_set_ps(tw2[l].im, tw2[l].re, tw2[l].im, tw2[l].re, tw2[l].im, tw2[l].re, tw2[l].im, tw2[l].re);
            w3[l] = _mm256_set_ps(tw3[l].im, tw3[l].re, tw3[l].im, tw3[l].re, tw3[l].im, tw3[l].re, tw3[l].im, tw3[l].re);
        }
        let w = [w1, w2, w3];

        let mut g = 0usize;
        while g + 4 <= groups {
            // Stage 0 (twiddle-free) for the four A-groups, four groups per
            // register over g.
            let mut a = [[_mm256_setzero_ps(); 4]; 4];
            for k in 0..4 {
                let off = g + k * groups;
                let x0 = _mm256_loadu_ps(src.as_ptr().add(off) as *const f32);
                let x1 = _mm256_loadu_ps(src.as_ptr().add(off + stride) as *const f32);
                let x2 = _mm256_loadu_ps(src.as_ptr().add(off + 2 * stride) as *const f32);
                let x3 = _mm256_loadu_ps(src.as_ptr().add(off + 3 * stride) as *const f32);
                let (y0, y1, y2, y3) = radix4_aos_f32(x0, x1, x2, x3, id_sign);
                a[k] = [y0, y1, y2, y3];
            }
            // Stage 1: butterfly over k per component l. Each output
            // register holds groups g..g+4 for one (b, l); the four groups
            // land sixteen elements apart in `dst`.
            for l in 0..4 {
                let x0 = a[0][l];
                let x1 = cmul_aos_f32(a[1][l], w[0][l]);
                let x2 = cmul_aos_f32(a[2][l], w[1][l]);
                let x3 = cmul_aos_f32(a[3][l], w[2][l]);
                let ys = radix4_aos_f32(x0, x1, x2, x3, id_sign);
                for (b, y) in [ys.0, ys.1, ys.2, ys.3].iter().enumerate() {
                    let mut tmp = [0.0f32; 8];
                    _mm256_storeu_ps(tmp.as_mut_ptr(), *y);
                    for j in 0..4 {
                        dst[(g + j) * 16 + b * 4 + l] =
                            Complex::new(tmp[2 * j], tmp[2 * j + 1]);
                    }
                }
            }
            g += 4;
        }
        while g < groups {
            let h0 = g;
            let h1 = g + groups;
            let h2 = g + 2 * groups;
            let h3 = g + 3 * groups;
            let mut a = [[Complex::new(0.0, 0.0); 4]; 4];
            for (k, h) in [h0, h1, h2, h3].iter().enumerate() {
                let x0 = src[*h];
                let x1 = src[*h + stride];
                let x2 = src[*h + 2 * stride];
                let x3 = src[*h + 3 * stride];
                let a_ = x0 + x2;
                let b_ = x0 - x2;
                let c_ = x1 + x3;
                let d_ = x1 - x3;
                let id = Complex::new(-d_.im * dir_sign, d_.re * dir_sign);
                a[k] = [a_ + c_, b_ + id, a_ - c_, b_ - id];
            }
            let base = g * 16;
            for l in 0..4 {
                let x0 = a[0][l];
                let x1 = a[1][l] * tw1[l];
                let x2 = a[2][l] * tw2[l];
                let x3 = a[3][l] * tw3[l];
                let a_ = x0 + x2;
                let b_ = x0 - x2;
                let c_ = x1 + x3;
                let d_ = x1 - x3;
                let id = Complex::new(-d_.im * dir_sign, d_.re * dir_sign);
                dst[base + l] = a_ + c_;
                dst[base + 4 + l] = b_ + id;
                dst[base + 8 + l] = a_ - c_;
                dst[base + 12 + l] = b_ - id;
            }
            g += 1;
        }
    }
}

/// AoS radix-4 butterfly over four interleaved `Complex<f32>` lanes.
#[inline]
unsafe fn radix4_aos_f32(
    x0: __m256,
    x1: __m256,
    x2: __m256,
    x3: __m256,
    id_sign: __m256,
) -> (__m256, __m256, __m256, __m256) {
    unsafe {
        let a = _mm256_add_ps(x0, x2);
        let b = _mm256_sub_ps(x0, x2);
        let c = _mm256_add_ps(x1, x3);
        let d = _mm256_sub_ps(x1, x3);
        let id = _mm256_mul_ps(_mm256_shuffle_ps(d, d, 0xB1), id_sign);
        (
            _mm256_add_ps(a, c),
            _mm256_add_ps(b, id),
            _mm256_sub_ps(a, c),
            _mm256_sub_ps(b, id),
        )
    }
}

/// AVX2/FMA fused final pair (f32): a radix-4 stage (`stage_a`, two or
/// four groups) followed by the closing radix-2/radix-4 stage (`stage_b`,
/// `s == 1`), executed in place on `buffer`. See
/// [`fused_final_pair_f64`] for the contract.
///
/// # Safety
///
/// The caller must provide AVX2 and FMA at runtime, `buffer.len()` must
/// equal `4 * stage_a.m * stage_a.s`, `stage_b` must be the Stockham
/// successor of `stage_a`, and `stage_a.radix == 4`.
#[allow(clippy::needless_range_loop)]
#[target_feature(enable = "avx2,fma")]
unsafe fn fused_final_pair_f32_core(
    src: *const Complex<f32>,
    dst: *mut Complex<f32>,
    stage_a: &StageTwiddles<f32>,
    stage_b: &StageTwiddles<f32>,
    dir_sign: f32,
) {
    unsafe {
        let m_a = stage_a.m;
        let s_a = stage_a.s;
        let n = 4 * m_a * s_a;
        debug_assert_eq!(stage_a.radix, 4);
        debug_assert_eq!(n, 4 * m_a * s_a);
        debug_assert_eq!(stage_b.s, 1);
        let stride = n / 4;
        let id_sign = _mm256_set_ps(
            dir_sign, -dir_sign, dir_sign, -dir_sign, dir_sign, -dir_sign, dir_sign, -dir_sign,
        );
        let tw_a1 = stage_a.k_slice(1);
        let tw_a2 = stage_a.k_slice(2);
        let tw_a3 = stage_a.k_slice(3);

        let mut l = 0usize;
        while l + 4 <= m_a {
            let mut p = [[_mm256_setzero_ps(); 4]; 4];
            for (a, point) in p.iter_mut().enumerate().take(s_a) {
                let off = a * m_a + l;
                let x0 = _mm256_loadu_ps(src.add(off) as *const f32);
                let x1 = _mm256_loadu_ps(src.add(off + stride) as *const f32);
                let x2 = _mm256_loadu_ps(src.add(off + 2 * stride) as *const f32);
                let x3 = _mm256_loadu_ps(src.add(off + 3 * stride) as *const f32);
                let x1 = cmul_aos_f32(x1, _mm256_loadu_ps(tw_a1.as_ptr().add(l) as *const f32));
                let x2 = cmul_aos_f32(x2, _mm256_loadu_ps(tw_a2.as_ptr().add(l) as *const f32));
                let x3 = cmul_aos_f32(x3, _mm256_loadu_ps(tw_a3.as_ptr().add(l) as *const f32));
                let (y0, y1, y2, y3) = radix4_aos_f32(x0, x1, x2, x3, id_sign);
                *point = [y0, y1, y2, y3];
            }

            if stage_b.radix == 4 {
                let tw_b1 = stage_b.k_slice(1);
                let tw_b2 = stage_b.k_slice(2);
                let tw_b3 = stage_b.k_slice(3);
                for c in 0..4 {
                    let off = c * m_a + l;
                    let x0 = p[0][c];
                    let x1 = cmul_aos_f32(
                        p[1][c],
                        _mm256_loadu_ps(tw_b1.as_ptr().add(off) as *const f32),
                    );
                    let x2 = cmul_aos_f32(
                        p[2][c],
                        _mm256_loadu_ps(tw_b2.as_ptr().add(off) as *const f32),
                    );
                    let x3 = cmul_aos_f32(
                        p[3][c],
                        _mm256_loadu_ps(tw_b3.as_ptr().add(off) as *const f32),
                    );
                    let (y0, y1, y2, y3) = radix4_aos_f32(x0, x1, x2, x3, id_sign);
                    _mm256_storeu_ps(dst.add(off) as *mut f32, y0);
                    _mm256_storeu_ps(dst.add(off + 4 * m_a) as *mut f32, y1);
                    _mm256_storeu_ps(dst.add(off + 8 * m_a) as *mut f32, y2);
                    _mm256_storeu_ps(dst.add(off + 12 * m_a) as *mut f32, y3);
                }
            } else {
                debug_assert_eq!(stage_b.radix, 2);
                debug_assert_eq!(s_a, 2);
                let tw_b = stage_b.k_slice(1);
                for c in 0..4 {
                    let off = c * m_a + l;
                    let b = cmul_aos_f32(
                        p[1][c],
                        _mm256_loadu_ps(tw_b.as_ptr().add(off) as *const f32),
                    );
                    let a = p[0][c];
                    _mm256_storeu_ps(dst.add(off) as *mut f32, _mm256_add_ps(a, b));
                    _mm256_storeu_ps(dst.add(off + n / 2) as *mut f32, _mm256_sub_ps(a, b));
                }
            }
            l += 4;
        }
        while l < m_a {
            let mut p = [[Complex::new(0.0, 0.0); 4]; 4];
            for (a, point) in p.iter_mut().enumerate().take(s_a) {
                let off = a * m_a + l;
                let x0 = *src.add(off);
                let x1 = *src.add(off + stride) * tw_a1[l];
                let x2 = *src.add(off + 2 * stride) * tw_a2[l];
                let x3 = *src.add(off + 3 * stride) * tw_a3[l];
                let a_ = x0 + x2;
                let b_ = x0 - x2;
                let c_ = x1 + x3;
                let d_ = x1 - x3;
                let id = Complex::new(-d_.im * dir_sign, d_.re * dir_sign);
                *point = [a_ + c_, b_ + id, a_ - c_, b_ - id];
            }
            if stage_b.radix == 4 {
                let tw_b1 = stage_b.k_slice(1);
                let tw_b2 = stage_b.k_slice(2);
                let tw_b3 = stage_b.k_slice(3);
                for c in 0..4 {
                    let off = c * m_a + l;
                    let x0 = p[0][c];
                    let x1 = p[1][c] * tw_b1[off];
                    let x2 = p[2][c] * tw_b2[off];
                    let x3 = p[3][c] * tw_b3[off];
                    let a_ = x0 + x2;
                    let b_ = x0 - x2;
                    let c_ = x1 + x3;
                    let d_ = x1 - x3;
                    let id = Complex::new(-d_.im * dir_sign, d_.re * dir_sign);
                    *dst.add(off) = a_ + c_;
                    *dst.add(off + 4 * m_a) = b_ + id;
                    *dst.add(off + 8 * m_a) = a_ - c_;
                    *dst.add(off + 12 * m_a) = b_ - id;
                }
            } else {
                let tw_b = stage_b.k_slice(1);
                for c in 0..4 {
                    let off = c * m_a + l;
                    let b = p[1][c] * tw_b[off];
                    *dst.add(off) = p[0][c] + b;
                    *dst.add(off + n / 2) = p[0][c] - b;
                }
            }
            l += 1;
        }
    }
}

#[target_feature(enable = "avx2,fma")]
pub unsafe fn fused_final_pair_f32(
    buffer: &mut [Complex<f32>],
    stage_a: &StageTwiddles<f32>,
    stage_b: &StageTwiddles<f32>,
    dir_sign: f32,
) {
    unsafe {
        fused_final_pair_f32_core(
            buffer.as_ptr(),
            buffer.as_mut_ptr(),
            stage_a,
            stage_b,
            dir_sign,
        );
    }
}

/// AVX2/FMA radix-4 stage for the `m == 1` case (the first stage of a
/// power-of-two plan): one butterfly per group, no twiddles (the only
/// twiddle is `w^0 = 1`). Four groups ride one AoS register per input
/// stream — the butterfly needs no deinterleave — and a 4×4 register
/// transpose produces the Stockham output order `dst[4g + a]`.
///
/// # Safety
///
/// The caller must provide AVX2 and FMA at runtime. `stage_tw` must
/// describe a radix-4 stage with `stage_tw.m == 1`, and source and
/// destination slices must be disjoint with equal length `4 * s`.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn stage_radix4_f32_m1(
    src: &[Complex<f32>],
    dst: &mut [Complex<f32>],
    stage_tw: &StageTwiddles<f32>,
    dir_sign: f32,
) {
    unsafe {
        debug_assert_eq!(stage_tw.radix, 4);
        debug_assert_eq!(stage_tw.m, 1);
        let s = stage_tw.s;
        let stride = s;
        // Per-complex multiplier turning (d.re, d.im) into
        // (-d.im * dir, d.re * dir) after an in-pair swap.
        let id_sign = _mm256_set_ps(
            dir_sign, -dir_sign, dir_sign, -dir_sign, dir_sign, -dir_sign, dir_sign, -dir_sign,
        );

        let mut g = 0usize;
        while g + 4 <= s {
            let x0 = _mm256_loadu_ps(src.as_ptr().add(g) as *const f32);
            let x1 = _mm256_loadu_ps(src.as_ptr().add(stride + g) as *const f32);
            let x2 = _mm256_loadu_ps(src.as_ptr().add(2 * stride + g) as *const f32);
            let x3 = _mm256_loadu_ps(src.as_ptr().add(3 * stride + g) as *const f32);

            let a = _mm256_add_ps(x0, x2);
            let b = _mm256_sub_ps(x0, x2);
            let c = _mm256_add_ps(x1, x3);
            let d = _mm256_sub_ps(x1, x3);
            // id = (-d.im * dir, d.re * dir) per complex pair.
            let d_swap = _mm256_shuffle_ps(d, d, 0xB1);
            let id = _mm256_mul_ps(d_swap, id_sign);

            let y0 = _mm256_add_ps(a, c);
            let y1 = _mm256_add_ps(b, id);
            let y2 = _mm256_sub_ps(a, c);
            let y3 = _mm256_sub_ps(b, id);

            // Transpose 4 streams × 4 groups (complex = one 64-bit lane):
            // O_j = [Y0(gj) Y1(gj) Y2(gj) Y3(gj)].
            let u0 = _mm256_unpacklo_pd(_mm256_castps_pd(y0), _mm256_castps_pd(y1));
            let u1 = _mm256_unpackhi_pd(_mm256_castps_pd(y0), _mm256_castps_pd(y1));
            let u2 = _mm256_unpacklo_pd(_mm256_castps_pd(y2), _mm256_castps_pd(y3));
            let u3 = _mm256_unpackhi_pd(_mm256_castps_pd(y2), _mm256_castps_pd(y3));
            let out = dst.as_mut_ptr().add(4 * g) as *mut f32;
            _mm256_storeu_ps(out, _mm256_castpd_ps(_mm256_permute2f128_pd(u0, u2, 0x20)));
            _mm256_storeu_ps(
                out.add(8),
                _mm256_castpd_ps(_mm256_permute2f128_pd(u1, u3, 0x20)),
            );
            _mm256_storeu_ps(
                out.add(16),
                _mm256_castpd_ps(_mm256_permute2f128_pd(u0, u2, 0x31)),
            );
            _mm256_storeu_ps(
                out.add(24),
                _mm256_castpd_ps(_mm256_permute2f128_pd(u1, u3, 0x31)),
            );
            g += 4;
        }
        while g < s {
            let x0 = src[g];
            let x1 = src[stride + g];
            let x2 = src[2 * stride + g];
            let x3 = src[3 * stride + g];
            let a = x0 + x2;
            let b = x0 - x2;
            let c = x1 + x3;
            let d = x1 - x3;
            let id = Complex::new(-d.im * dir_sign, d.re * dir_sign);
            dst[4 * g] = a + c;
            dst[4 * g + 1] = b + id;
            dst[4 * g + 2] = a - c;
            dst[4 * g + 3] = b - id;
            g += 1;
        }
    }
}

/// AVX2/FMA radix-4 stage for the `m == 1` case, f64: two groups per
/// register; see [`stage_radix4_f32_m1`] for the layout contract.
///
/// # Safety
///
/// The caller must provide AVX2 and FMA at runtime. `stage_tw` must
/// describe a radix-4 stage with `stage_tw.m == 1`, and source and
/// destination slices must be disjoint with equal length `4 * s`.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn stage_radix4_f64_m1(
    src: &[Complex<f64>],
    dst: &mut [Complex<f64>],
    stage_tw: &StageTwiddles<f64>,
    dir_sign: f64,
) {
    unsafe {
        debug_assert_eq!(stage_tw.radix, 4);
        debug_assert_eq!(stage_tw.m, 1);
        let s = stage_tw.s;
        let stride = s;
        let id_sign = _mm256_set_pd(dir_sign, -dir_sign, dir_sign, -dir_sign);

        let mut g = 0usize;
        while g + 2 <= s {
            let x0 = _mm256_loadu_pd(src.as_ptr().add(g) as *const f64);
            let x1 = _mm256_loadu_pd(src.as_ptr().add(stride + g) as *const f64);
            let x2 = _mm256_loadu_pd(src.as_ptr().add(2 * stride + g) as *const f64);
            let x3 = _mm256_loadu_pd(src.as_ptr().add(3 * stride + g) as *const f64);

            let a = _mm256_add_pd(x0, x2);
            let b = _mm256_sub_pd(x0, x2);
            let c = _mm256_add_pd(x1, x3);
            let d = _mm256_sub_pd(x1, x3);
            let d_swap = _mm256_shuffle_pd(d, d, 0x05);
            let id = _mm256_mul_pd(d_swap, id_sign);

            let y0 = _mm256_add_pd(a, c);
            let y1 = _mm256_add_pd(b, id);
            let y2 = _mm256_sub_pd(a, c);
            let y3 = _mm256_sub_pd(b, id);

            // O = [Y0(g) Y1(g) | Y2(g) Y3(g)] then the same for g + 1:
            // each output register is one group's four results.
            let out = dst.as_mut_ptr().add(4 * g) as *mut f64;
            _mm256_storeu_pd(out, _mm256_permute2f128_pd(y0, y1, 0x20));
            _mm256_storeu_pd(out.add(4), _mm256_permute2f128_pd(y2, y3, 0x20));
            _mm256_storeu_pd(out.add(8), _mm256_permute2f128_pd(y0, y1, 0x31));
            _mm256_storeu_pd(out.add(12), _mm256_permute2f128_pd(y2, y3, 0x31));
            g += 2;
        }
        while g < s {
            let x0 = src[g];
            let x1 = src[stride + g];
            let x2 = src[2 * stride + g];
            let x3 = src[3 * stride + g];
            let a = x0 + x2;
            let b = x0 - x2;
            let c = x1 + x3;
            let d = x1 - x3;
            let id = Complex::new(-d.im * dir_sign, d.re * dir_sign);
            dst[4 * g] = a + c;
            dst[4 * g + 1] = b + id;
            dst[4 * g + 2] = a - c;
            dst[4 * g + 3] = b - id;
            g += 1;
        }
    }
}

/// AVX2/FMA radix-2 stage for two independent f64 FFTs interleaved as
/// `[fft0[0], fft1[0], fft0[1], fft1[1], …]`.
///
/// The logical `stride` and stage dimensions are expressed in elements of
/// one FFT; the physical slices therefore contain twice as many `Complex`
/// values. Broadcasting each logical twiddle keeps both transforms in the
/// two lanes of an SSE register while the surrounding AVX2 stage kernels
/// remain available for single-transform execution.
///
/// # Safety
///
/// The caller must provide AVX2 and FMA at runtime. `stage_tw` must describe
/// a radix-2 stage, `src.len() == dst.len() == 2 * (2 * stage_tw.m *
/// stage_tw.s)`, and the source and destination slices must be disjoint.
/// `stride` must equal the logical distance between the two radix inputs
/// (`2 * stage_tw.m * stage_tw.s / 2`).
#[target_feature(enable = "avx2,fma")]
pub unsafe fn stage_radix2_f64_batched2(
    stride: usize,
    src: &[Complex<f64>],
    dst: &mut [Complex<f64>],
    stage_tw: &StageTwiddles<f64>,
) {
    unsafe {
        let m = stage_tw.m;
        let s = stage_tw.s;
        let m_new = m * 2;
        let tw1 = stage_tw.k_slice(1);

        for g in 0..s {
            let base_in = g * m;
            let base_out = g * m_new;
            for (l, tw) in tw1.iter().copied().enumerate().take(m) {
                let (a_re, a_im) = load_complex2(src.as_ptr().add(2 * (base_in + l)));
                let (b_re, b_im) = load_complex2(src.as_ptr().add(2 * (base_in + l + stride)));
                let tw_re = _mm_set1_pd(tw.re);
                let tw_im = _mm_set1_pd(tw.im);
                let (b_re, b_im) = cmul2(b_re, b_im, tw_re, tw_im);
                let y0_re = _mm_add_pd(a_re, b_re);
                let y0_im = _mm_add_pd(a_im, b_im);
                let y1_re = _mm_sub_pd(a_re, b_re);
                let y1_im = _mm_sub_pd(a_im, b_im);
                store_complex2(dst.as_mut_ptr().add(2 * (base_out + l)), y0_re, y0_im);
                store_complex2(dst.as_mut_ptr().add(2 * (base_out + m + l)), y1_re, y1_im);
            }
        }
    }
}

/// AVX2/FMA radix-4 stage for two independent f64 FFTs interleaved as
/// `[fft0[0], fft1[0], fft0[1], fft1[1], …]`.
///
/// # Safety
///
/// The caller must provide AVX2 and FMA at runtime. `stage_tw` must describe
/// a radix-4 stage, `src.len() == dst.len() == 2 * (4 * stage_tw.m *
/// stage_tw.s)`, and the source and destination slices must be disjoint.
/// `stride` is the logical distance between radix inputs for one FFT.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn stage_radix4_f64_batched2(
    stride: usize,
    src: &[Complex<f64>],
    dst: &mut [Complex<f64>],
    stage_tw: &StageTwiddles<f64>,
    dir_sign: f64,
) {
    unsafe {
        let m = stage_tw.m;
        let s = stage_tw.s;
        let m_new = m * 4;
        let tw1 = stage_tw.k_slice(1);
        let tw2 = stage_tw.k_slice(2);
        let tw3 = stage_tw.k_slice(3);
        let dir_v = _mm_set1_pd(dir_sign);
        let sign_flip = _mm_set1_pd(-0.0);

        for g in 0..s {
            let base_in = g * m;
            let base_out = g * m_new;
            for (l, ((t1, t2), t3)) in tw1
                .iter()
                .copied()
                .zip(tw2.iter().copied())
                .zip(tw3.iter().copied())
                .enumerate()
                .take(m)
            {
                let (x0_re, x0_im) = load_complex2(src.as_ptr().add(2 * (base_in + l)));
                let (mut x1_re, mut x1_im) =
                    load_complex2(src.as_ptr().add(2 * (base_in + l + stride)));
                let (mut x2_re, mut x2_im) =
                    load_complex2(src.as_ptr().add(2 * (base_in + l + 2 * stride)));
                let (mut x3_re, mut x3_im) =
                    load_complex2(src.as_ptr().add(2 * (base_in + l + 3 * stride)));

                (x1_re, x1_im) = cmul2(x1_re, x1_im, _mm_set1_pd(t1.re), _mm_set1_pd(t1.im));
                (x2_re, x2_im) = cmul2(x2_re, x2_im, _mm_set1_pd(t2.re), _mm_set1_pd(t2.im));
                (x3_re, x3_im) = cmul2(x3_re, x3_im, _mm_set1_pd(t3.re), _mm_set1_pd(t3.im));

                let a_re = _mm_add_pd(x0_re, x2_re);
                let a_im = _mm_add_pd(x0_im, x2_im);
                let b_re = _mm_sub_pd(x0_re, x2_re);
                let b_im = _mm_sub_pd(x0_im, x2_im);
                let c_re = _mm_add_pd(x1_re, x3_re);
                let c_im = _mm_add_pd(x1_im, x3_im);
                let d_re = _mm_sub_pd(x1_re, x3_re);
                let d_im = _mm_sub_pd(x1_im, x3_im);
                let id_re = _mm_xor_pd(_mm_mul_pd(d_im, dir_v), sign_flip);
                let id_im = _mm_mul_pd(d_re, dir_v);

                let y0_re = _mm_add_pd(a_re, c_re);
                let y0_im = _mm_add_pd(a_im, c_im);
                let y2_re = _mm_sub_pd(a_re, c_re);
                let y2_im = _mm_sub_pd(a_im, c_im);
                let y1_re = _mm_add_pd(b_re, id_re);
                let y1_im = _mm_add_pd(b_im, id_im);
                let y3_re = _mm_sub_pd(b_re, id_re);
                let y3_im = _mm_sub_pd(b_im, id_im);

                store_complex2(dst.as_mut_ptr().add(2 * (base_out + l)), y0_re, y0_im);
                store_complex2(dst.as_mut_ptr().add(2 * (base_out + m + l)), y1_re, y1_im);
                store_complex2(
                    dst.as_mut_ptr().add(2 * (base_out + 2 * m + l)),
                    y2_re,
                    y2_im,
                );
                store_complex2(
                    dst.as_mut_ptr().add(2 * (base_out + 3 * m + l)),
                    y3_re,
                    y3_im,
                );
            }
        }
    }
}

/// AVX2/FMA radix-2 stage for two independent f32 FFTs interleaved as
/// `[fft0[0], fft1[0], fft0[1], fft1[1], …]`.
///
/// The two logical transforms occupy the two useful lanes in each `__m128`
/// real/imaginary vector. The remaining duplicate lanes make the complex
/// arithmetic uniform while the load/store helpers preserve the public
/// interleaved `Complex<f32>` layout.
///
/// # Safety
///
/// The caller must provide AVX2 and FMA at runtime. `stage_tw` must describe
/// a radix-2 stage, `src.len() == dst.len() == 2 * n` where `n` is the logical
/// FFT length, and the source and destination slices must be disjoint.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn stage_radix2_f32_batched2(
    stride: usize,
    src: &[Complex<f32>],
    dst: &mut [Complex<f32>],
    stage_tw: &StageTwiddles<f32>,
) {
    unsafe {
        let m = stage_tw.m;
        let s = stage_tw.s;
        let m_new = m * 2;
        let tw1 = stage_tw.k_slice(1);

        for g in 0..s {
            let base_in = g * m;
            let base_out = g * m_new;
            for (l, tw) in tw1.iter().copied().enumerate().take(m) {
                let (a_re, a_im) = load_complex2_f32(src.as_ptr().add(2 * (base_in + l)));
                let (b_re, b_im) = load_complex2_f32(src.as_ptr().add(2 * (base_in + l + stride)));
                let (b_re, b_im) = cmul_f32(b_re, b_im, _mm_set1_ps(tw.re), _mm_set1_ps(tw.im));
                let y0_re = _mm_add_ps(a_re, b_re);
                let y0_im = _mm_add_ps(a_im, b_im);
                let y1_re = _mm_sub_ps(a_re, b_re);
                let y1_im = _mm_sub_ps(a_im, b_im);
                store_complex2_f32(dst.as_mut_ptr().add(2 * (base_out + l)), y0_re, y0_im);
                store_complex2_f32(dst.as_mut_ptr().add(2 * (base_out + m + l)), y1_re, y1_im);
            }
        }
    }
}

/// AVX2/FMA radix-4 stage for two independent f32 FFTs in the same
/// interleaved layout as [`stage_radix2_f32_batched2`].
///
/// # Safety
///
/// The caller must provide AVX2 and FMA at runtime. `stage_tw` must describe
/// a radix-4 stage, `src.len() == dst.len() == 2 * n`, and source and
/// destination slices must be disjoint.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn stage_radix4_f32_batched2(
    stride: usize,
    src: &[Complex<f32>],
    dst: &mut [Complex<f32>],
    stage_tw: &StageTwiddles<f32>,
    dir_sign: f32,
) {
    unsafe {
        let m = stage_tw.m;
        let s = stage_tw.s;
        let m_new = m * 4;
        let tw1 = stage_tw.k_slice(1);
        let tw2 = stage_tw.k_slice(2);
        let tw3 = stage_tw.k_slice(3);
        let dir_v = _mm_set1_ps(dir_sign);
        let sign_flip = _mm_set1_ps(-0.0);

        for g in 0..s {
            let base_in = g * m;
            let base_out = g * m_new;
            for (l, ((t1, t2), t3)) in tw1
                .iter()
                .copied()
                .zip(tw2.iter().copied())
                .zip(tw3.iter().copied())
                .enumerate()
                .take(m)
            {
                let (x0_re, x0_im) = load_complex2_f32(src.as_ptr().add(2 * (base_in + l)));
                let (mut x1_re, mut x1_im) =
                    load_complex2_f32(src.as_ptr().add(2 * (base_in + l + stride)));
                let (mut x2_re, mut x2_im) =
                    load_complex2_f32(src.as_ptr().add(2 * (base_in + l + 2 * stride)));
                let (mut x3_re, mut x3_im) =
                    load_complex2_f32(src.as_ptr().add(2 * (base_in + l + 3 * stride)));

                (x1_re, x1_im) = cmul_f32(x1_re, x1_im, _mm_set1_ps(t1.re), _mm_set1_ps(t1.im));
                (x2_re, x2_im) = cmul_f32(x2_re, x2_im, _mm_set1_ps(t2.re), _mm_set1_ps(t2.im));
                (x3_re, x3_im) = cmul_f32(x3_re, x3_im, _mm_set1_ps(t3.re), _mm_set1_ps(t3.im));

                let a_re = _mm_add_ps(x0_re, x2_re);
                let a_im = _mm_add_ps(x0_im, x2_im);
                let b_re = _mm_sub_ps(x0_re, x2_re);
                let b_im = _mm_sub_ps(x0_im, x2_im);
                let c_re = _mm_add_ps(x1_re, x3_re);
                let c_im = _mm_add_ps(x1_im, x3_im);
                let d_re = _mm_sub_ps(x1_re, x3_re);
                let d_im = _mm_sub_ps(x1_im, x3_im);
                let id_re = _mm_xor_ps(_mm_mul_ps(d_im, dir_v), sign_flip);
                let id_im = _mm_mul_ps(d_re, dir_v);

                let y0_re = _mm_add_ps(a_re, c_re);
                let y0_im = _mm_add_ps(a_im, c_im);
                let y2_re = _mm_sub_ps(a_re, c_re);
                let y2_im = _mm_sub_ps(a_im, c_im);
                let y1_re = _mm_add_ps(b_re, id_re);
                let y1_im = _mm_add_ps(b_im, id_im);
                let y3_re = _mm_sub_ps(b_re, id_re);
                let y3_im = _mm_sub_ps(b_im, id_im);

                store_complex2_f32(dst.as_mut_ptr().add(2 * (base_out + l)), y0_re, y0_im);
                store_complex2_f32(dst.as_mut_ptr().add(2 * (base_out + m + l)), y1_re, y1_im);
                store_complex2_f32(
                    dst.as_mut_ptr().add(2 * (base_out + 2 * m + l)),
                    y2_re,
                    y2_im,
                );
                store_complex2_f32(
                    dst.as_mut_ptr().add(2 * (base_out + 3 * m + l)),
                    y3_re,
                    y3_im,
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codelets::scalar as scalar_codelets;
    use approx::assert_relative_eq;
    use num_complex::Complex64;

    fn scalar_stage_radix2(
        stride: usize,
        src: &[Complex64],
        dst: &mut [Complex64],
        stage_tw: &StageTwiddles<f64>,
    ) {
        let m = stage_tw.m;
        let s = stage_tw.s;
        let m_new = m * 2;
        let tw1 = stage_tw.k_slice(1);
        for g in 0..s {
            let base_in = g * m;
            let base_out = g * m_new;
            for l in 0..m {
                let a = src[base_in + l];
                let b = src[base_in + l + stride] * tw1[l];
                let (y0, y1) = scalar_codelets::radix2(a, b);
                dst[base_out + l] = y0;
                dst[base_out + m + l] = y1;
            }
        }
    }

    fn scalar_stage_radix4(
        stride: usize,
        src: &[Complex64],
        dst: &mut [Complex64],
        stage_tw: &StageTwiddles<f64>,
        dir_sign: f64,
    ) {
        let m = stage_tw.m;
        let s = stage_tw.s;
        let m_new = m * 4;
        let tw1 = stage_tw.k_slice(1);
        let tw2 = stage_tw.k_slice(2);
        let tw3 = stage_tw.k_slice(3);
        for g in 0..s {
            let base_in = g * m;
            let base_out = g * m_new;
            for l in 0..m {
                let x0 = src[base_in + l];
                let x1 = src[base_in + l + stride] * tw1[l];
                let x2 = src[base_in + l + 2 * stride] * tw2[l];
                let x3 = src[base_in + l + 3 * stride] * tw3[l];
                let (y0, y1, y2, y3) = scalar_codelets::radix4(x0, x1, x2, x3, dir_sign);
                dst[base_out + l] = y0;
                dst[base_out + m + l] = y1;
                dst[base_out + 2 * m + l] = y2;
                dst[base_out + 3 * m + l] = y3;
            }
        }
    }

    fn assert_close(a: &[Complex64], b: &[Complex64]) {
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b.iter()) {
            assert_relative_eq!(x.re, y.re, max_relative = 1e-13, epsilon = 1e-13);
            assert_relative_eq!(x.im, y.im, max_relative = 1e-13, epsilon = 1e-13);
        }
    }

    fn rand_complex(seed: u64, n: usize) -> Vec<Complex64> {
        let mut s = seed
            .wrapping_mul(2862933555777941757)
            .wrapping_add(3037000493);
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let re = ((s >> 11) as f64) / ((1u64 << 53) as f64) - 0.5;
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let im = ((s >> 11) as f64) / ((1u64 << 53) as f64) - 0.5;
                Complex64::new(re, im)
            })
            .collect()
    }

    fn make_stage_twiddles_radix2(n: usize, m: usize, s: usize) -> StageTwiddles<f64> {
        use std::f64::consts::TAU;
        let mut flat = Vec::with_capacity(m);
        for l in 0..m {
            let j = (l * s) % n;
            let theta = -TAU * (j as f64) / (n as f64);
            let (sv, cv) = theta.sin_cos();
            flat.push(Complex64::new(cv, sv));
        }
        StageTwiddles::from_parts(2, m, s, flat)
    }

    fn make_stage_twiddles_radix4(
        n: usize,
        m: usize,
        s: usize,
        dir_sign: f64,
    ) -> StageTwiddles<f64> {
        use std::f64::consts::TAU;
        let mut flat = Vec::with_capacity(3 * m);
        for k in 1..=3 {
            for l in 0..m {
                let j = (k * l * s) % n;
                let theta = dir_sign * TAU * (j as f64) / (n as f64);
                let (sv, cv) = theta.sin_cos();
                flat.push(Complex64::new(cv, sv));
            }
        }
        StageTwiddles::from_parts(4, m, s, flat)
    }

    fn check_radix2_for(m: usize, s: usize) {
        if !std::is_x86_feature_detected!("avx2") || !std::is_x86_feature_detected!("fma") {
            return;
        }
        let n = m * 2 * s;
        let stride = n / 2;
        let stage_tw = make_stage_twiddles_radix2(n.max(2), m, s);
        let src = rand_complex(m as u64 * 1009 + s as u64 * 251, n);
        let mut dst_scalar = vec![Complex64::new(0.0, 0.0); n];
        let mut dst_avx2 = vec![Complex64::new(0.0, 0.0); n];
        scalar_stage_radix2(stride, &src, &mut dst_scalar, &stage_tw);
        unsafe {
            stage_radix2_f64_aos(stride, &src, &mut dst_avx2, &stage_tw);
        }
        assert_close(&dst_scalar, &dst_avx2);
    }

    fn check_radix4_for(m: usize, s: usize, dir_sign: f64) {
        if !std::is_x86_feature_detected!("avx2") || !std::is_x86_feature_detected!("fma") {
            return;
        }
        let n = m * 4 * s;
        let stride = n / 4;
        let stage_tw = make_stage_twiddles_radix4(n.max(4), m, s, dir_sign);
        let src = rand_complex(m as u64 * 1013 + s as u64 * 257, n);
        let mut dst_scalar = vec![Complex64::new(0.0, 0.0); n];
        let mut dst_avx2 = vec![Complex64::new(0.0, 0.0); n];
        scalar_stage_radix4(stride, &src, &mut dst_scalar, &stage_tw, dir_sign);
        unsafe {
            stage_radix4_f64_aos(stride, &src, &mut dst_avx2, &stage_tw, dir_sign);
        }
        assert_close(&dst_scalar, &dst_avx2);
    }

    fn check_radix4_s1_for(m: usize, dir_sign: f64) {
        if !std::is_x86_feature_detected!("avx2") || !std::is_x86_feature_detected!("fma") {
            return;
        }
        let n = m * 4;
        let stage_tw = make_stage_twiddles_radix4(n.max(4), m, 1, dir_sign);
        let src = rand_complex(m as u64 * 1031 + 271, n);
        let mut dst_scalar = vec![Complex64::new(0.0, 0.0); n];
        let mut dst_avx2 = vec![Complex64::new(0.0, 0.0); n];
        scalar_stage_radix4(n / 4, &src, &mut dst_scalar, &stage_tw, dir_sign);
        unsafe {
            stage_radix4_f64_s1_aos(&src, &mut dst_avx2, &stage_tw, dir_sign);
        }
        assert_close(&dst_scalar, &dst_avx2);
    }

    fn check_batched2_f64_for(m: usize, s: usize, dir_sign: f64) {
        if !std::is_x86_feature_detected!("avx2") || !std::is_x86_feature_detected!("fma") {
            return;
        }
        let radix = if m.is_multiple_of(2) { 2 } else { 4 };
        let n = m * radix * s;
        let stride = n / radix;
        let stage_tw = if radix == 2 {
            make_stage_twiddles_radix2(n.max(2), m, s)
        } else {
            make_stage_twiddles_radix4(n.max(4), m, s, dir_sign)
        };
        let src = rand_complex(0xB2F6_4000 + n as u64 + dir_sign as u64, 2 * n);
        let mut expected = vec![Complex64::new(0.0, 0.0); 2 * n];
        let mut actual = vec![Complex64::new(0.0, 0.0); 2 * n];
        for g in 0..s {
            let base_in = g * m;
            let base_out = g * m * radix;
            for l in 0..m {
                for transform in 0..2 {
                    let lane = transform;
                    let x0 = src[2 * (base_in + l) + lane];
                    if radix == 2 {
                        let x1 = src[2 * (base_in + l + stride) + lane] * stage_tw.k_slice(1)[l];
                        let (y0, y1) = scalar_codelets::radix2(x0, x1);
                        expected[2 * (base_out + l) + lane] = y0;
                        expected[2 * (base_out + m + l) + lane] = y1;
                    } else {
                        let tw1 = stage_tw.k_slice(1);
                        let tw2 = stage_tw.k_slice(2);
                        let tw3 = stage_tw.k_slice(3);
                        let x1 = src[2 * (base_in + l + stride) + lane] * tw1[l];
                        let x2 = src[2 * (base_in + l + 2 * stride) + lane] * tw2[l];
                        let x3 = src[2 * (base_in + l + 3 * stride) + lane] * tw3[l];
                        let (y0, y1, y2, y3) = scalar_codelets::radix4(x0, x1, x2, x3, dir_sign);
                        expected[2 * (base_out + l) + lane] = y0;
                        expected[2 * (base_out + m + l) + lane] = y1;
                        expected[2 * (base_out + 2 * m + l) + lane] = y2;
                        expected[2 * (base_out + 3 * m + l) + lane] = y3;
                    }
                }
            }
        }
        unsafe {
            if radix == 2 {
                stage_radix2_f64_batched2(stride, &src, &mut actual, &stage_tw);
            } else {
                stage_radix4_f64_batched2(stride, &src, &mut actual, &stage_tw, dir_sign);
            }
        }
        assert_close(&actual, &expected);
    }

    fn check_batched2_f32_for(m: usize, s: usize, dir_sign: f32, radix: usize) {
        if !std::is_x86_feature_detected!("avx2") || !std::is_x86_feature_detected!("fma") {
            return;
        }
        let n = m * radix * s;
        let stride = n / radix;
        let stage_tw_f64 = if radix == 2 {
            make_stage_twiddles_radix2(n.max(2), m, s)
        } else {
            make_stage_twiddles_radix4(n.max(4), m, s, dir_sign as f64)
        };
        let mut twiddles = Vec::with_capacity((radix - 1) * m);
        for k in 1..radix {
            twiddles.extend(
                stage_tw_f64
                    .k_slice(k)
                    .iter()
                    .map(|value| Complex::new(value.re as f32, value.im as f32)),
            );
        }
        let stage_tw = StageTwiddles::from_parts(radix, m, s, twiddles);
        let src64 = rand_complex(0xB2F6_3200 + n as u64 + radix as u64, 2 * n);
        let src: Vec<_> = src64
            .into_iter()
            .map(|value| Complex::new(value.re as f32, value.im as f32))
            .collect();
        let mut expected = vec![Complex::new(0.0_f32, 0.0_f32); 2 * n];
        let mut actual = vec![Complex::new(0.0_f32, 0.0_f32); 2 * n];
        for g in 0..s {
            let base_in = g * m;
            let base_out = g * m * radix;
            for l in 0..m {
                for transform in 0..2 {
                    let lane = transform;
                    let x0 = src[2 * (base_in + l) + lane];
                    if radix == 2 {
                        let x1 = src[2 * (base_in + l + stride) + lane] * stage_tw.k_slice(1)[l];
                        let (y0, y1) = scalar_codelets::radix2(x0, x1);
                        expected[2 * (base_out + l) + lane] = y0;
                        expected[2 * (base_out + m + l) + lane] = y1;
                    } else {
                        let x1 = src[2 * (base_in + l + stride) + lane] * stage_tw.k_slice(1)[l];
                        let x2 =
                            src[2 * (base_in + l + 2 * stride) + lane] * stage_tw.k_slice(2)[l];
                        let x3 =
                            src[2 * (base_in + l + 3 * stride) + lane] * stage_tw.k_slice(3)[l];
                        let (y0, y1, y2, y3) = scalar_codelets::radix4(x0, x1, x2, x3, dir_sign);
                        expected[2 * (base_out + l) + lane] = y0;
                        expected[2 * (base_out + m + l) + lane] = y1;
                        expected[2 * (base_out + 2 * m + l) + lane] = y2;
                        expected[2 * (base_out + 3 * m + l) + lane] = y3;
                    }
                }
            }
        }
        unsafe {
            if radix == 2 {
                stage_radix2_f32_batched2(stride, &src, &mut actual, &stage_tw);
            } else {
                stage_radix4_f32_batched2(stride, &src, &mut actual, &stage_tw, dir_sign);
            }
        }
        for (actual, expected) in actual.iter().zip(expected) {
            assert_relative_eq!(actual.re, expected.re, max_relative = 5e-5, epsilon = 5e-5);
            assert_relative_eq!(actual.im, expected.im, max_relative = 5e-5, epsilon = 5e-5);
        }
    }

    #[test]
    fn avx2_radix2_matches_scalar_various_shapes() {
        for &(m, s) in &[(1, 1), (4, 1), (8, 4), (16, 2), (64, 8), (7, 3), (15, 5)] {
            check_radix2_for(m, s);
        }
    }

    #[test]
    fn avx2_radix4_matches_scalar_forward() {
        for &(m, s) in &[(1, 1), (4, 1), (8, 8), (16, 4), (64, 4), (5, 2), (7, 3)] {
            check_radix4_for(m, s, -1.0);
        }
    }

    #[test]
    fn avx2_radix4_matches_scalar_inverse() {
        for &(m, s) in &[(4, 1), (16, 4), (64, 4), (7, 3)] {
            check_radix4_for(m, s, 1.0);
        }
    }

    #[test]
    fn avx2_radix4_s1_matches_scalar_both_directions() {
        for &m in &[1, 2, 3, 4, 7, 16, 63, 257] {
            check_radix4_s1_for(m, -1.0);
            check_radix4_s1_for(m, 1.0);
        }
    }

    #[test]
    fn avx2_batched2_radix_stages_match_scalar_both_directions() {
        for &(m, s) in &[(1usize, 1usize), (2, 1), (3, 2), (4, 3), (7, 5), (15, 2)] {
            for &dir_sign in &[-1.0, 1.0] {
                check_batched2_f64_for(m, s, dir_sign);
            }
        }
    }

    #[test]
    fn avx2_batched2_f32_radix_stages_match_scalar_both_directions() {
        for &(radix, m, s) in &[
            (2usize, 1usize, 1usize),
            (2, 5, 3),
            (4, 1, 1),
            (4, 3, 2),
            (4, 7, 3),
        ] {
            for &dir_sign in &[-1.0_f32, 1.0_f32] {
                check_batched2_f32_for(m, s, dir_sign, radix);
            }
        }
    }
}
