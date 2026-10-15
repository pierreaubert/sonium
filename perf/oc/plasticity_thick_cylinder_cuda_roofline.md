# plasticity_thick_cylinder CUDA roofline (Phase G0 proxy, 2026-10-15)

Verdict: **GPU not recommended for this deck.** A persistent-kernel proxy of
the native LAW36 explicit run (624 nodes, 288 bricks, 5880 steps at 0.8x
native mean dt) runs **2.75x slower than 1-thread CPU** per step (609 us vs
222 us sonium native, 2.95x slower than the engine serial wall). Limiter:
**912-way deck parallelism against one SM's FP64 stream**, not bandwidth.
No full CUDA port is warranted; this proxy report stands as the deck's
distance-to-roofline record. (Closest deck of the campaign; the bound below
shows why even a perfect port stays short.)

Method: G0 proxy (see `oc-opt-20261015.md` methodology). No `sonium.cuda`
explicit arm exists, and `ncu` counters are unavailable on this box
(`ERR_NVGPUCTRPERM`: needs root `NVreg_RestrictProfilingToAdminHost=0`), so
the roofline rests on device-timed walls, measured machine ceilings, and
differential timing instead of counter replay. Raw outputs and sources sit
beside this file (`plasticity_thick_cylinder_cuda_*.txt`,
`plasticity_thick_cylinder_cuda_*.cu`).

## Machine ceilings (RTX PRO 6000 Blackwell, CC 12.0, 188 SMs, 128 MB L2)

Re-measured same-day, device-timed (necking machine source):

| Quantity | Value |
| -------- | ----- |
| Empty kernel launch | 6.85 us |
| Launch+sync roundtrip (host-stepped floor) | 8.27 us/step |
| 4 KB H2D+D2H pinned roundtrip | 14.8 us |
| H2D / D2H pinned BW (256 MB) | 22.0 / 28.0 GB/s |
| STREAM triad (DRAM ceiling) | 2160 GB/s |
| FP64 MAD (compute ceiling) | 479.9 GFLOP/s |
| FP64 ridge point | 0.22 FLOP/byte |
| Clocks under proxy load | 2692 MHz SM (7% below the foam runs; ratio noise ~+-10%) |

Host-stepped verdict first: a launch+sync per step (8.3 us) fits inside
the 222 us/step CPU budget, so the floor alone does not kill a GPU port
here. The proxy is still a persistent on-device loop (the design a real
port would use), so the comparison is fair, not floor-bound.

## Proxy design (deck-faithful persistent kernel)

`plasticity_thick_cylinder_cuda_proxy.cu`: 1 block x 128 threads over
the exact deck mesh (624 nodes, 288 bricks, deck masses, greedy 4-color
ordering), per element per step: current-config centroid Jacobian +
gradients, uniform strain increment, quadratic-only bulk viscosity,
LAW36 tabulated J2 return (perfectly plastic 250 MPa table,
80-iteration bisection when yielding; the card's rupture/deletion/
tensile legs are off-scale and omitted -- documented), V B^T
(sigma - q I) scatter, Flanagan-Belytschko hourglass; 88 bore-pressure
CLOADs on the deck FUNCT/1 curve; all-node axial pins (plane strain);
velocity-Verlet at fixed dt = 0.8x native mean (5880 steps to 3 ms);
per-step bore/outer DX + SY histories booked on-device.

Fidelity checks (measured, not assumed) at the full deck:

| Probe | Proxy | Native | Match |
| ----- | ----- | ------ | ----- |
| bore DX at 50 MPa | 2.414070e-5 | 2.414243e-5 | 0.007% |
| outer DX at 50 MPa | 1.594107e-5 | 1.594237e-5 | 0.008% |
| bore hoop residual SY | -8.566989e7 | -8.754235e7 | 2.1% |
| outer hoop residual SY | 1.182575e7 | 1.103160e7 | 7.2% (inside the 1e6 abs band) |
| eroded | 0/288 | 0/288 | exact |
| max ep | 6.572361e-4 | 6.574544e-4 | 0.03% |

N=1 first step is quiescent (all zeros, correct); bit-identical
repeats. All four probes land inside their bench bands. Physics
validated BEFORE timing. Documented simplifications: fixed dt,
off-scale LAW36 legs omitted, consistent-tangent FLOP booked in the
estimate (~2%), no TH/energy bookkeeping.

## Roofline numbers

Per-step proxy wall (device-timed, 2 repeats within 490 ppm):
**609 us/step**. Full deck measured, not projected: **3.58 s**.

| Quantity | Value |
| -------- | ----- |
| Work estimate | ~0.53 MFLOP/step (+-40%; mostly elastic ~1.6k/element) |
| Achieved throughput | 0.87 GFLOP/s |
| Distance to FP64 ceiling | **~550x** (0.18%; best of the campaign) |
| Traffic estimate | ~80 KB state, L2-resident (~0.2 MB/step over passes) |
| Arithmetic intensity | ~2.6 FLOP/byte (12x above ridge: compute-side) |
| Empty-force floor (cloads+update+loop) | 18.2 us/step (3.0%) |
| Force-math share | **97.0%** |
| Launch config | 1 block x 128, <=72 active, 4 color phases |

Perfect-port bound (kills the "bad kernel" objection without writing
G1): the deck offers 288 elements + 624 nodes = 912 work items --
under one SM's 1536 thread slots. Step work (0.53 MFLOP) at 100% of
one SM's FP64 stream (2.55 GFLOP/s same-day) floors at 208 us vs the
CPU's 222 us: parity needs >=94% of one SM sustained. That is
implausible for this loop (elastic/yielding divergence in the bore
ring, 4 color syncs, CLOAD/update/history phases, no second SM to
spread to -- the deck cannot fill two). A realistic multi-block G1
(2-3x over this 1-block proxy) projects ~1.4-2x short of CPU; the
CPU wall itself carries ~10-20% scaffolding (per-step dt_crit pass,
energy ledger) the proxy skips, which widens rather than narrows the
gap. Measured proxy is 2.75x short (222 us/step CPU).

## Reading

- The deck is small-and-slow: 912-way parallelism cannot feed the
  GPU's FP64 pipes, while the CPU's clocks, IPC, and caches sustain
  2.4 GFLOP/s effective on the same loop. Bandwidth is irrelevant
  (L2-resident, compute-side by 12x).
- The G1 gate fails 2.75x short with ~1.4-2x realistic headroom --
  closest of the campaign, but parity needs >=94% of one SM and the
  prize is a 1-second deck the engine already runs in 1.2 s serial.
  Revisit only if G1 tooling clears the one-SM efficiency bar with
  measured evidence.
- Reproduce: `nvcc -O2 -arch=native -o cyl_proxy
  plasticity_thick_cylinder_cuda_proxy.cu && ./cyl_proxy 5880` (4 s,
  idle GPU).
