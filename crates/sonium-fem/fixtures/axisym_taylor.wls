(* Independent Wolfram reference for the axisymmetric Taylor-analog bar.
   Single QUAD ring spanning a copper bar cross-section (r in [0, R],
   z in [0, L], R = 3.175 mm, L = 25.4 mm, matching the sonium Taylor
   bar constants) with an edge on the axis. Derives, at 50-digit
   working precision, the centroid operator rows, hourglass vector,
   area, centroid radius, ring volume (= Pi R^2 L exactly), lumped
   nodal mass, characteristic length, critical step, the elastic
   uniaxial-strain response to an impact-rate driver (V = 227 m/s over
   dt = 1 ns), the shock viscosity of that compressive step, the
   impact-face resultant identity (-Pi R^2 (szz - q)), and the
   single-step J2 radial-return response (linear hardening) to the
   same driver over dt = 1 us. All inputs are exact rationals; roots
   and Pi stay exact until N[..., 50].
   The Rust side consumes axisym_taylor_golden.json; this script is
   the derivation. Full mushrooming needs contact plus a multi-element
   mesh and stays out of scope: this fixture pins the axisymmetric
   bulk response the Taylor bar is built on.
   Regen: wolframscript -script fixtures/axisym_taylor.wls > fixtures/axisym_taylor_golden.json
   Engine: Wolfram Language 15.0.0 for Linux x86 (64-bit) (May 26, 2026). *)

prec = 50;
dec[x_] := ToString[CForm[N[x, prec]]];

(* Bar cross-section rectangle, nodes (0,0),(R,0),(R,L),(0,L). *)
radius = 3175/10^6;
len = 254/10^4;
dnDr = {-1/(2*radius), 1/(2*radius), 1/(2*radius), -1/(2*radius)};
dnDz = {-1/(2*len), -1/(2*len), 1/(2*len), 1/(2*len)};

(* (h.r) = (h.z) = 0 on the rectangle, so gamma = h exactly. *)
h = {1, -1, 1, -1};
rs = {0, radius, radius, 0};
zs = {0, 0, len, len};
gamma = h - (h.rs)*dnDr - (h.zs)*dnDz;

area = radius*len;
rbar = radius/2;
volume = 2*Pi*rbar*area;
rho = 8930;
nodalMass = rho*volume/4;
charLen = area/Sqrt[radius^2 + len^2];
sound = 4500;
dtCrit = charLen/sound;

(* OFHC-copper-like J2: E = 117 GPa, nu = 0.35. *)
e = 117*10^9;
nu = 35/100;
lame = e*nu/((1 + nu)*(1 - 2*nu));
mu = e/(2*(1 + nu));
sigY = 300*10^6;
hard = 10^9;

(* Impact-rate driver: bottom face vz = V, top face vz = 0, vr = 0,
   so the axial rate is uniform -V/L and the step compresses. *)
vel = 227;
rate = -vel/len;
dtEl = 1/10^9;
ezzEl = rate*dtEl;
szzEl = (lame + 2*mu)*ezzEl;
srrEl = lame*ezzEl;
sthEl = lame*ezzEl;

(* Shock viscosity of the compressive step (d < 0):
   q = rho L (c0 L d^2 - c1 c d), evaluated at the step sound-speed
   input c = 4500 exactly as the native call receives it. *)
c0 = 11/10;
c1 = 5/100;
bulkQ = rho*charLen*(c0*charLen*rate^2 - c1*sound*rate);
(* Impact-face (bottom nodes) resultant: -Pi R^2 (szz - q).
   The viscous pressure enters the tension-positive stress as
   sigma - q I, so the shock term subtracts from szz. *)
faceArea = Pi*radius^2;
bottomEl = -faceArea*(szzEl - bulkQ);

(* Plastic single step: same driver over dt = 1 us. Trial is
   uniaxial strain; radial return is exact for linear hardening:
   dep = (vmTr - sigY)/(3 mu + H), deviatoric scaled by
   (sigY + H dep)/vmTr about the trial pressure. *)
dtPl = 1/10^6;
ezzPl = rate*dtPl;
szzTr = (lame + 2*mu)*ezzPl;
srrTr = lame*ezzPl;
pTr = (2*srrTr + szzTr)/3;
vmTr = 2*mu*Abs[ezzPl];
depPl = (vmTr - sigY)/(3*mu + hard);
flow = sigY + hard*depPl;
scale = flow/vmTr;
szzPl = pTr + (szzTr - pTr)*scale;
srrPl = pTr + (srrTr - pTr)*scale;
sthPl = srrPl;
bottomPl = -faceArea*(szzPl - bulkQ);

golden = <|
  "meta" -> <|
    "engine" -> "Wolfram Language " <> $Version,
    "protocol" -> "solids-golden-v1",
    "case" -> "axisym_taylor",
    "units" -> "SI"
  |>,
  "inputs" -> <|
    "bar" -> "r in [0, R], z in [0, L], R = 3.175 mm, L = 25.4 mm",
    "density" -> 8930,
    "sound_speed" -> 4500,
    "youngs" -> 117.0*^9,
    "poisson" -> 0.35,
    "yield" -> 300.0*^6,
    "hardening" -> 1.0*^9,
    "impact_velocity" -> 227,
    "dt_elastic" -> 1.0*^-9,
    "dt_plastic" -> 1.0*^-6
  |>,
  "values" -> <|
    "b_r_row" -> <|
      "values" -> dec /@ dnDr,
      "exact" -> "dN/dr row, exact over 2R"
    |>,
    "b_z_row" -> <|
      "values" -> dec /@ dnDz,
      "exact" -> "dN/dz row, exact over 2L"
    |>,
    "gamma" -> <|
      "values" -> dec /@ gamma,
      "exact" -> "h - (h.r)br - (h.z)bz = {1,-1,1,-1}"
    |>,
    "area" -> <|
      "values" -> {dec[area]},
      "exact" -> "R L"
    |>,
    "radius_bar" -> <|
      "values" -> {dec[rbar]},
      "exact" -> "R/2"
    |>,
    "volume" -> <|
      "values" -> {dec[volume]},
      "exact" -> "2 Pi rbar A = Pi R^2 L"
    |>,
    "nodal_mass" -> <|
      "values" -> {dec[nodalMass]},
      "exact" -> "rho V / 4"
    |>,
    "char_length" -> <|
      "values" -> {dec[charLen]},
      "exact" -> "A / Sqrt[R^2 + L^2]"
    |>,
    "critical_step" -> <|
      "values" -> {dec[dtCrit]},
      "exact" -> "L / c at c = 4500"
    |>,
    "elastic_szz" -> <|
      "values" -> {dec[szzEl]},
      "exact" -> "(lambda + 2 mu) ezz"
    |>,
    "elastic_srr" -> <|
      "values" -> {dec[srrEl]},
      "exact" -> "lambda ezz"
    |>,
    "elastic_sth" -> <|
      "values" -> {dec[sthEl]},
      "exact" -> "lambda ezz"
    |>,
    "bulk_q" -> <|
      "values" -> {dec[bulkQ]},
      "exact" -> "rho L (c0 L d^2 - c1 c d) at c = 4500"
    |>,
    "bottom_el" -> <|
      "values" -> {dec[bottomEl]},
      "exact" -> "-Pi R^2 (szz - q)"
    |>,
    "plastic_szz" -> <|
      "values" -> {dec[szzPl]},
      "exact" -> "radial return about trial pressure"
    |>,
    "plastic_srr" -> <|
      "values" -> {dec[srrPl]},
      "exact" -> "radial return about trial pressure"
    |>,
    "plastic_sth" -> <|
      "values" -> {dec[sthPl]},
      "exact" -> "radial return, equals srr"
    |>,
    "plastic_dep" -> <|
      "values" -> {dec[depPl]},
      "exact" -> "(2 mu |e| - sigY)/(3 mu + H)"
    |>,
    "bottom_pl" -> <|
      "values" -> {dec[bottomPl]},
      "exact" -> "-Pi R^2 (szz - q)"
    |>
  |>,
  "tolerances" -> <|
    "b_r_row_abs" -> 1.0*^-12,
    "b_r_row_rel" -> 1.0*^-12,
    "b_z_row_abs" -> 1.0*^-12,
    "b_z_row_rel" -> 1.0*^-12,
    "gamma_abs" -> 1.0*^-15,
    "gamma_rel" -> 1.0*^-12,
    "area_abs" -> 1.0*^-15,
    "area_rel" -> 1.0*^-12,
    "radius_bar_abs" -> 1.0*^-18,
    "radius_bar_rel" -> 1.0*^-12,
    "volume_abs" -> 1.0*^-15,
    "volume_rel" -> 1.0*^-12,
    "nodal_mass_abs" -> 1.0*^-12,
    "nodal_mass_rel" -> 1.0*^-12,
    "char_length_abs" -> 1.0*^-15,
    "char_length_rel" -> 1.0*^-12,
    "critical_step_abs" -> 1.0*^-18,
    "critical_step_rel" -> 1.0*^-12,
    "elastic_szz_abs" -> 1.0*^-3,
    "elastic_szz_rel" -> 1.0*^-9,
    "elastic_srr_abs" -> 1.0*^-3,
    "elastic_srr_rel" -> 1.0*^-9,
    "elastic_sth_abs" -> 1.0*^-3,
    "elastic_sth_rel" -> 1.0*^-9,
    "bulk_q_abs" -> 1.0*^-3,
    "bulk_q_rel" -> 1.0*^-9,
    "bottom_el_abs" -> 1.0*^-6,
    "bottom_el_rel" -> 1.0*^-9,
    "plastic_szz_abs" -> 1.0*^-3,
    "plastic_szz_rel" -> 1.0*^-9,
    "plastic_srr_abs" -> 1.0*^-3,
    "plastic_srr_rel" -> 1.0*^-9,
    "plastic_sth_abs" -> 1.0*^-3,
    "plastic_sth_rel" -> 1.0*^-9,
    "plastic_dep_abs" -> 1.0*^-12,
    "plastic_dep_rel" -> 1.0*^-9,
    "bottom_pl_abs" -> 1.0*^-6,
    "bottom_pl_rel" -> 1.0*^-9
  |>
|>;

WriteString[$Output, ExportString[golden, "JSON"]];
