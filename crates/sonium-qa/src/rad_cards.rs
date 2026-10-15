//! OpenCourant deck card families and header normalization.
//!
//! This module owns the [`CardKind`] enum covering the 319 normalized card
//! headers in `sonium/miniqa_cards.txt` plus eight structural extras needed
//! for a complete corpus parse, and the [`parse_header`] normalization that
//! maps raw deck header lines onto `(CardKind, id)` pairs.
//!
//! A raw header carries a family path plus an optional trailing numeric
//! identifier (`/PART/1`, `/GRNOD/NODE/2`), with a few documented special
//! cases: `/RUN/<name>/<id>` run stamps, `/PRINT` frequency parameters,
//! the `/INIVEL/T+G` ALE type, `/UNIT` work-system suffixes, and the
//! double-numeric `/TH/NODE/1/0` form. Unknown headers map to `None` so the
//! lexer in `crate::rad_parse` can fail closed with file and line context.
//!
//! # Examples
//!
//! ```rust
//! use sonium_qa::rad_cards::{CardKind, parse_header};
//!
//! let (kind, id, sub) = parse_header("/PART/1").expect("known header");
//! assert_eq!(kind, CardKind::Part);
//! assert_eq!(id, Some(1));
//! assert_eq!(sub, None);
//! assert!(parse_header("/NOPE/1").is_none());
//! ```

// Rust guideline compliant 2026-02-21

use std::fmt;

/// Normalized inventory headers in `sonium/miniqa_cards.txt`.
///
/// The count is asserted by [`inventory_matches_enum`] so the enum cannot
/// drift from the coordinator-owned inventory without a loud test failure.
pub const INVENTORY_COUNT: usize = 319;

/// Declares [`CardKind`] plus its canonical-text conversions from one table.
///
/// A single table row generates the enum variant, the `as_str` arm, the
/// `parse` arm, and the [`CardKind::ALL`] entry, so adding a family touches
/// exactly one line. Variant docs name the deck header each variant covers.
macro_rules! define_kinds {
    ($($variant:ident => $text:literal),* $(,)?) => {
        /// OpenCourant deck card family.
        ///
        /// Each variant names one normalized deck header (see
        /// `sonium/miniqa_cards.txt`). The eight [`CardKind::is_extra`]
        /// variants cover headers the inventory misses: include-only
        /// cards, submodel markers, the `/INIVEL/T+G` ALE type, the
        /// writer-emitted `/TH/TITLE` and `/MAT/LAW2` cards, and the
        /// bare `/DT/BRICK` dt-scale card our foam engine decks carry.
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
        pub enum CardKind {
            $(
                #[doc = concat!("Card `", $text, "` deck header.")]
                $variant,
            )*
        }

        impl CardKind {
            /// Every known card family in inventory order.
            pub const ALL: &'static [Self] = &[$(Self::$variant),*];

            /// Canonical header text for this family.
            #[must_use]
            pub const fn as_str(self) -> &'static str {
                match self {
                    $(Self::$variant => $text,)*
                }
            }

            /// Parses canonical header text into a family.
            ///
            /// Returns `None` for unknown headers; use [`parse_header`]
            /// for raw deck lines carrying identifiers or parameters.
            #[must_use]
            pub fn parse(text: &str) -> Option<Self> {
                match text {
                    $($text => Some(Self::$variant),)*
                    _ => None,
                }
            }
        }
    };
}

define_kinds! {
    Accel => "/ACCEL",
    Activ => "/ACTIV",
    Admas0 => "/ADMAS/0",
    AleBcs => "/ALE/BCS",
    AleGridDonea => "/ALE/GRID/DONEA",
    AleGridZero => "/ALE/GRID/ZERO",
    AleMat => "/ALE/MAT",
    AleSolver => "/ALE/SOLVER",
    Ams => "/AMS",
    Analy => "/ANALY",
    AnimBrickDens => "/ANIM/BRICK/DENS",
    AnimBrickEner => "/ANIM/BRICK/ENER",
    AnimBrickEpsp => "/ANIM/BRICK/EPSP",
    AnimBrickHourg => "/ANIM/BRICK/HOURG",
    AnimBrickP => "/ANIM/BRICK/P",
    AnimBrickSigx => "/ANIM/BRICK/SIGX",
    AnimBrickSigxy => "/ANIM/BRICK/SIGXY",
    AnimBrickSigy => "/ANIM/BRICK/SIGY",
    AnimBrickSigyz => "/ANIM/BRICK/SIGYZ",
    AnimBrickSigz => "/ANIM/BRICK/SIGZ",
    AnimBrickSigzx => "/ANIM/BRICK/SIGZX",
    AnimBrickTensStrain => "/ANIM/BRICK/TENS/STRAIN",
    AnimBrickTensStress => "/ANIM/BRICK/TENS/STRESS",
    AnimBrickVonm => "/ANIM/BRICK/VONM",
    AnimDt => "/ANIM/DT",
    AnimElemAms => "/ANIM/ELEM/AMS",
    AnimElemDam1 => "/ANIM/ELEM/DAM1",
    AnimElemDam2 => "/ANIM/ELEM/DAM2",
    AnimElemDam3 => "/ANIM/ELEM/DAM3",
    AnimElemDens => "/ANIM/ELEM/DENS",
    AnimElemEner => "/ANIM/ELEM/ENER",
    AnimElemEpsd => "/ANIM/ELEM/EPSD",
    AnimElemEpsp => "/ANIM/ELEM/EPSP",
    AnimElemHour => "/ANIM/ELEM/HOUR",
    AnimElemHourg => "/ANIM/ELEM/HOURG",
    AnimElemP => "/ANIM/ELEM/P",
    AnimElemPres => "/ANIM/ELEM/PRES",
    AnimElemSigx => "/ANIM/ELEM/SIGX",
    AnimElemSigxy => "/ANIM/ELEM/SIGXY",
    AnimElemSigy => "/ANIM/ELEM/SIGY",
    AnimElemSigyz => "/ANIM/ELEM/SIGYZ",
    AnimElemSigz => "/ANIM/ELEM/SIGZ",
    AnimElemSigzx => "/ANIM/ELEM/SIGZX",
    AnimElemTemp => "/ANIM/ELEM/TEMP",
    AnimElemThic => "/ANIM/ELEM/THIC",
    AnimElemUser1 => "/ANIM/ELEM/USER1",
    AnimElemUser2 => "/ANIM/ELEM/USER2",
    AnimElemUser3 => "/ANIM/ELEM/USER3",
    AnimElemUser4 => "/ANIM/ELEM/USER4",
    AnimElemVonm => "/ANIM/ELEM/VONM",
    AnimGzip => "/ANIM/GZIP",
    AnimMass => "/ANIM/MASS",
    AnimNodaDens => "/ANIM/NODA/DENS",
    AnimNodaDiner => "/ANIM/NODA/DINER",
    AnimNodaDmas => "/ANIM/NODA/DMAS",
    AnimNodaDmass => "/ANIM/NODA/DMASS",
    AnimNodaDt => "/ANIM/NODA/DT",
    AnimNodaP => "/ANIM/NODA/P",
    AnimNodaTemp => "/ANIM/NODA/TEMP",
    AnimShellTensEpsdotAll => "/ANIM/SHELL/TENS/EPSDOT/ALL",
    AnimShellTensStrainAll => "/ANIM/SHELL/TENS/STRAIN/ALL",
    AnimShellTensStrainLower => "/ANIM/SHELL/TENS/STRAIN/LOWER",
    AnimShellTensStrainMemb => "/ANIM/SHELL/TENS/STRAIN/MEMB",
    AnimShellTensStrainUpper => "/ANIM/SHELL/TENS/STRAIN/UPPER",
    AnimShellTensStressAll => "/ANIM/SHELL/TENS/STRESS/ALL",
    AnimShellTensStressLower => "/ANIM/SHELL/TENS/STRESS/LOWER",
    AnimShellTensStressMemb => "/ANIM/SHELL/TENS/STRESS/MEMB",
    AnimShellTensStressUpper => "/ANIM/SHELL/TENS/STRESS/UPPER",
    AnimSpringForc => "/ANIM/SPRING/FORC",
    AnimTensStrainUpper => "/ANIM/TENS/STRAIN/UPPER",
    AnimTensStressMemb => "/ANIM/TENS/STRESS/MEMB",
    AnimVectAcc => "/ANIM/VECT/ACC",
    AnimVectAccel => "/ANIM/VECT/ACCEL",
    AnimVectCont => "/ANIM/VECT/CONT",
    AnimVectDisp => "/ANIM/VECT/DISP",
    AnimVectFext => "/ANIM/VECT/FEXT",
    AnimVectFint => "/ANIM/VECT/FINT",
    AnimVectFopt => "/ANIM/VECT/FOPT",
    AnimVectFvel => "/ANIM/VECT/FVEL",
    AnimVectPcont => "/ANIM/VECT/PCONT",
    AnimVectVel => "/ANIM/VECT/VEL",
    AnimVers => "/ANIM/VERS",
    Bcs => "/BCS",
    Beam => "/BEAM",
    Begin => "/BEGIN",
    BoxCylin => "/BOX/CYLIN",
    BoxRecta => "/BOX/RECTA",
    BoxRecta3 => "/BOX/RECTA/3",
    BoxSpher => "/BOX/SPHER",
    Brick => "/BRICK",
    Caa => "/CAA",
    Cload => "/CLOAD",
    Damp => "/DAMP",
    DefShell => "/DEF_SHELL",
    DefSolid => "/DEF_SOLID",
    DelShell => "/DEL/SHELL",
    DfsDetpoin => "/DFS/DETPOIN",
    Dt => "/DT",
    DtAms => "/DT/AMS",
    DtBrick => "/DT/BRICK",
    DtBrickCst => "/DT/BRICK/CST",
    DtBrickStop => "/DT/BRICK/STOP",
    DtCstAms => "/DT/CST_AMS",
    DtInterCst => "/DT/INTER/CST",
    DtInterDel => "/DT/INTER/DEL",
    Dtix => "/DTIX",
    DtNoda => "/DT/NODA",
    DtNodaCst => "/DT/NODA/CST",
    DtNodaCst1 => "/DT/NODA/CST1",
    DtSh3nCst => "/DT/SH_3N/CST",
    DtShell => "/DT/SHELL",
    DtShellCst => "/DT/SHELL/CST",
    EbcsIniv => "/EBCS/INIV",
    End => "/END",
    EndEngine => "/END/ENGINE",
    EosPolynomial => "/EOS/POLYNOMIAL",
    EosPuff => "/EOS/PUFF",
    FailChang => "/FAIL/CHANG",
    FailHashin => "/FAIL/HASHIN",
    FailJohnson => "/FAIL/JOHNSON",
    FailOrthstrain => "/FAIL/ORTHSTRAIN",
    FailTab => "/FAIL/TAB",
    FailWierzbicki => "/FAIL/WIERZBICKI",
    FrameFix => "/FRAME/FIX",
    Funct => "/FUNCT",
    FunctPython => "/FUNCT_PYTHON",
    Grav => "/GRAV",
    GrbricBox => "/GRBRIC/BOX",
    GrbricBox2 => "/GRBRIC/BOX2",
    GrbricBox21003 => "/GRBRIC/BOX2/1003",
    GrbricBric => "/GRBRIC/BRIC",
    GrnodGrnod => "/GRNOD/GRNOD",
    GrnodGrshel => "/GRNOD/GRSHEL",
    GrnodNode => "/GRNOD/NODE",
    GrnodPart => "/GRNOD/PART",
    GrnodSubset => "/GRNOD/SUBSET",
    GrnodSurf => "/GRNOD/SURF",
    GrquadQuad => "/GRQUAD/QUAD",
    GrshelBox => "/GRSHEL/BOX",
    GrshelShel => "/GRSHEL/SHEL",
    Impdisp => "/IMPDISP",
    ImpdispFgeo => "/IMPDISP/FGEO",
    Impvel => "/IMPVEL",
    InibriStress => "/INIBRI/STRESS",
    InivelRot => "/INIVEL/ROT",
    InivelT => "/INIVEL/T",
    InivelTra => "/INIVEL/TRA",
    InivTraY => "/INIV/TRA/Y",
    InivTraZ => "/INIV/TRA/Z",
    InterType11 => "/INTER/TYPE11",
    InterType12 => "/INTER/TYPE12",
    InterType19 => "/INTER/TYPE19",
    InterType2 => "/INTER/TYPE2",
    InterType22 => "/INTER/TYPE22",
    InterType25 => "/INTER/TYPE25",
    InterType3 => "/INTER/TYPE3",
    InterType5 => "/INTER/TYPE5",
    InterType7 => "/INTER/TYPE7",
    Ioflag => "/IOFLAG",
    Kerel => "/KEREL",
    LeakMat => "/LEAK/MAT",
    LineEdge => "/LINE/EDGE",
    LineSurf => "/LINE/SURF",
    MatBoltzman => "/MAT/BOLTZMAN",
    MatBound => "/MAT/BOUND",
    MatCompsh => "/MAT/COMPSH",
    MatCowper => "/MAT/COWPER",
    MatElast => "/MAT/ELAST",
    MatFabrA => "/MAT/FABR_A",
    MatFabri => "/MAT/FABRI",
    MatFoamPlas => "/MAT/FOAM_PLAS",
    MatFoamVisc => "/MAT/FOAM_VISC",
    MatGasPredef => "/MAT/GAS/PREDEF",
    MatHoneycomb => "/MAT/HONEYCOMB",
    MatHydJcook => "/MAT/HYD_JCOOK",
    MatHydpla => "/MAT/HYDPLA",
    MatHydVisc => "/MAT/HYD_VISC",
    MatLaw01 => "/MAT/LAW01",
    MatLaw1 => "/MAT/LAW1",
    MatLaw12 => "/MAT/LAW12",
    MatLaw14 => "/MAT/LAW14",
    MatLaw19 => "/MAT/LAW19",
    MatLaw2 => "/MAT/LAW2",
    MatLaw21 => "/MAT/LAW21",
    MatLaw22 => "/MAT/LAW22",
    MatLaw23 => "/MAT/LAW23",
    MatLaw28 => "/MAT/LAW28",
    MatLaw3 => "/MAT/LAW3",
    MatLaw36 => "/MAT/LAW36",
    MatLaw4 => "/MAT/LAW4",
    MatLaw49 => "/MAT/LAW49",
    MatLaw50 => "/MAT/LAW50",
    MatLaw51 => "/MAT/LAW51",
    MatLaw62 => "/MAT/LAW62",
    MatLaw66 => "/MAT/LAW66",
    MatLaw68 => "/MAT/LAW68",
    MatLaw70 => "/MAT/LAW70",
    MatLaw76 => "/MAT/LAW76",
    MatOgden => "/MAT/OGDEN",
    MatPlasBrit => "/MAT/PLAS_BRIT",
    MatPlasJohns => "/MAT/PLAS_JOHNS",
    MatPlasTab => "/MAT/PLAS_TAB",
    MatRigid => "/MAT/RIGID",
    MatViscTab => "/MAT/VISC_TAB",
    MatVoid => "/MAT/VOID",
    Memory => "/MEMORY",
    MonOff => "/MON/OFF",
    MonOn => "/MON/ON",
    MonvolAirbag1 => "/MONVOL/AIRBAG1",
    MonvolCommu => "/MONVOL/COMMU",
    MoveFunct => "/MOVE_FUNCT",
    Node => "/NODE",
    OutpDt => "/OUTP/DT",
    OutpElemEner => "/OUTP/ELEM/ENER",
    OutpElemEpsp => "/OUTP/ELEM/EPSP",
    OutpElemVonm => "/OUTP/ELEM/VONM",
    OutpNodaDiner => "/OUTP/NODA/DINER",
    OutpNodaDmas => "/OUTP/NODA/DMAS",
    OutpVectCont => "/OUTP/VECT/CONT",
    OutpVectVel => "/OUTP/VECT/VEL",
    ParithOn => "/PARITH/ON",
    Part => "/PART",
    Pload => "/PLOAD",
    Print => "/PRINT",
    Print100 => "/PRINT/100",
    Proc => "/PROC",
    Proc1 => "/PROC/1",
    Proc8 => "/PROC/8",
    PropBeam => "/PROP/BEAM",
    PropInject2 => "/PROP/INJECT2",
    PropShell => "/PROP/SHELL",
    PropShFabr => "/PROP/SH_FABR",
    PropShOrth => "/PROP/SH_ORTH",
    PropShSandw => "/PROP/SH_SANDW",
    PropSolid => "/PROP/SOLID",
    PropSolOrth => "/PROP/SOL_ORTH",
    PropSprBeam => "/PROP/SPR_BEAM",
    PropSprGene => "/PROP/SPR_GENE",
    PropSpring => "/PROP/SPRING",
    PropSprPre => "/PROP/SPR_PRE",
    PropTruss => "/PROP/TRUSS",
    PropType1 => "/PROP/TYPE1",
    PropType13 => "/PROP/TYPE13",
    PropType14 => "/PROP/TYPE14",
    PropType25 => "/PROP/TYPE25",
    PropType34 => "/PROP/TYPE34",
    PropType4 => "/PROP/TYPE4",
    PropType45 => "/PROP/TYPE45",
    PropType6 => "/PROP/TYPE6",
    PropType8 => "/PROP/TYPE8",
    PropType9 => "/PROP/TYPE9",
    PropVoid => "/PROP/VOID",
    Quad => "/QUAD",
    Random => "/RANDOM",
    Rbody => "/RBODY",
    RbodyOff => "/RBODY/OFF",
    RbodyOn => "/RBODY/ON",
    Refsta => "/REFSTA",
    Rfile => "/RFILE",
    Run => "/RUN",
    RwallCyl => "/RWALL/CYL",
    RwallParal => "/RWALL/PARAL",
    RwallPlane => "/RWALL/PLANE",
    RwallSpher => "/RWALL/SPHER",
    Sect => "/SECT",
    SensorSens => "/SENSOR/SENS",
    SensorTime => "/SENSOR/TIME",
    Sh3n => "/SH3N",
    Shell => "/SHELL",
    ShfraV4 => "/SHFRA/V4",
    SkewFix => "/SKEW/FIX",
    SkewMov => "/SKEW/MOV",
    SphbcsSlide => "/SPHBCS/SLIDE",
    SphbcsTied => "/SPHBCS/TIED",
    Sphglo => "/SPHGLO",
    Spmd => "/SPMD",
    Spring => "/SPRING",
    Stop => "/STOP",
    Subset => "/SUBSET",
    SurfGrshel => "/SURF/GRSHEL",
    SurfPart => "/SURF/PART",
    SurfPartAll => "/SURF/PART/ALL",
    SurfSeg => "/SURF/SEG",
    Tetra4 => "/TETRA4",
    Tfile => "/TFILE",
    ThAccel => "/TH/ACCEL",
    ThBeam => "/TH/BEAM",
    ThBric => "/TH/BRIC",
    ThBrick => "/TH/BRICK",
    ThInter => "/TH/INTER",
    ThMonv => "/TH/MONV",
    ThMonvol => "/TH/MONVOL",
    ThNode => "/TH/NODE",
    ThPart => "/TH/PART",
    ThQuad => "/TH/QUAD",
    ThRbody => "/TH/RBODY",
    ThRwall => "/TH/RWALL",
    ThSectio => "/TH/SECTIO",
    ThSh3n => "/TH/SH3N",
    ThShel => "/TH/SHEL",
    ThShell => "/TH/SHELL",
    ThSphcel => "/TH/SPHCEL",
    ThSpring => "/TH/SPRING",
    ThSubs => "/TH/SUBS",
    ThTitle => "/TH/TITLE",
    ThTruss => "/TH/TRUSS",
    ThVers => "/TH/VERS",
    Title => "/TITLE",
    TransformRot => "/TRANSFORM/ROT",
    TransformSca => "/TRANSFORM/SCA",
    TransformSym => "/TRANSFORM/SYM",
    TransformTra => "/TRANSFORM/TRA",
    Truss => "/TRUSS",
    Unit => "/UNIT",
    UnitLength => "/UNIT/LENGTH",
    UnitMass => "/UNIT/MASS",
    UnitMassM => "/UNIT/MASS/M",
    UnitTime => "/UNIT/TIME",
    VelTraXyz => "/VEL/TRA/XYZ",
    VelTraZ => "/VEL/TRA/Z",
    Vers => "/VERS",
    Xref => "/XREF",
    // Structural extras below: headers the 319-line inventory misses.
    // See CardKind::is_extra for rationale.
    EulerMat => "/EULER/MAT",
    Sphcel => "/SPHCEL",
    Submodel => "//SUBMODEL",
    Endsub => "//ENDSUB",
    InivelTG => "/INIVEL/T+G",
}

impl CardKind {
    /// Reports inventory extras needed for corpus coverage.
    ///
    /// `/EULER/MAT` and `/SPHCEL` appear only inside `#include` files,
    /// `//SUBMODEL` and `//ENDSUB` are submodel section markers,
    /// `/INIVEL/T+G` is a distinct ALE type (Reference Guide `/INIVEL`,
    /// p.2070) that the inventory generator truncated to `/INIVEL/T`,
    /// and `/TH/TITLE` plus `/MAT/LAW2` are writer-emitted,
    /// starter-accepted cards no miniqa deck uses (gate amendment:
    /// our engine decks request `/TH/TITLE` for full channel names).
    /// Bare `/DT/BRICK` is the engine dt-scale card our foam decks
    /// carry; no miniqa deck uses it, so it stays an extra.
    #[must_use]
    pub const fn is_extra(self) -> bool {
        matches!(
            self,
            Self::EulerMat
                | Self::Sphcel
                | Self::Submodel
                | Self::Endsub
                | Self::InivelTG
                | Self::ThTitle
                | Self::MatLaw2
                | Self::DtBrick
        )
    }

    /// Reports families whose first data line is always a title.
    ///
    /// Most cards detect titles per block (a leading non-numeric line),
    /// but `/PART`, `/TH/*`, and `/SKEW/*` blocks in the corpus carry
    /// purely numeric titles (`1`, `11`), so they take the first data
    /// line as the title unconditionally, as do `/TITLE` and `/BEGIN`
    /// whose first line is definitionally the title and run name.
    #[must_use]
    pub const fn force_title(self) -> bool {
        matches!(
            self,
            Self::Part
                | Self::Title
                | Self::Begin
                | Self::SkewFix
                | Self::SkewMov
                | Self::ThAccel
                | Self::ThBeam
                | Self::ThBric
                | Self::ThBrick
                | Self::ThInter
                | Self::ThMonv
                | Self::ThMonvol
                | Self::ThNode
                | Self::ThPart
                | Self::ThQuad
                | Self::ThRbody
                | Self::ThRwall
                | Self::ThSectio
                | Self::ThSh3n
                | Self::ThShel
                | Self::ThShell
                | Self::ThSphcel
                | Self::ThSpring
                | Self::ThSubs
                | Self::ThTitle
                | Self::ThTruss
                | Self::ThVers
        )
    }
}

impl fmt::Display for CardKind {
    /// Renders the canonical header text.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Splits a raw header into segments, trimming inner padding.
///
/// Returns `None` for empty middle segments so malformed headers such as
/// `/A//B` fail closed instead of silently collapsing.
fn split_header(line: &str) -> Option<Vec<&str>> {
    let mut segments = Vec::new();
    for (index, part) in line.split('/').enumerate() {
        if index == 0 {
            continue;
        }
        let trimmed = part.trim();
        if trimmed.is_empty() {
            return None;
        }
        segments.push(trimmed);
    }
    if segments.is_empty() {
        return None;
    }
    Some(segments)
}

/// Parses a header segment as a block identifier.
fn parse_id(segment: &str) -> Option<i64> {
    segment.parse::<i64>().ok()
}

/// Normalizes a raw deck header line into a family and identifier.
///
/// Trims padding, drops a trailing slash, uppercases the family path for
/// matching, and applies the documented special cases (`/RUN` names,
/// `/PRINT` parameters, `/INIVEL/T+G`, `/UNIT` suffixes, double-numeric
/// `/TH` headers) before exact matching and stripping one trailing
/// numeric identifier.
///
/// # Examples
///
/// ```rust
/// use sonium_qa::rad_cards::{CardKind, parse_header};
///
/// assert_eq!(
///     parse_header("/RUN/CT3AV4/1"),
///     Some((CardKind::Run, Some(1), None))
/// );
/// assert_eq!(
///     parse_header("/TH/NODE    /         1/         0"),
///     Some((CardKind::ThNode, Some(1), Some(0)))
/// );
/// assert_eq!(
///     parse_header("/GRNOD/NODE/1/1"),
///     Some((CardKind::GrnodNode, Some(1), Some(1)))
/// );
/// ```
#[must_use]
pub fn parse_header(raw: &str) -> Option<(CardKind, Option<i64>, Option<i64>)> {
    let line = raw.trim().trim_end_matches('/');
    if !line.starts_with('/') {
        return None;
    }
    if line.starts_with("//") {
        return parse_submodel(line);
    }
    let segments = split_header(line)?;
    let upper: Vec<String> = segments.iter().map(|s| s.to_ascii_uppercase()).collect();
    let joined = format!("/{}", upper.join("/"));
    if joined == "/RUN" || joined.starts_with("/RUN/") {
        let id = upper.last().and_then(|s| parse_id(s));
        return Some((CardKind::Run, id, None));
    }
    if joined == "/PRINT" || joined.starts_with("/PRINT/") {
        let kind = if upper.get(1).is_some_and(|s| s == "100") {
            CardKind::Print100
        } else {
            CardKind::Print
        };
        return Some((kind, None, None));
    }
    if joined == "/INIVEL/T+G" || joined.starts_with("/INIVEL/T+G/") {
        let id = upper.last().and_then(|s| parse_id(s));
        return Some((CardKind::InivelTG, id, None));
    }
    if let Some(kind) = unit_kind(&upper) {
        return Some((kind, None, None));
    }
    if joined.starts_with("/TH/") {
        return parse_th(&upper);
    }
    if let Some(kind) = CardKind::parse(&joined) {
        return Some((kind, None, None));
    }
    if let Some(last) = upper.last()
        && let Some(id) = parse_id(last)
    {
        let head = format!("/{}", upper[..upper.len() - 1].join("/"));
        if let Some(kind) = CardKind::parse(&head) {
            return Some((kind, Some(id), None));
        }
        // Double-numeric headers (`/GRNOD/NODE/1/1`, `/MAT/LAW2/1/1`):
        // vendor starter accepts them and our writer emits them, so the
        // second qualifier is kept as the sub identifier. One-strip is
        // tried first so kinds with embedded numbers
        // (`/GRBRIC/BOX2/1003/1`, `/PROC/1/0`) keep their readings.
        if upper.len() >= 3
            && let Some(first) = parse_id(&upper[upper.len() - 2])
        {
            let head = format!("/{}", upper[..upper.len() - 2].join("/"));
            if let Some(kind) = CardKind::parse(&head) {
                return Some((kind, Some(first), Some(id)));
            }
        }
    }
    None
}

/// Parses `//SUBMODEL` section markers.
fn parse_submodel(line: &str) -> Option<(CardKind, Option<i64>, Option<i64>)> {
    let upper = line.to_ascii_uppercase();
    if upper == "//SUBMODEL" || upper.starts_with("//SUBMODEL/") {
        let id = upper.rsplit('/').next().and_then(parse_id);
        return Some((CardKind::Submodel, id, None));
    }
    if upper == "//ENDSUB" {
        return Some((CardKind::Endsub, None, None));
    }
    None
}

/// Matches `/UNIT` work-system suffixes such as `/UNIT/MASS/Mg`.
///
/// Numeric suffixes (`/UNIT/1`) fall through to generic identifier
/// handling, as does the exact inventory header `/UNIT/MASS/M`; only
/// other non-numeric suffixes map here with no identifier.
fn unit_kind(upper: &[String]) -> Option<CardKind> {
    if upper.len() != 3 || upper[0] != "UNIT" {
        return None;
    }
    if parse_id(&upper[2]).is_some() || upper[2] == "M" {
        return None;
    }
    match upper[1].as_str() {
        "LENGTH" => Some(CardKind::UnitLength),
        "MASS" => Some(CardKind::UnitMass),
        "TIME" => Some(CardKind::UnitTime),
        _ => None,
    }
}

/// Parses `/TH` headers with one or two trailing numeric segments.
///
/// `/TH/NODE/1/0` keeps the first numeric as the group identifier and the
/// trailing qualifier as the sub identifier; canonical output re-emits
/// both. A third or later numeric is dropped by design (no corpus case).
fn parse_th(upper: &[String]) -> Option<(CardKind, Option<i64>, Option<i64>)> {
    let mut head = upper.to_vec();
    let mut ids = Vec::new();
    while head.len() > 2 {
        let Some(id) = head.last().and_then(|s| parse_id(s)) else {
            break;
        };
        ids.push(id);
        head.pop();
    }
    let kind = CardKind::parse(&format!("/{}", head.join("/")))?;
    ids.reverse();
    let mut ids = ids.into_iter();
    Some((kind, ids.next(), ids.next()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    /// Coordinator inventory embedded so the enum cannot drift silently.
    const INVENTORY: &str = include_str!("../../../miniqa_cards.txt");

    #[test]
    fn inventory_matches_enum() {
        let lines: Vec<&str> = INVENTORY.lines().collect();
        assert_eq!(lines.len(), INVENTORY_COUNT);
        let mut kinds = HashSet::new();
        for line in &lines {
            let kind = CardKind::parse(line)
                .unwrap_or_else(|| panic!("inventory header has no variant: {line}"));
            assert_eq!(kind.as_str(), *line);
            assert!(kinds.insert(kind), "duplicate variant for {line}");
            assert!(!kind.is_extra(), "inventory header marked extra: {line}");
        }
        assert_eq!(CardKind::ALL.len(), INVENTORY_COUNT + 8);
        assert_eq!(CardKind::ALL.iter().filter(|k| k.is_extra()).count(), 8);
    }

    #[test]
    fn header_normalization() {
        let cases: &[(&str, CardKind, Option<i64>)] = &[
            ("/PART/1", CardKind::Part, Some(1)),
            ("/GRNOD/NODE/2", CardKind::GrnodNode, Some(2)),
            ("/MAT/HYD_VISC/0601", CardKind::MatHydVisc, Some(601)),
            ("/FUNCT/       9", CardKind::Funct, Some(9)),
            ("/SHELL/1", CardKind::Shell, Some(1)),
            ("/NODE", CardKind::Node, None),
            ("/END", CardKind::End, None),
            ("/END/ENGINE", CardKind::EndEngine, None),
            ("/BEGIN/", CardKind::Begin, None),
            ("/BEGIN/0", CardKind::Begin, Some(0)),
            ("/RANDOM/0", CardKind::Random, Some(0)),
            ("/TITLE ", CardKind::Title, None),
            ("/TFILE/4", CardKind::Tfile, Some(4)),
            ("/TFILE   ", CardKind::Tfile, None),
            ("/VERS/2019", CardKind::Vers, Some(2019)),
            ("/RUN/CT3AV4/1", CardKind::Run, Some(1)),
            ("/RUN/model_00/1/", CardKind::Run, Some(1)),
            ("/RUN/model", CardKind::Run, None),
            ("/PRINT/-100/55", CardKind::Print, None),
            ("/PRINT/-1", CardKind::Print, None),
            ("/PRINT/ -10", CardKind::Print, None),
            ("/PRINT/100", CardKind::Print100, None),
            ("/PRINT/100/55", CardKind::Print100, None),
            ("/PRINT/1000", CardKind::Print, None),
            ("/INIVEL/T+G/2", CardKind::InivelTG, Some(2)),
            ("/UNIT/MASS/Mg", CardKind::UnitMass, None),
            ("/UNIT/LENGTH/mm", CardKind::UnitLength, None),
            ("/UNIT/TIME/s", CardKind::UnitTime, None),
            ("/UNIT/MASS/M", CardKind::UnitMassM, None),
            ("/UNIT/1", CardKind::Unit, Some(1)),
            ("/TH/NODE/2", CardKind::ThNode, Some(2)),
            // NOTE: "/TH/NODE/1/0" moved to double_numeric_headers_keep_sub_id.
            ("/TH/VERS/41", CardKind::ThVers, Some(41)),
            ("/ANIM/VERS/41", CardKind::AnimVers, Some(41)),
            ("/ADMAS/0/1", CardKind::Admas0, Some(1)),
            ("/BOX/RECTA/3/1", CardKind::BoxRecta3, Some(1)),
            ("/BOX/RECTA/2", CardKind::BoxRecta, Some(2)),
            ("/GRBRIC/BOX2/1003/1", CardKind::GrbricBox21003, Some(1)),
            ("/GRBRIC/BOX2/1002", CardKind::GrbricBox2, Some(1002)),
            ("/GRBRIC/BOX2/1003", CardKind::GrbricBox21003, None),
            ("/PROC/1/0", CardKind::Proc1, Some(0)),
            ("/PROC/8/0", CardKind::Proc8, Some(0)),
            ("/PROC/4", CardKind::Proc, Some(4)),
            ("/IMPVEL/         2/", CardKind::Impvel, Some(2)),
            ("/PROP/SH_SANDW/         2/", CardKind::PropShSandw, Some(2)),
            ("/EULER/MAT/0601", CardKind::EulerMat, Some(601)),
            ("/SPHCEL/1", CardKind::Sphcel, Some(1)),
            ("//SUBMODEL/1", CardKind::Submodel, Some(1)),
            ("//ENDSUB  ", CardKind::Endsub, None),
            ("/DT/BRICK", CardKind::DtBrick, None),
            ("/DT/BRICK/CST/0", CardKind::DtBrickCst, Some(0)),
            ("/DT/NODA/CST/0", CardKind::DtNodaCst, Some(0)),
        ];
        for (raw, kind, id) in cases {
            assert_eq!(parse_header(raw), Some((*kind, *id, None)), "raw: {raw}");
        }
    }

    #[test]
    fn double_numeric_headers_keep_sub_id() {
        let cases: &[(&str, CardKind, Option<i64>, Option<i64>)] = &[
            ("/GRNOD/NODE/1/1", CardKind::GrnodNode, Some(1), Some(1)),
            ("/MAT/LAW36/1/1", CardKind::MatLaw36, Some(1), Some(1)),
            ("/PROP/SOLID/1/1", CardKind::PropSolid, Some(1), Some(1)),
            ("/CLOAD/1/1", CardKind::Cload, Some(1), Some(1)),
            ("/PART/1/2", CardKind::Part, Some(1), Some(2)),
            (
                "/TH/NODE    /         1/         0",
                CardKind::ThNode,
                Some(1),
                Some(0),
            ),
        ];
        for (raw, kind, id, sub) in cases {
            assert_eq!(parse_header(raw), Some((*kind, *id, *sub)), "raw: {raw}");
        }
    }

    #[test]
    fn unknown_headers_fail_closed() {
        for raw in [
            "/NOPE",
            "/MAT/LAW999/1",
            "/TH/NODE/X",
            "/A//B",
            "/",
            "NODE",
            "",
            "//FROB/1",
            "/UNIT/MASS/Mg/1",
            // NOTE: "/PART/1/2" used to fail here; double-numeric headers
            // now parse (gate fix: our writer emits them, starter accepts).
        ] {
            assert_eq!(parse_header(raw), None, "raw: {raw}");
        }
    }

    #[test]
    fn display_matches_canonical() {
        assert_eq!(CardKind::ThNode.to_string(), "/TH/NODE");
        assert_eq!(CardKind::Submodel.to_string(), "//SUBMODEL");
    }
}
