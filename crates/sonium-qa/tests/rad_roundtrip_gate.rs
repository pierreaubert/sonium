//! Phase-4 gate cross-checks: writer output re-parses under the reader.
//!
//! Lanes 4A/4B/4C were built independently (no cross-lane imports); these
//! maintained tests prove the composition the gate requires: meshes built
//! by 4C and extra cards written by 4B both parse under the 4A reader with
//! the expected cards present, and both survive a parse/emit round trip.

use sonium_qa::rad::{
    RadDamping, RadExtra, cantilever_case, hertz_sphere_case, shell_clamped_plate_case,
    shell_scordelis_lo_case, write_engine, write_extended_engine, write_extended_starter,
    write_starter, write_starter_extra,
};
use sonium_qa::rad_cards::CardKind;
use sonium_qa::rad_mesh::{crash_box_case, shell_plate_case, taylor_bar_case};
use sonium_qa::rad_parse::RadDeck;

fn has_kind(deck: &RadDeck, kind: CardKind) -> bool {
    deck.cards.iter().any(|card| card.kind == kind)
}

/// Mesher case decks carry parseable mesh cards and round-trip stably.
#[test]
fn mesh_case_decks_reparse_with_mesh_cards() {
    let case = taylor_bar_case();
    assert!(case.mesh.num_nodes() > 0);
    assert!(case.mesh.num_bricks() > 0);

    let starter = write_starter(&case);
    let deck = RadDeck::parse_str(&starter).expect("starter parses");
    assert!(has_kind(&deck, CardKind::Node));
    assert!(has_kind(&deck, CardKind::Brick));
    let node_lines: usize = deck
        .cards
        .iter()
        .filter(|card| card.kind == CardKind::Node)
        .map(|card| card.fields.len())
        .sum();
    assert!(
        node_lines >= case.mesh.num_nodes(),
        "node lines {node_lines} cover {} mesh nodes",
        case.mesh.num_nodes()
    );
    let again = deck.round_trip().expect("starter round-trips");
    assert_eq!(again.cards.len(), deck.cards.len());

    let engine = write_engine(&case);
    let engine_deck = RadDeck::parse_str(&engine).expect("engine parses");
    let engine_again = engine_deck.round_trip().expect("engine round-trips");
    assert_eq!(engine_again.cards.len(), engine_deck.cards.len());
}

/// Every case builder's starter and engine decks parse and round-trip.
#[test]
fn all_case_builders_reparse() {
    for case in [
        cantilever_case(),
        hertz_sphere_case(),
        taylor_bar_case(),
        crash_box_case(),
    ] {
        let starter = write_starter(&case);
        let deck = RadDeck::parse_str(&starter).unwrap_or_else(|e| panic!("starter parses: {e}"));
        assert_eq!(
            deck.round_trip().expect("round-trips").cards.len(),
            deck.cards.len()
        );
        let engine = write_engine(&case);
        let engine_deck =
            RadDeck::parse_str(&engine).unwrap_or_else(|e| panic!("engine parses: {e}"));
        assert_eq!(
            engine_deck.round_trip().expect("round-trips").cards.len(),
            engine_deck.cards.len()
        );
    }
    for case in [
        shell_scordelis_lo_case(),
        shell_clamped_plate_case(),
        shell_plate_case(),
    ] {
        let starter = write_extended_starter(&case);
        let deck =
            RadDeck::parse_str(&starter).unwrap_or_else(|e| panic!("ext starter parses: {e}"));
        assert_eq!(
            deck.round_trip().expect("round-trips").cards.len(),
            deck.cards.len()
        );
        let engine = write_extended_engine(&case);
        let engine_deck =
            RadDeck::parse_str(&engine).unwrap_or_else(|e| panic!("ext engine parses: {e}"));
        assert_eq!(
            engine_deck.round_trip().expect("round-trips").cards.len(),
            engine_deck.cards.len()
        );
    }
}

/// Writer-extra cards parse under the reader with the card present.
#[test]
fn writer_extra_damping_reparses() {
    let mut extra = RadExtra::default();
    extra.damps.push(RadDamping {
        id: 1,
        title: "gate".to_string(),
        alpha: 1.5,
        beta_s: 0.0,
        grnd: 1,
        skew: 0,
        tstart_s: 0.0,
        tstop_s: 1.0,
    });
    let text = write_starter_extra(&extra, 1);
    assert!(text.contains("/DAMP"));
    let deck = RadDeck::parse_str(&text).expect("extra parses");
    assert!(has_kind(&deck, CardKind::Damp));
    let again = deck.round_trip().expect("extra round-trips");
    assert_eq!(again.cards.len(), deck.cards.len());
}

/// Bare `/DT/BRICK` dt-scale cards parse (foam engine legs carry one).
#[test]
fn bare_dt_brick_card_parses() {
    // Byte-faithful to decks/foam76_compression/foam76_compression_0001.rad.
    let text = "#RADIOSS ENGINE\n/RUN/foam76_compression/1\n  3.000000000000E-03\n/TFILE\n  2.500000000000E-05\n/DT/BRICK\n  5.000000000000E-02  0.000000000000E+00\n/PRINT/-1000\n";
    let deck = RadDeck::parse_str(text).expect("foam engine text parses");
    let card = deck
        .cards
        .iter()
        .find(|card| card.kind == CardKind::DtBrick)
        .expect("DT/BRICK card present");
    assert_eq!(
        card.fields,
        vec![vec![
            "5.000000000000E-02".to_string(),
            "0.000000000000E+00".to_string()
        ]]
    );
    let again = deck.round_trip().expect("engine round-trips");
    assert_eq!(again.cards.len(), deck.cards.len());
}
