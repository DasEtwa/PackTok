use packtok_packs::semantic::*;
use packtok_format::semantic::*;
fn main() {
    let r = SemanticRegistry::builtin();
    for s in [format!("xlanguage:{}fr", " ".repeat(119)), format!("xlanguage:{}fr", " ".repeat(118)), "fr\u{000b}bro".to_owned(), "fr\u{000c}bro".to_owned(), "language: 'fr'".to_owned(), "'language': \"fr\"".to_owned(), "\"language\": \"fr\"".to_owned()] {
        println!("{:?}: {:?}",s,r.annotate(&s));
    }
    let alphabet = ['i','d','k','f','r','b','o',' ','\n','\r','\u{000b}','\u{0085}','é','\u{0301}','👩','\u{200d}','\u{2019}','\'','"','\0'];
    let mut state = 0x5316_4832_u32;
    for len in 0..4000 {
        let mut s = String::new();
        for _ in 0..len % 200 {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            s.push(alphabet[state as usize % alphabet.len()]);
        }
        let a = r.annotate(&s);
        assert_eq!(a, r.annotate(&s));
        let mut end = 0;
        for span in a {
            assert!(span.start >= end && span.end > span.start);
            assert!(s.get(span.start..span.end).is_some());
            assert!(!span.candidates.is_empty());
            assert!(span.candidates.iter().all(|c|r.concept(*c).is_some()));
            assert_eq!(span.decision == SemanticDecision::Resolved,span.resolved_id().is_some());
            end = span.end;
        }
        let side = SemanticSidecar::annotate(&s,r);
        let bytes = side.to_bytes().unwrap();
        assert_eq!(SemanticSidecar::from_bytes(&bytes,&s,r).unwrap(),side);
    }
    println!("4000 generated UTF8 invariants + canonical sidecar roundtrips passed");
}
