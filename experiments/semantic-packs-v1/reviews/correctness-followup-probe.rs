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
    let mut checked = 0;
    for key in ["language", "lang", "sprache"] {
        for prefix in ["x", "_", "0", "é", "\u{0301}", "\u{200d}", "漢"] {
            for whitespace in [" ", "\t", "\u{00a0}", "\u{2003}"] {
                for count in 0..=140 {
                    let text = format!("{prefix}{key}:{}fr",whitespace.repeat(count));
                    let a = r.annotate(&text);
                    assert_eq!(a.len(),1,"{text:?}");
                    assert_eq!(a[0].resolved_id(),None,"{text:?}");
                    checked += 1;
                }
            }
        }
    }
    println!("{checked} near-limit metadata prefix/context probes passed");
    for sep in ['\u{000b}','\u{000c}','\r','\n','\u{0085}','\u{2028}','\u{2029}'] {
        let s = format!("fr{sep}bro");
        assert_eq!(r.annotate(&s)[0].resolved_id(),None,"{s:?}");
        let s = format!("language:{sep}fr");
        assert_eq!(r.annotate(&s)[0].resolved_id(),None,"{s:?}");
        let s = format!("keine{sep}Ahnung");
        assert!(r.annotate(&s).is_empty(),"{s:?}");
    }
    println!("All 7 vertical separator families rejected as phrase/context whitespace");
    for (open,close) in [("'","'"),("’","’"),("\"","\"")] {
        let s = format!("{open}language{close}: {open}fr{close}");
        assert_eq!(r.annotate(&s)[0].resolved_id(),Some(FRENCH_LANGUAGE),"{s:?}");
        for prefix in ["x","é","_","漢"] {
            let s = format!("{prefix}{open}language{close}: {open}fr{close}");
            let a = r.annotate(&s);
            assert!(a.is_empty() || a[0].resolved_id().is_none(),"{s:?}");
        }
    }
    for s in ["idk's", "idk’s", "idk'x", "idk’x", "x'idk", "x’idk", "é'idk", "idk'é"] {
        assert!(r.annotate(s).is_empty(),"{s:?}");
    }
    println!("Simple quote delimiters accepted; intra-word apostrophe protections preserved");
    use packtok_tokenizer::{ByteFallbackTokenizer, encode_with_semantics};
    let custom = SemanticRegistry::new("review.custom",1,vec![SemanticConcept{id:UNCERTAINTY,name:"CUSTOM".into(),description:"Review identity collision".into()}],vec![SemanticVariant{pattern:"idk".into(),candidates:vec![UNCERTAINTY],rule:ResolutionRule::Direct}]).unwrap();
    let t=ByteFallbackTokenizer::default();
    let a=encode_with_semantics(&t,"idk",Some(r)).unwrap();
    let b=encode_with_semantics(&t,"idk",Some(&custom)).unwrap();
    assert_eq!(a.lexical_tokens,b.lexical_tokens);
    assert_eq!(a.semantic_annotations,b.semantic_annotations);
    assert_ne!(a,b);
    assert_eq!(a.semantic_registry,Some(r.identity()));
    let off=encode_with_semantics(&t,"idk",None).unwrap();
    assert_eq!(off.semantic_registry,None);
    assert_eq!(off.semantic_annotations,None);
    println!("Registry identity collision and disabled wrapper checks passed");
}
