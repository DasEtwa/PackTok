//! Read-only reproductions against the reviewed public APIs. No production patch.
use packtok_core::TokenId;
use packtok_model::{CausalLm, ModelConfig, ModelError, PackVocabulary};

fn invalid_prompt(model: &CausalLm, valid: TokenId, invalid: TokenId, label: &str) {
    assert!(model.hidden_states(&[invalid]).is_err());
    let prompt = [invalid, valid, valid];
    let generated = model.greedy_generate(&prompt, 1).unwrap();
    assert_eq!(generated[0], invalid);
    let zero = model.greedy_generate(&[invalid], 0).unwrap();
    assert_eq!(zero, vec![invalid]);
    println!(
        "generation {label}: invalid ID {invalid:?} is rejected by hidden_states but retained in successful generation; prompt_len={} context=2 new_tokens=1 output_len={} zero_new_tokens_also_accepts_invalid=true",
        prompt.len(),
        generated.len()
    );
}

fn main() {
    let config = ModelConfig {
        hidden_size: 2,
        context_length: 2,
    };
    invalid_prompt(
        &CausalLm::new_flat(config, 256, 19).unwrap(),
        TokenId::new(0, 97),
        TokenId::new(0, 256),
        "flat",
    );
    invalid_prompt(
        &CausalLm::new_factorized(
            config,
            vec![PackVocabulary {
                pack_id: 7,
                token_count: 256,
            }],
            19,
        )
        .unwrap(),
        TokenId::new(7, 97),
        TokenId::new(99, 0),
        "factorized",
    );
    // Sequential scopes keep the two large models from coexisting in memory.
    let config = ModelConfig {
        hidden_size: 8,
        context_length: 6,
    };
    {
        let model = CausalLm::new_flat(config, 986_888, 19).unwrap();
        let count = model.parameter_counts().total;
        assert_eq!(count, 16_777_216);
        assert_eq!(model.to_bytes(), Err(ModelError::ModelTooLarge));
        println!(
            "serialization flat: hidden=8 context=6 vocabulary=986888 seed=19 constructor=Ok parameters={count} required_wire_bytes={} file_cap_bytes=67108864 to_bytes=ModelTooLarge",
            count * 4 + 46
        );
    }
    {
        let model = CausalLm::new_factorized(
            config,
            vec![PackVocabulary {
                pack_id: 7,
                token_count: 986_887,
            }],
            19,
        )
        .unwrap();
        let count = model.parameter_counts().total;
        assert_eq!(count, 16_777_208);
        assert_eq!(model.to_bytes(), Err(ModelError::ModelTooLarge));
        println!(
            "serialization factorized: hidden=8 context=6 pack=7 vocabulary=986887 seed=19 constructor=Ok parameters={count} required_wire_bytes={} file_cap_bytes=67108864 to_bytes=ModelTooLarge",
            count * 4 + 46
        );
    }
    println!(
        "All review reproductions confirmed; no implementation or frozen artifact was changed."
    );
}
