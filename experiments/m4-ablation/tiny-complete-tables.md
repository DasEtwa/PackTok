# Derived M4 tables

Input: experiments/m4-ablation/runs/tiny-final-20261007-v2/run-summary.txt. Arithmetic means and population SD; rounded log values are used. Raw logs and Rust summaries retain additional precision. Peak memory is the shared process high-water mark at each result, not isolated model memory. MACs exclude non-matrix work.

## schedule regime

| Variant | Test bits/byte ± SD | Parameters | Updates per seed | MACs ± SD | Train ms ± SD | Test ms ± SD | Process peak B range |
|---|---:|---:|---|---:|---:|---:|---|
| A | 6.875202 ± 0.146236 | 17424 | 120, 120, 120 | 194519040.0 ± 0.0 | 289.144 ± 5.095 | 10.186 ± 0.281 | 5951488–6213632 |
| B | 6.510746 ± 0.154847 | 17475 | 120, 120, 120 | 69806048.0 ± 1034.2 | 150.228 ± 4.238 | 6.254 ± 0.083 | 6017024–6213632 |
| C | 5.973783 ± 0.006542 | 17424 | 120, 120, 120 | 194519040.0 ± 0.0 | 289.848 ± 2.394 | 11.361 ± 0.363 | 6037504–6213632 |
| D | 6.007689 ± 0.054984 | 17475 | 120, 120, 120 | 93921472.0 ± 35400.8 | 187.154 ± 4.958 | 9.155 ± 1.209 | 6066176–6213632 |

| Seed | B-A | C-A | D-A | (D-C)-(B-A) |
|---|---:|---:|---:|---:|
| 20261007 | -0.23775460 | -0.72826900 | -0.76200759 | 0.20401601 |
| 20261008 | -0.34984910 | -1.09486228 | -1.07332671 | 0.37138467 |
| 20261009 | -0.50576492 | -0.88112562 | -0.76720315 | 0.61968739 |
| mean ± population SD | -0.36445621 ± 0.10990119 | -0.90141897 ± 0.15034743 | -0.86751248 ± 0.14554809 | 0.39836269 ± 0.17076599 |

## mac regime

| Variant | Test bits/byte ± SD | Parameters | Updates per seed | MACs ± SD | Train ms ± SD | Test ms ± SD | Process peak B range |
|---|---:|---:|---|---:|---:|---:|---|
| A | 6.875202 ± 0.146236 | 17424 | 120, 120, 120 | 194519040.0 ± 0.0 | 284.549 ± 2.731 | 10.543 ± 0.381 | 6066176–6213632 |
| B | 7.647585 ± 0.077164 | 17475 | 335, 335, 335 | 194876016.0 ± 296.8 | 362.642 ± 5.002 | 6.746 ± 0.142 | 6066176–6213632 |
| C | 5.973783 ± 0.006542 | 17424 | 120, 120, 120 | 194519040.0 ± 0.0 | 292.575 ± 4.080 | 11.228 ± 0.274 | 6066176–6213632 |
| D | 6.095338 ± 0.187750 | 17475 | 249, 249, 249 | 194836320.0 ± 159738.2 | 333.653 ± 2.351 | 8.159 ± 0.267 | 6070272–6225920 |

| Seed | B-A | C-A | D-A | (D-C)-(B-A) |
|---|---:|---:|---:|---:|
| 20261007 | 0.86706309 | -0.72826900 | -0.35077282 | -0.48956691 |
| 20261008 | 0.54330743 | -1.09486228 | -1.08753463 | -0.53597978 |
| 20261009 | 0.90678067 | -0.88112562 | -0.90128469 | -0.92693974 |
| mean ± population SD | 0.77238373 ± 0.16279094 | -0.90141897 ± 0.15034743 | -0.77986405 ± 0.31279570 | -0.65082881 ± 0.19615720 |

## Per-seed measurements

| Regime | Variant | Seed | Test bits/byte | NLL/byte | Loss/token | B/token | Updates | Targets | Train target bytes | MACs | Train ms | Curve validation ms | Final validation ms | Test ms | Model B | Head params | Parameter/Adam/gradient B |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| schedule | A | 20261007 | 6.71100401 | 4.65171351 | 7.18901178 | 1.54659950 | 120 | 7680 | 23098 | 194519040 | 292.627 | 49.075 | 9.850 | 10.068 | 69742 | 8704 | 278784 |
| schedule | B | 20261007 | 6.47324941 | 4.48691458 | 6.93432253 | 1.54659950 | 120 | 7680 | 23098 | 69805440 | 156.202 | 33.694 | 6.175 | 6.192 | 69958 | 8755 | 279600 |
| schedule | C | 20261007 | 5.98273501 | 4.14691591 | 5.98564277 | 1.44470588 | 120 | 7680 | 16303 | 194519040 | 293.042 | 53.773 | 10.834 | 10.966 | 69742 | 8704 | 278784 |
| schedule | D | 20261007 | 5.94899642 | 4.12353010 | 5.95188778 | 1.44470588 | 120 | 7680 | 16303 | 93874368 | 189.301 | 38.550 | 7.671 | 8.294 | 69958 | 8755 | 279600 |
| mac | A | 20261007 | 6.71100401 | 4.65171351 | 7.18901178 | 1.54659950 | 120 | 7680 | 23098 | 194519040 | 287.927 | 49.757 | 10.406 | 10.627 | 69742 | 8704 | 278784 |
| mac | B | 20261007 | 7.57806710 | 5.25271584 | 8.11783358 | 1.54659950 | 335 | 21440 | 64343 | 194876176 | 367.751 | 30.159 | 6.179 | 6.567 | 69958 | 8755 | 279600 |
| mac | C | 20261007 | 5.98273501 | 4.14691591 | 5.98564277 | 1.44470588 | 120 | 7680 | 16303 | 194519040 | 291.024 | 53.288 | 10.795 | 11.307 | 69742 | 8704 | 278784 |
| mac | D | 20261007 | 6.36023119 | 4.40857631 | 6.36332242 | 1.44470588 | 249 | 15936 | 34129 | 194691360 | 330.361 | 39.576 | 7.785 | 8.157 | 69958 | 8755 | 279600 |
| schedule | A | 20261008 | 7.06618987 | 4.89790958 | 7.56949663 | 1.54659950 | 120 | 7680 | 23272 | 194519040 | 292.865 | 49.976 | 10.232 | 10.574 | 69742 | 8704 | 278784 |
| schedule | B | 20261008 | 6.71634077 | 4.65541267 | 7.19472867 | 1.54659950 | 120 | 7680 | 23272 | 69807504 | 146.823 | 29.830 | 5.852 | 6.199 | 69958 | 8755 | 279600 |
| schedule | C | 20261008 | 5.97132759 | 4.13900888 | 5.97422980 | 1.44470588 | 120 | 7680 | 16413 | 194519040 | 289.221 | 53.068 | 10.880 | 11.274 | 69742 | 8704 | 278784 |
| schedule | D | 20261008 | 5.99286316 | 4.15393620 | 5.99577584 | 1.44470588 | 120 | 7680 | 16413 | 93930336 | 191.861 | 41.109 | 10.840 | 10.864 | 69958 | 8755 | 279600 |
| mac | A | 20261008 | 7.06618987 | 4.89790958 | 7.56949663 | 1.54659950 | 120 | 7680 | 23272 | 194519040 | 284.481 | 48.932 | 9.660 | 10.040 | 69742 | 8704 | 278784 |
| mac | B | 20261008 | 7.60949730 | 5.27450160 | 8.15150247 | 1.54659950 | 335 | 21440 | 64382 | 194875600 | 355.850 | 29.328 | 6.400 | 6.913 | 69958 | 8755 | 279600 |
| mac | C | 20261008 | 5.97132759 | 4.13900888 | 5.97422980 | 1.44470588 | 120 | 7680 | 16413 | 194519040 | 298.163 | 58.012 | 11.559 | 11.518 | 69742 | 8704 | 278784 |
| mac | D | 20261008 | 5.97865524 | 4.14408802 | 5.98156101 | 1.44470588 | 249 | 15936 | 34112 | 195058848 | 335.706 | 39.754 | 8.215 | 8.487 | 69958 | 8755 | 279600 |
| schedule | A | 20261009 | 6.84841138 | 4.74695704 | 7.33620633 | 1.54659950 | 120 | 7680 | 22968 | 194519040 | 281.939 | 48.023 | 9.480 | 9.916 | 69742 | 8704 | 278784 |
| schedule | B | 20261009 | 6.34264646 | 4.39638751 | 6.79441706 | 1.54659950 | 120 | 7680 | 22968 | 69805200 | 147.658 | 30.378 | 5.802 | 6.372 | 69958 | 8755 | 279600 |
| schedule | C | 20261009 | 5.96728576 | 4.13620730 | 5.97018600 | 1.44470588 | 120 | 7680 | 16587 | 194519040 | 287.280 | 53.075 | 10.848 | 11.843 | 69742 | 8704 | 278784 |
| schedule | D | 20261009 | 6.08120823 | 4.21517234 | 6.08416384 | 1.44470588 | 120 | 7680 | 16587 | 93959712 | 180.299 | 38.684 | 8.060 | 8.306 | 69958 | 8755 | 279600 |
| mac | A | 20261009 | 6.84841138 | 4.74695704 | 7.33620633 | 1.54659950 | 120 | 7680 | 22968 | 194519040 | 281.239 | 46.513 | 10.153 | 10.962 | 69742 | 8704 | 278784 |
| mac | B | 20261009 | 7.75519205 | 5.37548950 | 8.30757469 | 1.54659950 | 335 | 21440 | 63937 | 194876272 | 364.325 | 30.171 | 6.342 | 6.758 | 69958 | 8755 | 279600 |
| mac | C | 20261009 | 5.96728576 | 4.13620730 | 5.97018600 | 1.44470588 | 120 | 7680 | 16587 | 194519040 | 288.537 | 53.634 | 10.786 | 10.860 | 69742 | 8704 | 278784 |
| mac | D | 20261009 | 5.94712669 | 4.12223410 | 5.95001714 | 1.44470588 | 249 | 15936 | 34357 | 194758752 | 334.892 | 39.687 | 7.925 | 7.832 | 69958 | 8755 | 279600 |

## Raw tokenizer, head and generation observations

| Regime | Variant | NLL/byte mean ± SD | Loss/token mean ± SD | Final validation bits/byte mean ± SD | Final validation ms mean ± SD |
|---|---|---:|---:|---:|---:|
| schedule | A | 4.765527 ± 0.101363 | 7.364905 ± 0.156652 | 6.206782 ± 0.125027 | 9.854000 ± 0.307016 |
| schedule | B | 4.512905 ± 0.107332 | 6.974489 ± 0.165876 | 5.944352 ± 0.122157 | 5.943000 ± 0.165314 |
| schedule | C | 4.140711 ± 0.004534 | 5.976686 ± 0.006545 | 6.014272 ± 0.002306 | 10.854000 ± 0.019253 |
| schedule | D | 4.164213 ± 0.038112 | 6.010609 ± 0.055011 | 6.044873 ± 0.077277 | 8.857000 ± 1.411157 |
| mac | A | 4.765527 ± 0.101363 | 7.364905 ± 0.156652 | 6.206782 ± 0.125027 | 10.073000 ± 0.309762 |
| mac | B | 5.300902 ± 0.053486 | 8.192304 ± 0.082660 | 7.143241 ± 0.104303 | 6.307000 ± 0.093556 |
| mac | C | 4.140711 ± 0.004534 | 5.976686 ± 0.006545 | 6.014272 ± 0.002306 | 11.046667 ± 0.362293 |
| mac | D | 4.224966 ± 0.130138 | 6.098300 ± 0.187841 | 6.069545 ± 0.167201 | 7.975000 ± 0.179072 |

| Regime | Factorized variant | Pack accuracy mean ± SD | Local accuracy under gold pack mean ± SD | Active local rows | Total logits/target | Total test logits | Head parameters |
|---|---|---:|---:|---:|---:|---:|---:|
| schedule | B | 0.303872 ± 0.014628 | 0.039562 ± 0.007241 | 170.659091 | 173.659091 | 68769 | 8755 |
| schedule | D | 0.516509 ± 0.140839 | 0.238208 ± 0.000000 | 246.683962 | 249.683962 | 105866 | 8755 |
| mac | B | 0.356061 ± 0.016366 | 0.037037 ± 0.004762 | 170.659091 | 173.659091 | 68769 | 8755 |
| mac | D | 0.590409 ± 0.001112 | 0.263364 ± 0.008017 | 246.683962 | 249.683962 | 105866 | 8755 |

```text
tokenizer_training m1_ms=31.827 m2_ms=15.401 m1_bytes=3949 m2_bytes=4464 both_trained_twice_identical=true m2_allocation=[FactorizedPackTrainingStats { pack_id: 65535, spans: 0, routed_bytes: 0, learned_merges: 0, local_vocab_size: 0 }, FactorizedPackTrainingStats { pack_id: 0, spans: 473, routed_bytes: 2032, learned_merges: 253, local_vocab_size: 253 }, FactorizedPackTrainingStats { pack_id: 1, spans: 4, routed_bytes: 8, learned_merges: 0, local_vocab_size: 0 }, FactorizedPackTrainingStats { pack_id: 2, spans: 477, routed_bytes: 565, learned_merges: 3, local_vocab_size: 3 }]
encode tokenizer=M1 split=0 bytes=2605 tokens=872 bytes_per_token=2.98738532 ms=0.366 mib_per_s=6.785910
encode tokenizer=M2 split=0 bytes=2605 tokens=1213 bytes_per_token=2.14756801 ms=0.742 mib_per_s=3.346789
encode tokenizer=M1 split=1 bytes=604 tokens=375 bytes_per_token=1.61066667 ms=0.044 mib_per_s=12.944254
encode tokenizer=M2 split=1 bytes=604 tokens=418 bytes_per_token=1.44497608 ms=0.214 mib_per_s=2.694197
encode tokenizer=M1 split=2 bytes=614 tokens=397 bytes_per_token=1.54659950 ms=0.042 mib_per_s=13.842932
encode tokenizer=M2 split=2 bytes=614 tokens=425 bytes_per_token=1.44470588 ms=0.231 mib_per_s=2.538171
mapping variant=B rows=512 packs=[PackVocabulary { pack_id: 0, token_count: 171 }, PackVocabulary { pack_id: 1, token_count: 171 }, PackVocabulary { pack_id: 2, token_count: 170 }]
mapping variant=C rows=512 packs=[PackVocabulary { pack_id: 0, token_count: 253 }, PackVocabulary { pack_id: 2, token_count: 3 }, PackVocabulary { pack_id: 65535, token_count: 256 }]
heads regime=schedule variant=A seed=20261007 output_rows=512 logits_per_target=512 head_parameters=8704 measured_scoring_ms=10.068
generation regime=schedule variant=A seed=20261007 prompt_exact=true valid_ids=true decode_success=true selected_pack_counts={0: 8} hex=41742074686520686172626f7273207320732073207320732073207320
heads regime=schedule variant=B seed=20261007 pack_accuracy=0.323232 local_accuracy_given_gold_pack=0.047980 active_local_head_size_mean=170.659091 pack_head_logit_fraction=0.017275 local_head_logit_fraction=0.982725 pack_transitions=255 byte_fallback_targets=0/396 byte_fallback_fraction=0.000000 pack[0]={targets:104,accuracy:0.576923,local_count:171} pack[1]={targets:157,accuracy:0.433121,local_count:171} pack[2]={targets:135,accuracy:0.000000,local_count:170}
generation regime=schedule variant=B seed=20261007 prompt_exact=true valid_ids=true decode_success=true selected_pack_counts={0: 7, 1: 1} hex=41742074686520686172626f726572732073207070707070
heads regime=schedule variant=C seed=20261007 output_rows=512 logits_per_target=512 head_parameters=8704 measured_scoring_ms=10.966
generation regime=schedule variant=C seed=20261007 prompt_exact=true valid_ids=true decode_success=true selected_pack_counts={0: 8} hex=41742074686520686172626f722020202020202020
heads regime=schedule variant=D seed=20261007 pack_accuracy=0.582547 local_accuracy_given_gold_pack=0.238208 active_local_head_size_mean=246.683962 pack_head_logit_fraction=0.012015 local_head_logit_fraction=0.987985 pack_transitions=225 byte_fallback_targets=274/424 byte_fallback_fraction=0.646226 pack[0]={targets:136,accuracy:0.316176,local_count:253} pack[2]={targets:14,accuracy:0.000000,local_count:3} pack[65535]={targets:274,accuracy:0.744526,local_count:256}
generation regime=schedule variant=D seed=20261007 prompt_exact=true valid_ids=true decode_success=true selected_pack_counts={0: 4, 65535: 4} hex=41742074686520686172626f7220746865207468652074686520746865
heads regime=mac variant=A seed=20261007 output_rows=512 logits_per_target=512 head_parameters=8704 measured_scoring_ms=10.627
generation regime=mac variant=A seed=20261007 prompt_exact=true valid_ids=true decode_success=true selected_pack_counts={0: 8} hex=41742074686520686172626f7273207320732073207320732073207320
heads regime=mac variant=B seed=20261007 pack_accuracy=0.348485 local_accuracy_given_gold_pack=0.030303 active_local_head_size_mean=170.659091 pack_head_logit_fraction=0.017275 local_head_logit_fraction=0.982725 pack_transitions=255 byte_fallback_targets=0/396 byte_fallback_fraction=0.000000 pack[0]={targets:104,accuracy:0.557692,local_count:171} pack[1]={targets:157,accuracy:0.318471,local_count:171} pack[2]={targets:135,accuracy:0.222222,local_count:170}
generation regime=mac variant=B seed=20261007 prompt_exact=true valid_ids=true decode_success=true selected_pack_counts={0: 5, 2: 3} hex=41742074686520686172626f727770617468206a6f696e7320636f696e
heads regime=mac variant=C seed=20261007 output_rows=512 logits_per_target=512 head_parameters=8704 measured_scoring_ms=11.307
generation regime=mac variant=C seed=20261007 prompt_exact=true valid_ids=true decode_success=true selected_pack_counts={0: 8} hex=41742074686520686172626f722020202020202020
heads regime=mac variant=D seed=20261007 pack_accuracy=0.589623 local_accuracy_given_gold_pack=0.266509 active_local_head_size_mean=246.683962 pack_head_logit_fraction=0.012015 local_head_logit_fraction=0.987985 pack_transitions=225 byte_fallback_targets=274/424 byte_fallback_fraction=0.646226 pack[0]={targets:136,accuracy:0.367647,local_count:253} pack[2]={targets:14,accuracy:0.000000,local_count:3} pack[65535]={targets:274,accuracy:0.729927,local_count:256}
generation regime=mac variant=D seed=20261007 prompt_exact=true valid_ids=true decode_success=true selected_pack_counts={0: 4, 65535: 4} hex=41742074686520686172626f7220746865207468652074686520746865
heads regime=schedule variant=A seed=20261008 output_rows=512 logits_per_target=512 head_parameters=8704 measured_scoring_ms=10.574
generation regime=schedule variant=A seed=20261008 prompt_exact=true valid_ids=true decode_success=true selected_pack_counts={0: 8} hex=41742074686520686172626f7272657265726572657265726572657265
heads regime=schedule variant=B seed=20261008 pack_accuracy=0.287879 local_accuracy_given_gold_pack=0.030303 active_local_head_size_mean=170.659091 pack_head_logit_fraction=0.017275 local_head_logit_fraction=0.982725 pack_transitions=255 byte_fallback_targets=0/396 byte_fallback_fraction=0.000000 pack[0]={targets:104,accuracy:0.740385,local_count:171} pack[1]={targets:157,accuracy:0.114650,local_count:171} pack[2]={targets:135,accuracy:0.140741,local_count:170}
generation regime=schedule variant=B seed=20261008 prompt_exact=true valid_ids=true decode_success=true selected_pack_counts={0: 4, 1: 2, 2: 2} hex=41742074686520686172626f726572706f6f61746572726573207320
heads regime=schedule variant=C seed=20261008 output_rows=512 logits_per_target=512 head_parameters=8704 measured_scoring_ms=11.274
generation regime=schedule variant=C seed=20261008 prompt_exact=true valid_ids=true decode_success=true selected_pack_counts={0: 8} hex=41742074686520686172626f722020202020202020
heads regime=schedule variant=D seed=20261008 pack_accuracy=0.646226 local_accuracy_given_gold_pack=0.238208 active_local_head_size_mean=246.683962 pack_head_logit_fraction=0.012015 local_head_logit_fraction=0.987985 pack_transitions=225 byte_fallback_targets=274/424 byte_fallback_fraction=0.646226 pack[0]={targets:136,accuracy:0.007353,local_count:253} pack[2]={targets:14,accuracy:0.000000,local_count:3} pack[65535]={targets:274,accuracy:0.996350,local_count:256}
generation regime=schedule variant=D seed=20261008 prompt_exact=true valid_ids=true decode_success=true selected_pack_counts={65535: 8} hex=41742074686520686172626f722020202020202020
heads regime=mac variant=A seed=20261008 output_rows=512 logits_per_target=512 head_parameters=8704 measured_scoring_ms=10.040
generation regime=mac variant=A seed=20261008 prompt_exact=true valid_ids=true decode_success=true selected_pack_counts={0: 8} hex=41742074686520686172626f7272657265726572657265726572657265
heads regime=mac variant=B seed=20261008 pack_accuracy=0.378788 local_accuracy_given_gold_pack=0.040404 active_local_head_size_mean=170.659091 pack_head_logit_fraction=0.017275 local_head_logit_fraction=0.982725 pack_transitions=255 byte_fallback_targets=0/396 byte_fallback_fraction=0.000000 pack[0]={targets:104,accuracy:0.567308,local_count:171} pack[1]={targets:157,accuracy:0.382166,local_count:171} pack[2]={targets:135,accuracy:0.229630,local_count:170}
generation regime=mac variant=B seed=20261008 prompt_exact=true valid_ids=true decode_success=true selected_pack_counts={0: 1, 1: 4, 2: 3} hex=41742074686520686172626f72726f626f742063617272696573207468652063757020746f20746865207461626c652e20
heads regime=mac variant=C seed=20261008 output_rows=512 logits_per_target=512 head_parameters=8704 measured_scoring_ms=11.518
generation regime=mac variant=C seed=20261008 prompt_exact=true valid_ids=true decode_success=true selected_pack_counts={0: 8} hex=41742074686520686172626f722020202020202020
heads regime=mac variant=D seed=20261008 pack_accuracy=0.591981 local_accuracy_given_gold_pack=0.271226 active_local_head_size_mean=246.683962 pack_head_logit_fraction=0.012015 local_head_logit_fraction=0.987985 pack_transitions=225 byte_fallback_targets=274/424 byte_fallback_fraction=0.646226 pack[0]={targets:136,accuracy:0.375000,local_count:253} pack[2]={targets:14,accuracy:0.000000,local_count:3} pack[65535]={targets:274,accuracy:0.729927,local_count:256}
generation regime=mac variant=D seed=20261008 prompt_exact=true valid_ids=true decode_success=true selected_pack_counts={0: 4, 65535: 4} hex=41742074686520686172626f7220746865207468652074686520746865
heads regime=schedule variant=A seed=20261009 output_rows=512 logits_per_target=512 head_parameters=8704 measured_scoring_ms=9.916
generation regime=schedule variant=A seed=20261009 prompt_exact=true valid_ids=true decode_success=true selected_pack_counts={0: 8} hex=41742074686520686172626f727070707070707070
heads regime=schedule variant=B seed=20261009 pack_accuracy=0.300505 local_accuracy_given_gold_pack=0.040404 active_local_head_size_mean=170.659091 pack_head_logit_fraction=0.017275 local_head_logit_fraction=0.982725 pack_transitions=255 byte_fallback_targets=0/396 byte_fallback_fraction=0.000000 pack[0]={targets:104,accuracy:0.884615,local_count:171} pack[1]={targets:157,accuracy:0.140127,local_count:171} pack[2]={targets:135,accuracy:0.037037,local_count:170}
generation regime=schedule variant=B seed=20261009 prompt_exact=true valid_ids=true decode_success=true selected_pack_counts={0: 8} hex=41742074686520686172626f727070707070707070
heads regime=schedule variant=C seed=20261009 output_rows=512 logits_per_target=512 head_parameters=8704 measured_scoring_ms=11.843
generation regime=schedule variant=C seed=20261009 prompt_exact=true valid_ids=true decode_success=true selected_pack_counts={0: 8} hex=41742074686520686172626f722020202020202020
heads regime=schedule variant=D seed=20261009 pack_accuracy=0.320755 local_accuracy_given_gold_pack=0.238208 active_local_head_size_mean=246.683962 pack_head_logit_fraction=0.012015 local_head_logit_fraction=0.987985 pack_transitions=225 byte_fallback_targets=274/424 byte_fallback_fraction=0.646226 pack[0]={targets:136,accuracy:1.000000,local_count:253} pack[2]={targets:14,accuracy:0.000000,local_count:3} pack[65535]={targets:274,accuracy:0.000000,local_count:256}
generation regime=schedule variant=D seed=20261009 prompt_exact=true valid_ids=true decode_success=true selected_pack_counts={0: 8} hex=41742074686520686172626f72746865746865746865746865746865746865746865746865
heads regime=mac variant=A seed=20261009 output_rows=512 logits_per_target=512 head_parameters=8704 measured_scoring_ms=10.962
generation regime=mac variant=A seed=20261009 prompt_exact=true valid_ids=true decode_success=true selected_pack_counts={0: 8} hex=41742074686520686172626f727070707070707070
heads regime=mac variant=B seed=20261009 pack_accuracy=0.340909 local_accuracy_given_gold_pack=0.040404 active_local_head_size_mean=170.659091 pack_head_logit_fraction=0.017275 local_head_logit_fraction=0.982725 pack_transitions=255 byte_fallback_targets=0/396 byte_fallback_fraction=0.000000 pack[0]={targets:104,accuracy:0.548077,local_count:171} pack[1]={targets:157,accuracy:0.292994,local_count:171} pack[2]={targets:135,accuracy:0.237037,local_count:170}
generation regime=mac variant=B seed=20261009 prompt_exact=true valid_ids=true decode_success=true selected_pack_counts={0: 5, 1: 2, 2: 1} hex=41742074686520686172626f7220616f6f7265656e696e67207261696e
heads regime=mac variant=C seed=20261009 output_rows=512 logits_per_target=512 head_parameters=8704 measured_scoring_ms=10.860
generation regime=mac variant=C seed=20261009 prompt_exact=true valid_ids=true decode_success=true selected_pack_counts={0: 8} hex=41742074686520686172626f722020202020202020
heads regime=mac variant=D seed=20261009 pack_accuracy=0.589623 local_accuracy_given_gold_pack=0.252358 active_local_head_size_mean=246.683962 pack_head_logit_fraction=0.012015 local_head_logit_fraction=0.987985 pack_transitions=225 byte_fallback_targets=274/424 byte_fallback_fraction=0.646226 pack[0]={targets:136,accuracy:0.382353,local_count:253} pack[2]={targets:14,accuracy:0.000000,local_count:3} pack[65535]={targets:274,accuracy:0.722628,local_count:256}
generation regime=mac variant=D seed=20261009 prompt_exact=true valid_ids=true decode_success=true selected_pack_counts={0: 4, 65535: 4} hex=41742074686520686172626f7220746865207468652074686520746865
process_peak_memory_bytes=Ok(6225920)
```

## Learning curves

```text
curve regime=schedule variant=A seed=20261007 updates=1 macs=1620992 train_loss_token=6.23911607 validation_loss_token=6.23550607 validation_bits_byte=5.60746533 validation_ms=9.291
curve regime=schedule variant=A seed=20261007 updates=30 macs=48629760 train_loss_token=5.52098756 validation_loss_token=6.68693468 validation_bits_byte=6.01342601 validation_ms=10.240
curve regime=schedule variant=A seed=20261007 updates=60 macs=97259520 train_loss_token=5.71181066 validation_loss_token=6.98964141 validation_bits_byte=6.28564409 validation_ms=9.846
curve regime=schedule variant=A seed=20261007 updates=90 macs=145889280 train_loss_token=5.29891735 validation_loss_token=6.97402463 validation_bits_byte=6.27160024 validation_ms=9.976
curve regime=schedule variant=A seed=20261007 updates=120 macs=194519040 train_loss_token=5.12394968 validation_loss_token=6.74683180 validation_bits_byte=6.06729029 validation_ms=9.721
curve regime=schedule variant=B seed=20261007 updates=1 macs=581888 train_loss_token=6.23865554 validation_loss_token=6.23553368 validation_bits_byte=5.60749016 validation_ms=8.835
curve regime=schedule variant=B seed=20261007 updates=30 macs=17451024 train_loss_token=5.50491398 validation_loss_token=6.71972563 validation_bits_byte=6.04291425 validation_ms=6.428
curve regime=schedule variant=B seed=20261007 updates=60 macs=34902480 train_loss_token=5.63713599 validation_loss_token=6.83035516 validation_bits_byte=6.14240116 validation_ms=6.234
curve regime=schedule variant=B seed=20261007 updates=90 macs=52352976 train_loss_token=5.28248399 validation_loss_token=6.79849436 validation_bits_byte=6.11374939 validation_ms=6.242
curve regime=schedule variant=B seed=20261007 updates=120 macs=69805440 train_loss_token=5.01548877 validation_loss_token=6.57068311 validation_bits_byte=5.90888331 validation_ms=5.955
curve regime=schedule variant=C seed=20261007 updates=1 macs=1620992 train_loss_token=6.23679109 validation_loss_token=6.22803283 validation_bits_byte=6.23429021 validation_ms=10.240
curve regime=schedule variant=C seed=20261007 updates=30 macs=48629760 train_loss_token=4.23192625 validation_loss_token=5.70204923 validation_bits_byte=5.70777815 validation_ms=10.790
curve regime=schedule variant=C seed=20261007 updates=60 macs=97259520 train_loss_token=4.29048527 validation_loss_token=5.88058753 validation_bits_byte=5.88649583 validation_ms=10.750
curve regime=schedule variant=C seed=20261007 updates=90 macs=145889280 train_loss_token=3.90722904 validation_loss_token=5.85644520 validation_bits_byte=5.86232925 validation_ms=11.057
curve regime=schedule variant=C seed=20261007 updates=120 macs=194519040 train_loss_token=4.04368010 validation_loss_token=6.00819309 validation_bits_byte=6.01422960 validation_ms=10.937
curve regime=schedule variant=D seed=20261007 updates=1 macs=778304 train_loss_token=6.28957320 validation_loss_token=6.49466156 validation_bits_byte=6.50118683 validation_ms=6.635
curve regime=schedule variant=D seed=20261007 updates=30 macs=23595312 train_loss_token=4.22502251 validation_loss_token=5.78839241 validation_bits_byte=5.79420808 validation_ms=7.945
curve regime=schedule variant=D seed=20261007 updates=60 macs=47083200 train_loss_token=4.31112160 validation_loss_token=5.97889463 validation_bits_byte=5.98490170 validation_ms=8.177
curve regime=schedule variant=D seed=20261007 updates=90 macs=70472208 train_loss_token=3.86887577 validation_loss_token=5.97579869 validation_bits_byte=5.98180264 validation_ms=7.825
curve regime=schedule variant=D seed=20261007 updates=120 macs=93874368 train_loss_token=3.90600308 validation_loss_token=5.94977699 validation_bits_byte=5.95575480 validation_ms=7.968
curve regime=mac variant=A seed=20261007 updates=1 macs=1620992 train_loss_token=6.23911607 validation_loss_token=6.23550607 validation_bits_byte=5.60746533 validation_ms=8.776
curve regime=mac variant=A seed=20261007 updates=30 macs=48629760 train_loss_token=5.52098756 validation_loss_token=6.68693468 validation_bits_byte=6.01342601 validation_ms=9.543
curve regime=mac variant=A seed=20261007 updates=60 macs=97259520 train_loss_token=5.71181066 validation_loss_token=6.98964141 validation_bits_byte=6.28564409 validation_ms=9.982
curve regime=mac variant=A seed=20261007 updates=90 macs=145889280 train_loss_token=5.29891735 validation_loss_token=6.97402463 validation_bits_byte=6.27160024 validation_ms=10.179
curve regime=mac variant=A seed=20261007 updates=120 macs=194519040 train_loss_token=5.12394968 validation_loss_token=6.74683180 validation_bits_byte=6.06729029 validation_ms=11.277
curve regime=mac variant=B seed=20261007 updates=1 macs=581888 train_loss_token=6.23865554 validation_loss_token=6.23553368 validation_bits_byte=5.60749016 validation_ms=5.681
curve regime=mac variant=B seed=20261007 updates=84 macs=48862560 train_loss_token=5.13003598 validation_loss_token=6.81133328 validation_bits_byte=6.12529517 validation_ms=6.459
curve regime=mac variant=B seed=20261007 updates=168 macs=97728480 train_loss_token=4.29846692 validation_loss_token=6.64243757 validation_bits_byte=5.97341065 validation_ms=5.665
curve regime=mac variant=B seed=20261007 updates=251 macs=146012176 train_loss_token=3.09810824 validation_loss_token=7.28679025 validation_bits_byte=6.55286408 validation_ms=6.263
curve regime=mac variant=B seed=20261007 updates=335 macs=194876176 train_loss_token=1.92749673 validation_loss_token=7.89437302 validation_bits_byte=7.09925105 validation_ms=6.091
curve regime=mac variant=C seed=20261007 updates=1 macs=1620992 train_loss_token=6.23679109 validation_loss_token=6.22803283 validation_bits_byte=6.23429021 validation_ms=9.841
curve regime=mac variant=C seed=20261007 updates=30 macs=48629760 train_loss_token=4.23192625 validation_loss_token=5.70204923 validation_bits_byte=5.70777815 validation_ms=10.948
curve regime=mac variant=C seed=20261007 updates=60 macs=97259520 train_loss_token=4.29048527 validation_loss_token=5.88058753 validation_bits_byte=5.88649583 validation_ms=10.819
curve regime=mac variant=C seed=20261007 updates=90 macs=145889280 train_loss_token=3.90722904 validation_loss_token=5.85644520 validation_bits_byte=5.86232925 validation_ms=10.797
curve regime=mac variant=C seed=20261007 updates=120 macs=194519040 train_loss_token=4.04368010 validation_loss_token=6.00819309 validation_bits_byte=6.01422960 validation_ms=10.883
curve regime=mac variant=D seed=20261007 updates=1 macs=778304 train_loss_token=6.28957320 validation_loss_token=6.49466156 validation_bits_byte=6.50118683 validation_ms=8.002
curve regime=mac variant=D seed=20261007 updates=62 macs=48663952 train_loss_token=4.01784782 validation_loss_token=5.99274158 validation_bits_byte=5.99876256 validation_ms=7.808
curve regime=mac variant=D seed=20261007 updates=125 macs=97863040 train_loss_token=4.10433734 validation_loss_token=5.98792188 validation_bits_byte=5.99393802 validation_ms=7.861
curve regime=mac variant=D seed=20261007 updates=187 macs=146315072 train_loss_token=3.35268886 validation_loss_token=6.18612612 validation_bits_byte=6.19234140 validation_ms=7.894
curve regime=mac variant=D seed=20261007 updates=249 macs=194691360 train_loss_token=2.99705490 validation_loss_token=6.29965481 validation_bits_byte=6.30598415 validation_ms=8.011
curve regime=schedule variant=A seed=20261008 updates=1 macs=1620992 train_loss_token=6.23881117 validation_loss_token=6.23690062 validation_bits_byte=5.60871942 validation_ms=9.072
curve regime=schedule variant=A seed=20261008 updates=30 macs=48629760 train_loss_token=5.49079786 validation_loss_token=6.73834048 validation_bits_byte=6.05965421 validation_ms=10.196
curve regime=schedule variant=A seed=20261008 updates=60 macs=97259520 train_loss_token=5.19838683 validation_loss_token=6.77790313 validation_bits_byte=6.09523211 validation_ms=10.455
curve regime=schedule variant=A seed=20261008 updates=90 macs=145889280 train_loss_token=5.39374316 validation_loss_token=7.05329393 validation_bits_byte=6.34288552 validation_ms=10.001
curve regime=schedule variant=A seed=20261008 updates=120 macs=194519040 train_loss_token=4.97707356 validation_loss_token=7.08414003 validation_bits_byte=6.37062480 validation_ms=10.253
curve regime=schedule variant=B seed=20261008 updates=1 macs=581840 train_loss_token=6.23796062 validation_loss_token=6.23674958 validation_bits_byte=5.60858359 validation_ms=5.215
curve regime=schedule variant=B seed=20261008 updates=30 macs=17451600 train_loss_token=5.48240944 validation_loss_token=6.71875599 validation_bits_byte=6.04204228 validation_ms=6.165
curve regime=schedule variant=B seed=20261008 updates=60 macs=34903104 train_loss_token=5.17794257 validation_loss_token=6.89291095 validation_bits_byte=6.19865633 validation_ms=6.143
curve regime=schedule variant=B seed=20261008 updates=90 macs=52355568 train_loss_token=5.44177837 validation_loss_token=7.11823669 validation_bits_byte=6.40128724 validation_ms=6.216
curve regime=schedule variant=B seed=20261008 updates=120 macs=69807504 train_loss_token=4.98174097 validation_loss_token=6.79266875 validation_bits_byte=6.10851053 validation_ms=6.091
curve regime=schedule variant=C seed=20261008 updates=1 macs=1620992 train_loss_token=6.23904564 validation_loss_token=6.22781125 validation_bits_byte=6.23406841 validation_ms=9.522
curve regime=schedule variant=C seed=20261008 updates=30 macs=48629760 train_loss_token=4.13140104 validation_loss_token=5.61960525 validation_bits_byte=5.62525133 validation_ms=11.034
curve regime=schedule variant=C seed=20261008 updates=60 macs=97259520 train_loss_token=3.94769482 validation_loss_token=5.83825509 validation_bits_byte=5.84412086 validation_ms=10.795
curve regime=schedule variant=C seed=20261008 updates=90 macs=145889280 train_loss_token=4.02456124 validation_loss_token=5.96699740 validation_bits_byte=5.97299251 validation_ms=10.704
curve regime=schedule variant=C seed=20261008 updates=120 macs=194519040 train_loss_token=4.00896238 validation_loss_token=6.00543504 validation_bits_byte=6.01146877 validation_ms=11.014
curve regime=schedule variant=D seed=20261008 updates=1 macs=779024 train_loss_token=6.28973938 validation_loss_token=6.48477567 validation_bits_byte=6.49129101 validation_ms=6.976
curve regime=schedule variant=D seed=20261008 updates=30 macs=23571888 train_loss_token=4.13641774 validation_loss_token=5.68545886 validation_bits_byte=5.69117111 validation_ms=9.131
curve regime=schedule variant=D seed=20261008 updates=60 macs=47117472 train_loss_token=3.96307333 validation_loss_token=5.96353542 validation_bits_byte=5.96952706 validation_ms=8.192
curve regime=schedule variant=D seed=20261008 updates=90 macs=70590048 train_loss_token=4.01365806 validation_loss_token=6.11852415 validation_bits_byte=6.12467151 validation_ms=8.716
curve regime=schedule variant=D seed=20261008 updates=120 macs=93930336 train_loss_token=4.00300288 validation_loss_token=6.02859296 validation_bits_byte=6.03464997 validation_ms=8.095
curve regime=mac variant=A seed=20261008 updates=1 macs=1620992 train_loss_token=6.23881117 validation_loss_token=6.23690062 validation_bits_byte=5.60871942 validation_ms=10.173
curve regime=mac variant=A seed=20261008 updates=30 macs=48629760 train_loss_token=5.49079786 validation_loss_token=6.73834048 validation_bits_byte=6.05965421 validation_ms=9.787
curve regime=mac variant=A seed=20261008 updates=60 macs=97259520 train_loss_token=5.19838683 validation_loss_token=6.77790313 validation_bits_byte=6.09523211 validation_ms=9.591
curve regime=mac variant=A seed=20261008 updates=90 macs=145889280 train_loss_token=5.39374316 validation_loss_token=7.05329393 validation_bits_byte=6.34288552 validation_ms=9.756
curve regime=mac variant=A seed=20261008 updates=120 macs=194519040 train_loss_token=4.97707356 validation_loss_token=7.08414003 validation_bits_byte=6.37062480 validation_ms=9.625
curve regime=mac variant=B seed=20261008 updates=1 macs=581840 train_loss_token=6.23796062 validation_loss_token=6.23674958 validation_bits_byte=5.60858359 validation_ms=5.334
curve regime=mac variant=B seed=20261008 updates=84 macs=48864672 train_loss_token=5.21499987 validation_loss_token=7.20505853 validation_bits_byte=6.47936438 validation_ms=6.256
curve regime=mac variant=B seed=20261008 updates=168 macs=97728720 train_loss_token=4.23811792 validation_loss_token=6.83908589 validation_bits_byte=6.15025254 validation_ms=5.683
curve regime=mac variant=B seed=20261008 updates=251 macs=146009632 train_loss_token=3.17977718 validation_loss_token=7.25915525 validation_bits_byte=6.52801247 validation_ms=5.976
curve regime=mac variant=B seed=20261008 updates=335 macs=194875600 train_loss_token=2.24704309 validation_loss_token=7.83215962 validation_bits_byte=7.04330379 validation_ms=6.078
curve regime=mac variant=C seed=20261008 updates=1 macs=1620992 train_loss_token=6.23904564 validation_loss_token=6.22781125 validation_bits_byte=6.23406841 validation_ms=11.224
curve regime=mac variant=C seed=20261008 updates=30 macs=48629760 train_loss_token=4.13140104 validation_loss_token=5.61960525 validation_bits_byte=5.62525133 validation_ms=11.145
curve regime=mac variant=C seed=20261008 updates=60 macs=97259520 train_loss_token=3.94769482 validation_loss_token=5.83825509 validation_bits_byte=5.84412086 validation_ms=11.652
curve regime=mac variant=C seed=20261008 updates=90 macs=145889280 train_loss_token=4.02456124 validation_loss_token=5.96699740 validation_bits_byte=5.97299251 validation_ms=12.387
curve regime=mac variant=C seed=20261008 updates=120 macs=194519040 train_loss_token=4.00896238 validation_loss_token=6.00543504 validation_bits_byte=6.01146877 validation_ms=11.604
curve regime=mac variant=D seed=20261008 updates=1 macs=779024 train_loss_token=6.28973938 validation_loss_token=6.48477567 validation_bits_byte=6.49129101 validation_ms=6.602
curve regime=mac variant=D seed=20261008 updates=62 macs=48698656 train_loss_token=3.98495796 validation_loss_token=5.97098025 validation_bits_byte=5.97697937 validation_ms=8.024
curve regime=mac variant=D seed=20261008 updates=125 macs=97884304 train_loss_token=3.94257026 validation_loss_token=5.99148662 validation_bits_byte=5.99750634 validation_ms=8.336
curve regime=mac variant=D seed=20261008 updates=187 macs=146312048 train_loss_token=3.33519228 validation_loss_token=5.93725718 validation_bits_byte=5.94322242 validation_ms=8.221
curve regime=mac variant=D seed=20261008 updates=249 macs=195058848 train_loss_token=3.40359578 validation_loss_token=5.94275774 validation_bits_byte=5.94872850 validation_ms=8.571
curve regime=schedule variant=A seed=20261009 updates=1 macs=1620992 train_loss_token=6.23821813 validation_loss_token=6.23424132 validation_bits_byte=5.60632797 validation_ms=9.147
curve regime=schedule variant=A seed=20261009 updates=30 macs=48629760 train_loss_token=5.46844483 validation_loss_token=6.87458236 validation_bits_byte=6.18217380 validation_ms=9.787
curve regime=schedule variant=A seed=20261009 updates=60 macs=97259520 train_loss_token=5.38118455 validation_loss_token=7.03558533 validation_bits_byte=6.32696053 validation_ms=10.139
curve regime=schedule variant=A seed=20261009 updates=90 macs=145889280 train_loss_token=5.21902177 validation_loss_token=6.90311938 validation_bits_byte=6.20783657 validation_ms=9.571
curve regime=schedule variant=A seed=20261009 updates=120 macs=194519040 train_loss_token=5.10123811 validation_loss_token=6.87486706 validation_bits_byte=6.18242982 validation_ms=9.379
curve regime=schedule variant=B seed=20261009 updates=1 macs=581312 train_loss_token=6.23794150 validation_loss_token=6.23620622 validation_bits_byte=5.60809496 validation_ms=5.441
curve regime=schedule variant=B seed=20261009 updates=30 macs=17451456 train_loss_token=5.48548462 validation_loss_token=6.84315263 validation_bits_byte=6.15390968 validation_ms=6.608
curve regime=schedule variant=B seed=20261009 updates=60 macs=34902672 train_loss_token=5.39046266 validation_loss_token=7.05703886 validation_bits_byte=6.34625326 validation_ms=6.362
curve regime=schedule variant=B seed=20261009 updates=90 macs=52354656 train_loss_token=5.21390065 validation_loss_token=6.76258475 validation_bits_byte=6.08145660 validation_ms=6.203
curve regime=schedule variant=B seed=20261009 updates=120 macs=69805200 train_loss_token=5.02843174 validation_loss_token=6.46702244 validation_bits_byte=5.81566335 validation_ms=5.764
curve regime=schedule variant=C seed=20261009 updates=1 macs=1620992 train_loss_token=6.23771369 validation_loss_token=6.22604470 validation_bits_byte=6.23230008 validation_ms=9.732
curve regime=schedule variant=C seed=20261009 updates=30 macs=48629760 train_loss_token=4.16231413 validation_loss_token=5.79523605 validation_bits_byte=5.80105860 validation_ms=10.938
curve regime=schedule variant=C seed=20261009 updates=60 macs=97259520 train_loss_token=4.07184229 validation_loss_token=5.81587352 validation_bits_byte=5.82171680 validation_ms=10.898
curve regime=schedule variant=C seed=20261009 updates=90 macs=145889280 train_loss_token=3.88456754 validation_loss_token=5.90036642 validation_bits_byte=5.90629459 validation_ms=10.769
curve regime=schedule variant=C seed=20261009 updates=120 macs=194519040 train_loss_token=4.07150622 validation_loss_token=6.01107775 validation_bits_byte=6.01711716 validation_ms=10.738
curve regime=schedule variant=D seed=20261009 updates=1 macs=778592 train_loss_token=6.29199875 validation_loss_token=6.49936803 validation_bits_byte=6.50589803 validation_ms=7.021
curve regime=schedule variant=D seed=20261009 updates=30 macs=23433552 train_loss_token=4.20494037 validation_loss_token=5.80093282 validation_bits_byte=5.80676108 validation_ms=7.830
curve regime=schedule variant=D seed=20261009 updates=60 macs=47063424 train_loss_token=4.05618634 validation_loss_token=6.01545107 validation_bits_byte=6.02149487 validation_ms=7.995
curve regime=schedule variant=D seed=20261009 updates=90 macs=70534560 train_loss_token=3.88771123 validation_loss_token=6.06050969 validation_bits_byte=6.06659876 validation_ms=7.849
curve regime=schedule variant=D seed=20261009 updates=120 macs=93959712 train_loss_token=4.03396466 validation_loss_token=6.13804773 validation_bits_byte=6.14421470 validation_ms=7.989
curve regime=mac variant=A seed=20261009 updates=1 macs=1620992 train_loss_token=6.23821813 validation_loss_token=6.23424132 validation_bits_byte=5.60632797 validation_ms=8.481
curve regime=mac variant=A seed=20261009 updates=30 macs=48629760 train_loss_token=5.46844483 validation_loss_token=6.87458236 validation_bits_byte=6.18217380 validation_ms=9.637
curve regime=mac variant=A seed=20261009 updates=60 macs=97259520 train_loss_token=5.38118455 validation_loss_token=7.03558533 validation_bits_byte=6.32696053 validation_ms=9.557
curve regime=mac variant=A seed=20261009 updates=90 macs=145889280 train_loss_token=5.21902177 validation_loss_token=6.90311938 validation_bits_byte=6.20783657 validation_ms=9.645
curve regime=mac variant=A seed=20261009 updates=120 macs=194519040 train_loss_token=5.10123811 validation_loss_token=6.87486706 validation_bits_byte=6.18242982 validation_ms=9.193
curve regime=mac variant=B seed=20261009 updates=1 macs=581312 train_loss_token=6.23794150 validation_loss_token=6.23620622 validation_bits_byte=5.60809496 validation_ms=5.763
curve regime=mac variant=B seed=20261009 updates=84 macs=48863808 train_loss_token=5.27448971 validation_loss_token=6.82306087 validation_bits_byte=6.13584156 validation_ms=6.350
curve regime=mac variant=B seed=20261009 updates=168 macs=97728000 train_loss_token=5.08005943 validation_loss_token=6.75451697 validation_bits_byte=6.07420141 validation_ms=5.472
curve regime=mac variant=B seed=20261009 updates=251 macs=146013136 train_loss_token=2.99476399 validation_loss_token=7.28847009 validation_bits_byte=6.55437471 validation_ms=6.137
curve regime=mac variant=B seed=20261009 updates=335 macs=194876272 train_loss_token=1.83816232 validation_loss_token=8.10333601 validation_bits_byte=7.28716727 validation_ms=6.449
curve regime=mac variant=C seed=20261009 updates=1 macs=1620992 train_loss_token=6.23771369 validation_loss_token=6.22604470 validation_bits_byte=6.23230008 validation_ms=9.402
curve regime=mac variant=C seed=20261009 updates=30 macs=48629760 train_loss_token=4.16231413 validation_loss_token=5.79523605 validation_bits_byte=5.80105860 validation_ms=11.094
curve regime=mac variant=C seed=20261009 updates=60 macs=97259520 train_loss_token=4.07184229 validation_loss_token=5.81587352 validation_bits_byte=5.82171680 validation_ms=10.908
curve regime=mac variant=C seed=20261009 updates=90 macs=145889280 train_loss_token=3.88456754 validation_loss_token=5.90036642 validation_bits_byte=5.90629459 validation_ms=11.156
curve regime=mac variant=C seed=20261009 updates=120 macs=194519040 train_loss_token=4.07150622 validation_loss_token=6.01107775 validation_bits_byte=6.01711716 validation_ms=11.074
curve regime=mac variant=D seed=20261009 updates=1 macs=778592 train_loss_token=6.29199875 validation_loss_token=6.49936803 validation_bits_byte=6.50589803 validation_ms=7.453
curve regime=mac variant=D seed=20261009 updates=63 macs=49374192 train_loss_token=4.11183598 validation_loss_token=6.04949002 validation_bits_byte=6.05556802 validation_ms=8.313
curve regime=mac variant=D seed=20261009 updates=125 macs=97948960 train_loss_token=4.13903991 validation_loss_token=6.08130719 validation_bits_byte=6.08741716 validation_ms=8.285
curve regime=mac variant=D seed=20261009 updates=187 macs=146218112 train_loss_token=3.64186681 validation_loss_token=5.91678019 validation_bits_byte=5.92272485 validation_ms=7.814
curve regime=mac variant=D seed=20261009 updates=249 macs=194758752 train_loss_token=3.27860051 validation_loss_token=5.94794505 validation_bits_byte=5.95392103 validation_ms=7.822
```
