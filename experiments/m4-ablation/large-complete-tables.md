# Derived M4 tables

Input: experiments/m4-ablation/runs/large-final-20261007/run-summary.txt. Arithmetic means and population SD; rounded log values are used. Raw logs and Rust summaries retain additional precision. Peak memory is the shared process high-water mark at each result, not isolated model memory. MACs exclude non-matrix work.

## schedule regime

| Variant | Test bits/byte ± SD | Parameters | Updates per seed | MACs ± SD | Train ms ± SD | Test ms ± SD | Process peak B range |
|---|---:|---:|---|---:|---:|---:|---|
| A | 2.612470 ± 0.001542 | 17424 | 2000, 2000, 2000 | 3241984000.0 ± 0.0 | 9039.564 ± 131.156 | 1074.113 ± 6.790 | 81297408–81297408 |
| B | 2.622904 ± 0.002963 | 17492 | 2000, 2000, 2000 | 907264000.0 ± 0.0 | 4713.490 ± 17.097 | 654.022 ± 11.985 | 81297408–81297408 |
| C | 2.581360 ± 0.001000 | 17424 | 2000, 2000, 2000 | 3241984000.0 ± 0.0 | 9877.250 ± 109.302 | 1237.723 ± 5.378 | 81297408–81297408 |
| D | 2.592482 ± 0.005545 | 17492 | 2000, 2000, 2000 | 1458011328.0 ± 721828.9 | 6284.384 ± 60.865 | 860.290 ± 8.108 | 81297408–81297408 |

| Seed | B-A | C-A | D-A | (D-C)-(B-A) |
|---|---:|---:|---:|---:|
| 20261007 | 0.00882033 | -0.02772808 | -0.01043339 | 0.00847436 |
| 20261008 | 0.01334663 | -0.03289478 | -0.02646305 | -0.00691490 |
| 20261009 | 0.00913312 | -0.03270715 | -0.02306756 | 0.00050647 |
| mean ± population SD | 0.01043336 ± 0.00206395 | -0.03111000 ± 0.00239261 | -0.01998800 ± 0.00689687 | 0.00068864 ± 0.00628396 |

## mac regime

| Variant | Test bits/byte ± SD | Parameters | Updates per seed | MACs ± SD | Train ms ± SD | Test ms ± SD | Process peak B range |
|---|---:|---:|---|---:|---:|---:|---|
| A | 2.612470 ± 0.001542 | 17424 | 2000, 2000, 2000 | 3241984000.0 ± 0.0 | 8955.431 ± 46.838 | 1062.902 ± 4.574 | 81297408–81297408 |
| B | 2.514732 ± 0.013810 | 17492 | 7147, 7147, 7147 | 3242107904.0 ± 0.0 | 9161.516 ± 15.224 | 644.720 ± 5.403 | 81297408–81297408 |
| C | 2.581360 ± 0.001000 | 17424 | 2000, 2000, 2000 | 3241984000.0 ± 0.0 | 9877.331 ± 43.605 | 1244.129 ± 4.268 | 81297408–81297408 |
| D | 2.491482 ± 0.012420 | 17492 | 4446, 4444, 4451 | 3242441280.0 ± 292134.2 | 9120.821 ± 33.225 | 852.352 ± 7.517 | 81297408–81297408 |

| Seed | B-A | C-A | D-A | (D-C)-(B-A) |
|---|---:|---:|---:|---:|
| 20261007 | -0.08451379 | -0.02772808 | -0.10138409 | 0.01085778 |
| 20261008 | -0.08998827 | -0.03289478 | -0.13175801 | -0.00887496 |
| 20261009 | -0.11871462 | -0.03270715 | -0.12982361 | 0.02159816 |
| mean ± population SD | -0.09773889 ± 0.01499952 | -0.03111000 ± 0.00239261 | -0.12098857 ± 0.01388494 | 0.00786033 ± 0.01261986 |

## Per-seed measurements

| Regime | Variant | Seed | Test bits/byte | NLL/byte | Loss/token | B/token | Updates | Targets | Train target bytes | MACs | Train ms | Curve validation ms | Final validation ms | Test ms | Model B | Head params | Parameter/Adam/gradient B |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| schedule | A | 20261007 | 2.61035376 | 1.80935935 | 3.74878529 | 2.07193146 | 2000 | 128000 | 260048 | 3241984000 | 9000.251 | 5058.062 | 1027.680 | 1083.453 | 69742 | 8704 | 278784 |
| schedule | B | 20261007 | 2.61917409 | 1.81547314 | 3.76145236 | 2.07193146 | 2000 | 128000 | 260048 | 907264000 | 4714.376 | 2948.560 | 631.210 | 638.503 | 70032 | 8772 | 279872 |
| schedule | C | 20261007 | 2.58262568 | 1.79013971 | 3.22061173 | 1.79910890 | 2000 | 128000 | 229982 | 3241984000 | 10026.022 | 5989.960 | 1187.535 | 1240.625 | 69742 | 8704 | 278784 |
| schedule | D | 20261007 | 2.59992037 | 1.80212747 | 3.24217873 | 1.79910890 | 2000 | 128000 | 229982 | 1458695872 | 6370.060 | 4087.392 | 876.106 | 856.873 | 70032 | 8772 | 279872 |
| mac | A | 20261007 | 2.61035376 | 1.80935935 | 3.74878529 | 2.07193146 | 2000 | 128000 | 260048 | 3241984000 | 8906.376 | 4988.369 | 1015.909 | 1060.255 | 69742 | 8704 | 278784 |
| mac | B | 20261007 | 2.52583997 | 1.75077886 | 3.62741322 | 2.07193146 | 7147 | 457408 | 930171 | 3242107904 | 9175.299 | 2949.737 | 611.142 | 648.507 | 70032 | 8772 | 279872 |
| mac | C | 20261007 | 2.58262568 | 1.79013971 | 3.22061173 | 1.79910890 | 2000 | 128000 | 229982 | 3241984000 | 9848.845 | 5823.637 | 1184.613 | 1250.118 | 69742 | 8704 | 278784 |
| mac | D | 20261007 | 2.50896967 | 1.73908526 | 3.12876048 | 1.79910890 | 4446 | 284544 | 510370 | 3242723376 | 9160.230 | 4024.535 | 844.585 | 860.267 | 70032 | 8772 | 279872 |
| schedule | A | 20261008 | 2.61307628 | 1.81124646 | 3.75269517 | 2.07193146 | 2000 | 128000 | 261601 | 3241984000 | 9216.204 | 5218.142 | 1024.637 | 1067.510 | 69742 | 8704 | 278784 |
| schedule | B | 20261008 | 2.62642291 | 1.82049763 | 3.77186254 | 2.07193146 | 2000 | 128000 | 261601 | 907264000 | 4733.972 | 2959.248 | 623.101 | 667.684 | 70032 | 8772 | 279872 |
| schedule | C | 20261008 | 2.58018150 | 1.78844553 | 3.21756376 | 1.79910890 | 2000 | 128000 | 230842 | 3241984000 | 9766.523 | 5826.406 | 1197.489 | 1230.184 | 69742 | 8704 | 278784 |
| schedule | D | 20261008 | 2.58661323 | 1.79290367 | 3.22558432 | 1.79910890 | 2000 | 128000 | 230842 | 1458324880 | 6234.363 | 3982.186 | 831.306 | 852.519 | 70032 | 8772 | 279872 |
| mac | A | 20261008 | 2.61307628 | 1.81124646 | 3.75269517 | 2.07193146 | 2000 | 128000 | 261601 | 3241984000 | 8941.412 | 5024.628 | 1012.878 | 1059.115 | 69742 | 8704 | 278784 |
| mac | B | 20261008 | 2.52308801 | 1.74887134 | 3.62346107 | 2.07193146 | 7147 | 457408 | 937274 | 3242107904 | 9140.300 | 2949.192 | 622.804 | 648.573 | 70032 | 8772 | 279872 |
| mac | C | 20261008 | 2.58018150 | 1.78844553 | 3.21756376 | 1.79910890 | 2000 | 128000 | 230842 | 3241984000 | 9938.939 | 5924.278 | 1199.705 | 1240.487 | 69742 | 8704 | 278784 |
| mac | D | 20261008 | 2.48131827 | 1.71991876 | 3.09427835 | 1.79910890 | 4444 | 284416 | 512112 | 3242038832 | 9078.956 | 4011.162 | 818.983 | 854.541 | 70032 | 8772 | 279872 |
| schedule | A | 20261009 | 2.61398118 | 1.81187368 | 3.75399471 | 2.07193146 | 2000 | 128000 | 260766 | 3241984000 | 8902.238 | 4994.730 | 1020.913 | 1071.377 | 69742 | 8704 | 278784 |
| schedule | B | 20261009 | 2.62311430 | 1.81820428 | 3.76711098 | 2.07193146 | 2000 | 128000 | 260766 | 907264000 | 4692.121 | 2947.914 | 609.378 | 655.879 | 70032 | 8772 | 279872 |
| schedule | C | 20261009 | 2.58127403 | 1.78920282 | 3.21892618 | 1.79910890 | 2000 | 128000 | 230675 | 3241984000 | 9839.205 | 5882.717 | 1191.183 | 1242.361 | 69742 | 8704 | 278784 |
| schedule | D | 20261009 | 2.59091362 | 1.79588447 | 3.23094704 | 1.79910890 | 2000 | 128000 | 230675 | 1457013232 | 6248.729 | 3993.310 | 819.749 | 871.477 | 70032 | 8772 | 279872 |
| mac | A | 20261009 | 2.61398118 | 1.81187368 | 3.75399471 | 2.07193146 | 2000 | 128000 | 260766 | 3241984000 | 9018.505 | 5043.023 | 1056.269 | 1069.337 | 69742 | 8704 | 278784 |
| mac | B | 20261009 | 2.49526656 | 1.72958698 | 3.58350609 | 2.07193146 | 7147 | 457408 | 937751 | 3242107904 | 9168.949 | 2957.150 | 609.904 | 637.079 | 70032 | 8772 | 279872 |
| mac | C | 20261009 | 2.58127403 | 1.78920282 | 3.21892618 | 1.79910890 | 2000 | 128000 | 230675 | 3241984000 | 9844.209 | 5851.688 | 1186.774 | 1241.782 | 69742 | 8704 | 278784 |
| mac | D | 20261009 | 2.48415757 | 1.72188681 | 3.09781904 | 1.79910890 | 4451 | 284864 | 513550 | 3242561632 | 9123.276 | 4019.351 | 834.748 | 842.248 | 70032 | 8772 | 279872 |

## Raw tokenizer, head and generation observations

| Regime | Variant | NLL/byte mean ± SD | Loss/token mean ± SD | Final validation bits/byte mean ± SD | Final validation ms mean ± SD |
|---|---|---:|---:|---:|---:|
| schedule | A | 1.810826 ± 0.001069 | 3.751825 ± 0.002214 | 2.585382 ± 0.001558 | 1024.410000 ± 2.767275 |
| schedule | B | 1.818058 ± 0.002054 | 3.766809 ± 0.004255 | 2.596988 ± 0.000919 | 621.229667 ± 9.010567 |
| schedule | C | 1.789263 ± 0.000693 | 3.219034 ± 0.001247 | 2.553898 ± 0.005325 | 1192.069000 ± 4.111713 |
| schedule | D | 1.796972 ± 0.003843 | 3.232903 ± 0.006914 | 2.571116 ± 0.007312 | 842.387000 ± 24.305271 |
| mac | A | 1.810826 ± 0.001069 | 3.751825 ± 0.002214 | 2.585382 ± 0.001558 | 1028.352000 ± 19.779045 |
| mac | B | 1.743079 ± 0.009572 | 3.611460 ± 0.019832 | 2.487374 ± 0.011373 | 614.616667 ± 5.811338 |
| mac | C | 1.789263 ± 0.000693 | 3.219034 ± 0.001247 | 2.553898 ± 0.005325 | 1190.364000 ± 6.663742 |
| mac | D | 1.726964 ± 0.008609 | 3.106953 ± 0.015488 | 2.469165 ± 0.007704 | 832.772000 ± 10.544952 |

| Regime | Factorized variant | Pack accuracy mean ± SD | Local accuracy under gold pack mean ± SD | Active local rows | Total logits/target | Total test logits | Head parameters |
|---|---|---:|---:|---:|---:|---:|---:|
| schedule | B | 0.360111 ± 0.004806 | 0.301599 ± 0.001423 | 128.000000 | 132.000000 | 5530800 | 8772 |
| schedule | D | 0.694305 ± 0.002310 | 0.366042 ± 0.003510 | 218.463278 | 222.463278 | 10734743 | 8772 |
| mac | B | 0.355338 ± 0.003613 | 0.333246 ± 0.004261 | 128.000000 | 132.000000 | 5530800 | 8772 |
| mac | D | 0.691977 ± 0.001494 | 0.383809 ± 0.001477 | 218.463278 | 222.463278 | 10734743 | 8772 |

```text
tokenizer_training m1_ms=22460.618 m2_ms=10158.575 m1_bytes=3984 m2_bytes=4543 both_trained_twice_identical=true m2_allocation=[FactorizedPackTrainingStats { pack_id: 65535, spans: 0, routed_bytes: 0, learned_merges: 0, local_vocab_size: 0 }, FactorizedPackTrainingStats { pack_id: 0, spans: 264001, routed_bytes: 1298392, learned_merges: 214, local_vocab_size: 214 }, FactorizedPackTrainingStats { pack_id: 1, spans: 22231, routed_bytes: 77187, learned_merges: 9, local_vocab_size: 9 }, FactorizedPackTrainingStats { pack_id: 2, spans: 285995, routed_bytes: 481510, learned_merges: 33, local_vocab_size: 33 }]
encode tokenizer=M1 split=0 bytes=1857089 tokens=908767 bytes_per_token=2.04352601 ms=438.797 mib_per_s=4.036171
encode tokenizer=M2 split=0 bytes=1857089 tokens=1031748 bytes_per_token=1.79994437 ms=530.012 mib_per_s=3.341541
encode tokenizer=M1 split=1 bytes=82984 tokens=39979 bytes_per_token=2.07568974 ms=9.548 mib_per_s=8.288182
encode tokenizer=M2 split=1 bytes=82984 tokens=46308 bytes_per_token=1.79200138 ms=24.348 mib_per_s=3.250424
encode tokenizer=M1 split=2 bytes=86816 tokens=41901 bytes_per_token=2.07193146 ms=10.484 mib_per_s=7.896893
encode tokenizer=M2 split=2 bytes=86816 tokens=48255 bytes_per_token=1.79910890 ms=25.079 mib_per_s=3.301296
mapping variant=B rows=512 packs=[PackVocabulary { pack_id: 0, token_count: 128 }, PackVocabulary { pack_id: 1, token_count: 128 }, PackVocabulary { pack_id: 2, token_count: 128 }, PackVocabulary { pack_id: 3, token_count: 128 }]
mapping variant=C rows=512 packs=[PackVocabulary { pack_id: 0, token_count: 214 }, PackVocabulary { pack_id: 1, token_count: 9 }, PackVocabulary { pack_id: 2, token_count: 33 }, PackVocabulary { pack_id: 65535, token_count: 256 }]
heads regime=schedule variant=A seed=20261007 output_rows=512 logits_per_target=512 head_parameters=8704 measured_scoring_ms=1083.453
generation regime=schedule variant=A seed=20261007 prompt_exact=true valid_ids=true decode_success=true selected_pack_counts={0: 8} hex=41742074686520686172626f7220776172206d65792073
heads regime=schedule variant=B seed=20261007 pack_accuracy=0.363341 local_accuracy_given_gold_pack=0.299833 active_local_head_size_mean=128.000000 pack_head_logit_fraction=0.030303 local_head_logit_fraction=0.969697 pack_transitions=30806 byte_fallback_targets=0/41900 byte_fallback_fraction=0.000000 pack[0]={targets:10641,accuracy:0.398177,local_count:128} pack[1]={targets:9315,accuracy:0.224477,local_count:128} pack[2]={targets:11438,accuracy:0.388967,local_count:128} pack[3]={targets:10506,accuracy:0.423282,local_count:128}
generation regime=schedule variant=B seed=20261007 prompt_exact=true valid_ids=true decode_success=true selected_pack_counts={0: 2, 1: 1, 2: 1, 3: 4} hex=41742074686520686172626f72742e20486f722077686172
heads regime=schedule variant=C seed=20261007 output_rows=512 logits_per_target=512 head_parameters=8704 measured_scoring_ms=1240.625
generation regime=schedule variant=C seed=20261007 prompt_exact=true valid_ids=true decode_success=true selected_pack_counts={0: 8} hex=41742074686520686172626f726420746f207468652065766572
heads regime=schedule variant=D seed=20261007 pack_accuracy=0.693683 local_accuracy_given_gold_pack=0.361276 active_local_head_size_mean=218.463278 pack_head_logit_fraction=0.017980 local_head_logit_fraction=0.982020 pack_transitions=32554 byte_fallback_targets=24930/48254 byte_fallback_fraction=0.516641 pack[0]={targets:18783,accuracy:0.710057,local_count:214} pack[1]={targets:407,accuracy:0.272727,local_count:9} pack[2]={targets:4134,accuracy:0.297533,local_count:33} pack[65535]={targets:24930,accuracy:0.753911,local_count:256}
generation regime=schedule variant=D seed=20261007 prompt_exact=true valid_ids=true decode_success=true selected_pack_counts={0: 4, 65535: 4} hex=41742074686520686172626f7220746f2074686520746865207265
heads regime=mac variant=A seed=20261007 output_rows=512 logits_per_target=512 head_parameters=8704 measured_scoring_ms=1060.255
generation regime=mac variant=A seed=20261007 prompt_exact=true valid_ids=true decode_success=true selected_pack_counts={0: 8} hex=41742074686520686172626f7220776172206d65792073
heads regime=mac variant=B seed=20261007 pack_accuracy=0.353652 local_accuracy_given_gold_pack=0.330573 active_local_head_size_mean=128.000000 pack_head_logit_fraction=0.030303 local_head_logit_fraction=0.969697 pack_transitions=30806 byte_fallback_targets=0/41900 byte_fallback_fraction=0.000000 pack[0]={targets:10641,accuracy:0.275538,local_count:128} pack[1]={targets:9315,accuracy:0.380247,local_count:128} pack[2]={targets:11438,accuracy:0.450428,local_count:128} pack[3]={targets:10506,accuracy:0.303826,local_count:128}
generation regime=mac variant=B seed=20261007 prompt_exact=true valid_ids=true decode_success=true selected_pack_counts={0: 1, 1: 3, 2: 1, 3: 3} hex=41742074686520686172626f72206469650d0a77686f6d20
heads regime=mac variant=C seed=20261007 output_rows=512 logits_per_target=512 head_parameters=8704 measured_scoring_ms=1250.118
generation regime=mac variant=C seed=20261007 prompt_exact=true valid_ids=true decode_success=true selected_pack_counts={0: 8} hex=41742074686520686172626f726420746f207468652065766572
heads regime=mac variant=D seed=20261007 pack_accuracy=0.692399 local_accuracy_given_gold_pack=0.381730 active_local_head_size_mean=218.463278 pack_head_logit_fraction=0.017980 local_head_logit_fraction=0.982020 pack_transitions=32554 byte_fallback_targets=24930/48254 byte_fallback_fraction=0.516641 pack[0]={targets:18783,accuracy:0.722994,local_count:214} pack[1]={targets:407,accuracy:0.385749,local_count:9} pack[2]={targets:4134,accuracy:0.317126,local_count:33} pack[65535]={targets:24930,accuracy:0.736582,local_count:256}
generation regime=mac variant=D seed=20261007 prompt_exact=true valid_ids=true decode_success=true selected_pack_counts={0: 4, 65535: 4} hex=41742074686520686172626f72207a75206765676572207a75
heads regime=schedule variant=A seed=20261008 output_rows=512 logits_per_target=512 head_parameters=8704 measured_scoring_ms=1067.510
generation regime=schedule variant=A seed=20261008 prompt_exact=true valid_ids=true decode_success=true selected_pack_counts={0: 8} hex=41742074686520686172626f7220746865206d61636b657920
heads regime=schedule variant=B seed=20261008 pack_accuracy=0.363675 local_accuracy_given_gold_pack=0.303317 active_local_head_size_mean=128.000000 pack_head_logit_fraction=0.030303 local_head_logit_fraction=0.969697 pack_transitions=30806 byte_fallback_targets=0/41900 byte_fallback_fraction=0.000000 pack[0]={targets:10641,accuracy:0.248849,local_count:128} pack[1]={targets:9315,accuracy:0.234138,local_count:128} pack[2]={targets:11438,accuracy:0.451390,local_count:128} pack[3]={targets:10506,accuracy:0.499334,local_count:128}
generation regime=schedule variant=B seed=20261008 prompt_exact=true valid_ids=true decode_success=true selected_pack_counts={0: 1, 1: 2, 2: 1, 3: 4} hex=41742074686520686172626f7220776861726c6f74656e2047
heads regime=schedule variant=C seed=20261008 output_rows=512 logits_per_target=512 head_parameters=8704 measured_scoring_ms=1230.184
generation regime=schedule variant=C seed=20261008 prompt_exact=true valid_ids=true decode_success=true selected_pack_counts={0: 8} hex=41742074686520686172626f722074686520746865207468652061
heads regime=schedule variant=D seed=20261008 pack_accuracy=0.691839 local_accuracy_given_gold_pack=0.369627 active_local_head_size_mean=218.463278 pack_head_logit_fraction=0.017980 local_head_logit_fraction=0.982020 pack_transitions=32554 byte_fallback_targets=24930/48254 byte_fallback_fraction=0.516641 pack[0]={targets:18783,accuracy:0.719853,local_count:214} pack[1]={targets:407,accuracy:0.007371,local_count:9} pack[2]={targets:4134,accuracy:0.288341,local_count:33} pack[65535]={targets:24930,accuracy:0.748817,local_count:256}
generation regime=schedule variant=D seed=20261008 prompt_exact=true valid_ids=true decode_success=true selected_pack_counts={0: 2, 65535: 6} hex=41742074686520686172626f7220696e2065786520696e
heads regime=mac variant=A seed=20261008 output_rows=512 logits_per_target=512 head_parameters=8704 measured_scoring_ms=1059.115
generation regime=mac variant=A seed=20261008 prompt_exact=true valid_ids=true decode_success=true selected_pack_counts={0: 8} hex=41742074686520686172626f7220746865206d61636b657920
heads regime=mac variant=B seed=20261008 pack_accuracy=0.360358 local_accuracy_given_gold_pack=0.329905 active_local_head_size_mean=128.000000 pack_head_logit_fraction=0.030303 local_head_logit_fraction=0.969697 pack_transitions=30806 byte_fallback_targets=0/41900 byte_fallback_fraction=0.000000 pack[0]={targets:10641,accuracy:0.425336,local_count:128} pack[1]={targets:9315,accuracy:0.266345,local_count:128} pack[2]={targets:11438,accuracy:0.478668,local_count:128} pack[3]={targets:10506,accuracy:0.249096,local_count:128}
generation regime=mac variant=B seed=20261008 prompt_exact=true valid_ids=true decode_success=true selected_pack_counts={0: 4, 2: 2, 3: 2} hex=41742074686520686172626f727465657665206d6163
heads regime=mac variant=C seed=20261008 output_rows=512 logits_per_target=512 head_parameters=8704 measured_scoring_ms=1240.487
generation regime=mac variant=C seed=20261008 prompt_exact=true valid_ids=true decode_success=true selected_pack_counts={0: 8} hex=41742074686520686172626f722074686520746865207468652061
heads regime=mac variant=D seed=20261008 pack_accuracy=0.689974 local_accuracy_given_gold_pack=0.385025 active_local_head_size_mean=218.463278 pack_head_logit_fraction=0.017980 local_head_logit_fraction=0.982020 pack_transitions=32554 byte_fallback_targets=24930/48254 byte_fallback_fraction=0.516641 pack[0]={targets:18783,accuracy:0.722728,local_count:214} pack[1]={targets:407,accuracy:0.294840,local_count:9} pack[2]={targets:4134,accuracy:0.279874,local_count:33} pack[65535]={targets:24930,accuracy:0.739751,local_count:256}
generation regime=mac variant=D seed=20261008 prompt_exact=true valid_ids=true decode_success=true selected_pack_counts={0: 4, 65535: 4} hex=41742074686520686172626f7220696e206661726872656e20
heads regime=schedule variant=A seed=20261009 output_rows=512 logits_per_target=512 head_parameters=8704 measured_scoring_ms=1071.377
generation regime=schedule variant=A seed=20261009 prompt_exact=true valid_ids=true decode_success=true selected_pack_counts={0: 8} hex=41742074686520686172626f7220736f20736f20736f2073
heads regime=schedule variant=B seed=20261009 pack_accuracy=0.353317 local_accuracy_given_gold_pack=0.301647 active_local_head_size_mean=128.000000 pack_head_logit_fraction=0.030303 local_head_logit_fraction=0.969697 pack_transitions=30806 byte_fallback_targets=0/41900 byte_fallback_fraction=0.000000 pack[0]={targets:10641,accuracy:0.416220,local_count:128} pack[1]={targets:9315,accuracy:0.255824,local_count:128} pack[2]={targets:11438,accuracy:0.463805,local_count:128} pack[3]={targets:10506,accuracy:0.255759,local_count:128}
generation regime=schedule variant=B seed=20261009 prompt_exact=true valid_ids=true decode_success=true selected_pack_counts={0: 5, 1: 1, 2: 1, 3: 1} hex=41742074686520686172626f72207765727474657474
heads regime=schedule variant=C seed=20261009 output_rows=512 logits_per_target=512 head_parameters=8704 measured_scoring_ms=1242.361
generation regime=schedule variant=C seed=20261009 prompt_exact=true valid_ids=true decode_success=true selected_pack_counts={0: 8} hex=41742074686520686172626f7220746865207468652074686520746865
heads regime=schedule variant=D seed=20261009 pack_accuracy=0.697393 local_accuracy_given_gold_pack=0.367223 active_local_head_size_mean=218.463278 pack_head_logit_fraction=0.017980 local_head_logit_fraction=0.982020 pack_transitions=32554 byte_fallback_targets=24930/48254 byte_fallback_fraction=0.516641 pack[0]={targets:18783,accuracy:0.712240,local_count:214} pack[1]={targets:407,accuracy:0.429975,local_count:9} pack[2]={targets:4134,accuracy:0.287131,local_count:33} pack[65535]={targets:24930,accuracy:0.758604,local_count:256}
generation regime=schedule variant=D seed=20261009 prompt_exact=true valid_ids=true decode_success=true selected_pack_counts={0: 3, 65535: 5} hex=41742074686520686172626f72206f6620696e207468652068
heads regime=mac variant=A seed=20261009 output_rows=512 logits_per_target=512 head_parameters=8704 measured_scoring_ms=1069.337
generation regime=mac variant=A seed=20261009 prompt_exact=true valid_ids=true decode_success=true selected_pack_counts={0: 8} hex=41742074686520686172626f7220736f20736f20736f2073
heads regime=mac variant=B seed=20261009 pack_accuracy=0.352005 local_accuracy_given_gold_pack=0.339260 active_local_head_size_mean=128.000000 pack_head_logit_fraction=0.030303 local_head_logit_fraction=0.969697 pack_transitions=30806 byte_fallback_targets=0/41900 byte_fallback_fraction=0.000000 pack[0]={targets:10641,accuracy:0.332394,local_count:128} pack[1]={targets:9315,accuracy:0.357703,local_count:128} pack[2]={targets:11438,accuracy:0.431457,local_count:128} pack[3]={targets:10506,accuracy:0.280316,local_count:128}
generation regime=mac variant=B seed=20261009 prompt_exact=true valid_ids=true decode_success=true selected_pack_counts={1: 5, 3: 3} hex=41742074686520686172626f72206f6620686973206f662074686520636f6d
heads regime=mac variant=C seed=20261009 output_rows=512 logits_per_target=512 head_parameters=8704 measured_scoring_ms=1241.782
generation regime=mac variant=C seed=20261009 prompt_exact=true valid_ids=true decode_success=true selected_pack_counts={0: 8} hex=41742074686520686172626f7220746865207468652074686520746865
heads regime=mac variant=D seed=20261009 pack_accuracy=0.693559 local_accuracy_given_gold_pack=0.384673 active_local_head_size_mean=218.463278 pack_head_logit_fraction=0.017980 local_head_logit_fraction=0.982020 pack_transitions=32554 byte_fallback_targets=24930/48254 byte_fallback_fraction=0.516641 pack[0]={targets:18783,accuracy:0.744184,local_count:214} pack[1]={targets:407,accuracy:0.090909,local_count:9} pack[2]={targets:4134,accuracy:0.297291,local_count:33} pack[65535]={targets:24930,accuracy:0.730967,local_count:256}
generation regime=mac variant=D seed=20261009 prompt_exact=true valid_ids=true decode_success=true selected_pack_counts={0: 2, 65535: 6} hex=41742074686520686172626f722074686520707265612070
process_peak_memory_bytes=Ok(81297408)
```

## Learning curves

```text
curve regime=schedule variant=A seed=20261007 updates=1 macs=1620992 train_loss_token=6.23866541 validation_loss_token=6.23069904 validation_bits_byte=4.33065625 validation_ms=931.419
curve regime=schedule variant=A seed=20261007 updates=500 macs=810496000 train_loss_token=4.67917161 validation_loss_token=4.47657993 validation_bits_byte=3.11145326 validation_ms=1011.581
curve regime=schedule variant=A seed=20261007 updates=1000 macs=1620992000 train_loss_token=3.82494275 validation_loss_token=3.95905305 validation_bits_byte=2.75174547 validation_ms=1083.826
curve regime=schedule variant=A seed=20261007 updates=1500 macs=2431488000 train_loss_token=3.69209975 validation_loss_token=3.80843877 validation_bits_byte=2.64706079 validation_ms=1022.675
curve regime=schedule variant=A seed=20261007 updates=2000 macs=3241984000 train_loss_token=3.88036008 validation_loss_token=3.71740034 validation_bits_byte=2.58378440 validation_ms=1008.561
curve regime=schedule variant=B seed=20261007 updates=1 macs=453632 train_loss_token=6.23892589 validation_loss_token=6.23108509 validation_bits_byte=4.33092457 validation_ms=516.159
curve regime=schedule variant=B seed=20261007 updates=500 macs=226816000 train_loss_token=4.47788494 validation_loss_token=4.35170202 validation_bits_byte=3.02465669 validation_ms=597.266
curve regime=schedule variant=B seed=20261007 updates=1000 macs=453632000 train_loss_token=3.88979844 validation_loss_token=3.92604641 validation_bits_byte=2.72880415 validation_ms=613.734
curve regime=schedule variant=B seed=20261007 updates=1500 macs=680448000 train_loss_token=3.67943050 validation_loss_token=3.83839515 validation_bits_byte=2.66788202 validation_ms=614.537
curve regime=schedule variant=B seed=20261007 updates=2000 macs=907264000 train_loss_token=3.91395268 validation_loss_token=3.73539119 validation_bits_byte=2.59628897 validation_ms=606.864
curve regime=schedule variant=C seed=20261007 updates=1 macs=1620992 train_loss_token=6.23834576 validation_loss_token=6.22737397 validation_bits_byte=5.01351402 validation_ms=1055.634
curve regime=schedule variant=C seed=20261007 updates=500 macs=810496000 train_loss_token=4.49230228 validation_loss_token=3.98285892 validation_bits_byte=3.20650713 validation_ms=1331.177
curve regime=schedule variant=C seed=20261007 updates=1000 macs=1620992000 train_loss_token=4.12661156 validation_loss_token=3.42505896 validation_bits_byte=2.75743535 validation_ms=1192.711
curve regime=schedule variant=C seed=20261007 updates=1500 macs=2431488000 train_loss_token=2.95065538 validation_loss_token=3.27440332 validation_bits_byte=2.63614600 validation_ms=1202.451
curve regime=schedule variant=C seed=20261007 updates=2000 macs=3241984000 train_loss_token=3.90185962 validation_loss_token=3.17835599 validation_bits_byte=2.55882053 validation_ms=1207.987
curve regime=schedule variant=D seed=20261007 updates=1 macs=723536 train_loss_token=6.63937905 validation_loss_token=6.63853016 validation_bits_byte=5.34452631 validation_ms=790.106
curve regime=schedule variant=D seed=20261007 updates=500 macs=364510672 train_loss_token=4.41454120 validation_loss_token=3.87314627 validation_bits_byte=3.11818003 validation_ms=818.983
curve regime=schedule variant=D seed=20261007 updates=1000 macs=728881184 train_loss_token=4.18339990 validation_loss_token=3.43326169 validation_bits_byte=2.76403918 validation_ms=831.156
curve regime=schedule variant=D seed=20261007 updates=1500 macs=1093398816 train_loss_token=3.04201254 validation_loss_token=3.27457265 validation_bits_byte=2.63628232 validation_ms=823.847
curve regime=schedule variant=D seed=20261007 updates=2000 macs=1458695872 train_loss_token=3.96765155 validation_loss_token=3.20552103 validation_bits_byte=2.58069047 validation_ms=823.300
curve regime=mac variant=A seed=20261007 updates=1 macs=1620992 train_loss_token=6.23866541 validation_loss_token=6.23069904 validation_bits_byte=4.33065625 validation_ms=931.142
curve regime=mac variant=A seed=20261007 updates=500 macs=810496000 train_loss_token=4.67917161 validation_loss_token=4.47657993 validation_bits_byte=3.11145326 validation_ms=1005.600
curve regime=mac variant=A seed=20261007 updates=1000 macs=1620992000 train_loss_token=3.82494275 validation_loss_token=3.95905305 validation_bits_byte=2.75174547 validation_ms=1013.862
curve regime=mac variant=A seed=20261007 updates=1500 macs=2431488000 train_loss_token=3.69209975 validation_loss_token=3.80843877 validation_bits_byte=2.64706079 validation_ms=1026.385
curve regime=mac variant=A seed=20261007 updates=2000 macs=3241984000 train_loss_token=3.88036008 validation_loss_token=3.71740034 validation_bits_byte=2.58378440 validation_ms=1011.380
curve regime=mac variant=B seed=20261007 updates=1 macs=453632 train_loss_token=6.23892589 validation_loss_token=6.23108509 validation_bits_byte=4.33092457 validation_ms=505.008
curve regime=mac variant=B seed=20261007 updates=1787 macs=810640384 train_loss_token=3.80108347 validation_loss_token=3.74472633 validation_bits_byte=2.60277738 validation_ms=605.618
curve regime=mac variant=B seed=20261007 updates=3574 macs=1621280768 train_loss_token=2.69421911 validation_loss_token=3.64675858 validation_bits_byte=2.53468475 validation_ms=608.527
curve regime=mac variant=B seed=20261007 updates=5361 macs=2431921152 train_loss_token=3.30110336 validation_loss_token=3.59115711 validation_bits_byte=2.49603887 validation_ms=613.305
curve regime=mac variant=B seed=20261007 updates=7147 macs=3242107904 train_loss_token=3.46442002 validation_loss_token=3.59692420 validation_bits_byte=2.50004729 validation_ms=617.279
curve regime=mac variant=C seed=20261007 updates=1 macs=1620992 train_loss_token=6.23834576 validation_loss_token=6.22737397 validation_bits_byte=5.01351402 validation_ms=1091.242
curve regime=mac variant=C seed=20261007 updates=500 macs=810496000 train_loss_token=4.49230228 validation_loss_token=3.98285892 validation_bits_byte=3.20650713 validation_ms=1189.723
curve regime=mac variant=C seed=20261007 updates=1000 macs=1620992000 train_loss_token=4.12661156 validation_loss_token=3.42505896 validation_bits_byte=2.75743535 validation_ms=1191.562
curve regime=mac variant=C seed=20261007 updates=1500 macs=2431488000 train_loss_token=2.95065538 validation_loss_token=3.27440332 validation_bits_byte=2.63614600 validation_ms=1173.842
curve regime=mac variant=C seed=20261007 updates=2000 macs=3241984000 train_loss_token=3.90185962 validation_loss_token=3.17835599 validation_bits_byte=2.55882053 validation_ms=1177.268
curve regime=mac variant=D seed=20261007 updates=1 macs=723536 train_loss_token=6.63937905 validation_loss_token=6.63853016 validation_bits_byte=5.34452631 validation_ms=718.661
curve regime=mac variant=D seed=20261007 updates=1113 macs=810969312 train_loss_token=4.13947022 validation_loss_token=3.40083620 validation_bits_byte=2.73793417 validation_ms=818.595
curve regime=mac variant=D seed=20261007 updates=2223 macs=1621412928 train_loss_token=3.72438846 validation_loss_token=3.18137336 validation_bits_byte=2.56124974 validation_ms=841.315
curve regime=mac variant=D seed=20261007 updates=3335 macs=2432105584 train_loss_token=3.71767844 validation_loss_token=3.13451483 validation_bits_byte=2.52352503 validation_ms=823.272
curve regime=mac variant=D seed=20261007 updates=4446 macs=3242723376 train_loss_token=3.32988766 validation_loss_token=3.08005142 validation_bits_byte=2.47967780 validation_ms=822.692
curve regime=schedule variant=A seed=20261008 updates=1 macs=1620992 train_loss_token=6.23865247 validation_loss_token=6.23138436 validation_bits_byte=4.33113258 validation_ms=917.843
curve regime=schedule variant=A seed=20261008 updates=500 macs=810496000 train_loss_token=4.56857371 validation_loss_token=4.45380194 validation_bits_byte=3.09562139 validation_ms=1163.534
curve regime=schedule variant=A seed=20261008 updates=1000 macs=1620992000 train_loss_token=4.50248901 validation_loss_token=3.95218731 validation_bits_byte=2.74697343 validation_ms=1076.158
curve regime=schedule variant=A seed=20261008 updates=1500 macs=2431488000 train_loss_token=3.83664018 validation_loss_token=3.81671502 validation_bits_byte=2.65281321 validation_ms=1050.930
curve regime=schedule variant=A seed=20261008 updates=2000 macs=3241984000 train_loss_token=3.34704785 validation_loss_token=3.71895856 validation_bits_byte=2.58486745 validation_ms=1009.677
curve regime=schedule variant=B seed=20261008 updates=1 macs=453632 train_loss_token=6.23773264 validation_loss_token=6.23025952 validation_bits_byte=4.33035076 validation_ms=541.244
curve regime=schedule variant=B seed=20261008 updates=500 macs=226816000 train_loss_token=4.49506223 validation_loss_token=4.37757566 validation_bits_byte=3.04264020 validation_ms=601.892
curve regime=schedule variant=B seed=20261008 updates=1000 macs=453632000 train_loss_token=4.52960124 validation_loss_token=3.94495535 validation_bits_byte=2.74194684 validation_ms=603.125
curve regime=schedule variant=B seed=20261008 updates=1500 macs=680448000 train_loss_token=3.80033011 validation_loss_token=3.81158101 validation_bits_byte=2.64924482 validation_ms=606.305
curve regime=schedule variant=B seed=20261008 updates=2000 macs=907264000 train_loss_token=3.60332317 validation_loss_token=3.73826614 validation_bits_byte=2.59828721 validation_ms=606.683
curve regime=schedule variant=C seed=20261008 updates=1 macs=1620992 train_loss_token=6.23867776 validation_loss_token=6.22873672 validation_bits_byte=5.01461114 validation_ms=1067.488
curve regime=schedule variant=C seed=20261008 updates=500 macs=810496000 train_loss_token=4.46039426 validation_loss_token=3.99390485 validation_bits_byte=3.21539995 validation_ms=1181.864
curve regime=schedule variant=C seed=20261008 updates=1000 macs=1620992000 train_loss_token=3.04901639 validation_loss_token=3.45970320 validation_bits_byte=2.78532663 validation_ms=1177.788
curve regime=schedule variant=C seed=20261008 updates=1500 macs=2431488000 train_loss_token=3.49738866 validation_loss_token=3.28476324 validation_bits_byte=2.64448653 validation_ms=1187.382
curve regime=schedule variant=C seed=20261008 updates=2000 macs=3241984000 train_loss_token=2.46170277 validation_loss_token=3.17531647 validation_bits_byte=2.55637348 validation_ms=1211.883
curve regime=schedule variant=D seed=20261008 updates=1 macs=760304 train_loss_token=6.75920251 validation_loss_token=6.63610595 validation_bits_byte=5.34257465 validation_ms=702.230
curve regime=schedule variant=D seed=20261008 updates=500 macs=364779568 train_loss_token=4.35420452 validation_loss_token=3.84119041 validation_bits_byte=3.09245311 validation_ms=810.125
curve regime=schedule variant=D seed=20261008 updates=1000 macs=728949632 train_loss_token=3.12230256 validation_loss_token=3.42899826 validation_bits_byte=2.76060679 validation_ms=825.697
curve regime=schedule variant=D seed=20261008 updates=1500 macs=1093022208 train_loss_token=3.47482895 validation_loss_token=3.28194898 validation_bits_byte=2.64222083 validation_ms=822.649
curve regime=schedule variant=D seed=20261008 updates=2000 macs=1458324880 train_loss_token=2.52682305 validation_loss_token=3.19188618 validation_bits_byte=2.56971337 validation_ms=821.485
curve regime=mac variant=A seed=20261008 updates=1 macs=1620992 train_loss_token=6.23865247 validation_loss_token=6.23138436 validation_bits_byte=4.33113258 validation_ms=938.242
curve regime=mac variant=A seed=20261008 updates=500 macs=810496000 train_loss_token=4.56857371 validation_loss_token=4.45380194 validation_bits_byte=3.09562139 validation_ms=1007.680
curve regime=mac variant=A seed=20261008 updates=1000 macs=1620992000 train_loss_token=4.50248901 validation_loss_token=3.95218731 validation_bits_byte=2.74697343 validation_ms=1020.214
curve regime=mac variant=A seed=20261008 updates=1500 macs=2431488000 train_loss_token=3.83664018 validation_loss_token=3.81671502 validation_bits_byte=2.65281321 validation_ms=1028.151
curve regime=mac variant=A seed=20261008 updates=2000 macs=3241984000 train_loss_token=3.34704785 validation_loss_token=3.71895856 validation_bits_byte=2.58486745 validation_ms=1030.342
curve regime=mac variant=B seed=20261008 updates=1 macs=453632 train_loss_token=6.23773264 validation_loss_token=6.23025952 validation_bits_byte=4.33035076 validation_ms=512.515
curve regime=mac variant=B seed=20261008 updates=1787 macs=810640384 train_loss_token=3.42866873 validation_loss_token=3.74421951 validation_bits_byte=2.60242511 validation_ms=610.767
curve regime=mac variant=B seed=20261008 updates=3574 macs=1621280768 train_loss_token=3.74263350 validation_loss_token=3.64466537 validation_bits_byte=2.53322986 validation_ms=606.013
curve regime=mac variant=B seed=20261008 updates=5361 macs=2431921152 train_loss_token=3.81327901 validation_loss_token=3.59587303 validation_bits_byte=2.49931668 validation_ms=611.928
curve regime=mac variant=B seed=20261008 updates=7147 macs=3242107904 train_loss_token=3.49503793 validation_loss_token=3.58191361 validation_bits_byte=2.48961416 validation_ms=607.969
curve regime=mac variant=C seed=20261008 updates=1 macs=1620992 train_loss_token=6.23867776 validation_loss_token=6.22873672 validation_bits_byte=5.01461114 validation_ms=1069.337
curve regime=mac variant=C seed=20261008 updates=500 macs=810496000 train_loss_token=4.46039426 validation_loss_token=3.99390485 validation_bits_byte=3.21539995 validation_ms=1224.397
curve regime=mac variant=C seed=20261008 updates=1000 macs=1620992000 train_loss_token=3.04901639 validation_loss_token=3.45970320 validation_bits_byte=2.78532663 validation_ms=1207.194
curve regime=mac variant=C seed=20261008 updates=1500 macs=2431488000 train_loss_token=3.49738866 validation_loss_token=3.28476324 validation_bits_byte=2.64448653 validation_ms=1198.959
curve regime=mac variant=C seed=20261008 updates=2000 macs=3241984000 train_loss_token=2.46170277 validation_loss_token=3.17531647 validation_bits_byte=2.55637348 validation_ms=1224.392
curve regime=mac variant=D seed=20261008 updates=1 macs=760304 train_loss_token=6.75920251 validation_loss_token=6.63610595 validation_bits_byte=5.34257465 validation_ms=702.738
curve regime=mac variant=D seed=20261008 updates=1113 macs=810518400 train_loss_token=2.18041899 validation_loss_token=3.38817692 validation_bits_byte=2.72774248 validation_ms=823.762
curve regime=mac variant=D seed=20261008 updates=2224 macs=1621370144 train_loss_token=3.79491322 validation_loss_token=3.19603715 validation_bits_byte=2.57305522 validation_ms=824.917
curve regime=mac variant=D seed=20261008 updates=3335 macs=2431802992 train_loss_token=2.94048169 validation_loss_token=3.12012670 validation_bits_byte=2.51194148 validation_ms=819.536
curve regime=mac variant=D seed=20261008 updates=4444 macs=3242038832 train_loss_token=3.35051260 validation_loss_token=3.06354143 validation_bits_byte=2.46638599 validation_ms=840.208
curve regime=schedule variant=A seed=20261009 updates=1 macs=1620992 train_loss_token=6.23855517 validation_loss_token=6.23314545 validation_bits_byte=4.33235663 validation_ms=913.113
curve regime=schedule variant=A seed=20261009 updates=500 macs=810496000 train_loss_token=4.60788457 validation_loss_token=4.41166639 validation_bits_byte=3.06633502 validation_ms=1013.508
curve regime=schedule variant=A seed=20261009 updates=1000 macs=1620992000 train_loss_token=4.10116966 validation_loss_token=3.94103519 validation_bits_byte=2.73922213 validation_ms=1022.262
curve regime=schedule variant=A seed=20261009 updates=1500 macs=2431488000 train_loss_token=4.41351320 validation_loss_token=3.79313484 validation_bits_byte=2.63642378 validation_ms=1018.298
curve regime=schedule variant=A seed=20261009 updates=2000 macs=3241984000 train_loss_token=2.97603801 validation_loss_token=3.72273970 validation_bits_byte=2.58749553 validation_ms=1027.550
curve regime=schedule variant=B seed=20261009 updates=1 macs=453632 train_loss_token=6.23927834 validation_loss_token=6.23352731 validation_bits_byte=4.33262205 validation_ms=506.621
curve regime=schedule variant=B seed=20261009 updates=500 macs=226816000 train_loss_token=4.55728472 validation_loss_token=4.41940602 validation_bits_byte=3.07171445 validation_ms=603.499
curve regime=schedule variant=B seed=20261009 updates=1000 macs=453632000 train_loss_token=4.09588411 validation_loss_token=3.95306554 validation_bits_byte=2.74758384 validation_ms=607.853
curve regime=schedule variant=B seed=20261009 updates=1500 macs=680448000 train_loss_token=4.48896395 validation_loss_token=3.80783206 validation_bits_byte=2.64663910 validation_ms=623.695
curve regime=schedule variant=B seed=20261009 updates=2000 macs=907264000 train_loss_token=3.07072134 validation_loss_token=3.73553500 validation_bits_byte=2.59638893 validation_ms=606.245
curve regime=schedule variant=C seed=20261009 updates=1 macs=1620992 train_loss_token=6.23818703 validation_loss_token=6.22913904 validation_bits_byte=5.01493505 validation_ms=1056.905
curve regime=schedule variant=C seed=20261009 updates=500 macs=810496000 train_loss_token=3.92749549 validation_loss_token=3.95271907 validation_bits_byte=3.18224224 validation_ms=1190.962
curve regime=schedule variant=C seed=20261009 updates=1000 macs=1620992000 train_loss_token=4.11376546 validation_loss_token=3.43172678 validation_bits_byte=2.76280346 validation_ms=1206.318
curve regime=schedule variant=C seed=20261009 updates=1500 macs=2431488000 train_loss_token=3.49195106 validation_loss_token=3.26059519 validation_bits_byte=2.62502940 validation_ms=1217.614
curve regime=schedule variant=C seed=20261009 updates=2000 macs=3241984000 train_loss_token=3.49615718 validation_loss_token=3.16305306 validation_bits_byte=2.54650049 validation_ms=1210.918
curve regime=schedule variant=D seed=20261009 updates=1 macs=727904 train_loss_token=6.62646427 validation_loss_token=6.63940145 validation_bits_byte=5.34522777 validation_ms=700.982
curve regime=schedule variant=D seed=20261009 updates=500 macs=364451056 train_loss_token=3.81549395 validation_loss_token=3.79237876 validation_bits_byte=3.05315599 validation_ms=831.461
curve regime=schedule variant=D seed=20261009 updates=1000 macs=728485472 train_loss_token=4.19529449 validation_loss_token=3.44721284 validation_bits_byte=2.77527093 validation_ms=819.997
curve regime=schedule variant=D seed=20261009 updates=1500 macs=1092428112 train_loss_token=3.53824956 validation_loss_token=3.27582555 validation_bits_byte=2.63729100 validation_ms=816.960
curve regime=schedule variant=D seed=20261009 updates=2000 macs=1457013232 train_loss_token=3.41389996 validation_loss_token=3.18347981 validation_bits_byte=2.56294560 validation_ms=823.909
curve regime=mac variant=A seed=20261009 updates=1 macs=1620992 train_loss_token=6.23855517 validation_loss_token=6.23314545 validation_bits_byte=4.33235663 validation_ms=969.899
curve regime=mac variant=A seed=20261009 updates=500 macs=810496000 train_loss_token=4.60788457 validation_loss_token=4.41166639 validation_bits_byte=3.06633502 validation_ms=1020.835
curve regime=mac variant=A seed=20261009 updates=1000 macs=1620992000 train_loss_token=4.10116966 validation_loss_token=3.94103519 validation_bits_byte=2.73922213 validation_ms=1011.509
curve regime=mac variant=A seed=20261009 updates=1500 macs=2431488000 train_loss_token=4.41351320 validation_loss_token=3.79313484 validation_bits_byte=2.63642378 validation_ms=1025.565
curve regime=mac variant=A seed=20261009 updates=2000 macs=3241984000 train_loss_token=2.97603801 validation_loss_token=3.72273970 validation_bits_byte=2.58749553 validation_ms=1015.215
curve regime=mac variant=B seed=20261009 updates=1 macs=453632 train_loss_token=6.23927834 validation_loss_token=6.23352731 validation_bits_byte=4.33262205 validation_ms=514.542
curve regime=mac variant=B seed=20261009 updates=1787 macs=810640384 train_loss_token=3.74959618 validation_loss_token=3.76146186 validation_bits_byte=2.61440943 validation_ms=606.881
curve regime=mac variant=B seed=20261009 updates=3574 macs=1621280768 train_loss_token=3.00618516 validation_loss_token=3.62236317 validation_bits_byte=2.51772869 validation_ms=616.755
curve regime=mac variant=B seed=20261009 updates=5361 macs=2431921152 train_loss_token=3.14761119 validation_loss_token=3.57502596 validation_bits_byte=2.48482689 validation_ms=606.258
curve regime=mac variant=B seed=20261009 updates=7147 macs=3242107904 train_loss_token=3.29304782 validation_loss_token=3.55723312 validation_bits_byte=2.47245995 validation_ms=612.714
curve regime=mac variant=C seed=20261009 updates=1 macs=1620992 train_loss_token=6.23818703 validation_loss_token=6.22913904 validation_bits_byte=5.01493505 validation_ms=1086.726
curve regime=mac variant=C seed=20261009 updates=500 macs=810496000 train_loss_token=3.92749549 validation_loss_token=3.95271907 validation_bits_byte=3.18224224 validation_ms=1188.809
curve regime=mac variant=C seed=20261009 updates=1000 macs=1620992000 train_loss_token=4.11376546 validation_loss_token=3.43172678 validation_bits_byte=2.76280346 validation_ms=1182.294
curve regime=mac variant=C seed=20261009 updates=1500 macs=2431488000 train_loss_token=3.49195106 validation_loss_token=3.26059519 validation_bits_byte=2.62502940 validation_ms=1213.900
curve regime=mac variant=C seed=20261009 updates=2000 macs=3241984000 train_loss_token=3.49615718 validation_loss_token=3.16305306 validation_bits_byte=2.54650049 validation_ms=1179.958
curve regime=mac variant=D seed=20261009 updates=1 macs=727904 train_loss_token=6.62646427 validation_loss_token=6.63940145 validation_bits_byte=5.34522777 validation_ms=704.351
curve regime=mac variant=D seed=20261009 updates=1113 macs=810624336 train_loss_token=3.65846501 validation_loss_token=3.35424438 validation_bits_byte=2.70042418 validation_ms=858.752
curve regime=mac variant=D seed=20261009 updates=2226 macs=1621202688 train_loss_token=3.54369441 validation_loss_token=3.16619051 validation_bits_byte=2.54902638 validation_ms=819.238
curve regime=mac variant=D seed=20261009 updates=3339 macs=2431742640 train_loss_token=3.94959782 validation_loss_token=3.10804255 validation_bits_byte=2.50221281 validation_ms=826.078
curve regime=mac variant=D seed=20261009 updates=4451 macs=3242561632 train_loss_token=2.80117392 validation_loss_token=3.05738769 validation_bits_byte=2.46143176 validation_ms=810.932
```
