# Subset Study

The min-faithful subset is the smallest size where the full-split winner wins with a 95% CI excluding 0 for all three seeds.

## Min-Faithful Subset

| Strategy | Min-faithful subset |
|---|---|
| random | 50 |
| label_stratified | 50 |
| density | 50 |
| length | 50 |
| hardness | 10 |

## F1 Variance By Cell

| Strategy | Size | Model | Seeds | Mean F1 | Variance | Min F1 | Max F1 |
|---|---|---|---:|---:|---:|---:|---:|
| random | 10 | cyner | 3 | 0.0899 | 0.000386 | 0.0678 | 0.1053 |
| random | 10 | securebert | 3 | 0.3274 | 0.013517 | 0.2564 | 0.4615 |
| random | 50 | cyner | 3 | 0.0941 | 0.000318 | 0.0738 | 0.1074 |
| random | 50 | securebert | 3 | 0.2784 | 0.001594 | 0.2428 | 0.3216 |
| random | 100 | cyner | 3 | 0.0899 | 0.000174 | 0.0751 | 0.1004 |
| random | 100 | securebert | 3 | 0.2843 | 0.001218 | 0.2451 | 0.3118 |
| random | 250 | cyner | 3 | 0.1067 | 0.000011 | 0.1031 | 0.1096 |
| random | 250 | securebert | 3 | 0.2862 | 0.000286 | 0.2754 | 0.3057 |
| random | all | cyner | 3 | 0.1038 | 0.000000 | 0.1038 | 0.1038 |
| random | all | securebert | 3 | 0.2823 | 0.000000 | 0.2823 | 0.2823 |
| label_stratified | 10 | cyner | 3 | 0.1527 | 0.000645 | 0.1351 | 0.1818 |
| label_stratified | 10 | securebert | 3 | 0.3370 | 0.008966 | 0.2810 | 0.4463 |
| label_stratified | 50 | cyner | 3 | 0.1151 | 0.000008 | 0.1121 | 0.1176 |
| label_stratified | 50 | securebert | 3 | 0.3477 | 0.003378 | 0.2828 | 0.3949 |
| label_stratified | 100 | cyner | 3 | 0.0962 | 0.000260 | 0.0832 | 0.1142 |
| label_stratified | 100 | securebert | 3 | 0.3178 | 0.000191 | 0.3023 | 0.3289 |
| label_stratified | 250 | cyner | 3 | 0.1089 | 0.000140 | 0.0978 | 0.1214 |
| label_stratified | 250 | securebert | 3 | 0.3005 | 0.000065 | 0.2937 | 0.3094 |
| label_stratified | all | cyner | 3 | 0.1038 | 0.000000 | 0.1038 | 0.1038 |
| label_stratified | all | securebert | 3 | 0.2823 | 0.000000 | 0.2823 | 0.2823 |
| density | 10 | cyner | 3 | 0.0562 | 0.001508 | 0.0227 | 0.0988 |
| density | 10 | securebert | 3 | 0.1791 | 0.007968 | 0.1000 | 0.2759 |
| density | 50 | cyner | 3 | 0.0714 | 0.000205 | 0.0594 | 0.0872 |
| density | 50 | securebert | 3 | 0.2190 | 0.000989 | 0.1933 | 0.2541 |
| density | 100 | cyner | 3 | 0.0692 | 0.000348 | 0.0547 | 0.0902 |
| density | 100 | securebert | 3 | 0.2473 | 0.001033 | 0.2129 | 0.2765 |
| density | 250 | cyner | 3 | 0.0931 | 0.000213 | 0.0828 | 0.1098 |
| density | 250 | securebert | 3 | 0.2647 | 0.000361 | 0.2514 | 0.2864 |
| density | all | cyner | 3 | 0.1038 | 0.000000 | 0.1038 | 0.1038 |
| density | all | securebert | 3 | 0.2823 | 0.000000 | 0.2823 | 0.2823 |
| length | 10 | cyner | 3 | 0.1037 | 0.004121 | 0.0357 | 0.1633 |
| length | 10 | securebert | 3 | 0.3386 | 0.026471 | 0.2254 | 0.5250 |
| length | 50 | cyner | 3 | 0.1062 | 0.002175 | 0.0673 | 0.1579 |
| length | 50 | securebert | 3 | 0.2776 | 0.004626 | 0.2371 | 0.3562 |
| length | 100 | cyner | 3 | 0.1055 | 0.000691 | 0.0864 | 0.1355 |
| length | 100 | securebert | 3 | 0.2891 | 0.000793 | 0.2673 | 0.3209 |
| length | 250 | cyner | 3 | 0.1052 | 0.000008 | 0.1023 | 0.1081 |
| length | 250 | securebert | 3 | 0.3001 | 0.000000 | 0.2998 | 0.3005 |
| length | all | cyner | 3 | 0.1038 | 0.000000 | 0.1038 | 0.1038 |
| length | all | securebert | 3 | 0.2823 | 0.000000 | 0.2823 | 0.2823 |
| hardness | 10 | cyner | 3 | 0.0244 | 0.000000 | 0.0244 | 0.0244 |
| hardness | 10 | securebert | 3 | 0.2830 | 0.000000 | 0.2830 | 0.2830 |
| hardness | 50 | cyner | 3 | 0.0178 | 0.000000 | 0.0178 | 0.0178 |
| hardness | 50 | securebert | 3 | 0.2434 | 0.000000 | 0.2434 | 0.2434 |
| hardness | 100 | cyner | 3 | 0.0487 | 0.000000 | 0.0483 | 0.0492 |
| hardness | 100 | securebert | 3 | 0.3012 | 0.000001 | 0.3000 | 0.3020 |
| hardness | 250 | cyner | 3 | 0.0732 | 0.000000 | 0.0730 | 0.0734 |
| hardness | 250 | securebert | 3 | 0.3039 | 0.000004 | 0.3024 | 0.3060 |
| hardness | all | cyner | 3 | 0.1038 | 0.000000 | 0.1038 | 0.1038 |
| hardness | all | securebert | 3 | 0.2823 | 0.000000 | 0.2823 | 0.2823 |

## Per-Seed Decisions

| Strategy | Size | Seed | Model | Samples | Strict F1 | Gap | 95% CI | Flip? |
|---|---|---:|---|---:|---:|---:|---|---|
| random | 10 | 1 | cyner | 10 | 0.0968 | -0.1674 | [-0.3962, 0.0149] | yes |
| random | 10 | 1 | securebert | 10 | 0.2642 | 0.1674 | [-0.0149, 0.3962] | yes |
| random | 10 | 2 | cyner | 10 | 0.1053 | -0.1511 | [-0.3044, 0.0186] | yes |
| random | 10 | 2 | securebert | 10 | 0.2564 | 0.1511 | [-0.0186, 0.3044] | yes |
| random | 10 | 3 | cyner | 10 | 0.0678 | -0.3937 | [-0.5914, -0.1316] | no |
| random | 10 | 3 | securebert | 10 | 0.4615 | 0.3937 | [0.1316, 0.5914] | no |
| random | 50 | 1 | cyner | 50 | 0.1074 | -0.1635 | [-0.2620, -0.0713] | no |
| random | 50 | 1 | securebert | 50 | 0.2708 | 0.1635 | [0.0713, 0.2620] | no |
| random | 50 | 2 | cyner | 50 | 0.1010 | -0.2206 | [-0.3253, -0.1224] | no |
| random | 50 | 2 | securebert | 50 | 0.3216 | 0.2206 | [0.1224, 0.3253] | no |
| random | 50 | 3 | cyner | 50 | 0.0738 | -0.1690 | [-0.2727, -0.0724] | no |
| random | 50 | 3 | securebert | 50 | 0.2428 | 0.1690 | [0.0724, 0.2727] | no |
| random | 100 | 1 | cyner | 100 | 0.1004 | -0.1446 | [-0.2139, -0.0817] | no |
| random | 100 | 1 | securebert | 100 | 0.2451 | 0.1446 | [0.0817, 0.2139] | no |
| random | 100 | 2 | cyner | 100 | 0.0751 | -0.2367 | [-0.3090, -0.1660] | no |
| random | 100 | 2 | securebert | 100 | 0.3118 | 0.2367 | [0.1660, 0.3090] | no |
| random | 100 | 3 | cyner | 100 | 0.0942 | -0.2020 | [-0.2921, -0.1192] | no |
| random | 100 | 3 | securebert | 100 | 0.2961 | 0.2020 | [0.1192, 0.2921] | no |
| random | 250 | 1 | cyner | 250 | 0.1074 | -0.1679 | [-0.2106, -0.1266] | no |
| random | 250 | 1 | securebert | 250 | 0.2754 | 0.1679 | [0.1266, 0.2106] | no |
| random | 250 | 2 | cyner | 250 | 0.1031 | -0.2026 | [-0.2468, -0.1601] | no |
| random | 250 | 2 | securebert | 250 | 0.3057 | 0.2026 | [0.1601, 0.2468] | no |
| random | 250 | 3 | cyner | 250 | 0.1096 | -0.1679 | [-0.2184, -0.1188] | no |
| random | 250 | 3 | securebert | 250 | 0.2775 | 0.1679 | [0.1188, 0.2184] | no |
| random | all | 1 | cyner | 664 | 0.1038 | -0.1785 | [-0.2056, -0.1512] | no |
| random | all | 1 | securebert | 664 | 0.2823 | 0.1785 | [0.1512, 0.2056] | no |
| random | all | 2 | cyner | 664 | 0.1038 | -0.1785 | [-0.2055, -0.1514] | no |
| random | all | 2 | securebert | 664 | 0.2823 | 0.1785 | [0.1514, 0.2055] | no |
| random | all | 3 | cyner | 664 | 0.1038 | -0.1785 | [-0.2065, -0.1509] | no |
| random | all | 3 | securebert | 664 | 0.2823 | 0.1785 | [0.1509, 0.2065] | no |
| label_stratified | 10 | 1 | cyner | 10 | 0.1818 | -0.0992 | [-0.3322, 0.1141] | yes |
| label_stratified | 10 | 1 | securebert | 10 | 0.2810 | 0.0992 | [-0.1141, 0.3322] | yes |
| label_stratified | 10 | 2 | cyner | 10 | 0.1351 | -0.3111 | [-0.5135, -0.0700] | no |
| label_stratified | 10 | 2 | securebert | 10 | 0.4463 | 0.3111 | [0.0700, 0.5135] | no |
| label_stratified | 10 | 3 | cyner | 10 | 0.1412 | -0.1424 | [-0.3583, 0.0631] | yes |
| label_stratified | 10 | 3 | securebert | 10 | 0.2836 | 0.1424 | [-0.0631, 0.3583] | yes |
| label_stratified | 50 | 1 | cyner | 50 | 0.1157 | -0.2792 | [-0.3783, -0.1746] | no |
| label_stratified | 50 | 1 | securebert | 50 | 0.3949 | 0.2792 | [0.1746, 0.3783] | no |
| label_stratified | 50 | 2 | cyner | 50 | 0.1176 | -0.1651 | [-0.2472, -0.0815] | no |
| label_stratified | 50 | 2 | securebert | 50 | 0.2828 | 0.1651 | [0.0815, 0.2472] | no |
| label_stratified | 50 | 3 | cyner | 50 | 0.1121 | -0.2533 | [-0.3499, -0.1585] | no |
| label_stratified | 50 | 3 | securebert | 50 | 0.3654 | 0.2533 | [0.1585, 0.3499] | no |
| label_stratified | 100 | 1 | cyner | 100 | 0.1142 | -0.2081 | [-0.2756, -0.1426] | no |
| label_stratified | 100 | 1 | securebert | 100 | 0.3223 | 0.2081 | [0.1426, 0.2756] | no |
| label_stratified | 100 | 2 | cyner | 100 | 0.0832 | -0.2191 | [-0.2935, -0.1492] | no |
| label_stratified | 100 | 2 | securebert | 100 | 0.3023 | 0.2191 | [0.1492, 0.2935] | no |
| label_stratified | 100 | 3 | cyner | 100 | 0.0910 | -0.2378 | [-0.3060, -0.1698] | no |
| label_stratified | 100 | 3 | securebert | 100 | 0.3289 | 0.2378 | [0.1698, 0.3060] | no |
| label_stratified | 250 | 1 | cyner | 250 | 0.1214 | -0.1770 | [-0.2206, -0.1341] | no |
| label_stratified | 250 | 1 | securebert | 250 | 0.2984 | 0.1770 | [0.1341, 0.2206] | no |
| label_stratified | 250 | 2 | cyner | 250 | 0.0978 | -0.2116 | [-0.2554, -0.1688] | no |
| label_stratified | 250 | 2 | securebert | 250 | 0.3094 | 0.2116 | [0.1688, 0.2554] | no |
| label_stratified | 250 | 3 | cyner | 250 | 0.1075 | -0.1862 | [-0.2274, -0.1461] | no |
| label_stratified | 250 | 3 | securebert | 250 | 0.2937 | 0.1862 | [0.1461, 0.2274] | no |
| label_stratified | all | 1 | cyner | 664 | 0.1038 | -0.1785 | [-0.2056, -0.1512] | no |
| label_stratified | all | 1 | securebert | 664 | 0.2823 | 0.1785 | [0.1512, 0.2056] | no |
| label_stratified | all | 2 | cyner | 664 | 0.1038 | -0.1785 | [-0.2055, -0.1514] | no |
| label_stratified | all | 2 | securebert | 664 | 0.2823 | 0.1785 | [0.1514, 0.2055] | no |
| label_stratified | all | 3 | cyner | 664 | 0.1038 | -0.1785 | [-0.2065, -0.1509] | no |
| label_stratified | all | 3 | securebert | 664 | 0.2823 | 0.1785 | [0.1509, 0.2065] | no |
| density | 10 | 1 | cyner | 10 | 0.0988 | -0.0012 | [-0.1176, 0.1433] | yes |
| density | 10 | 1 | securebert | 10 | 0.1000 | 0.0012 | [-0.1433, 0.1176] | yes |
| density | 10 | 2 | cyner | 10 | 0.0227 | -0.1386 | [-0.3258, -0.0242] | no |
| density | 10 | 2 | securebert | 10 | 0.1613 | 0.1386 | [0.0242, 0.3258] | no |
| density | 10 | 3 | cyner | 10 | 0.0471 | -0.2288 | [-0.5978, -0.0202] | no |
| density | 10 | 3 | securebert | 10 | 0.2759 | 0.2288 | [0.0202, 0.5978] | no |
| density | 50 | 1 | cyner | 50 | 0.0872 | -0.1224 | [-0.2282, -0.0304] | no |
| density | 50 | 1 | securebert | 50 | 0.2096 | 0.1224 | [0.0304, 0.2282] | no |
| density | 50 | 2 | cyner | 50 | 0.0594 | -0.1339 | [-0.2216, -0.0646] | no |
| density | 50 | 2 | securebert | 50 | 0.1933 | 0.1339 | [0.0646, 0.2216] | no |
| density | 50 | 3 | cyner | 50 | 0.0674 | -0.1866 | [-0.3041, -0.0926] | no |
| density | 50 | 3 | securebert | 50 | 0.2541 | 0.1866 | [0.0926, 0.3041] | no |
| density | 100 | 1 | cyner | 100 | 0.0902 | -0.1226 | [-0.1895, -0.0628] | no |
| density | 100 | 1 | securebert | 100 | 0.2129 | 0.1226 | [0.0628, 0.1895] | no |
| density | 100 | 2 | cyner | 100 | 0.0625 | -0.2140 | [-0.2875, -0.1449] | no |
| density | 100 | 2 | securebert | 100 | 0.2765 | 0.2140 | [0.1449, 0.2875] | no |
| density | 100 | 3 | cyner | 100 | 0.0547 | -0.1978 | [-0.2728, -0.1286] | no |
| density | 100 | 3 | securebert | 100 | 0.2526 | 0.1978 | [0.1286, 0.2728] | no |
| density | 250 | 1 | cyner | 250 | 0.1098 | -0.1416 | [-0.1894, -0.0950] | no |
| density | 250 | 1 | securebert | 250 | 0.2514 | 0.1416 | [0.0950, 0.1894] | no |
| density | 250 | 2 | cyner | 250 | 0.0866 | -0.1696 | [-0.2183, -0.1250] | no |
| density | 250 | 2 | securebert | 250 | 0.2562 | 0.1696 | [0.1250, 0.2183] | no |
| density | 250 | 3 | cyner | 250 | 0.0828 | -0.2036 | [-0.2471, -0.1616] | no |
| density | 250 | 3 | securebert | 250 | 0.2864 | 0.2036 | [0.1616, 0.2471] | no |
| density | all | 1 | cyner | 664 | 0.1038 | -0.1785 | [-0.2056, -0.1512] | no |
| density | all | 1 | securebert | 664 | 0.2823 | 0.1785 | [0.1512, 0.2056] | no |
| density | all | 2 | cyner | 664 | 0.1038 | -0.1785 | [-0.2055, -0.1514] | no |
| density | all | 2 | securebert | 664 | 0.2823 | 0.1785 | [0.1514, 0.2055] | no |
| density | all | 3 | cyner | 664 | 0.1038 | -0.1785 | [-0.2065, -0.1509] | no |
| density | all | 3 | securebert | 664 | 0.2823 | 0.1785 | [0.1509, 0.2065] | no |
| length | 10 | 1 | cyner | 10 | 0.1633 | -0.3617 | [-0.4967, -0.2013] | no |
| length | 10 | 1 | securebert | 10 | 0.5250 | 0.3617 | [0.2013, 0.4967] | no |
| length | 10 | 2 | cyner | 10 | 0.0357 | -0.2296 | [-0.4176, -0.0711] | no |
| length | 10 | 2 | securebert | 10 | 0.2653 | 0.2296 | [0.0711, 0.4176] | no |
| length | 10 | 3 | cyner | 10 | 0.1121 | -0.1132 | [-0.3642, 0.0630] | yes |
| length | 10 | 3 | securebert | 10 | 0.2254 | 0.1132 | [-0.0630, 0.3642] | yes |
| length | 50 | 1 | cyner | 50 | 0.1579 | -0.1983 | [-0.2911, -0.1105] | no |
| length | 50 | 1 | securebert | 50 | 0.3562 | 0.1983 | [0.1105, 0.2911] | no |
| length | 50 | 2 | cyner | 50 | 0.0673 | -0.1698 | [-0.2399, -0.1055] | no |
| length | 50 | 2 | securebert | 50 | 0.2371 | 0.1698 | [0.1055, 0.2399] | no |
| length | 50 | 3 | cyner | 50 | 0.0935 | -0.1462 | [-0.2389, -0.0650] | no |
| length | 50 | 3 | securebert | 50 | 0.2397 | 0.1462 | [0.0650, 0.2389] | no |
| length | 100 | 1 | cyner | 100 | 0.1355 | -0.1854 | [-0.2469, -0.1235] | no |
| length | 100 | 1 | securebert | 100 | 0.3209 | 0.1854 | [0.1235, 0.2469] | no |
| length | 100 | 2 | cyner | 100 | 0.0864 | -0.1809 | [-0.2460, -0.1194] | no |
| length | 100 | 2 | securebert | 100 | 0.2673 | 0.1809 | [0.1194, 0.2460] | no |
| length | 100 | 3 | cyner | 100 | 0.0946 | -0.1844 | [-0.2541, -0.1210] | no |
| length | 100 | 3 | securebert | 100 | 0.2790 | 0.1844 | [0.1210, 0.2541] | no |
| length | 250 | 1 | cyner | 250 | 0.1081 | -0.1919 | [-0.2381, -0.1472] | no |
| length | 250 | 1 | securebert | 250 | 0.3000 | 0.1919 | [0.1472, 0.2381] | no |
| length | 250 | 2 | cyner | 250 | 0.1053 | -0.1951 | [-0.2453, -0.1486] | no |
| length | 250 | 2 | securebert | 250 | 0.3005 | 0.1951 | [0.1486, 0.2453] | no |
| length | 250 | 3 | cyner | 250 | 0.1023 | -0.1975 | [-0.2407, -0.1551] | no |
| length | 250 | 3 | securebert | 250 | 0.2998 | 0.1975 | [0.1551, 0.2407] | no |
| length | all | 1 | cyner | 664 | 0.1038 | -0.1785 | [-0.2056, -0.1512] | no |
| length | all | 1 | securebert | 664 | 0.2823 | 0.1785 | [0.1512, 0.2056] | no |
| length | all | 2 | cyner | 664 | 0.1038 | -0.1785 | [-0.2055, -0.1514] | no |
| length | all | 2 | securebert | 664 | 0.2823 | 0.1785 | [0.1514, 0.2055] | no |
| length | all | 3 | cyner | 664 | 0.1038 | -0.1785 | [-0.2065, -0.1509] | no |
| length | all | 3 | securebert | 664 | 0.2823 | 0.1785 | [0.1509, 0.2065] | no |
| hardness | 10 | 1 | cyner | 10 | 0.0244 | -0.2586 | [-0.3499, -0.1712] | no |
| hardness | 10 | 1 | securebert | 10 | 0.2830 | 0.2586 | [0.1712, 0.3499] | no |
| hardness | 10 | 2 | cyner | 10 | 0.0244 | -0.2586 | [-0.3507, -0.1717] | no |
| hardness | 10 | 2 | securebert | 10 | 0.2830 | 0.2586 | [0.1717, 0.3507] | no |
| hardness | 10 | 3 | cyner | 10 | 0.0244 | -0.2586 | [-0.3486, -0.1711] | no |
| hardness | 10 | 3 | securebert | 10 | 0.2830 | 0.2586 | [0.1711, 0.3486] | no |
| hardness | 50 | 1 | cyner | 50 | 0.0178 | -0.2256 | [-0.2802, -0.1725] | no |
| hardness | 50 | 1 | securebert | 50 | 0.2434 | 0.2256 | [0.1725, 0.2802] | no |
| hardness | 50 | 2 | cyner | 50 | 0.0178 | -0.2256 | [-0.2796, -0.1719] | no |
| hardness | 50 | 2 | securebert | 50 | 0.2434 | 0.2256 | [0.1719, 0.2796] | no |
| hardness | 50 | 3 | cyner | 50 | 0.0178 | -0.2256 | [-0.2789, -0.1721] | no |
| hardness | 50 | 3 | securebert | 50 | 0.2434 | 0.2256 | [0.1721, 0.2789] | no |
| hardness | 100 | 1 | cyner | 100 | 0.0483 | -0.2533 | [-0.3014, -0.2037] | no |
| hardness | 100 | 1 | securebert | 100 | 0.3017 | 0.2533 | [0.2037, 0.3014] | no |
| hardness | 100 | 2 | cyner | 100 | 0.0492 | -0.2508 | [-0.3004, -0.2022] | no |
| hardness | 100 | 2 | securebert | 100 | 0.3000 | 0.2508 | [0.2022, 0.3004] | no |
| hardness | 100 | 3 | cyner | 100 | 0.0486 | -0.2533 | [-0.3018, -0.2047] | no |
| hardness | 100 | 3 | securebert | 100 | 0.3020 | 0.2533 | [0.2047, 0.3018] | no |
| hardness | 250 | 1 | cyner | 250 | 0.0734 | -0.2326 | [-0.2768, -0.1903] | no |
| hardness | 250 | 1 | securebert | 250 | 0.3060 | 0.2326 | [0.1903, 0.2768] | no |
| hardness | 250 | 2 | cyner | 250 | 0.0730 | -0.2302 | [-0.2736, -0.1875] | no |
| hardness | 250 | 2 | securebert | 250 | 0.3032 | 0.2302 | [0.1875, 0.2736] | no |
| hardness | 250 | 3 | cyner | 250 | 0.0732 | -0.2292 | [-0.2729, -0.1867] | no |
| hardness | 250 | 3 | securebert | 250 | 0.3024 | 0.2292 | [0.1867, 0.2729] | no |
| hardness | all | 1 | cyner | 664 | 0.1038 | -0.1785 | [-0.2056, -0.1512] | no |
| hardness | all | 1 | securebert | 664 | 0.2823 | 0.1785 | [0.1512, 0.2056] | no |
| hardness | all | 2 | cyner | 664 | 0.1038 | -0.1785 | [-0.2055, -0.1514] | no |
| hardness | all | 2 | securebert | 664 | 0.2823 | 0.1785 | [0.1514, 0.2055] | no |
| hardness | all | 3 | cyner | 664 | 0.1038 | -0.1785 | [-0.2065, -0.1509] | no |
| hardness | all | 3 | securebert | 664 | 0.2823 | 0.1785 | [0.1509, 0.2065] | no |
