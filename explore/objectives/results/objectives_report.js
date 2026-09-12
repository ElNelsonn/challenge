window.OBJECTIVES_REPORT = {
  "dataset": "C:\\Users\\santb\\OneDrive\\Escritorio\\Challange\\wnn_connectome_starterpack\\datasets\\train.parquet",
  "sequences_analyzed": 10607,
  "sequences_available": 10607,
  "selected_rows": 211089907,
  "selection": {
    "include_warmup": false,
    "scored_only": false
  },
  "notes": {
    "exact": "moments, ranges, threshold rates and correlations",
    "approximate": "percentiles and outlier rates use a stratified sample",
    "correlation_type": "ordinary unweighted Pearson (not competition WP)"
  },
  "corr_t0_t1": -0.722240287325947,
  "targets": {
    "t0": {
      "count": 211089907,
      "mean": -0.2716546081732456,
      "std_population": 1.0060094211871948,
      "min": -125.56766510009766,
      "max": 74.49723815917969,
      "near_zero_pct": {
        "0.01": 21.870497579024466,
        "0.05": 21.870497579024466,
        "0.1": 21.870497579024466
      },
      "outside_minus2_plus2_pct": 3.1344411933442182,
      "autocorrelation": {
        "1": 0.9753063347174604,
        "5": 0.9243464693218127,
        "10": 0.8905083671790223
      },
      "autocorrelation_pair_counts": {
        "1": 211079300,
        "5": 211036872,
        "10": 210983837
      },
      "sample_size": 1000000,
      "percentiles_approx": {
        "0": -14.23845386505127,
        "0.1": -2.6546270847320557,
        "1": -1.9706029891967773,
        "5": -1.9706029891967773,
        "25": -0.6568676829338074,
        "50": -0.4685360789299011,
        "75": 0.0,
        "95": 1.4056081771850586,
        "99": 3.137286424636841,
        "99.9": 5.622432708740234,
        "100": 19.54770851135254
      },
      "sample_iqr": 0.6568676829338074,
      "sample_iqr_outlier_limits": [
        -1.6421692073345184,
        0.9853015244007111
      ],
      "sample_iqr_outlier_pct": 16.1407,
      "sample_abs_z_gt_3_pct": 1.3846
    },
    "t1": {
      "count": 211089907,
      "mean": -0.26678856388854094,
      "std_population": 1.0109143739927302,
      "min": -179.9178466796875,
      "max": 77.3084487915039,
      "near_zero_pct": {
        "0.01": 21.838184333465076,
        "0.05": 21.838184333465076,
        "0.1": 21.838184333465076
      },
      "outside_minus2_plus2_pct": 3.163598437702661,
      "autocorrelation": {
        "1": 0.9741300534427249,
        "5": 0.9225996762147218,
        "10": 0.8891149408587199
      },
      "autocorrelation_pair_counts": {
        "1": 211079300,
        "5": 211036872,
        "10": 210983837
      },
      "sample_size": 1000000,
      "percentiles_approx": {
        "0": -72.62309265136719,
        "0.1": -2.6546270847320557,
        "1": -1.9706029891967773,
        "5": -1.9706029891967773,
        "25": -0.6568676829338074,
        "50": -0.4685360789299011,
        "75": 0.0,
        "95": 1.4056081771850586,
        "99": 3.1358284950256348,
        "99.9": 5.622432708740234,
        "100": 19.818435668945312
      },
      "sample_iqr": 0.6568676829338074,
      "sample_iqr_outlier_limits": [
        -1.6421692073345184,
        0.9853015244007111
      ],
      "sample_iqr_outlier_pct": 16.2268,
      "sample_abs_z_gt_3_pct": 1.3507
    }
  }
};
