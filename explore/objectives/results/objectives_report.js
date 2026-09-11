window.OBJECTIVES_REPORT = {
  "dataset": "C:\\Users\\santb\\OneDrive\\Escritorio\\Challange\\wnn_connectome_starterpack\\datasets\\train.parquet",
  "sequences_analyzed": 200,
  "sequences_available": 10607,
  "selected_rows": 3980200,
  "selection": {
    "include_warmup": false,
    "scored_only": false
  },
  "notes": {
    "exact": "moments, ranges, threshold rates and correlations",
    "approximate": "percentiles and outlier rates use a stratified sample",
    "correlation_type": "ordinary unweighted Pearson (not competition WP)"
  },
  "corr_t0_t1": -0.7594418329575139,
  "targets": {
    "t0": {
      "count": 3980200,
      "mean": -0.2955370229980803,
      "std_population": 0.9845914047044267,
      "min": -9.170530319213867,
      "max": 8.654886245727539,
      "near_zero_pct": {
        "0.01": 22.591553188281996,
        "0.05": 22.591553188281996,
        "0.1": 22.591553188281996
      },
      "outside_minus2_plus2_pct": 2.9500778855333905,
      "autocorrelation": {
        "1": 0.9761968603356725,
        "5": 0.9245893635999629,
        "10": 0.8931056046576618
      },
      "autocorrelation_pair_counts": {
        "1": 3980000,
        "5": 3979200,
        "10": 3978200
      },
      "sample_size": 1000000,
      "percentiles_approx": {
        "0": -9.170530319213867,
        "0.1": -2.6274707317352295,
        "1": -1.9706029891967773,
        "5": -1.9706029891967773,
        "25": -0.6568676829338074,
        "50": -0.4685360789299011,
        "75": 0.0,
        "95": 1.4056081771850586,
        "99": 2.811216354370117,
        "99.9": 5.309254169464111,
        "100": 8.654886245727539
      },
      "sample_iqr": 0.6568676829338074,
      "sample_iqr_outlier_limits": [
        -1.6421692073345184,
        0.9853015244007111
      ],
      "sample_iqr_outlier_pct": 16.2148,
      "sample_abs_z_gt_3_pct": 1.2590000000000001
    },
    "t1": {
      "count": 3980200,
      "mean": -0.27927562875867584,
      "std_population": 0.9739406096772006,
      "min": -6.090969085693359,
      "max": 9.370721817016602,
      "near_zero_pct": {
        "0.01": 23.28641776795136,
        "0.05": 23.28641776795136,
        "0.1": 23.28641776795136
      },
      "outside_minus2_plus2_pct": 2.690166323300337,
      "autocorrelation": {
        "1": 0.9749728554516346,
        "5": 0.9223121082398408,
        "10": 0.8903775154619871
      },
      "autocorrelation_pair_counts": {
        "1": 3980000,
        "5": 3979200,
        "10": 3978200
      },
      "sample_size": 1000000,
      "percentiles_approx": {
        "0": -6.090969085693359,
        "0.1": -2.6546270847320557,
        "1": -1.9706029891967773,
        "5": -1.9706029891967773,
        "25": -0.6568676829338074,
        "50": -0.4685360789299011,
        "75": 0.0,
        "95": 1.4056081771850586,
        "99": 2.811216354370117,
        "99.9": 5.067924499511719,
        "100": 9.370721817016602
      },
      "sample_iqr": 0.6568676829338074,
      "sample_iqr_outlier_limits": [
        -1.6421692073345184,
        0.9853015244007111
      ],
      "sample_iqr_outlier_pct": 16.2584,
      "sample_abs_z_gt_3_pct": 1.2268
    }
  }
};
