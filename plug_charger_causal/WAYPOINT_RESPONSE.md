# PlugCharger-Causal Waypoint Response

This checks the geometry implied by the official PlugCharger motion-planning solution without invoking `mplib`. It compares phase waypoint transforms under isolated `do(dx)`, `do(dy)`, and `do(dyaw)` socket perturbations.

```json
{
  "delta_coordinates": "[delta_x_m, delta_y_m, delta_yaw_rad]",
  "h5": "maniskill_spectrum/plug_charger_causal/reset_datasets/isolated_grid/reset_states.h5",
  "response_coordinates": "body-frame log approximation [x_m, y_m, yaw_rad]",
  "rows": [
    {
      "approach_start_tcp": [
        0.0,
        3.469446951953614e-18,
        0.0
      ],
      "delta": [
        -0.02,
        0.0,
        0.0
      ],
      "generator": "dx",
      "insert_target": [
        -0.019954696430514465,
        -0.001335017915578468,
        0.0
      ],
      "pre_insert_target": [
        -0.01995469643051446,
        -0.001335017915578482,
        0.0
      ],
      "sample": "/sample_0"
    },
    {
      "approach_start_tcp": [
        0.0,
        3.469446951953614e-18,
        0.0
      ],
      "delta": [
        -0.01,
        0.0,
        0.0
      ],
      "generator": "dx",
      "insert_target": [
        -0.009977346356831271,
        -0.000667508833455982,
        0.0
      ],
      "pre_insert_target": [
        -0.009977346356831264,
        -0.0006675088334560098,
        0.0
      ],
      "sample": "/sample_1"
    },
    {
      "approach_start_tcp": [
        0.0,
        3.469446951953614e-18,
        0.0
      ],
      "delta": [
        0.0,
        0.0,
        0.0
      ],
      "generator": "dx",
      "insert_target": [
        -1.3877787807814457e-17,
        1.3877787807814457e-17,
        0.0
      ],
      "pre_insert_target": [
        -1.3877787807814457e-17,
        0.0,
        0.0
      ],
      "sample": "/sample_2"
    },
    {
      "approach_start_tcp": [
        0.0,
        3.469446951953614e-18,
        0.0
      ],
      "delta": [
        0.01,
        0.0,
        0.0
      ],
      "generator": "dx",
      "insert_target": [
        0.009977346356831257,
        0.0006675088334560236,
        0.0
      ],
      "pre_insert_target": [
        0.009977346356831243,
        0.0006675088334559959,
        0.0
      ],
      "sample": "/sample_3"
    },
    {
      "approach_start_tcp": [
        0.0,
        3.469446951953614e-18,
        0.0
      ],
      "delta": [
        0.02,
        0.0,
        0.0
      ],
      "generator": "dx",
      "insert_target": [
        0.019954700147366347,
        0.0013350181642449582,
        0.0
      ],
      "pre_insert_target": [
        0.019954700147366375,
        0.0013350181642449443,
        0.0
      ],
      "sample": "/sample_4"
    },
    {
      "approach_start_tcp": [
        0.0,
        3.469446951953614e-18,
        0.0
      ],
      "delta": [
        0.0,
        -0.02,
        0.0
      ],
      "generator": "dy",
      "insert_target": [
        -0.0013288984566523498,
        0.019945053200403184,
        0.0
      ],
      "pre_insert_target": [
        -0.0013288984566523568,
        0.019945053200403157,
        0.0
      ],
      "sample": "/sample_5"
    },
    {
      "approach_start_tcp": [
        0.0,
        3.469446951953614e-18,
        0.0
      ],
      "delta": [
        0.0,
        -0.01,
        0.0
      ],
      "generator": "dy",
      "insert_target": [
        -0.0006644489807995924,
        0.00997252288514655,
        0.0
      ],
      "pre_insert_target": [
        -0.0006644489807995924,
        0.009972522885146523,
        0.0
      ],
      "sample": "/sample_6"
    },
    {
      "approach_start_tcp": [
        0.0,
        3.469446951953614e-18,
        0.0
      ],
      "delta": [
        0.0,
        0.01,
        0.0
      ],
      "generator": "dy",
      "insert_target": [
        0.0006644489807995646,
        -0.009972522885146523,
        0.0
      ],
      "pre_insert_target": [
        0.0006644489807995785,
        -0.009972522885146537,
        0.0
      ],
      "sample": "/sample_7"
    },
    {
      "approach_start_tcp": [
        0.0,
        3.469446951953614e-18,
        0.0
      ],
      "delta": [
        0.0,
        0.02,
        0.0
      ],
      "generator": "dy",
      "insert_target": [
        0.001328898456652322,
        -0.019945053200403143,
        0.0
      ],
      "pre_insert_target": [
        0.001328898456652336,
        -0.01994505320040317,
        0.0
      ],
      "sample": "/sample_8"
    },
    {
      "approach_start_tcp": [
        0.0,
        3.469446951953614e-18,
        0.0
      ],
      "delta": [
        0.0,
        0.0,
        -0.2617993877991494
      ],
      "generator": "dyaw",
      "insert_target": [
        -0.04734994878225911,
        0.006887390325007056,
        0.2616444780034213
      ],
      "pre_insert_target": [
        -0.043532582153052765,
        -0.005586745302856674,
        0.2616444780034213
      ],
      "sample": "/sample_9"
    },
    {
      "approach_start_tcp": [
        0.0,
        3.469446951953614e-18,
        0.0
      ],
      "delta": [
        0.0,
        0.0,
        -0.1308996938995747
      ],
      "generator": "dyaw",
      "insert_target": [
        -0.02344863691228423,
        0.004994793210214357,
        0.130822785708967
      ],
      "pre_insert_target": [
        -0.02194898244343739,
        -0.001367456834212244,
        0.130822785708967
      ],
      "sample": "/sample_10"
    },
    {
      "approach_start_tcp": [
        0.0,
        3.469446951953614e-18,
        0.0
      ],
      "delta": [
        0.0,
        0.0,
        0.1308996938995747
      ],
      "generator": "dyaw",
      "insert_target": [
        0.022594746276613606,
        -0.008011529218789351,
        -0.13082850424172587
      ],
      "pre_insert_target": [
        0.021938809897610365,
        -0.0015077750569948145,
        -0.13082850424172587
      ],
      "sample": "/sample_11"
    },
    {
      "approach_start_tcp": [
        0.0,
        3.469446951953614e-18,
        0.0
      ],
      "delta": [
        0.0,
        0.0,
        0.2617993877991494
      ],
      "generator": "dyaw",
      "insert_target": [
        0.043948994765896435,
        -0.01890272192939163,
        -0.2616666823630376
      ],
      "pre_insert_target": [
        0.043492065627111234,
        -0.0058649908899569,
        -0.2616666823630376
      ],
      "sample": "/sample_12"
    }
  ],
  "summary": {
    "approach_start_tcp": {
      "diag": [
        0.0,
        0.0,
        0.0
      ],
      "matrix_rows_response_cols_delta": [
        [
          0.0,
          0.0,
          0.0
        ],
        [
          0.0,
          0.0,
          0.0
        ],
        [
          0.0,
          0.0,
          0.0
        ]
      ],
      "max_abs_response": [
        0.0,
        3.469446951953614e-18,
        0.0
      ],
      "rmse": [
        0.0,
        3.469446951953614e-18,
        0.0
      ]
    },
    "insert_target": {
      "diag": [
        0.9977348586942416,
        -0.9972525857190575,
        -0.9994474178734968
      ],
      "matrix_rows_response_cols_delta": [
        [
          0.9977348586942416,
          0.06644491788208502,
          0.1746690641313652
        ],
        [
          0.06675089826558861,
          -0.9972525857190575,
          -0.04934048737146125
        ],
        [
          0.0,
          0.0,
          -0.9994474178734968
        ]
      ],
      "max_abs_response": [
        0.04734994878225911,
        0.019945053200403184,
        0.2616666823630376
      ],
      "rmse": [
        0.0006911459380934149,
        0.0024296153055792906,
        4.559268155926991e-06
      ]
    },
    "pre_insert_target": {
      "diag": [
        0.9977348586942419,
        -0.997252585719057,
        -0.9994474178734968
      ],
      "matrix_rows_response_cols_delta": [
        [
          0.9977348586942419,
          0.06644491788208556,
          0.1664916711482729
        ],
        [
          0.06675089826558858,
          -0.997252585719057,
          -0.0005323231676291074
        ],
        [
          0.0,
          0.0,
          -0.9994474178734968
        ]
      ],
      "max_abs_response": [
        0.043532582153052765,
        0.01994505320040317,
        0.2616666823630376
      ],
      "rmse": [
        6.636910650015456e-05,
        0.002315576198448037,
        4.559268155926991e-06
      ]
    }
  }
}
```
