# Requirements and evidence

The original PDF has 20 FR rows (not 18). In that PDF the counts were 8
implemented, 2 partial, 10 planned. These counts describe the old presentation,
not this branch. Implementation and real-footage acceptance are separate below.

| ID | Implemented capability in this branch | Evidence / remaining acceptance work |
|---|---|---|
| FR1 | Decode frames, paced playback, q exit | Video/GUI checks pass; overloaded processing can be slower than source FPS |
| FR2 | Skip black start frames | Tested with black-prefix encoded video |
| FR3 | Click/save/reload polygons, explicit resolution scaling | Reload/scaling/validation tested; interactive clicking needs desktop review |
| FR4 | Draw overlay after analysis | Pipeline order prevents overlay feedback into motion detection |
| FR5 | MOG2 with explicit shadow-label removal | Regression test includes shadow value 127 with threshold 50 |
| FR6 | Separate per-zone area/variance/kernel/blur | Implemented and runtime-adjustable; wave robustness unmeasured |
| FR7 | Contours and model boxes, duplicate suppression | Cannot guarantee one box for each physical object in every scene |
| FR8 | Dock motion/person tracks orange; water motion/boat tracks green | Applied in rendering; motion-only boxes do not imply semantic class |
| FR9 | Active model tracks reclassified while stationary | Integration test; real occlusion/light behaviour unmeasured |
| FR10 | One-to-one track IDs and once-per-ID boat count | Tested; global identity after leaving or occlusion is not guaranteed |
| FR11 | Reviewed annotation UI, grouped manifest, YOLO dataset builder | Tools tested; independent reviewed real fishing sessions still needed |
| FR12 | Local pretrained-model fine-tuning and per-class held-out metrics | Actual synthetic training executes; useful own-footage fishing weights missing |
| FR13 | Motion and active-track crops gate model work | Empty gate/stationary re-inference tests; stationary objects at startup can be missed |
| FR14 | person/boat/custom fishing model class adapter | General inference verified; learned fishing accuracy unverified |
| FR15 | Time-confirmed episodes; negative/gap reset | Deterministic event tests; rod candidates deliberately separate from fishing |
| FR16 | CSV source, video seconds, ID, class, event type and duration | End-to-end CSV tests and controlled evaluator pass |
| FR17 | Exactly one snapshot per confirmed episode/crossing | Tested; no snapshot for ordinary negative observations |
| FR18 | Configurable directional counting line and once-per-track crossing | Synthetic crossing/direction tests; manual real-clip count comparison pending |
| FR19 | Playback/pause, parameter trackbars, event list and exports | GUI rendering tested under Xvfb; operator usability not benchmarked |
| FR20 | Reviewed-frame/event evaluation, count error, empty-dock hourly rates | Metric tests/CLI check; real labelled benchmark and empty-hour measurement pending |

Optional calibrated dock-plane positions are implemented and tested with known
correspondences. Real measured points have not been supplied. Body-height
measurement, Kalman prediction and reliable automatic identity/permission
decisions are not claimed.
