# bearing-telemetry

bearing fault diagnosis from vibration and acoustic telemetry, trained
from scratch on piblic data and evaluated on bearingss the model has 
nevber seen.

**Status: work in progess.**

## Data

University of Ottawa Rolling-element Dataset (UORED-VAFCLS, vs):
20 bearings, each recorded healthy, with a developing fault and with a
fault (inner race, outer race, ball or cage; 5 bearings each).
42 kHz, 10 s per recording. Licence: CC BY 4.0, see
[DATA_LICENSE.md](DATA_LICENSE.md).

## Findings from exploring the data

- Speed (1819 rpm) and load are stored once per recording, not as signals.
The temperature channel behaves like noise and is not used.
- Damage raised vibration in 19 of 20 bearings, but the developing stage
was louder than the faulty stage in 8 of 20. Vibration level shows
that something changed, not how far the damage has progeressed.
- Healthy bearings differ in vibration level by a factor of about 15, so
no fixed threshold seperates healthy from damaged across bearings.
([docs/survey.md](docs/survey.md))
- Recordings carry constant offsets of up to 178 units that depend on the
recording session. All inner-race fault recordings sit near 50 while
most healthy recordings sit near 1, so the offset alone would reveal
the class. A piezoelectric accelerometer cannot measure a constant, so
each window's mean is removed before training.
- Inner-race damage on bearing 1 produces impact bursts about 33 ms apart, once per shaft revolution at 1819 rpm.
([docs/bearing1_signals.png](docs/bearing1_signals.png))

## Method

- Windows of 4,096 samples (about 0.1 s, three shaft revolutions).
- 5 classes: healthy, inner race, outer race, ball, cage.
- Split by bearing: per fault type, 3 bearings for training, 1 for
  validation, 1 for testing. A bearing's healthy, developing and faulty
  recording always stay in the same set.

