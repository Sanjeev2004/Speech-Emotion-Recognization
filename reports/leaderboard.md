# Model comparison

Speaker-independent test sets. UAR = unweighted average recall (main metric).

## Trained and tested on combined

| Rank | Model | Accuracy | UAR | Macro-F1 | Params |
|---|---|---|---|---|---|
| 1 | hubert | 0.743 | 0.746 | 0.743 | 94.57M |
| 2 | wavlm | 0.735 | 0.732 | 0.736 | 94.58M |
| 3 | wav2vec2 | 0.721 | 0.723 | 0.721 | 94.57M |
| 4 | crnn_attention | 0.647 | 0.650 | 0.646 | 0.75M |
| 5 | cnn_lstm | 0.641 | 0.644 | 0.641 | 1.34M |
| 6 | cnn2d | 0.640 | 0.643 | 0.637 | 1.17M |
| 7 | bilstm | 0.559 | 0.559 | 0.556 | 0.63M |
| 8 | svm | 0.484 | 0.482 | 0.482 | - |
| 9 | random_forest | 0.468 | 0.469 | 0.453 | - |

## Trained and tested on cremad

| Rank | Model | Accuracy | UAR | Macro-F1 | Params |
|---|---|---|---|---|---|
| 1 | hubert | 0.728 | 0.730 | 0.728 | 94.57M |
| 2 | wavlm | 0.724 | 0.723 | 0.728 | 94.58M |
| 3 | wav2vec2 | 0.718 | 0.718 | 0.719 | 94.57M |
| 4 | crnn_attention | 0.626 | 0.627 | 0.626 | 0.75M |
| 5 | cnn_lstm | 0.593 | 0.595 | 0.591 | 1.34M |
| 6 | cnn2d | 0.584 | 0.585 | 0.583 | 1.17M |
| 7 | bilstm | 0.563 | 0.566 | 0.559 | 0.63M |
| 8 | svm | 0.499 | 0.498 | 0.496 | - |
| 9 | random_forest | 0.473 | 0.475 | 0.454 | - |

## Trained and tested on ravdess

| Rank | Model | Accuracy | UAR | Macro-F1 | Params |
|---|---|---|---|---|---|
| 1 | wavlm | 0.824 | 0.823 | 0.821 | 94.58M |
| 2 | hubert | 0.801 | 0.797 | 0.796 | 94.57M |
| 3 | wav2vec2 | 0.739 | 0.750 | 0.730 | 94.57M |
| 4 | cnn_lstm | 0.693 | 0.698 | 0.687 | 1.34M |
| 5 | crnn_attention | 0.665 | 0.677 | 0.648 | 0.75M |
| 6 | bilstm | 0.602 | 0.594 | 0.584 | 0.63M |
| 7 | cnn2d | 0.562 | 0.557 | 0.531 | 1.17M |
| 8 | random_forest | 0.511 | 0.479 | 0.474 | - |
| 9 | svm | 0.472 | 0.438 | 0.424 | - |

## Cross-dataset UAR (train -> test)

| model          |   ('combined', 'cremad') |   ('combined', 'ravdess') |   ('cremad', 'combined') |   ('cremad', 'ravdess') |   ('ravdess', 'combined') |   ('ravdess', 'cremad') |
|:---------------|-------------------------:|--------------------------:|-------------------------:|------------------------:|--------------------------:|------------------------:|
| bilstm         |                    0.559 |                     0.531 |                    0.522 |                   0.219 |                     0.261 |                   0.21  |
| cnn2d          |                    0.632 |                     0.714 |                    0.552 |                   0.312 |                     0.26  |                   0.216 |
| cnn_lstm       |                    0.633 |                     0.729 |                    0.564 |                   0.333 |                     0.344 |                   0.287 |
| crnn_attention |                    0.637 |                     0.729 |                    0.597 |                   0.375 |                     0.333 |                   0.281 |
| hubert         |                    0.742 |                     0.76  |                    0.725 |                   0.703 |                     0.458 |                   0.41  |
| random_forest  |                    0.477 |                     0.391 |                    0.441 |                   0.203 |                     0.251 |                   0.215 |
| svm            |                    0.487 |                     0.422 |                    0.467 |                   0.255 |                     0.251 |                   0.22  |
| wav2vec2       |                    0.717 |                     0.75  |                    0.7   |                   0.583 |                     0.443 |                   0.401 |
| wavlm          |                    0.728 |                     0.745 |                    0.709 |                   0.625 |                     0.512 |                   0.467 |
