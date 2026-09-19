# Learn ML Epistemology with Phoebe

**Why believe a model?** Sixteen 45-minute sessions on the question underneath every accuracy
figure: what licenses the belief that a model will work on data it has not seen.

Live: https://phoebefu6.github.io/learn-ml-epistemology-with-phoebe/

Thornbury's triage model scores 0.949 on patients it never trained on. A year later the hospital
group reorganises its wards and the same model, unchanged and never retrained, scores 0.765.
The split was honest, the metric was sensible, the number reproduced. What failed was an
assumption nobody had written down.

## Two tracks

**Leader, six sessions, no code.** What a test score entitles you to; why two equally good models
are not the same model; what selecting against a test set does to it; what a benchmark licenses;
how to treat evidence you did not produce; and the question to ask before your name goes on it.

**Practitioner, ten sessions, hands on.** The assumption under the split; the shortcut; twenty
seeds through one pipeline; stress tests; selecting on a set; how big a test set needs to be;
reading a benchmark; leakage at the scale of a field; a finding that did not survive its second
data draw; and a capstone bench.

## Both benches are real

Nothing on these pages is a stored result. `assets/ep-live.js` generates patients from
relationships written out in the file, trains three-input twelve-hidden-unit networks by
stochastic gradient descent in the browser, and counts every accuracy on data the model did not
train on.

- **Pane A, underspecification.** Twenty models differing only in their random seed. Held-out
  spread 0.003 to 0.019 across draws; spread on the shifted set 0.034 to 0.051. Two to eleven
  times wider where you did not look.
- **Pane B, selection.** Pick the best of k candidates on a test set and that set stops being a
  test set. At k=25 the reported number reaches 0.9562 while the honest number on fresh data is
  0.9494.

## Running it

No build step. Any static server:

```
python3 -m http.server 8679
```

Then open http://localhost:8679/

## Credits

by Phoebe Fu. Part of [Learn with Phoebe](https://phoebefu6.github.io/learn-with-phoebe/).

Memorization and double descent live in
[learn deep learning](https://phoebefu6.github.io/learn-deep-learning-with-phoebe/), leak
mechanics in [learn model risk](https://phoebefu6.github.io/learn-model-risk-with-phoebe/), and
the metrics themselves in
[learn model evaluation](https://phoebefu6.github.io/learn-model-evaluation-with-phoebe/).
