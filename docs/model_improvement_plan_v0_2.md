# Model Improvement Plan for Release 0.2

## Evidence-Driven Diagnosis

The adapted VQ-VAE improved substantially over the base checkpoint, so further training can change wildfire-relevant retention. It still trailed JPEG2000 by 11.85 SUS points and 0.064 dNBR-proxy Dice at the frozen 1,200-byte ceiling. Its predicted-positive area on negative images also rose by 0.284 relative to the base model.

This points to two immediate problems:

1. the decoder does not preserve enough wildfire-relevant structure at severe token loss;
2. training rewards positive-region consistency without sufficiently controlling detector activation on negative scenes.

## Next Candidate Within the Existing Architecture

Train one new VQ-VAE candidate using a **fresh, event-disjoint training/development source**. Keep the existing encoder, decoder, codebook size, token format and utility selector. Change only the training objective and checkpoint-selection rule:

- train with the same wire reconstruction used at evaluation time;
- sample multiple payload budgets during training instead of one retention level;
- include positive and negative images in every batch;
- retain the current reconstruction and semantic-consistency losses;
- add a specificity penalty for predicted wildfire area on labelled negative images;
- select checkpoints on a joint development score requiring SUS, independent-mask Dice and negative-area tolerance;
- count the same serialised mask, token and header bytes used by the benchmark.

This is a bounded improvement to the current model, not a new compression architecture.

## Predeclared Advancement Rule

Before scoring a new development cohort, freeze thresholds requiring all of the following:

1. event-paired SUS advantage over the current adapted checkpoint;
2. event-paired independent-mask Dice advantage;
3. no material increase in negative predicted-positive area;
4. compliance with the same actual payload tolerance;
5. no overlap in event, source scene or footprint with training data.

Only a candidate that passes every development criterion should proceed to a new external test. The existing 37-pair sealed cohort must remain closed because the prior gate failed.

## Improvements That Would Not Strengthen the Evidence

- training longer on the same 600 events without a fresh development source;
- selecting a checkpoint using the scored 18-event or 60-event cohorts;
- changing the SUS weights after observing results;
- opening the sealed cohort to search for a positive result;
- reporting token count without complete serialised payload bytes;
- increasing model size without measuring edge latency and memory.

## Expected Research Value

A positive outcome would show that specificity-aware, rate-conditioned training improves the current VQ-VAE without changing the system architecture. A negative outcome would identify a reproducible limitation of discrete semantic tokens relative to JPEG2000 at severe rates. Either result is suitable for the proposed PhD when evaluated on independent labels and reported transparently.
