# CompressAI Experimental Results Report

## Semantic Utility-Aware Compression for Wildfire-Centric Earth Observation

### 1. Research Aim

This report summarises the current experimental validation of the CompressAI research platform. The purpose of the experiments is to test whether semantic utility-aware token prioritisation can preserve wildfire-relevant Earth Observation information under severe communication constraints. The central idea is that a satellite or edge imaging system should not always transmit image information uniformly. During a wildfire event, the most valuable information is not necessarily the visually best reconstruction of the whole image, but the preservation of regions containing fire, smoke, burn scars, affected terrain, and other mission-relevant evidence.

The experiments therefore evaluate compression from two perspectives. The first is conventional image quality, measured using PSNR, SSIM, and LPIPS. The second is mission utility, measured using the Semantic Utility Score (SUS), detector retention, bandwidth saved, and compression ratio. This distinction is important because a method can produce lower global image fidelity while still preserving the information most relevant to wildfire monitoring. The research does not claim that utility-aware compression replaces JPEG or operational codecs in all conditions. Instead, it investigates whether learned semantic token transmission is useful under extreme bandwidth constraints, where communication efficiency is a primary mission requirement.

### 2. Experimental Pipeline

The CompressAI platform follows a semantic communication pipeline. An input image is first processed by a wildfire utility detector. This detector estimates fire, smoke, burn scar, and wildfire relevance signals and produces a semantic utility map. The image is then encoded using a VQ-VAE-style learned representation, which converts the image into discrete latent tokens. These tokens are ranked according to their relationship with wildfire utility. High-utility tokens are retained with priority, while lower-utility tokens are removed more aggressively when the bandwidth budget is limited.

After token selection, the retained tokens are transmitted or simulated as transmitted. The system reconstructs the image from the retained token set and evaluates the result. This creates a controlled way to study token retention rates and communication trade-offs. The benchmark includes retention levels from low token budgets to full reconstruction, although the main reported operating point for cross-dataset comparison is the 45 percent utility-aware token retention setting.

The experimental framework compares multiple methods. The baselines include JPEG at quality levels Q20, Q40, Q60, and Q80; full VQ-VAE reconstruction; VQ-VAE with random token selection; VQ-VAE with entropy-based token selection; and VQ-VAE with utility-aware token selection. This design allows the research to separate three questions: whether learned token compression works, whether token selection strategy matters, and whether semantic utility improves over random or entropy-only token retention.

### 3. Datasets

The current experiments use three validation sources. The first is DFire, a wildfire image dataset containing visible fire and smoke examples. The current benchmark uses 50 DFire images. This dataset is useful for testing whether the semantic utility detector and token ranking system can preserve clear wildfire cues.

The second dataset is FLAME, represented in the current validation by 18 wildfire images. FLAME provides an additional wildfire image source and helps test whether the method generalises beyond a single fire/smoke dataset. Because the sample size is smaller, FLAME results are treated as preliminary and interpreted with more caution.

The third dataset is a Sentinel-2/CEMS Earth Observation benchmark containing 100 fixed 512-pixel image patches. This is the most important validation set for the Earth Observation claim because it moves the work from generic wildfire imagery into satellite imagery. The Sentinel-2 benchmark allows the project to test whether semantic utility-aware token prioritisation can operate on satellite-style scenes and produce measurable communication savings.

### 4. Metrics

The evaluation uses both image-quality and mission-utility metrics. PSNR measures pixel-level reconstruction error, where higher values indicate closer reconstruction. SSIM measures structural similarity, again with higher values indicating stronger similarity. LPIPS estimates perceptual difference using learned visual features, where lower values are generally better.

The mission-oriented metrics are more central to this research. Detector retention measures the ratio of wildfire detector confidence after reconstruction to the detector confidence before compression. A value near 1 means that the reconstructed image retains most of the detector-visible wildfire evidence. Compression ratio measures how much smaller the transmitted representation is compared with the original representation. Bandwidth saved reports the percentage reduction in transmitted data.

The main proposed metric is the Semantic Utility Score, or SUS. SUS combines detector retention, object retention, relevance retention, and important-region preservation into a 0 to 100 score. It is designed to answer a mission-specific question: how much wildfire-relevant information survives compression and reconstruction? SUS is not intended to replace PSNR or SSIM. Instead, it complements conventional metrics by measuring whether the reconstructed image remains useful for wildfire interpretation.

### 5. Cross-Dataset Results

At the 45 percent utility-aware token retention setting, the system produced strong communication savings across all three datasets. On DFire, the mean SUS was 83.14, detector retention was 0.849, bandwidth saved was 95.10 percent, and compression ratio was 25.30x. This indicates that the method preserved a high level of wildfire utility while reducing the transmitted representation substantially.

On FLAME, the mean SUS was lower at 65.84, while detector retention remained strong at 0.834. Bandwidth saved was 96.27 percent, with a compression ratio of 55.51x. The lower SUS suggests that the method is sensitive to dataset characteristics, image composition, and utility-map quality. This result is valuable because it shows that the method should not be judged from one dataset alone. FLAME demonstrates both promise and the need for broader wildfire validation.

On Sentinel-2, the utility-aware method achieved SUS = 79.52, detector retention = 0.741, bandwidth saved = 99.35 percent, and compression ratio = 155.77x. This is the strongest result for the Earth Observation objective. Although detector retention is lower than in DFire and FLAME, the communication savings are much higher. The result suggests that utility-aware token prioritisation can preserve measurable wildfire relevance even when the transmitted representation is extremely reduced.

The cross-dataset pattern is therefore clear. DFire shows strong wildfire utility preservation. FLAME shows that performance can vary across wildfire imagery. Sentinel-2 shows that the approach is relevant to satellite communication, especially when bandwidth savings are prioritised.

### 6. Sentinel-2 Baseline Comparison

The Sentinel-2 benchmark provides the most complete comparison between methods. JPEG baselines performed very strongly in semantic utility and detector retention. JPEG Q20 achieved SUS = 97.10, detector retention = 0.985, compression ratio = 27.86x, and bandwidth saved = 96.15 percent. JPEG Q40 achieved SUS = 98.32, detector retention = 0.994, compression ratio = 16.32x, and bandwidth saved = 93.54 percent. JPEG Q60 and Q80 achieved even higher detector retention but lower compression ratios.

These results show that JPEG remains a powerful baseline. It preserves visual and detector information very well, especially when the compression ratio is moderate. Therefore, the research should not claim that utility-aware token compression outperforms JPEG in general-purpose reconstruction. The stronger and more defensible claim is that utility-aware token compression operates in a different regime: extreme compression and mission-aware prioritisation.

Compared with full VQ-VAE reconstruction, the utility-aware 45 percent method had lower SUS and detector retention, but higher compression. Full VQ-VAE achieved SUS = 92.81, detector retention = 0.947, compression ratio = 99.08x, and bandwidth saved = 98.98 percent. Utility-aware 45 percent token retention achieved SUS = 79.52 and detector retention = 0.741, but increased compression ratio to 155.77x and bandwidth saved to 99.35 percent. This reveals a direct trade-off: full reconstruction preserves more utility, while token pruning improves communication efficiency.

The most important comparison is between token selection strategies. Random token selection at 45 percent achieved SUS = 70.91 and detector retention = 0.673. Entropy-based token selection achieved SUS = 74.92 and detector retention = 0.694. Utility-aware token selection achieved SUS = 79.52 and detector retention = 0.741. This shows that semantic utility-aware ranking preserves more wildfire-relevant information than random or entropy-only selection under the same learned-token compression regime.

### 7. Statistical Interpretation

The statistical analysis supports the main internal comparison. Utility-aware token selection improved semantic utility over random token selection and entropy-based token selection. For Sentinel-2, the bootstrap confidence interval for the utility-aware improvement over random selection in SUS was positive, with an approximate range from 5.17 to 12.07 SUS points. Against entropy selection, the SUS improvement was also positive, with an approximate interval from 2.67 to 6.47 points. Detector retention improvements were also positive against both random and entropy selection.

The comparison against JPEG is different. JPEG Q20 and Q40 achieved higher SUS, detector retention, PSNR, and SSIM than utility-aware 45 percent token retention. However, utility-aware token retention achieved much higher compression ratios and slightly higher bandwidth savings. For example, compared with JPEG Q20, utility-aware token retention had a much higher compression ratio, but lower semantic utility and image fidelity. This means JPEG is currently stronger when reconstruction quality and detector preservation are the main goals, while utility-aware token compression is stronger when extreme data reduction is required.

This is an important and honest result. It strengthens the research because it defines the operating regime precisely. The objective is not to beat JPEG everywhere. The objective is to investigate whether mission utility can be preserved efficiently when only a small semantic token budget is available.

### 8. Operating Point Analysis

The operating point analysis shows that the best token retention level depends on the mission objective. If the objective is maximum utility, higher retention is preferred. If the objective is extreme bandwidth reduction, lower retention becomes attractive but sacrifices reconstruction quality and detector confidence. A practical operating point identified in the current analysis is around 80 percent token retention, where Sentinel-2 results showed strong SUS and detector retention while still maintaining very high bandwidth savings.

The 45 percent retention point is useful as a stress test. It demonstrates that even under aggressive token pruning, the system retains measurable wildfire utility. However, for operational use, the ideal retention point may depend on mission mode. Emergency alerting may accept lower visual fidelity if fire evidence is retained quickly. Detailed mapping may require higher retention or full reconstruction.

### 9. Main Findings

The first finding is that semantic token prioritisation matters. Utility-aware selection consistently outperformed random and entropy-only token selection in the Sentinel-2 benchmark. This supports the core hypothesis that mission-aware ranking can preserve more relevant information than non-semantic token selection.

The second finding is that semantic utility and image fidelity are not identical. JPEG performed strongly on both fidelity and detector retention, but utility-aware token compression provided a different advantage: much higher compression ratio under learned token transmission. This distinction is central to the PhD argument.

The third finding is that Earth Observation validation is feasible. The 100-scene Sentinel-2/CEMS benchmark demonstrates that the platform can move beyond natural wildfire imagery and produce satellite-relevant evidence.

The fourth finding is that the method is promising but not complete. FLAME performance was lower than DFire, and Sentinel-2 detector retention was lower than JPEG and full VQ-VAE. This indicates that future work should expand datasets, improve utility-map validation, and integrate stronger geospatial labels such as FIRMS detections and burned-area masks.

### 10. Conclusion

The experiments completed so far provide a credible foundation for a PhD-level research programme on semantic utility-aware compression for wildfire-centric Earth Observation. The strongest evidence is that utility-aware token selection preserves more wildfire-relevant information than random or entropy-only token selection while achieving very high communication savings. The Sentinel-2 benchmark is especially important because it connects the work to real Earth Observation imagery and satellite communication constraints.

At the same time, the results show that JPEG remains a very strong baseline. This should be treated as a strength of the evaluation rather than a weakness of the project. The research contribution is not a universal replacement for conventional codecs. It is a mission-aware semantic communication framework for extreme bandwidth-constrained scenarios. The current evidence supports continued development toward larger Sentinel-2 validation, FIRMS-aligned labels, burned-area evaluation, and edge-device benchmarking.

Overall, the CompressAI experiments show that semantic utility-aware token transmission is a scientifically defensible direction for wildfire-focused Earth Observation. The work is suitable for PhD applications, supervisor outreach, preprint development, and further publication-focused validation.
