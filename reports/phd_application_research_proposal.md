# PhD Research Proposal

## Semantic Utility-Aware Adaptive Communication for Wildfire-Centric Earth Observation Systems Under Resource Constraints

**Applicant:** Athul Jayakumar

**Research area:** Earth Observation AI, semantic communication, learned image compression, wildfire monitoring, and satellite edge intelligence

**Proposed PhD direction:** AI-powered semantic compression for resource-constrained Earth Observation and spaceborne wildfire monitoring

---

## Abstract

Earth Observation systems are producing imagery at a scale that increasingly exceeds available downlink capacity, onboard storage, and energy budgets. This challenge is especially important for wildfire monitoring, where the operational value of an image depends not only on visual quality but on whether fire, smoke, burn scars, and affected terrain remain detectable after transmission. Conventional codecs such as JPEG, JPEG2000, and operational space compression standards are efficient and reliable, but they are designed primarily around rate-distortion performance. They do not explicitly optimise for mission-relevant information preservation.

This PhD proposes a semantic utility-aware adaptive communication framework for wildfire-centric Earth Observation. The central hypothesis is that semantic utility-aware token prioritisation can preserve wildfire-relevant information more effectively than non-semantic token selection under severe bandwidth constraints. The research investigates compression as a mission-aware communication problem: instead of transmitting all image regions uniformly, a satellite or edge system should prioritise image tokens that carry wildfire-relevant evidence.

Preliminary experiments have been conducted using the CompressAI research platform on DFire, FLAME, and a 100-scene Sentinel-2/CEMS benchmark. At a 45 percent utility-aware token retention setting, the system achieved SUS = 83.14 on DFire, SUS = 65.84 on FLAME, and SUS = 79.52 on Sentinel-2. On Sentinel-2, utility-aware token transmission achieved 99.35 percent bandwidth saving and 155.77x compression ratio while preserving measurable wildfire utility. The results also show that JPEG remains a strong baseline for general reconstruction quality and detector retention. Therefore, the proposed research does not claim to replace conventional codecs. Instead, it investigates a complementary high-compression mode for extreme communication constraints, where preserving mission utility may be more important than reconstructing every pixel.

The expected contribution is a scientifically rigorous framework that connects semantic utility metrics, learned token compression, wildfire-focused Earth Observation validation, and satellite communication analysis. The project is aligned with European research priorities in onboard AI, autonomous Earth Observation, semantic communications, CubeSat intelligence, and disaster-response monitoring.

---

## 1. Introduction

Earth Observation has become a critical infrastructure for environmental monitoring, disaster response, climate adaptation, agriculture, urban planning, and security. Modern satellite systems collect increasingly high-resolution, high-frequency, and multispectral imagery. Missions such as Sentinel-2, Landsat, MODIS, VIIRS, and commercial small-satellite constellations enable wide-area monitoring of wildfire, flooding, vegetation health, burned areas, coastal change, and infrastructure damage.

However, the ability to sense the Earth is growing faster than the ability to transmit all collected information. Satellites, especially CubeSats and small platforms, operate under restricted downlink windows, limited energy budgets, finite onboard storage, and intermittent connectivity. During an emergency, a spacecraft may capture mission-critical imagery but may not be able to transmit the full image at high quality in time for operational use. This creates a central research question: when communication resources are limited, which information should be transmitted first?

Wildfire monitoring is a strong case for studying this question. During an active wildfire, the most useful information is not necessarily a visually perfect image of the whole scene. Decision-makers need to know where fire and smoke are present, whether burned areas are expanding, and whether affected regions require urgent attention. A conventional compression system may allocate bits according to visual detail across the whole image, while a mission-aware system could allocate more transmission capacity to regions that contain wildfire evidence.

This proposal therefore reframes compression as semantic communication for Earth Observation. The aim is not only to reduce file size, but to preserve mission-relevant meaning. In this project, the mission is wildfire monitoring, and the meaning of interest is wildfire-relevant visual and satellite evidence. The proposed system estimates semantic utility, ranks learned image tokens according to that utility, transmits the most important tokens first, reconstructs the image, and evaluates whether wildfire-relevant information survives.

## 2. Research Problem

Most image compression methods are evaluated using distortion, bitrate, visual fidelity, or perceptual similarity. Metrics such as PSNR, SSIM, MS-SSIM, LPIPS, and compression ratio are valuable because they allow reproducible comparison between methods. However, they do not directly answer whether compressed Earth Observation imagery remains useful for a specific mission. A reconstructed image may have lower global fidelity but still preserve the fire or smoke evidence needed for rapid interpretation. Conversely, a visually strong reconstruction may spend bandwidth on regions that are irrelevant to wildfire monitoring.

The research problem is the mismatch between conventional compression objectives and mission utility. Wildfire-centric Earth Observation requires evaluation methods and transmission strategies that reflect the value of semantic evidence. This requires a framework that can identify important regions, connect them to a learned compressed representation, transmit high-utility information under constrained bandwidth, and evaluate the reconstruction using both conventional quality metrics and wildfire-specific utility metrics.

No existing framework has systematically evaluated utility-aware semantic compression for wildfire-centric Earth Observation using learned token representations, mission-oriented utility metrics, and satellite communication constraints within a unified experimental framework. This gap motivates the proposed PhD.

## 3. Research Aim and Objectives

The aim of this PhD is to develop and evaluate a semantic utility-aware adaptive communication framework for wildfire-centric Earth Observation under resource constraints.

The specific objectives are:

1. To define and validate a mission-oriented Semantic Utility Score for wildfire-relevant information preservation.
2. To develop utility-aware token prioritisation for learned image compression using VQ-VAE-style representations.
3. To evaluate whether utility-aware token selection preserves wildfire information better than random or entropy-based token selection.
4. To compare semantic token transmission against JPEG and full learned reconstruction baselines.
5. To validate the framework across wildfire image datasets and Sentinel-2 Earth Observation imagery.
6. To analyse the communication trade-offs between semantic utility, compression ratio, and bandwidth saved.
7. To build a reproducible experimental pipeline suitable for PhD research, preprint development, and IEEE-style publication.

## 4. Research Questions

**RQ1.** How can wildfire-relevant semantic utility be measured after image compression and reconstruction?

**RQ2.** Does utility-aware token prioritisation preserve wildfire-relevant information better than random or entropy-based token selection under equivalent token budgets?

**RQ3.** How does utility-aware learned token compression compare with JPEG and full VQ-VAE reconstruction in terms of SUS, detector retention, PSNR, SSIM, LPIPS, compression ratio, and bandwidth saved?

**RQ4.** How well do wildfire image dataset results generalise to Sentinel-2 Earth Observation imagery?

**RQ5.** What operating points provide the best trade-off between mission utility preservation and satellite communication efficiency?

## 5. Research Hypothesis

The main hypothesis is:

> Semantic utility-aware token prioritisation preserves wildfire-relevant Earth Observation information more effectively than non-semantic token selection under severe bandwidth constraints.

This hypothesis is deliberately focused. It does not claim that utility-aware token compression outperforms JPEG in all general-purpose image reconstruction tasks. Instead, it investigates whether mission utility can be preserved more efficiently when only a limited semantic token budget can be transmitted.

## 6. Literature Context

Traditional image compression methods such as JPEG and JPEG2000 remain highly effective and widely deployed. They are fast, standardised, and reliable, which makes them important baselines for any compression research. In space systems, CCSDS image compression standards are also important because operational missions require robustness and interoperability. However, these methods primarily optimise rate-distortion performance rather than mission-specific semantic preservation.

Learned image compression has created new opportunities for adaptive representation learning. Variational autoencoders, hyperprior models, recurrent compression networks, and attention-based learned codecs have demonstrated strong performance in rate-distortion optimisation. VQ-VAE models are particularly relevant to this proposal because they represent images as discrete latent tokens. These tokens make it possible to ask which parts of a learned representation should be transmitted first.

Semantic communication provides the conceptual foundation for this work. Instead of treating communication as perfect bit recovery, semantic communication focuses on transmitting information that is useful for a task. This is highly relevant to Earth Observation, where the value of data often depends on downstream interpretation: detecting wildfire, mapping burned areas, monitoring smoke, identifying floods, or supporting disaster response.

Remote sensing research has produced strong methods for wildfire detection, active-fire monitoring, burned-area mapping, and multispectral analysis. NASA FIRMS, MODIS, VIIRS, Sentinel-2, and Landsat have all contributed to wildfire observation. However, wildfire monitoring research rarely connects semantic utility with learned compression and satellite downlink constraints. This proposal addresses that missing connection.

## 7. Proposed Methodology

The proposed system follows a semantic utility-aware compression pipeline.

First, an input wildfire or Sentinel-2 image is processed by a wildfire utility detector. The detector estimates the relevance of image regions based on fire, smoke, burn scar, and affected-terrain evidence. The output is a utility map in which high values indicate mission-important regions.

Second, the image is encoded using a VQ-VAE-style learned representation. The encoder converts the image into discrete latent tokens. Each token corresponds to part of the image representation and can be ranked, retained, pruned, or transmitted.

Third, token importance is computed by combining token location and semantic utility. Tokens corresponding to high-utility regions are prioritised. Tokens corresponding to low-utility regions are more aggressively removed when the bandwidth budget is constrained.

Fourth, the retained tokens are transmitted or simulated as transmitted. The image is reconstructed from the retained token set.

Fifth, the reconstruction is evaluated using both conventional and mission-oriented metrics. Conventional metrics include PSNR, SSIM, and LPIPS. Communication metrics include compression ratio and bandwidth saved. Mission-oriented metrics include detector retention and Semantic Utility Score.

The Semantic Utility Score is the central mission metric. It combines detector retention, object retention, relevance retention, and important-region preservation into a 0 to 100 score. This allows the system to measure whether wildfire-relevant information survives compression, rather than only measuring whether the reconstructed pixels match the original image.

## 8. Preliminary Experiments

A working experimental platform has already been implemented and tested. The current validation uses three datasets: DFire, FLAME, and Sentinel-2/CEMS.

The DFire experiment used 50 wildfire images. At the 45 percent utility-aware token retention setting, the system achieved SUS = 83.14, detector retention = 0.849, bandwidth saved = 95.10 percent, and compression ratio = 25.30x. This result shows that the method can preserve strong wildfire utility while substantially reducing transmitted information.

The FLAME experiment used 18 wildfire images. The same utility-aware operating point achieved SUS = 65.84, detector retention = 0.834, bandwidth saved = 96.27 percent, and compression ratio = 55.51x. The lower SUS indicates that performance varies across datasets and that the method should be validated across broader wildfire imagery. This is an important scientific result because it shows both promise and limitation.

The Sentinel-2/CEMS benchmark used 100 fixed 512-pixel satellite image patches. At 45 percent utility-aware token retention, the system achieved SUS = 79.52, detector retention = 0.741, bandwidth saved = 99.35 percent, and compression ratio = 155.77x. This result is central to the Earth Observation contribution because it demonstrates measurable wildfire utility preservation under very high compression in satellite imagery.

## 9. Baseline Comparison and Interpretation

The Sentinel-2 benchmark compared utility-aware token selection against JPEG, full VQ-VAE reconstruction, random token selection, and entropy-based token selection.

JPEG performed very strongly. JPEG Q20 achieved SUS = 97.10, detector retention = 0.985, compression ratio = 27.86x, and bandwidth saved = 96.15 percent. JPEG Q40 achieved SUS = 98.32, detector retention = 0.994, compression ratio = 16.32x, and bandwidth saved = 93.54 percent. These results show that JPEG remains a strong baseline for reconstruction quality and detector preservation.

Full VQ-VAE reconstruction achieved SUS = 92.81, detector retention = 0.947, compression ratio = 99.08x, and bandwidth saved = 98.98 percent. This result shows that learned reconstruction can preserve substantial utility, but with lower compression ratio than utility-aware pruning.

The most important comparison is between token selection strategies. Random token selection at 45 percent achieved SUS = 70.91 and detector retention = 0.673. Entropy-based token selection achieved SUS = 74.92 and detector retention = 0.694. Utility-aware token selection achieved SUS = 79.52 and detector retention = 0.741. This indicates that utility-aware ranking preserves more wildfire-relevant information than random or entropy-only selection under the same learned-token compression regime.

Therefore, the current evidence supports a precise claim: utility-aware token prioritisation improves semantic preservation compared with non-semantic token selection, especially under severe bandwidth constraints. It does not support a broad claim that utility-aware compression is universally better than JPEG. The research contribution is a mission-aware high-compression mode, not a general-purpose codec replacement.

## 10. Statistical and Operating Point Analysis

The current statistical analysis supports the internal learned-token comparison. On Sentinel-2, utility-aware selection improved SUS over random token selection by an approximate bootstrap confidence interval of 5.17 to 12.07 SUS points. Against entropy selection, the approximate improvement was 2.67 to 6.47 SUS points. Detector retention improvements were also positive against random and entropy-based selection.

The analysis against JPEG shows a different pattern. JPEG produced higher semantic utility, detector retention, PSNR, and SSIM than the aggressive 45 percent utility-aware token setting. However, utility-aware token compression produced much higher compression ratios and slightly higher bandwidth savings. This means the strongest operating regime for the proposed method is not ordinary image reconstruction, but extreme compression and mission-prioritised transmission.

Operating point analysis suggests that higher token retention improves utility, while lower retention improves bandwidth savings. A practical operating point around 80 percent token retention appears promising for balancing utility and communication efficiency, while 45 percent retention is useful as a stress-test condition. Future work will formalise operating point selection using Pareto analysis across SUS, detector retention, compression ratio, and bandwidth saved.

## 11. Expected Contributions

The proposed PhD is expected to make five contributions.

First, it will formalise Semantic Utility Score as a wildfire-centred evaluation metric for compressed Earth Observation imagery.

Second, it will develop and evaluate utility-aware token prioritisation for learned image representations.

Third, it will provide a wildfire-centric Earth Observation benchmark using DFire, FLAME, Sentinel-2, and future FIRMS-aligned wildfire labels.

Fourth, it will connect compression evaluation to satellite communication constraints through bandwidth saving, compression ratio, and operating point analysis.

Fifth, it will produce a reproducible research platform capable of generating tables, figures, confidence intervals, statistical tests, and publication-ready reports.

## 12. Work Plan

**Year 1:** Consolidate the semantic utility metric, wildfire detector, token compression pipeline, and benchmark framework. Prepare the first paper on SUS and wildfire utility preservation.

**Year 2:** Expand validation on DFire, FLAME, and Sentinel-2. Integrate FIRMS active-fire detections and burned-area masks where available. Prepare a paper on utility-aware token prioritisation.

**Year 3:** Extend the system toward satellite communication and edge deployment analysis. Evaluate bandwidth, latency, memory, and simulated onboard constraints. Prepare a paper on satellite communication trade-offs.

**Year 4:** Complete large-scale Earth Observation validation, finalise statistical analysis, release reproducible code and reports, submit journal work, and write the thesis.

## 13. Feasibility and Risks

The project is feasible because an experimental platform already exists and has produced preliminary results. The main technical risks are dataset quality, detector bias, and baseline strength. FIRMS labels and burned-area masks may be spatially coarse or temporally imperfect. The wildfire utility detector may introduce bias into SUS. JPEG may remain stronger at many reconstruction-focused operating points.

These risks are manageable. Dataset uncertainty will be handled through transparent reporting and cross-dataset validation. Detector bias will be addressed by reporting SUS components separately and comparing with conventional metrics. JPEG strength will be treated honestly by defining the proposed method as a complementary high-compression semantic mode rather than a universal replacement.

## 14. Research Impact

The scientific impact of this work is a new framework for evaluating semantic utility preservation in compressed Earth Observation imagery. The engineering impact is a reproducible platform for semantic token compression and communication analysis. The Earth Observation impact is a wildfire-focused method for studying how mission-relevant information can be prioritised under satellite resource constraints.

The longer-term vision is aligned with onboard AI, autonomous Earth Observation systems, semantic communications for space missions, CubeSat intelligence, and ESA Phi-lab-style innovation. Future satellites may not transmit all observations uniformly. They may interpret scenes onboard, rank information by mission value, and transmit the most important semantic evidence first.

## 15. Conclusion

This proposal presents a focused PhD research programme on semantic utility-aware adaptive communication for wildfire-centric Earth Observation. The current experiments show that utility-aware token selection preserves more wildfire-relevant information than random or entropy-only token selection while achieving very high compression and bandwidth savings. The 100-scene Sentinel-2 benchmark demonstrates that the approach is relevant to satellite imagery and not only to generic wildfire photographs.

The proposal is scientifically balanced. JPEG remains a strong baseline, and the research does not claim to replace it. Instead, the contribution is to investigate mission-aware semantic token transmission under extreme communication constraints. This makes the project suitable for PhD research in Earth Observation AI, remote sensing, satellite communication, and spaceborne edge intelligence.

## References

Balle, J., Laparra, V., and Simoncelli, E. P. (2017). End-to-end optimized image compression. International Conference on Learning Representations.

Balle, J., Minnen, D., Singh, S., Hwang, S. J., and Johnston, N. (2018). Variational image compression with a scale hyperprior. International Conference on Learning Representations.

CCSDS. (2017). Image data compression. CCSDS 122.0-B-2. Consultative Committee for Space Data Systems.

Christopoulos, C., Skodras, A., and Ebrahimi, T. (2000). The JPEG2000 still image coding system: An overview. IEEE Transactions on Consumer Electronics, 46(4), 1103-1127.

Drusch, M., Del Bello, U., Carlier, S., Colin, O., Fernandez, V., Gascon, F., Hoersch, B., Isola, C., Laberinti, P., Martimort, P., Meygret, A., Spoto, F., Sy, O., Marchese, F., and Bargellini, P. (2012). Sentinel-2: ESA's optical high-resolution mission for GMES operational services. Remote Sensing of Environment, 120, 25-36.

Giglio, L., Schroeder, W., and Justice, C. O. (2016). The collection 6 MODIS active fire detection algorithm and fire products. Remote Sensing of Environment, 178, 31-41.

Kingma, D. P., and Welling, M. (2014). Auto-encoding variational Bayes. International Conference on Learning Representations.

Li, X., Lan, X., Liu, J., and Yuen, C. (2022). Semantic communications: Principles and challenges. IEEE Network, 36(5), 206-213.

Minnen, D., Balle, J., and Toderici, G. D. (2018). Joint autoregressive and hierarchical priors for learned image compression. Advances in Neural Information Processing Systems.

NASA FIRMS. (2023). Fire Information for Resource Management System: Active fire data. NASA.

Schroeder, W., Oliva, P., Giglio, L., and Csiszar, I. A. (2014). The new VIIRS 375 m active fire detection data product. Remote Sensing of Environment, 143, 85-96.

Shannon, C. E. (1948). A mathematical theory of communication. Bell System Technical Journal, 27(3), 379-423.

Van Den Oord, A., Vinyals, O., and Kavukcuoglu, K. (2017). Neural discrete representation learning. Advances in Neural Information Processing Systems.

Wallace, G. K. (1992). The JPEG still picture compression standard. IEEE Transactions on Consumer Electronics, 38(1), xviii-xxxiv.

Wang, Z., Bovik, A. C., Sheikh, H. R., and Simoncelli, E. P. (2004). Image quality assessment: From error visibility to structural similarity. IEEE Transactions on Image Processing, 13(4), 600-612.

Zhu, X. X., Tuia, D., Mou, L., Xia, G. S., Zhang, L., Xu, F., and Fraundorfer, F. (2017). Deep learning in remote sensing: A comprehensive review and list of resources. IEEE Geoscience and Remote Sensing Magazine, 5(4), 8-36.
