# Career Evidence Pack

This folder turns the CompressAI research platform into a structured evidence package for three related goals:

1. PhD applications and supervisor outreach
2. Innovator Founder visa / startup validation
3. Global Talent visa evidence planning

The public files in this folder are safe to share with supervisors, incubators, and collaborators. The private invention disclosure files are intentionally excluded from Git by `.gitignore`.

## Folder Map

| Folder | Purpose | Shareability |
|---|---|---|
| `phd/` | Research summary, technical evidence, PhD positioning | Share with supervisors |
| `visa/` | Innovator Founder and Global Talent evidence framing | Share selectively |
| `outreach/` | Email templates for supervisors and startup contacts | Share/edit freely |
| `patent/` | Public-safe IP strategy notes | Share cautiously |
| `private/` | Invention disclosure and prior-art notes | Do not publish before legal review |

## Current Evidence Base

The strongest validated result currently is the 500-patch Sentinel-2 checkpoint comparison:

- Original VQ-VAE: SUS 80.470, detector retention 0.8642, PSNR 21.305, SSIM 0.8170
- Regularized mixed-domain VQ-VAE: SUS 81.159, detector retention 0.8403, PSNR 22.631, SSIM 0.8354

The conclusion is academically strong because it is honest: additional satellite-domain fine-tuning improves visual reconstruction, but the original checkpoint remains safer for detector retention and headline mission-utility experiments.

## Important IP Warning

Do not publish the private invention disclosure before speaking to a patent attorney. UK guidance states that an invention must be new and not publicly available before filing, and that disclosure before application can harm patentability. Use NDAs when discussing technical details with commercial partners.

Useful official references:

- UK patent overview: https://www.gov.uk/patent-your-invention
- Before applying for a patent: https://www.gov.uk/patent-your-invention/before-you-apply
- NDA guidance: https://www.gov.uk/government/publications/non-disclosure-agreements
- Innovator Founder eligibility: https://www.gov.uk/innovator-founder-visa/eligibility
- Global Talent digital technology eligibility: https://www.gov.uk/global-talent-digital-technology/eligibility
