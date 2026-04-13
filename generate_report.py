#!/usr/bin/env python3
"""Generate a PDF report of issues found in the sandbagging/sycophancy steering experiment."""

from fpdf import FPDF

class Report(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font("Helvetica", "I", 9)
            self.set_text_color(120, 120, 120)
            self.cell(0, 8, "Issues Report -- Steering Persona Vectors Experiment", align="R", new_x="LMARGIN", new_y="NEXT")
            self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
            self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 9)
        self.set_text_color(120, 120, 120)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def section_title(self, title):
        self.set_font("Helvetica", "B", 15)
        self.set_text_color(0, 0, 0)
        self.ln(4)
        self.multi_cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(4)

    def issue_heading(self, number, title):
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(30, 30, 30)
        self.ln(3)
        self.multi_cell(0, 7, f"{number}. {title}", new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def sub_heading(self, text):
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(50, 50, 50)
        self.multi_cell(0, 6, text, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def body(self, text):
        self.set_font("Helvetica", "", 10.5)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 5.5, text, new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def code(self, text):
        self.set_font("Courier", "", 9)
        self.set_fill_color(240, 240, 240)
        self.set_text_color(40, 40, 40)
        x = self.get_x()
        self.set_x(x + 4)
        self.multi_cell(self.w - self.l_margin - self.r_margin - 8, 5, text,
                        fill=True, new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def bullet(self, text):
        self.set_font("Helvetica", "", 10.5)
        self.set_text_color(30, 30, 30)
        x = self.get_x()
        self.set_x(x + 6)
        self.cell(5, 5.5, "-")
        self.multi_cell(self.w - self.l_margin - self.r_margin - 11, 5.5, text,
                        new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def severity_badge(self, level):
        colors = {
            "CRITICAL": (180, 30, 30),
            "HIGH": (200, 100, 0),
            "MEDIUM": (180, 160, 0),
            "LOW": (80, 130, 80),
        }
        r, g, b = colors.get(level, (100, 100, 100))
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(r, g, b)
        self.cell(0, 5, f"Severity: {level}", new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(30, 30, 30)
        self.ln(1)

    def separator(self):
        self.ln(3)
        y = self.get_y()
        mid = self.w / 2
        self.set_draw_color(180, 180, 180)
        self.line(mid - 40, y, mid + 40, y)
        self.ln(5)


def build_report():
    pdf = Report()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    # ========== TITLE PAGE ==========
    pdf.ln(50)
    pdf.set_font("Helvetica", "B", 24)
    pdf.set_text_color(0, 0, 0)
    pdf.multi_cell(0, 12, "Issues Report", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    pdf.set_font("Helvetica", "", 14)
    pdf.set_text_color(80, 80, 80)
    pdf.multi_cell(0, 8, "Steering Persona Vectors to\nReduce Sycophancy in Gemma 2 27B",
                   align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(20)
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(60, 60, 60)
    pdf.multi_cell(0, 6,
        "This report identifies statistical, methodological, and code-level issues\n"
        "found in the experiment codebase, with emphasis on the correlation\n"
        "coefficient computations and statistical testing framework.\n\n"
        "Issues are organized into three sections:\n"
        "  I.   Statistical & Correlation Issues (the core analytical problems)\n"
        "  II.  Methodological & Experimental Design Issues\n"
        "  III. Code-Level & Reproducibility Issues\n\n"
        "Each issue includes: what is wrong, where in the code it occurs,\n"
        "why it matters, and a concrete fix.",
        align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(20)
    pdf.set_font("Helvetica", "I", 10)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 6, "Generated: 2026-04-13", align="C", new_x="LMARGIN", new_y="NEXT")

    # ========== SECTION I: STATISTICAL & CORRELATION ISSUES ==========
    pdf.add_page()
    pdf.section_title("I. Statistical & Correlation Issues")

    pdf.body(
        "These issues concern the core statistical analyses in 03_analysis.py and the "
        "results reported in statistical_tests.json and summary.txt. The correlation "
        "coefficient problems are the most consequential: they produce results that look "
        "impressive on paper but carry almost no real statistical information."
    )

    # --- Issue 1 ---
    pdf.issue_heading(1,
        "Spearman Dose-Response Correlation Computed on Only 6 Aggregate Points, "
        "Yielding a Mechanically Perfect rho = -1.0")
    pdf.severity_badge("CRITICAL")

    pdf.sub_heading("What is wrong")
    pdf.body(
        "In 03_analysis.py (lines 458-471), the Spearman dose-response correlation is computed "
        "between 6 positive coefficient values [200, 500, 1000, 2000, 3000, 5000] and their "
        "corresponding 6 mean_syc_logit values, per condition. With only 6 data points that "
        "happen to be monotonically decreasing (each is an average over 600 rows), you "
        "mechanically get rho = -1.000 for every non-random condition."
    )

    pdf.sub_heading("Where in the code")
    pdf.code(
        "# 03_analysis.py, lines 462-467\n"
        "pos_coeffs = [c for c in COEFFS if c > 0]  # only 6 values\n"
        "for cond in CONDITIONS:\n"
        "    logits_pos = [rates[cond][f\"{c}\"][\"mean_syc_logit\"]\n"
        "                  for c in pos_coeffs]\n"
        "    rho_p, pval_p = stats.spearmanr(pos_coeffs, logits_pos)"
    )

    pdf.sub_heading("Why it matters")
    pdf.body(
        "A Spearman rho of -1.0 tells you only that 6 averages are monotonically ordered. It "
        "does not test the strength of the dose-response relationship, does not account for "
        "within-coefficient variance, and grossly inflates statistical significance. The "
        "summary.txt reports this as evidence of a \"dose-dependent\" effect (Finding 4), "
        "but it is a tautology: any smooth perturbation of a neural network with increasing "
        "magnitude will produce monotonically increasing disruption when averaged over 600 "
        "samples. The random control also has rho = -0.77 by the same artifact. This "
        "correlation is not computed over all datasets -- it uses only the 6 discrete "
        "aggregate points rather than the underlying row-level data."
    )

    pdf.sub_heading("Fix")
    pdf.bullet(
        "Option A (best): Use row-level data. Compute a partial Spearman correlation between "
        "coefficient (as a continuous predictor) and per-row syc_logit across all 3,600 "
        "rows per condition (6 coefficients x 600 rows), with base_id as a clustering "
        "variable (clustered permutation test or mixed-effects model)."
    )
    pdf.bullet(
        "Option B: Fit a linear mixed-effects model: syc_logit ~ coefficient + (1|base_id), "
        "per condition. The t-statistic on the coefficient slope is a proper dose-response "
        "test that accounts for within-question correlation."
    )
    pdf.bullet(
        "Option C (minimum): If keeping the aggregate approach, compute the correlation "
        "across all conditions simultaneously (pool 6 coefficients x 7 conditions = 42 "
        "points), and acknowledge the n=6-per-condition limitation explicitly."
    )

    pdf.separator()

    # --- Issue 2 ---
    pdf.issue_heading(2,
        "Reported p-Values of Exactly 0.0 (Numerical Underflow)")
    pdf.severity_badge("HIGH")

    pdf.sub_heading("What is wrong")
    pdf.body(
        "The statistical_tests.json file reports p_value = 0.0 for all non-random "
        "dose-response correlations and p_value = 0.0 for many McNemar tests. A p-value of "
        "exactly 0.0 is mathematically impossible -- it represents floating-point underflow "
        "in float64. scipy.stats.spearmanr with n=6 and rho=-1.0 produces a value below "
        "float64 minimum representable positive number, which rounds to 0.0."
    )

    pdf.sub_heading("Where in the code")
    pdf.code(
        "# statistical_tests.json, lines 230-233\n"
        "\"dose_response_positive\": {\n"
        "    \"assistant_axis\": { \"rho\": -1.0, \"p_value\": 0.0 },\n"
        "    ..."
    )
    pdf.body(
        "The summary.txt then reports these as \"p = 0.0000\" which looks like a real "
        "result rather than a numerical artifact."
    )

    pdf.sub_heading("Fix")
    pdf.bullet("Report as \"p < 1e-15\" or \"p < machine epsilon\" instead of 0.0.")
    pdf.bullet(
        "Add a guard in 03_analysis.py: if p_value == 0.0, replace with a floor value "
        "and a note that it underflowed."
    )
    pdf.bullet(
        "Better yet, fix Issue 1 first -- once you compute the correlation properly "
        "on row-level data, the p-values will be real numbers, not underflows."
    )

    pdf.separator()

    # --- Issue 3 ---
    pdf.issue_heading(3,
        "Pearson Correlation for Projection Analysis Computed Only at Baseline, "
        "Not Pooled Across Steered Conditions")
    pdf.severity_badge("HIGH")

    pdf.sub_heading("What is wrong")
    pdf.body(
        "In 03_analysis.py (lines 162-174), the Pearson r between Assistant-Axis projection "
        "and per-base syc_logit is computed only on baseline (coefficient=0) data. At "
        "baseline, the variance in projection is entirely natural variance across questions "
        "(a narrow range around ~12800), which severely limits power. The steered conditions "
        "create a much wider range of projection values (shifted by coefficient * cos(vec, "
        "axis)), providing far more variance to detect a relationship."
    )
    pdf.body(
        "The README concludes this is a \"soft null\" (r = -0.088, p = 0.130), but this "
        "conclusion is premature -- the test has low power because it uses only the narrowest "
        "slice of the data."
    )

    pdf.sub_heading("Fix")
    pdf.bullet(
        "Pool across steered conditions: use all (condition, coefficient) cells and compute "
        "a partial correlation between post-steer projection and syc_logit, controlling for "
        "condition. This leverages the full dynamic range created by steering."
    )
    pdf.bullet(
        "Alternatively, use a multi-level regression: syc_logit ~ post_steer_projection + "
        "(1|condition) + (1|base_id)."
    )
    pdf.bullet(
        "Keep the baseline-only result as one panel, but add the pooled analysis as the "
        "primary test of whether projection predicts sycophancy."
    )

    pdf.separator()

    # --- Issue 4 ---
    pdf.issue_heading(4,
        "No Multiple Comparison Correction Across 30+ Statistical Tests")
    pdf.severity_badge("HIGH")

    pdf.sub_heading("What is wrong")
    pdf.body(
        "The analysis runs at least 35 separate hypothesis tests without any correction "
        "for multiple comparisons:"
    )
    pdf.bullet("7 McNemar tests (best vs baseline)")
    pdf.bullet("6 McNemar tests (axis vs role at matched magnitude)")
    pdf.bullet("6 McNemar tests (real vs random)")
    pdf.bullet("7 Spearman dose-response (positive coefficients)")
    pdf.bullet("7 Spearman dose-response (negative coefficients)")
    pdf.bullet("1 Pearson r (projection vs syc_logit)")
    pdf.bullet("1 Welch t-test (quartile projections)")
    pdf.bullet("1 Logistic regression")

    pdf.body(
        "While many p-values are very small (< 1e-7) and would survive any correction, "
        "at least one key comparison would not: contrarian vs assistant_axis at matched "
        "coefficient has p = 0.0485 (statistical_tests.json line 99). Under even a mild "
        "Holm correction with 35 tests, this would not be significant. The dose-response "
        "tests for negative coefficients also include p-values of 0.072 and 0.208 that are "
        "already non-significant before correction."
    )

    pdf.sub_heading("Fix")
    pdf.bullet(
        "Apply Holm-Bonferroni or Benjamini-Hochberg FDR correction within each family "
        "of tests (e.g., all 7 best-vs-baseline McNemar tests as one family)."
    )
    pdf.bullet(
        "Report both raw and adjusted p-values. Flag which results survive correction."
    )
    pdf.bullet(
        "At minimum, acknowledge the multiple comparison issue in the summary "
        "and note which borderline results (p > 0.01) are fragile."
    )

    pdf.separator()

    # --- Issue 5 ---
    pdf.issue_heading(5,
        "McNemar's Test Uses Chi-Square Approximation With Very Small "
        "Off-Diagonal Cells")
    pdf.severity_badge("MEDIUM")

    pdf.sub_heading("What is wrong")
    pdf.body(
        "In 03_analysis.py (line 411), mcnemar() is called with exact=False for all "
        "comparisons. The chi-square approximation for McNemar's test requires both "
        "off-diagonal cells (b01 and b10) to be reasonably large. Several comparisons "
        "have b01 = 1 or 2:"
    )
    pdf.code(
        "# From statistical_tests.json:\n"
        "skeptic vs baseline:  b10=65, b01=2   (b01 too small)\n"
        "judge vs baseline:    b10=59, b01=2   (b01 too small)\n"
        "scientist vs baseline: b10=51, b01=2  (b01 too small)\n"
        "skeptic vs axis:      b10=31, b01=1   (b01 too small)\n"
        "judge vs axis:        b10=25, b01=1   (b01 too small)"
    )
    pdf.body(
        "When one off-diagonal cell is below 5, the exact binomial test (exact=True) is "
        "the standard recommendation. The chi-square approximation with continuity "
        "correction is conservative but may not match the exact p-value well with such "
        "extreme asymmetry."
    )

    pdf.sub_heading("Fix")
    pdf.bullet(
        "Use exact=True when min(b01, b10) < 5, or use exact=True universally for "
        "consistency (the computational cost is negligible with these sample sizes)."
    )
    pdf.bullet(
        "Alternatively, use a mid-p McNemar test for less conservative results while "
        "maintaining validity."
    )

    pdf.separator()

    # --- Issue 6 ---
    pdf.issue_heading(6,
        "Random Condition Uses a Different Best-Coefficient Selection Criterion")
    pdf.severity_badge("MEDIUM")

    pdf.sub_heading("What is wrong")
    pdf.body(
        "In 03_analysis.py (lines 317-323), the \"best coefficient\" for each non-random "
        "condition is selected by maximizing the excess over random at the same coefficient. "
        "But for the random condition itself, the code switches to a different criterion: "
        "it picks the coefficient with the lowest mean_syc_logit."
    )

    pdf.code(
        "# 03_analysis.py, lines 317-323\n"
        "if cond == \"random\":\n"
        "    # Lowest absolute logit (different criterion!)\n"
        "    if best is None or s[\"mean_syc_logit\"] < best[\"mean_syc_logit\"]:\n"
        "        best = entry\n"
        "else:\n"
        "    # Highest excess over random (standard criterion)\n"
        "    if best is None or excess > best[\"excess_over_random\"]:\n"
        "        best = entry"
    )

    pdf.body(
        "This causes random's \"best\" to be coefficient -5000 (logit = +0.054), which is "
        "deep in the degradation regime. The McNemar test then compares this degraded "
        "output against baseline and reports p < 1e-14, but this just measures \"extreme "
        "perturbation differs from baseline,\" not \"random steering has a real directional "
        "effect.\" The results summary table presents this alongside the other conditions' "
        "results without flagging the criterion switch."
    )

    pdf.sub_heading("Fix")
    pdf.bullet(
        "Use a consistent selection criterion across all conditions, or clearly label "
        "random's entry as \"most degraded\" rather than \"best.\""
    )
    pdf.bullet(
        "Omit random from the \"best per condition\" table entirely. Its purpose is as a "
        "control at matched coefficients, not as a competitor."
    )

    # ========== SECTION II: METHODOLOGICAL ISSUES ==========
    pdf.add_page()
    pdf.section_title("II. Methodological & Experimental Design Issues")

    pdf.body(
        "These issues affect the experimental validity and the strength of the claims. "
        "Several were identified in a prior review (Issues.pdf in the repo); the issues "
        "below are either new or provide additional detail on previously noted problems."
    )

    # --- Issue 7 ---
    pdf.issue_heading(7,
        "Degradation Flags Are All False -- the Detector Never Fires")
    pdf.severity_badge("HIGH")

    pdf.sub_heading("What is wrong")
    pdf.body(
        "The degradation_flags.json file contains 91 entries (7 conditions x 13 "
        "coefficients), and every single one is false. The detection criterion in "
        "02_evaluate_steering.py (line 279) is near_tie_frac > 0.8, meaning more than "
        "80% of rows must have |logp(A) - logp(B)| < 0.05."
    )
    pdf.body(
        "The highest near_tie_frac in the actual data is 0.02 (2%). The threshold of 80% "
        "is unreachable in practice. The README explicitly states that at |coef| = 5000 "
        "the model degrades and random vectors collapse binary rate to 50%, yet the "
        "degradation detector does not flag any of these. The detector is effectively dead "
        "code."
    )

    pdf.sub_heading("Evidence from the data")
    pdf.body(
        "At random +5000: binary_rate = 50.0%, mean_syc_logit = +0.444, near_tie_frac = 0.0. "
        "The model is clearly degraded (chance-level binary rate), but the log-probability "
        "gap is still outside the 0.05 near-tie window. The model isn't producing ties -- "
        "it's producing random confident answers. The near-tie proxy fundamentally "
        "mischaracterizes what degradation looks like."
    )

    pdf.sub_heading("Fix")
    pdf.bullet(
        "Replace the single near-tie heuristic with a multi-signal degradation battery: "
        "answer entropy, top-2 margin distribution, perplexity shift, and comparison "
        "against the random control's behavior at the same coefficient."
    )
    pdf.bullet(
        "A simple effective threshold: flag degradation if the condition's mean_syc_logit "
        "is within 0.1 of random's mean_syc_logit at the same coefficient AND the binary "
        "rate is near 50%."
    )

    pdf.separator()

    # --- Issue 8 ---
    pdf.issue_heading(8,
        "Best Coefficient Uniformly +2000 for All Conditions Is Suspicious")
    pdf.severity_badge("MEDIUM")

    pdf.sub_heading("What is wrong")
    pdf.body(
        "Every non-random condition selects +2000 as its best coefficient. While this "
        "could be genuine (all role vectors might have similar effective coupling to the "
        "residual stream at this magnitude), it is also consistent with the selection "
        "criterion being dominated by a common artifact at that coefficient level."
    )
    pdf.body(
        "The selection picks the coefficient maximizing excess_over_random = "
        "random_logit - condition_logit. Random's logit at +2000 is 0.997 -- the highest "
        "in the positive range (random actually gets worse at low-to-moderate coefficients). "
        "So the excess is partly driven by random having an unusually high logit at +2000, "
        "not just by the conditions performing well there."
    )

    pdf.sub_heading("Fix")
    pdf.bullet(
        "Fit a curve (e.g., logistic or piecewise-linear) to each condition's "
        "dose-response and identify the optimal coefficient continuously rather than "
        "picking from the discrete sweep."
    )
    pdf.bullet(
        "Report results at multiple operating points (e.g., +1000, +2000, +3000) "
        "rather than only at the selected \"best.\""
    )

    pdf.separator()

    # --- Issue 9 ---
    pdf.issue_heading(9,
        "Primary Metric (syc_logit) and Primary Test (McNemar) Measure "
        "Different Things")
    pdf.severity_badge("MEDIUM")

    pdf.sub_heading("What is wrong")
    pdf.body(
        "The README and analysis declare mean_syc_logit as the primary metric (continuous, "
        "more sensitive, cancels A-bias). But the headline statistical tests are McNemar "
        "tests on binary classifications (chose_sycophantic yes/no). These test different "
        "hypotheses: McNemar tests whether the count of flipped decisions is symmetric, "
        "while the continuous logit captures magnitude shifts that may not cross the "
        "decision boundary."
    )
    pdf.body(
        "This mismatch means a condition could have a large, significant shift in "
        "mean_syc_logit (the primary metric) but a non-significant McNemar p-value "
        "(because few rows actually cross the boundary), or vice versa."
    )

    pdf.sub_heading("Fix")
    pdf.bullet(
        "Use paired t-tests or Wilcoxon signed-rank tests on per-base syc_logit "
        "(averaged across original/swapped variants) as the primary significance test."
    )
    pdf.bullet(
        "Keep McNemar as a secondary test for the binary rate, clearly labeled as such."
    )

    pdf.separator()

    # --- Issue 10 ---
    pdf.issue_heading(10,
        "Row-Level Independence Assumption Is Violated")
    pdf.severity_badge("MEDIUM")

    pdf.sub_heading("What is wrong")
    pdf.body(
        "Each base question produces two rows (original and swapped). These pairs share "
        "almost identical prompts -- only the A/B labels differ. Their activations, "
        "projections, and syc_logit values are highly correlated. Yet the McNemar tests "
        "and logistic regression treat all 600 rows as independent observations."
    )
    pdf.body(
        "The effective sample size is closer to 300 (base questions) than 600 (rows). "
        "McNemar with n=600 paired-but-not-independent rows produces overly narrow "
        "confidence intervals."
    )

    pdf.sub_heading("Fix")
    pdf.bullet(
        "Conduct all primary analyses at the base-question level (n=300): average each "
        "base question's syc_logit across its two variants, then run paired tests on "
        "these 300 averaged values."
    )
    pdf.bullet(
        "For the logistic regression, use base-level aggregated data or a mixed-effects "
        "model with base_id as a random intercept."
    )

    # ========== SECTION III: CODE-LEVEL ISSUES ==========
    pdf.add_page()
    pdf.section_title("III. Code-Level & Reproducibility Issues")

    # --- Issue 11 ---
    pdf.issue_heading(11,
        "setup_info.json Contains Stale Coefficient Range Metadata")
    pdf.severity_badge("MEDIUM")

    pdf.sub_heading("What is wrong")
    pdf.body(
        "data/setup_info.json (lines 111-125) records the coefficient_range as "
        "[-4, -3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 4]. These are small-"
        "magnitude coefficients from an earlier iteration of the experiment. The actual "
        "coefficients used in 02_evaluate_steering.py are "
        "[-5000, -3000, -2000, -1000, -500, -200, 0, 200, 500, 1000, 2000, 3000, 5000]."
    )
    pdf.body(
        "This metadata was written by 00_setup.py (which ran before the coefficient range "
        "was finalized) and never updated. Anyone inspecting setup_info.json to understand "
        "or reproduce the experiment will be misled."
    )

    pdf.sub_heading("Fix")
    pdf.bullet(
        "Move the coefficient_range constant into a shared config that is imported by "
        "both 00_setup.py and 02_evaluate_steering.py, ensuring they always agree."
    )
    pdf.bullet(
        "Or: have 02_evaluate_steering.py overwrite the coefficient_range field in "
        "setup_info.json at runtime."
    )

    pdf.separator()

    # --- Issue 12 ---
    pdf.issue_heading(12,
        "00_setup.py and 00b_rebuild_eval.py Produce Incompatible "
        "eval_data.json Formats")
    pdf.severity_badge("MEDIUM")

    pdf.sub_heading("What is wrong")
    pdf.body(
        "00_setup.py writes eval_data.json with 300 rows and fields: question_id, id, "
        "question_text, sycophantic_answer, non_sycophantic_answer. Then "
        "00b_rebuild_eval.py overwrites it with 600 counterbalanced rows that add base_id, "
        "orig_id, and variant fields."
    )
    pdf.body(
        "02_evaluate_steering.py reads row[\"base_id\"] and row[\"variant\"] (lines 131-132). "
        "If someone runs 00_setup.py but skips 00b_rebuild_eval.py, script 02 will crash "
        "with a KeyError. There is no guard, no warning, and no documentation of this "
        "dependency beyond the numbered script ordering in the README."
    )

    pdf.sub_heading("Fix")
    pdf.bullet(
        "Use distinct filenames: eval_data_raw.json (from 00_setup.py) and "
        "eval_data.json (from 00b, the one 02 actually needs)."
    )
    pdf.bullet(
        "Add a schema check at the top of 02_evaluate_steering.py that verifies "
        "base_id and variant fields exist, with a clear error message directing the user "
        "to run 00b_rebuild_eval.py."
    )

    pdf.separator()

    # --- Issue 13 ---
    pdf.issue_heading(13,
        "Hardcoded Paths and Model Constants Prevent Portability")
    pdf.severity_badge("LOW")

    pdf.sub_heading("What is wrong")
    pdf.body(
        "Every script hardcodes ROOT = \"/home/ubuntu/experiment\" and "
        "sys.path.insert(0, \"/home/ubuntu/assistant-axis\"). Token IDs (A_TOKEN_ID = "
        "235280, B_TOKEN_ID = 235305) are Gemma-specific constants repeated across "
        "multiple files. The target layer (22) is also hardcoded independently in each script."
    )

    pdf.sub_heading("Fix")
    pdf.bullet(
        "Create a config.py with ROOT, MODEL_NAME, TARGET_LAYER, A_TOKEN_ID, B_TOKEN_ID, "
        "COEFFICIENTS, and ASSISTANT_AXIS_PATH. Import it in all scripts."
    )
    pdf.bullet(
        "Use argparse or environment variables for paths so the experiment can run on "
        "different machines without editing source code."
    )

    pdf.separator()

    # --- Issue 14 ---
    pdf.issue_heading(14,
        "Logistic Regression Included Despite Being Acknowledged as Confounded")
    pdf.severity_badge("LOW")

    pdf.sub_heading("What is wrong")
    pdf.body(
        "03_analysis.py (lines 477-503) fits a logistic regression of chose_sycophantic "
        "on axis_projection at the row level (n=600). The summary.txt correctly notes: "
        "\"This is expected to be null because the row-level label is confounded by A-bias.\" "
        "The result is indeed null (p = 0.934)."
    )
    pdf.body(
        "Including a test that the authors acknowledge is confounded adds noise to the "
        "statistical report and gives reviewers a target: \"why did you run a test you knew "
        "was invalid?\" The Fisher information matrix inversion (line 491) is also wrapped "
        "in a bare try/except that silently sets SE and p to NaN if it fails, hiding "
        "potential numerical issues."
    )

    pdf.sub_heading("Fix")
    pdf.bullet(
        "Remove the row-level logistic regression, or move it to an appendix/supplementary "
        "section clearly labeled as a sanity check."
    )
    pdf.bullet(
        "If kept, replace the manual Fisher information computation with "
        "statsmodels.Logit which provides robust standard errors and proper Wald tests."
    )

    # ========== SUMMARY TABLE ==========
    pdf.add_page()
    pdf.section_title("Summary: All Issues at a Glance")

    pdf.set_font("Helvetica", "B", 9)
    col_widths = [8, 75, 18, 70]  # #, Issue, Severity, File(s)
    headers = ["#", "Issue", "Severity", "Primary Location"]
    for w, h in zip(col_widths, headers):
        pdf.cell(w, 7, h, border=1, align="C")
    pdf.ln()

    issues = [
        ("1", "Spearman rho = -1.0 on 6 aggregate points", "CRITICAL", "03_analysis.py:458-471"),
        ("2", "p-values of exactly 0.0 (underflow)", "HIGH", "statistical_tests.json"),
        ("3", "Pearson r only at baseline, not pooled", "HIGH", "03_analysis.py:162-174"),
        ("4", "No multiple comparison correction", "HIGH", "03_analysis.py:386-503"),
        ("5", "McNemar chi-sq with b01 = 1 or 2", "MEDIUM", "03_analysis.py:392-413"),
        ("6", "Random uses different selection criterion", "MEDIUM", "03_analysis.py:317-323"),
        ("7", "Degradation flags all false (dead code)", "HIGH", "02_evaluate_steering.py:279"),
        ("8", "Best coeff uniformly +2000 (suspicious)", "MEDIUM", "03_analysis.py:303-328"),
        ("9", "Primary metric / primary test mismatch", "MEDIUM", "03_analysis.py:386-423"),
        ("10", "Row-level independence violated (n=600 vs 300)", "MEDIUM", "03_analysis.py:477-503"),
        ("11", "Stale coefficient range in setup_info", "MEDIUM", "data/setup_info.json:111"),
        ("12", "eval_data.json overwrite confusion", "MEDIUM", "00_setup.py / 00b_rebuild"),
        ("13", "Hardcoded paths and constants", "LOW", "all scripts"),
        ("14", "Confounded logistic regression included", "LOW", "03_analysis.py:477-503"),
    ]

    pdf.set_font("Helvetica", "", 8.5)
    severity_colors = {
        "CRITICAL": (180, 30, 30),
        "HIGH": (200, 100, 0),
        "MEDIUM": (140, 130, 0),
        "LOW": (80, 130, 80),
    }
    for num, desc, sev, loc in issues:
        row_h = 6
        pdf.cell(col_widths[0], row_h, num, border=1, align="C")
        pdf.cell(col_widths[1], row_h, desc[:48], border=1)
        r, g, b = severity_colors[sev]
        pdf.set_text_color(r, g, b)
        pdf.set_font("Helvetica", "B", 8.5)
        pdf.cell(col_widths[2], row_h, sev, border=1, align="C")
        pdf.set_text_color(30, 30, 30)
        pdf.set_font("Courier", "", 7.5)
        pdf.cell(col_widths[3], row_h, loc[:42], border=1)
        pdf.set_font("Helvetica", "", 8.5)
        pdf.ln()

    pdf.ln(8)
    pdf.section_title("Recommended Priority Order")
    pdf.body(
        "If addressing these issues incrementally, the following order maximizes "
        "impact per unit of effort:"
    )
    pdf.set_font("Helvetica", "B", 10.5)
    pdf.cell(0, 7, "Tier 1: Must fix before any publication", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10.5)
    pdf.bullet("Issue 1: Replace aggregate Spearman with row-level dose-response test")
    pdf.bullet("Issue 3: Pool Pearson correlation across steered conditions")
    pdf.bullet("Issue 4: Apply multiple comparison correction")
    pdf.bullet("Issue 7: Replace broken degradation detector")
    pdf.bullet("Issue 10: Move primary analyses to base-question level (n=300)")

    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 10.5)
    pdf.cell(0, 7, "Tier 2: Should fix for methodological rigor", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10.5)
    pdf.bullet("Issue 2: Fix p-value underflow reporting")
    pdf.bullet("Issue 5: Use exact McNemar where appropriate")
    pdf.bullet("Issue 6: Consistent selection criterion or omit random from ranking")
    pdf.bullet("Issue 9: Align primary test with primary metric")
    pdf.bullet("Issue 11: Fix stale metadata in setup_info.json")

    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 10.5)
    pdf.cell(0, 7, "Tier 3: Nice to fix for code quality", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10.5)
    pdf.bullet("Issue 8: Investigate uniform best-coefficient selection")
    pdf.bullet("Issue 12: Disambiguate eval_data.json pipeline")
    pdf.bullet("Issue 13: Centralize config and remove hardcoded paths")
    pdf.bullet("Issue 14: Remove or isolate confounded logistic regression")

    return pdf


if __name__ == "__main__":
    pdf = build_report()
    out_path = "/Users/vikramkakaria/Desktop/Sandbagging/Issues_Report.pdf"
    pdf.output(out_path)
    print(f"Report written to {out_path}")
