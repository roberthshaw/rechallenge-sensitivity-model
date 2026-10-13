import math

import numpy as np
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Rechallenge Assumption Sensitivity",
    page_icon="🧪",
    layout="wide"
)

st.title("Rechallenge Assumption Sensitivity")

st.markdown(
    """
    ### How robust is our conclusion to the assumption that
    primary non-colonisers would have little or no colonisation
    if rechallenged?

    The central issue is that rechallenging only participants
    colonised at primary challenge requires an assumption about
    what would have happened to participants who resisted
    primary colonisation.

    This tool asks:

    **How wrong can that assumption be before our conclusion changes?**
    """
)


# ============================================================
# FIXED STUDY SIZE
# ============================================================

TOTAL_N = 156


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Study assumptions")

st.sidebar.metric(
    "Total participants",
    TOTAL_N
)


primary_ar_target = st.sidebar.slider(
    "Primary colonisation rate (%)",
    min_value=30,
    max_value=80,
    value=60,
    step=1,
    help=(
        "Expected or observed colonisation rate "
        "following primary challenge."
    )
)


# Convert percentage to an integer participant count
n_primary_positive = round(
    TOTAL_N * primary_ar_target / 100
)

n_primary_negative = (
    TOTAL_N - n_primary_positive
)

primary_ar = (
    100 * n_primary_positive / TOTAL_N
)

P = primary_ar / 100


st.sidebar.caption(
    f"""
    Realised primary rate:

    **{n_primary_positive}/{TOTAL_N}
    = {primary_ar:.1f}%**

    Primary non-colonisers:

    **{n_primary_negative}/{TOTAL_N}**
    """
)


st.sidebar.markdown("---")


secondary_positive_rate_target = st.sidebar.slider(
    "Secondary AR in primary colonisers, C (%)",
    min_value=1,
    max_value=99,
    value=90,
    step=1,
    help=(
        "Colonisation rate on rechallenge among participants "
        "who were colonised at primary challenge."
    )
)


# Convert C to an integer number
n_pp = round(
    n_primary_positive
    * secondary_positive_rate_target
    / 100
)

n_pn = (
    n_primary_positive - n_pp
)

C_realised = (
    100 * n_pp / n_primary_positive
)


st.sidebar.caption(
    f"""
    Realised C:

    **{n_pp}/{n_primary_positive}
    = {C_realised:.1f}%**
    """
)


st.sidebar.markdown("---")


outcome_scale = st.sidebar.radio(
    "Primary display",
    [
        "Absolute reduction in attack rate (percentage points)",
        "Relative apparent protective effect (%)"
    ],
    index=0
)


st.sidebar.markdown("---")

st.sidebar.markdown(
    """
    ### Transition notation

    **+/+**  
    Colonised at both challenges

    **+/−**  
    Colonised at primary, not at secondary

    **−/+**  
    Not colonised at primary, colonised at secondary

    **−/−**  
    Not colonised at either challenge
    """
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def wilson_interval(k, n, z=1.959963984540054):
    """
    Wilson score interval for a binomial proportion.
    Returns lower and upper bounds as proportions.
    """

    if n <= 0:
        return np.nan, np.nan

    p = k / n

    denominator = 1 + (z ** 2) / n

    centre = (
        p + (z ** 2) / (2 * n)
    ) / denominator

    half_width = (
        z
        * math.sqrt(
            p * (1 - p) / n
            + (z ** 2) / (4 * n ** 2)
        )
        / denominator
    )

    return (
        max(0, centre - half_width),
        min(1, centre + half_width)
    )


def paired_difference_newcombe(
    n_pn,
    n_np,
    n_total
):
    """
    Newcombe-style confidence interval for the paired
    difference in proportions.

    Difference is:

        Primary AR - Secondary AR

    which equals:

        (n_pn - n_np) / n_total

    where:

        n_pn = primary positive -> secondary negative
        n_np = primary negative -> secondary positive

    The CI is constructed from Wilson intervals for the two
    discordant proportions.

    Returns:
        estimate, lower, upper

    all as proportions.
    """

    p_pn = n_pn / n_total
    p_np = n_np / n_total

    estimate = (
        p_pn - p_np
    )

    pn_low, pn_high = wilson_interval(
        n_pn,
        n_total
    )

    np_low, np_high = wilson_interval(
        n_np,
        n_total
    )

    # Newcombe paired difference construction
    lower = (
        estimate
        - math.sqrt(
            (p_pn - pn_low) ** 2
            + (np_high - p_np) ** 2
        )
    )

    upper = (
        estimate
        + math.sqrt(
            (pn_high - p_pn) ** 2
            + (p_np - np_low) ** 2
        )
    )

    return estimate, lower, upper


def relative_effect_from_absolute(
    absolute_difference,
    primary_proportion
):
    """
    Convert absolute reduction in AR into relative
    apparent protective effect.
    """

    if primary_proportion <= 0:
        return np.nan

    return (
        absolute_difference
        / primary_proportion
    )


def format_signed(value, suffix="%"):
    return f"{value:+.1f}{suffix}"


# ============================================================
# INTRODUCTION
# ============================================================

st.header("1. The assumption being made")


left_intro, right_intro = st.columns([1.3, 1])


with left_intro:

    st.info(
        """
        **Working assumption**

        Participants who were **not colonised at primary challenge**
        would have **no meaningful colonisation on rechallenge**.

        If those participants are not rechallenged, this assumption
        cannot be directly tested.
        """
    )


with right_intro:

    st.metric(
        "Primary colonised",
        f"{n_primary_positive}/{TOTAL_N}",
        f"{primary_ar:.1f}%"
    )

    st.metric(
        "Primary non-colonised",
        f"{n_primary_negative}/{TOTAL_N}",
        f"{100-primary_ar:.1f}%"
    )


st.markdown(
    f"""
    Among the **{n_primary_positive} primary colonisers**,
    the assumed/observed secondary attack rate is
    **{C_realised:.1f}% ({n_pp}/{n_primary_positive})**.

    We now vary what might happen to the
    **{n_primary_negative} primary non-colonisers** if they were
    also rechallenged.
    """
)


st.markdown("---")


# ============================================================
# BUILD SENSITIVITY DATA
# ============================================================

# We vary the number of primary-negative participants who
# become colonised at secondary challenge from 0 to all.

scenario_rows = []


for n_np in range(
    0,
    n_primary_negative + 1
):

    n_nn = (
        n_primary_negative - n_np
    )

    N_rate = (
        100 * n_np / n_primary_negative
        if n_primary_negative > 0
        else 0
    )

    # Primary attack rate
    primary_prop = (
        n_primary_positive / TOTAL_N
    )

    # Secondary positive count:
    # +/+ plus -/+
    secondary_positive_count = (
        n_pp + n_np
    )

    secondary_prop = (
        secondary_positive_count / TOTAL_N
    )

    secondary_ar = (
        100 * secondary_prop
    )

    # Absolute reduction:
    # primary AR - secondary AR
    diff, diff_low, diff_high = (
        paired_difference_newcombe(
            n_pn=n_pn,
            n_np=n_np,
            n_total=TOTAL_N
        )
    )

    absolute_effect = (
        100 * diff
    )

    absolute_low = (
        100 * diff_low
    )

    absolute_high = (
        100 * diff_high
    )

    # Relative apparent protective effect
    relative_effect = (
        100
        * relative_effect_from_absolute(
            diff,
            primary_prop
        )
    )

    relative_low = (
        100
        * relative_effect_from_absolute(
            diff_low,
            primary_prop
        )
    )

    relative_high = (
        100
        * relative_effect_from_absolute(
            diff_high,
            primary_prop
        )
    )

    scenario_rows.append(
        {
            "n_np": n_np,
            "n_nn": n_nn,
            "N_rate": N_rate,
            "secondary_ar": secondary_ar,
            "absolute_effect": absolute_effect,
            "absolute_low": absolute_low,
            "absolute_high": absolute_high,
            "relative_effect": relative_effect,
            "relative_low": relative_low,
            "relative_high": relative_high
        }
    )


# Convert to arrays
N_rates = np.array(
    [r["N_rate"] for r in scenario_rows]
)

secondary_rates = np.array(
    [r["secondary_ar"] for r in scenario_rows]
)

absolute_effects = np.array(
    [r["absolute_effect"] for r in scenario_rows]
)

absolute_lows = np.array(
    [r["absolute_low"] for r in scenario_rows]
)

absolute_highs = np.array(
    [r["absolute_high"] for r in scenario_rows]
)

relative_effects = np.array(
    [r["relative_effect"] for r in scenario_rows]
)

relative_lows = np.array(
    [r["relative_low"] for r in scenario_rows]
)

relative_highs = np.array(
    [r["relative_high"] for r in scenario_rows]
)


# ============================================================
# FIND THRESHOLD 1:
# LOWER CI FIRST REACHES / CROSSES ZERO
# ============================================================

ci_cross_indices = np.where(
    absolute_lows <= 0
)[0]


if len(ci_cross_indices) > 0:

    ci_index = int(
        ci_cross_indices[0]
    )

    ci_threshold_n = (
        scenario_rows[ci_index]["n_np"]
    )

    ci_threshold_rate = (
        scenario_rows[ci_index]["N_rate"]
    )

else:

    ci_index = None
    ci_threshold_n = None
    ci_threshold_rate = None


# ============================================================
# FIND THRESHOLD 2:
# POINT ESTIMATE REACHES / CROSSES ZERO
# ============================================================

point_cross_indices = np.where(
    absolute_effects <= 0
)[0]


if len(point_cross_indices) > 0:

    point_index = int(
        point_cross_indices[0]
    )

    point_threshold_n = (
        scenario_rows[point_index]["n_np"]
    )

    point_threshold_rate = (
        scenario_rows[point_index]["N_rate"]
    )

else:

    point_index = None
    point_threshold_n = None
    point_threshold_rate = None


# ============================================================
# THRESHOLD SUMMARY
# ============================================================

st.header("2. How robust is the assumption?")


threshold_col1, threshold_col2 = st.columns(2)


with threshold_col1:

    st.markdown(
        "### 🟠 Threshold 1"
    )

    st.markdown(
        "**Loss of statistical evidence for protection**"
    )

    if ci_threshold_rate is not None:

        st.metric(
            "Primary non-coloniser secondary AR",
            f"{ci_threshold_rate:.1f}%",
            f"{ci_threshold_n}/{n_primary_negative} participants"
        )

        st.caption(
            """
            At this point the lower bound of the two-sided
            95% confidence interval reaches or falls below zero.
            """
        )

    else:

        st.metric(
            "Threshold",
            "Not reached"
        )


with threshold_col2:

    st.markdown(
        "### 🔴 Threshold 2"
    )

    st.markdown(
        "**Point estimate no longer indicates protection**"
    )

    if point_threshold_rate is not None:

        st.metric(
            "Primary non-coloniser secondary AR",
            f"{point_threshold_rate:.1f}%",
            f"{point_threshold_n}/{n_primary_negative} participants"
        )

        st.caption(
            """
            At this point the whole-population secondary
            attack rate reaches or exceeds the primary attack rate.
            """
        )

    else:

        st.metric(
            "Threshold",
            "Not reached"
        )


# ============================================================
# PLAIN LANGUAGE HEADLINE
# ============================================================

if (
    ci_threshold_rate is not None
    and point_threshold_rate is not None
):

    st.success(
        f"""
        ### Interpretation

        With **{primary_ar:.1f}% primary colonisation**
        and **{C_realised:.1f}% secondary colonisation among
        primary colonisers**:

        • If secondary colonisation among primary non-colonisers
        remains below approximately **{ci_threshold_rate:.1f}%**,
        the 95% CI remains above zero.

        • At approximately **{ci_threshold_rate:.1f}%**, statistical
        evidence for protection is lost.

        • At approximately **{point_threshold_rate:.1f}%**, the
        protective-effect point estimate itself reaches zero.

        • Above approximately **{point_threshold_rate:.1f}%**, the
        point estimate indicates a **higher secondary attack rate**
        than primary attack rate.
        """
    )


st.markdown("---")


# ============================================================
# CENTRAL SENSITIVITY GRAPH
# ============================================================

st.header(
    "3. What if the assumption about primary non-colonisers is wrong?"
)


st.markdown(
    """
    The horizontal axis represents the secondary attack rate that
    **would have been observed among primary non-colonisers** if
    they had also been rechallenged.

    The solid line is the estimated population effect.

    The shaded area is its **two-sided 95% confidence interval**.
    """
)


if outcome_scale == (
    "Absolute reduction in attack rate (percentage points)"
):

    graph_effect = absolute_effects
    graph_low = absolute_lows
    graph_high = absolute_highs

    y_title = (
        "Reduction in attack rate (percentage points)"
    )

    graph_title = (
        "Sensitivity of the estimated population effect "
        "to the missing rechallenge group"
    )

else:

    graph_effect = relative_effects
    graph_low = relative_lows
    graph_high = relative_highs

    y_title = (
        "Relative apparent protective effect (%)"
    )

    graph_title = (
        "Sensitivity of relative apparent protection "
        "to the missing rechallenge group"
    )


fig = go.Figure()


# ============================================================
# CONFIDENCE BAND
# ============================================================

fig.add_trace(
    go.Scatter(
        x=N_rates,
        y=graph_high,

        mode="lines",

        line=dict(
            width=0
        ),

        hoverinfo="skip",

        showlegend=False
    )
)


fig.add_trace(
    go.Scatter(
        x=N_rates,
        y=graph_low,

        mode="lines",

        line=dict(
            width=0
        ),

        fill="tonexty",

        fillcolor="rgba(100,100,100,0.20)",

        name="95% CI",

        hovertemplate=(
            "N = %{x:.1f}%<br>"
            "Lower 95% CI = %{y:.1f}"
            "<extra></extra>"
        )
    )
)


# ============================================================
# POINT ESTIMATE
# ============================================================

fig.add_trace(
    go.Scatter(
        x=N_rates,
        y=graph_effect,

        mode="lines+markers",

        line=dict(
            width=4,
            color="#2c3e50"
        ),

        marker=dict(
            size=5,
            color="#2c3e50"
        ),

        name="Point estimate",

        hovertemplate=(
            "Secondary AR in primary non-colonisers: "
            "%{x:.1f}%<br>"
            "Estimated effect: %{y:.1f}"
            "<extra></extra>"
        )
    )
)


# ============================================================
# ZERO EFFECT LINE
# ============================================================

fig.add_hline(
    y=0,

    line_width=3,

    line_dash="dash",

    line_color="black",

    annotation_text="No protection",

    annotation_position="top right"
)


# ============================================================
# THRESHOLD 1 LINE
# ============================================================

if ci_threshold_rate is not None:

    fig.add_vline(
        x=ci_threshold_rate,

        line_width=3,

        line_dash="dot",

        line_color="#f39c12",

        annotation_text=(
            "95% CI reaches 0"
        ),

        annotation_position="top left"
    )


# ============================================================
# THRESHOLD 2 LINE
# ============================================================

if point_threshold_rate is not None:

    fig.add_vline(
        x=point_threshold_rate,

        line_width=3,

        line_dash="dot",

        line_color="#c0392b",

        annotation_text=(
            "Point estimate reaches 0"
        ),

        annotation_position="top right"
    )


# ============================================================
# GRAPH LAYOUT
# ============================================================

fig.update_layout(

    title=dict(
        text=graph_title,
        font=dict(
            size=18
        )
    ),

    height=650,

    xaxis=dict(
        title=(
            "Secondary attack rate among "
            "primary non-colonisers, N (%)"
        ),
        range=[
            0,
            100
        ],
        dtick=10
    ),

    yaxis=dict(
        title=y_title,
        zeroline=False
    ),

    hovermode="x unified",

    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0
    ),

    margin=dict(
        l=70,
        r=50,
        t=100,
        b=70
    )
)


st.plotly_chart(
    fig,
    use_container_width=True,

    config={
        "displaylogo": False,
        "scrollZoom": True
    }
)


# ============================================================
# EXPLAIN THE THREE ZONES
# ============================================================

st.subheader("Three regions of interpretation")


zone1, zone2, zone3 = st.columns(3)


with zone1:

    st.success(
        f"""
        ### 🟢 Robust protection

        **N below ~{ci_threshold_rate:.1f}%**

        The point estimate indicates protection and the
        two-sided 95% CI remains above zero.

        The assumption about the unobserved group has not yet
        altered the statistical conclusion.
        """
        if ci_threshold_rate is not None
        else
        """
        ### 🟢 Robust protection

        The CI remains above zero throughout the modelled range.
        """
    )


with zone2:

    if (
        ci_threshold_rate is not None
        and point_threshold_rate is not None
    ):

        st.warning(
            f"""
            ### 🟠 Uncertain protection

            **N ~{ci_threshold_rate:.1f}% to
            {point_threshold_rate:.1f}%**

            The point estimate still indicates protection,
            but the 95% CI includes zero.

            Statistical evidence for protection has been lost.
            """
        )

    else:

        st.warning(
            """
            ### 🟠 Uncertain protection

            This region is not present under the current assumptions.
            """
        )


with zone3:

    if point_threshold_rate is not None:

        st.error(
            f"""
            ### 🔴 Direction changes

            **N ≥ ~{point_threshold_rate:.1f}%**

            The point estimate reaches or falls below zero.

            The population secondary attack rate is now at least
            as high as the primary attack rate.

            Beyond this threshold the point estimate is compatible
            with **increased susceptibility**, although this does
            not establish its biological cause.
            """
        )

    else:

        st.error(
            """
            ### 🔴 Direction changes

            This threshold is not reached under the current assumptions.
            """
        )


st.markdown("---")


# ============================================================
# INTERACTIVE ASSUMPTION TEST
# ============================================================

st.header("4. Test a specific assumption")


st.markdown(
    """
    Choose what you believe the secondary attack rate among
    primary non-colonisers might actually be.
    """
)


assumed_N_target = st.slider(
    "Assumed secondary AR in primary non-colonisers (%)",
    min_value=0,
    max_value=100,
    value=0,
    step=1
)


# Convert to nearest possible integer event count
assumed_n_np = round(
    n_primary_negative
    * assumed_N_target
    / 100
)


assumed_n_np = max(
    0,
    min(
        n_primary_negative,
        assumed_n_np
    )
)


assumed_row = scenario_rows[
    assumed_n_np
]


assumed_N_realised = (
    assumed_row["N_rate"]
)


# ============================================================
# DISPLAY TRANSITION TABLE
# ============================================================

n_nn_assumed = (
    n_primary_negative
    - assumed_n_np
)


st.subheader("What the complete rechallenge data would look like")


table_col1, table_col2 = st.columns(2)


with table_col1:

    st.markdown(
        f"""
        | Primary → Secondary | Participants |
        |---|---:|
        | Colonised → Colonised (+/+) | **{n_pp}** |
        | Colonised → Not colonised (+/−) | **{n_pn}** |
        | Not colonised → Colonised (−/+) | **{assumed_n_np}** |
        | Not colonised → Not colonised (−/−) | **{n_nn_assumed}** |
        | **Total** | **{TOTAL_N}** |
        """
    )


with table_col2:

    st.metric(
        "Realised N",
        f"{assumed_N_realised:.1f}%",
        f"{assumed_n_np}/{n_primary_negative}"
    )

    st.metric(
        "Whole-population secondary AR",
        f"{assumed_row['secondary_ar']:.1f}%"
    )


# ============================================================
# CURRENT EFFECT
# ============================================================

current_abs = (
    assumed_row["absolute_effect"]
)

current_abs_low = (
    assumed_row["absolute_low"]
)

current_abs_high = (
    assumed_row["absolute_high"]
)


current_rel = (
    assumed_row["relative_effect"]
)

current_rel_low = (
    assumed_row["relative_low"]
)

current_rel_high = (
    assumed_row["relative_high"]
)


result1, result2 = st.columns(2)


with result1:

    st.metric(
        "Absolute reduction in attack rate",
        f"{current_abs:+.1f} pp"
    )

    st.caption(
        f"""
        95% CI:
        {current_abs_low:+.1f} to
        {current_abs_high:+.1f} pp
        """
    )


with result2:

    st.metric(
        "Relative apparent protective effect",
        f"{current_rel:+.1f}%"
    )

    st.caption(
        f"""
        95% CI:
        {current_rel_low:+.1f}% to
        {current_rel_high:+.1f}%
        """
    )


# ============================================================
# CURRENT INTERPRETATION
# ============================================================

if current_abs_low > 0:

    st.success(
        """
        🟢 **The conclusion remains statistically consistent
        with protection.**

        The point estimate is above zero and the two-sided
        95% CI excludes zero.
        """
    )

elif current_abs > 0:

    st.warning(
        """
        🟠 **The point estimate still indicates protection,
        but the 95% CI includes zero.**

        Statistical evidence for protection has been lost.
        """
    )

elif current_abs == 0:

    st.error(
        """
        🔴 **The point estimate is at no protection.**

        The whole-population primary and secondary attack rates
        are equal.
        """
    )

else:

    st.error(
        """
        🔴 **The point estimate indicates a higher secondary
        attack rate than primary attack rate.**

        This is compatible with increased susceptibility or
        predisposition, but does not establish its mechanism.
        """
    )


st.markdown("---")


# ============================================================
# SELECTED-ONLY VERSUS COMPLETE RECHALLENGE
# ============================================================

st.header("5. What would selective rechallenge hide?")


# If we assume N = 0 because the group is not measured,
# then secondary positives are only +/+.
selected_secondary_ar = (
    100 * n_pp / TOTAL_N
)


selected_abs_effect = (
    primary_ar
    - selected_secondary_ar
)


selected_relative_effect = (
    100
    * (
        1
        - selected_secondary_ar
        / primary_ar
    )
)


comparison1, comparison2 = st.columns(2)


with comparison1:

    st.markdown(
        "### If primary non-colonisers are not rechallenged"
    )

    st.caption(
        "Implicit assumption: N ≈ 0%"
    )

    st.metric(
        "Apparent secondary AR",
        f"{selected_secondary_ar:.1f}%"
    )

    st.metric(
        "Apparent absolute protection",
        f"{selected_abs_effect:+.1f} pp"
    )

    st.metric(
        "Apparent relative protection",
        f"{selected_relative_effect:+.1f}%"
    )


with comparison2:

    st.markdown(
        "### If the assumed N is actually correct"
    )

    st.caption(
        f"N = {assumed_N_realised:.1f}%"
    )

    st.metric(
        "Whole-population secondary AR",
        f"{assumed_row['secondary_ar']:.1f}%"
    )

    st.metric(
        "Absolute protection",
        f"{current_abs:+.1f} pp"
    )

    st.metric(
        "Relative apparent protection",
        f"{current_rel:+.1f}%"
    )


absolute_bias = (
    selected_abs_effect
    - current_abs
)


st.info(
    f"""
    Under this scenario, assuming N ≈ 0 rather than observing
    the primary non-colonisers changes the estimated absolute
    protective effect by:

    ### {absolute_bias:+.1f} percentage points
    """
)


st.markdown("---")


# ============================================================
# MATHEMATICS
# ============================================================

with st.expander(
    "6. Show the mathematical and statistical model"
):

    st.markdown(
        """
        Let:

        - **P** = primary attack rate
        - **C** = secondary attack rate among primary colonisers
        - **N** = secondary attack rate among primary non-colonisers
        """
    )


    st.markdown(
        "### Whole-population secondary attack rate"
    )

    st.latex(
        r"""
        S = PC + (1-P)N
        """
    )


    st.markdown(
        "### Absolute reduction in attack rate"
    )

    st.latex(
        r"""
        \Delta AR = P-S
        """
    )


    st.markdown(
        "### Relative apparent protective effect"
    )

    st.latex(
        r"""
        PE =
        100
        \left(
        1-\frac{S}{P}
        \right)
        """
    )


    st.markdown(
        """
        Because primary and secondary outcomes are measured
        in the same participants, the change in attack rate is
        determined by the two discordant transition groups.
        """
    )


    st.latex(
        r"""
        \Delta AR =
        \frac{
        n_{+/-} - n_{-/+}
        }{n}
        """
    )


    st.markdown(
        """
        where:

        - \(n_{+/-}\) are participants colonised at primary
          challenge but not at secondary challenge;

        - \(n_{-/+}\) are participants not colonised at primary
          challenge but colonised at secondary challenge.

        The two-sided 95% confidence interval displayed here is
        constructed using Wilson score intervals for the two
        discordant proportions and a Newcombe-style paired
        difference interval.

        This avoids treating primary and secondary attack rates
        as independent proportions.
        """
    )


# ============================================================
# SCIENTIFIC QUALIFICATION
# ============================================================

st.markdown("---")


st.warning(
    """
    **Interpretation**

    The analysis is a sensitivity analysis of the assumption that
    participants who resist primary colonisation would have little
    or no colonisation if rechallenged.

    Crossing zero does not establish a biological mechanism.

    A negative protective-effect estimate means that the observed
    whole-population secondary attack rate is higher than the
    primary attack rate. This is compatible with increased
    susceptibility or predisposition, but could also reflect
    intrinsic heterogeneity, previous exposure, challenge-related
    biological changes, sampling variation, or other mechanisms.

    Rechallenging the primary non-colonisers provides the missing
    transition information needed to determine the population-level
    change in attack rate.
    """
)
