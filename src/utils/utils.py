import pandas as pd

PROMPT_TEMPLATE = """You are helping a healthcare claims analyst quickly triage a claim before submission.
 
Claim facts (grounded only in these values — do not invent or assume anything beyond them):
{claim_facts}
 
Model's top risk drivers for this claim: {risk_factors}
Predicted denial probability: {probability:.0%}
 
Write a 2-3 sentence explanation for the analyst that:
- Uses plain language, no insurance jargon the analyst would need to look up
- States which factor(s) above are driving the risk
- Includes exactly one specific, concrete recommended action
- Ends by noting this is a risk estimate, not a guarantee of denial
- Does not state or imply any fact not given above
"""



def build_claim_facts_from_data(row: pd.Series) -> str:
    """Turn a claim's raw field values into a short bullet list for the prompt."""
    lines = [
        f"- Payer: {row['payer_type']} ({row['payer_id']})",
        f"- Visit type: {row['visit_type']}",
        f"- Total billed: ${row['total_billed']:.2f}",
        f"- Prior auth required: {'Yes' if row['prior_auth_required'] else 'No'}; on file: {'Yes' if row['has_prior_auth'] else 'No'}",
        f"- Referral required: {'Yes' if row['referral_required'] else 'No'}; on file: {'Yes' if row['referral_present'] else 'No'}",
        f"- In-network: {'Yes' if row['is_in_network'] else 'No'}",
        f"- Documentation missing flag: {'Yes' if row['missing_documentation_flag'] else 'No'}",
        f"- Eligibility verified: {'Yes' if row['eligibility_verified'] else 'No'}",
        f"- Days to submit: {row['days_to_submit']}",
    ]
    return "\n".join(lines)


def build_prompt(claim_row: pd.Series, risk_factors: str, probability: float) -> str:
    return PROMPT_TEMPLATE.format(
        claim_facts=build_claim_facts_from_data(claim_row),
        risk_factors=risk_factors,
        probability=probability,
    )
