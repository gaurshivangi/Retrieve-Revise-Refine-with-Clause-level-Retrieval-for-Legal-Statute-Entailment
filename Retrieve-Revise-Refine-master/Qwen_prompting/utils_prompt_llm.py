prompt_llm_support_1 = lambda content_of_articles, statement: f"""Given the following legal article(s) and legal statement:
\nLegal article(s): ```{content_of_articles}```
\nLegal statement: ```{statement}```
\nIs it possible to verify the accuracy of the legal statement using the provided legal article(s), or is the content of the legal article(s) insufficient?
\nPlease respond with either "The statement is true" or "The statement is false" or "Not enough information". Explain first then answer later."""

prompt_llm_support_2 = lambda laws, claim: f"""Assessment of Legal Claim:
\nRelevant Law(s): ```{laws}```
\nClaim to Verify: ```{claim}```
\nIs the claim substantiated by the law(s) cited? Choose an appropriate response from: "Claim substantiated", "Claim unsubstantiated", or "More information required". Provide the reasoning first, followed by the answer."""

prompt_llm_support_3 = lambda statutes, assertion: f"""Law Conformity Assessment:
\nStatute(s) Provided: ```{statutes}```
\nAssertion: ```{assertion}```
\nDetermine if the assertion is supported by the given statute(s). Respond with "Assertion valid", "Assertion invalid", or "Insufficient legal context". Start with the reasoning, followed by presenting the conclusion at the end."""


prompt_llm_support_4 = lambda legal_texts, hypothesis: f"""Statute Compliance Test:
\nLegal Text(s): ```{legal_texts}```
\nHypothesis: ```{hypothesis}```
\nPlease assess if the hypothesis aligns with the legal text(s). Your options are "Hypothesis compliant", "Hypothesis non-compliant", or "Cannot determine compliance". Begin by explaining your reasoning, followed by your conclusion."""

prompt_llm_support_5 = lambda legal_provisions, conjecture: f"""Legal Verification:
\nLegal Provision(s): ```{legal_provisions}```
\nConjecture: ```{conjecture}```
\nIs the conjecture in accordance with the supplied legal provision(s)? Record your answer as "Conjecture verified", "Conjecture unverified", or "Unable to verify". Provide an explanation first and then state your conclusion."""
