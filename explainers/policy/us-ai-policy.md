# US AI Policy

**In one line:** The US has no comprehensive federal AI law; policy is set by presidential executive orders that have swung sharply between administrations, by a growing set of state laws, and by an unresolved fight over whether Washington should override those state laws.
**Last reviewed:** 2026-09-30

---

## The short version

- **No federal AI statute.** As of September 2026 Congress has not passed a general AI law. Federal direction comes from executive orders, which a new president can revoke on day one.
- **The federal approach flipped in January 2025.** The 2023 Biden order (EO 14110), focused on safety testing and risk management, was revoked on 20 January 2025. Its replacements (EO 14179 and the July 2025 AI Action Plan) prioritise removing barriers to AI development and US competitiveness.
- **States filled the gap.** California's SB 53 (in force since 1 January 2026) and New York's RAISE Act (effective 1 January 2027) regulate the largest "frontier" model developers. Colorado's broader anti-discrimination AI law was delayed twice and then replaced in May 2026 by a narrower law starting 1 January 2027.
- **Preemption is the live fight.** A 10-year moratorium on state AI laws was stripped from the 2025 budget bill by a 99-1 Senate vote. A December 2025 executive order set up a Justice Department task force to challenge state laws, and a bipartisan House discussion draft in June 2026 proposed a three-year preemption. None of this has produced a federal statute as of this review.

## The mental model: three layers pulling against each other

```
  FEDERAL EXECUTIVE           CONGRESS                 STATES
  (executive orders)          (statute)                (legislatures)
  -----------------           ---------                --------------
  Fast, reversible,           Slow, durable,           Many, varied,
  binds federal agencies      would bind everyone      binding within
  and contractors             - but nothing passed     each state
        |                          |                        |
        +---- tries to shape ------+---- tries to override -+
              state law via            state law via
              litigation, funding      preemption clauses
```

The key idea: in the US system an executive order mainly directs federal agencies. It cannot on its own repeal a state law; only Congress (by statute) or the courts (by ruling a law unconstitutional or preempted) can do that. That is why the preemption debate matters so much.

## Federal: the executive orders

### 2023: EO 14110 (Biden)

Executive Order 14110, "Safe, Secure, and Trustworthy Development and Use of Artificial Intelligence", was signed on 30 October 2023. It was the most extensive US federal AI action to that point. Among other things it used the Defense Production Act to require developers of the most compute-intensive models to report training plans and red-team results to the government, directed NIST to develop safety testing guidance, and tasked agencies with addressing AI risks in areas such as civil rights, critical infrastructure and biosecurity.

Supporters saw it as a proportionate first step to understand risks from the most capable models. Critics argued it stretched the Defense Production Act beyond its purpose and created reporting burdens without congressional authorisation.

### 2025: revocation and a new direction (Trump)

| Date | Action | What it did |
|---|---|---|
| 20 Jan 2025 | EO 14148, "Initial Rescissions of Harmful Executive Orders and Actions" | Revoked EO 14110 among many others |
| 23 Jan 2025 | EO 14179, "Removing Barriers to American Leadership in Artificial Intelligence" | Set a policy of sustaining US AI dominance; ordered a review of all actions taken under EO 14110, and suspension or rescission of those inconsistent with the new policy; called for an action plan |
| 23 Jul 2025 | "Winning the AI Race: America's AI Action Plan" | More than 90 federal policy actions under three pillars: accelerating innovation, building AI infrastructure, and international diplomacy and security. Accompanied by executive orders on data-center permitting, AI exports, and federal procurement of models that are "objective and free from top-down ideological bias" |
| 11 Dec 2025 | EO 14365, "Ensuring a National Policy Framework for Artificial Intelligence" | The main preemption push; see below |
| 20 Mar 2026 | White House "National Policy Framework for Artificial Intelligence" (legislative recommendations) | Proposals for Congress, including federal preemption, organised around child protection, energy, intellectual property, preventing AI-enabled censorship, and workforce, with a preference for lighter oversight over a new AI agency |

The strongest case for the 2025 approach, as its proponents put it: AI capability is a matter of economic and national-security competition, especially with China; heavy pre-emptive regulation slows US developers without stopping foreign ones; and existing law (consumer protection, civil rights, product liability) already applies to harms.

The strongest case against, as critics put it: removing federal safety testing and reporting leaves the most capable systems with only voluntary oversight; the "ideological bias" procurement rules put government in the business of judging model outputs' viewpoints; and blocking states without a federal replacement creates a regulatory vacuum.

## States: where binding rules actually exist

### California SB 53 (Transparency in Frontier Artificial Intelligence Act)

- **Signed:** 29 September 2025. **In force:** 1 January 2026.
- **Who it covers:** "frontier developers" training models with more than 10^26 floating-point operations; heavier duties fall on "large frontier developers" with more than $500 million in annual revenue.
- **What it requires:** publishing a frontier AI framework describing how the developer manages catastrophic risk; publishing transparency reports when releasing new frontier models; reporting "critical safety incidents" to the California Office of Emergency Services within 15 days (24 hours if there is imminent danger of death or serious injury); and whistleblower protections for employees raising catastrophic-risk concerns.
- **Enforcement:** the state Attorney General can seek civil penalties of up to $1 million per violation, including for failing to follow the developer's own published framework.

SB 53 followed the veto of a stricter bill (SB 1047) in 2024. It is a transparency law: it does not tell developers what safety measures to adopt, only that they must say what they do and then do it. See [Frontier safety frameworks](./frontier-safety-frameworks.md) for what those published frameworks contain.

### New York RAISE Act

- Signed in December 2025, then amended by a chapter amendment signed on **27 March 2026** that brought it closer to California's model. **Effective 1 January 2027.**
- Covers large frontier developers (over $500 million revenue, models above 10^26 operations).
- Requires published safety protocols and reporting of critical safety incidents within **72 hours** of determining one occurred (California allows 15 days), to a new office within the Department of Financial Services.
- Penalties of up to $1 million for a first violation and $3 million for subsequent ones.

### Colorado: a law delayed, then replaced

Colorado's SB 24-205, signed in May 2024, was the first broad US state law aimed at "algorithmic discrimination" by high-risk AI systems in decisions about jobs, housing, credit, healthcare and similar. Its history shows how unsettled this area is:

| Date | Event |
|---|---|
| May 2024 | SB 24-205 signed; original effective date 1 February 2026 |
| 28 Aug 2025 | Special session bill SB25B-004 delays it to 30 June 2026 |
| 9 Apr 2026 | xAI sues to block the law in federal court |
| 24 Apr 2026 | US Department of Justice intervenes in the suit |
| 27 Apr 2026 | A federal magistrate judge stays enforcement pending a ruling on a preliminary injunction |
| 14 May 2026 | Governor signs SB 26-189, repealing and replacing the law with a narrower "automated decision-making technology" framework |
| 1 Jan 2027 | SB 26-189 obligations begin: notice to consumers, a plain-language explanation within 30 days of an adverse decision, rights to correct data and request human review |

The Attorney General enforces the new law, with a 60-day opportunity to cure violations before 2030.

## Federal preemption: the unresolved question

Preemption means a federal law overriding state laws on the same subject. The arguments:

- **For preemption:** a patchwork of 50 state regimes raises compliance costs, especially for start-ups; AI models are deployed nationally and internationally, so a single rulebook fits the technology; and states regulating model development can effectively set national policy for everyone.
- **Against preemption:** states act as "laboratories of democracy" and are moving because Congress has not; preempting state law without a substantive federal replacement leaves consumers unprotected; and the existing state frontier laws (SB 53, RAISE) are mainly transparency rules with modest costs.

### Timeline

| Date | Event |
|---|---|
| May 2025 | House-passed budget reconciliation bill includes a 10-year moratorium on enforcing state AI laws |
| July 2025 | Senate votes 99-1 to strip the moratorium before final passage |
| 11 Dec 2025 | EO 14365 directs: an AI Litigation Task Force at the Justice Department to challenge state AI laws (for example on interstate commerce grounds); a Commerce Department review identifying "onerous" state laws; conditions on some federal broadband (BEAD) funding; FCC and FTC action on federal standards; and a legislative recommendation, with carve-outs for child safety, data-center permitting and state government use of AI |
| 9 Jan 2026 | Justice Department announces the AI Litigation Task Force |
| 20 Mar 2026 | White House releases legislative recommendations including preemption |
| Apr 2026 | Justice Department intervenes in xAI's challenge to Colorado's law |
| 4 Jun 2026 | Bipartisan House members (led by Reps. Jay Obernolte and Lori Trahan) release the "Great American Artificial Intelligence Act" as a discussion draft. It would preempt state laws specifically regulating AI model *development* for three years, with a sunset, while leaving state rules on AI *use and deployment* alone |

As of September 2026 the Great American AI Act remains a discussion draft, not an introduced bill, and no federal preemption statute has passed. An executive order cannot by itself void state laws, so the practical route for the administration is litigation, funding conditions and pressure on Congress.

## What to watch

- **The courts.** How judges rule in xAI's case (and any Task Force suits) will signal whether state AI laws survive constitutional challenge.
- **Congress.** Whether the Great American AI Act is formally introduced, and whether any preemption rides on must-pass legislation.
- **1 January 2027.** New York's RAISE Act and Colorado's SB 26-189 take effect on the same day.
- **SB 53 in practice.** The first published frameworks, transparency reports and incident reports under California's law, and whether the Attorney General brings any enforcement action.
- **The next administration.** Every federal action described here is an executive order and can be reversed by a future president, as EO 14110 was.

## Read next

- [The EU AI Act](./eu-ai-act.md) - the comprehensive, risk-tiered alternative
- [Frontier safety frameworks](./frontier-safety-frameworks.md) - what SB 53 and RAISE require developers to publish
- [Labs landscape](../ecosystem/labs-landscape.md) - who the "large frontier developers" are
- [Cost of training](../compute/cost-of-training.md) - context for the 10^26 FLOP threshold
- [Machines of Loving Grace](../../papers/essays/114-machines-of-loving-grace/summary.md) and [Situational Awareness](../../papers/essays/113-situational-awareness/summary.md) - two influential essays framing AI as a strategic race

## Sources

- Federal Register, EO 14179, "Removing Barriers to American Leadership in Artificial Intelligence" (signed 23 Jan 2025, published 31 Jan 2025): https://www.federalregister.gov/documents/2025/01/31/2025-02172/removing-barriers-to-american-leadership-in-artificial-intelligence
- White House (archived), fact sheet on EO 14110 (30 Oct 2023; Defense Production Act reporting, NIST red-team standards): https://bidenwhitehouse.archives.gov/briefing-room/statements-releases/2023/10/30/fact-sheet-president-biden-issues-executive-order-on-safe-secure-and-trustworthy-artificial-intelligence/
- White House, "White House Unveils America's AI Action Plan" (23 Jul 2025): https://www.whitehouse.gov/articles/2025/07/white-house-unveils-americas-ai-action-plan/
- White House, EO 14365 (11 Dec 2025): https://www.whitehouse.gov/presidential-actions/2025/12/eliminating-state-law-obstruction-of-national-artificial-intelligence-policy/
- Morgan Lewis, "White House AI Framework Puts Federal Preemption at the Center of the Debate" (March 2026, framework released 20 Mar 2026): https://www.morganlewis.com/pubs/2026/03/white-house-ai-framework-puts-federal-preemption-at-the-center-of-the-debate
- US Department of Justice, AI Litigation Task Force memorandum: https://www.justice.gov/ag/media/1422986/dl?inline= ; Baker Botts, "Inside the DOJ's New AI Litigation Task Force" (January 2026): https://www.bakerbotts.com/thought-leadership/publications/2026/january/inside-the-dojs-new-ai-litigation-task-force
- California SB 53 bill text: https://leginfo.legislature.ca.gov/faces/billTextClient.xhtml?bill_id=202520260SB53 ; Future of Privacy Forum explainer: https://fpf.org/blog/californias-sb-53-the-first-frontier-ai-law-explained/
- Wiley, "New York Finalizes RAISE Act for Frontier AI Models; Law Takes Effect January 1, 2027": https://www.wiley.law/alert-New-York-Finalizes-RAISE-Act-for-Frontier-AI-Models-Law-Takes-Effect-January-1-2027
- Colorado General Assembly, SB26-189 (signed 14 May 2026): https://leg.colorado.gov/bills/sb26-189
- Hunton, "Colorado AI Act Amended and Effective Date Delayed" (SB25B-004): https://www.hunton.com/privacy-and-cybersecurity-law-blog/colorado-ai-act-amended-and-effective-date-delayed
- Law and the Workplace (Proskauer), "Major Developments Put Colorado's AI Law on Ice Ahead of Implementation" (May 2026; xAI suit, DOJ intervention, stay): https://www.lawandtheworkplace.com/2026/05/major-developments-put-colorados-ai-law-on-ice-ahead-of-implementation/
- Roll Call, 4 Jun 2026, "Bipartisan AI draft proposes three-year preemption of state laws": https://rollcall.com/2026/06/04/bipartisan-ai-draft-proposes-three-year-preemption-of-state-laws/
- CASRAI, "Federal AI Preemption Fight: Where It Stands" (2025 moratorium and 99-1 Senate vote; status of the House draft as of September 2026): https://casrai.org/news/federal-ai-moratorium-state-preemption-fight-2026
