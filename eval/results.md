# Evaluation results

```
Questions: 28 (21 answerable, 7 not covered by the contract)
Settings: top-k=4, relevance threshold=0.6

Retrieval (answerable questions):
  evidence in top-1 chunk:             17/21 (81%)
  evidence in top-4 chunks:             21/21 (100%)
  evidence kept after the threshold:   21/21 (100%)

Answers:
  answerable questions answered correctly:  21/21 (100%)
  uncovered questions correctly refused:    7/7 (100%)
    stopped by the relevance threshold:     1
    refused by the model:                   6
```

| # | Contract | Question | Evidence rank | Answer |
|---|---|---|---|---|
| 1 | employment_offer.pdf | What is the annual base salary? | 1 | correct |
| 2 | employment_offer.pdf | How often is the salary paid? | 1 | correct |
| 3 | employment_offer.pdf | What is the maximum annual bonus? | 1 | correct |
| 4 | employment_offer.pdf | By what date must the employee start? | 1 | correct |
| 5 | employment_offer.pdf | How many stock options may be granted? | 2 | correct |
| 6 | employment_offer.pdf | Which law governs this agreement? | 1 | correct |
| 7 | employment_offer.pdf | How much notice must the employee give before starting a new business activity? | 1 | correct |
| 8 | employment_offer.pdf | By how long can the non-competition restrictions be extended if the employee breaches a fiduciary duty? | 1 | correct |
| 9 | employment_offer.pdf | How much vacation or time off does the company provide? | 1 | correct |
| 10 | employment_offer.pdf | What background checks is the offer contingent on? | 1 | correct |
| 11 | employment_offer.pdf | What is the limit of liability? | not in contract | correctly refused |
| 12 | employment_offer.pdf | Is there an arbitration clause? | not in contract | correctly refused |
| 13 | employment_offer.pdf | What relocation assistance is provided? | not in contract | correctly refused |
| 14 | license.pdf | How long is the initial term of the agreement? | 1 | correct |
| 15 | license.pdf | How much notice is needed to stop the agreement from renewing automatically? | 1 | correct |
| 16 | license.pdf | How long does a party have to cure a material breach? | 4 | correct |
| 17 | license.pdf | What interest is charged on late payments? | 1 | correct |
| 18 | license.pdf | When is payment for support services due? | 2 | correct |
| 19 | license.pdf | Which state's law governs the agreement? | 1 | correct |
| 20 | license.pdf | How many days after termination must the licensee return the software? | 3 | correct |
| 21 | license.pdf | When is a mailed notice considered delivered? | 1 | correct |
| 22 | license.pdf | Do payment obligations get excused by force majeure? | 1 | correct |
| 23 | license.pdf | What is RTL not liable for? | 1 | correct |
| 24 | license.pdf | What happens if a bankruptcy petition is filed against a party? | 1 | correct |
| 25 | license.pdf | Is there an arbitration clause? | not in contract | correctly refused |
| 26 | license.pdf | What insurance must the licensee carry? | not in contract | correctly refused |
| 27 | license.pdf | Is the source code held in escrow? | not in contract | correctly refused |
| 28 | license.pdf | What is the capital of France? | not in contract | correctly refused |
