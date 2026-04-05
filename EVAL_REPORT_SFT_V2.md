# MathGPT Evaluation Report

> Generated: 2026-04-05 08:32
> Source: sft | Model: d20 | Device: cuda
> GSM8K test problems: 100 | Samples/problem: 4 | Temperature: 1.0

## 1. GSM8K Pass@k Summary

| Step | Pass@1 | Pass@2 | Pass@4 | Solved (any) | Avg Rate |
|------|--------|--------|--------|-------------|----------|
| **2999** | 4.0% | 13.0% | 20.0% | 20/100 | 7.2% |

**Best Pass@1**: 4.0% at step 2999

## 2. Sample Outputs

### Step 2999

**Example 1** [CORRECT] (pass rate: 1/4)

**Q:** Indras has 6 letters in her name. Her sister's name has 4 more letters than half of the letters in Indras' name. How many letters are in Indras and her sister's names?

**Ground truth:** ...I = 6
Sister = 6/2 + 4 = 7
6 + 7 = 13 letters
Together Indras and her sister have 13 letters in their names.
#### 13

**Model output:**
```
Half of Indras' name is 6/2 = <|python_start|>6/2<|python_end|><|output_start|>3.0<|output_end|>3.0 letters.
 means Indras has 3.0 + 4 = <|python_start|>3.0+4<|python_end|><|output_start|>7.0<|output_end|>7.0 letters in her name.
Together, Indras and her sister's names have 7.0 + 6 = <|python_start|>7.0+6<|python_end|><|output_start|>13.0<|output_end|>13.0 letters.
#### 13
```

**Example 2** [CORRECT] (pass rate: 1/4)

**Q:** Gunther, the gorilla, had 48 bananas hidden under a fern branch.  When Gunther wasn't looking, Arnold, the chimpanzee, stole half of the bananas from the pile.  The next day, Gunther added another 25 bananas to his pile, but later that evening, Arnold stole another 12 of the bananas.  On the third d

**Ground truth:** ... brings the total to 24+25=49 bananas.
Arnold's second act of thievery resulted in 49-12=37 bananas remaining.
Gunther adds another 6 bananas, resulting in the final number of 37+6=43 bananas.
#### 43

**Model output:**
```
Arnold, the gorilla, had 48 bananas in his pile, so the chaos occurred 48 bananas / 4 = <|python_start|>48/4<|python_end|><|output_start|>12.0<|output_end|>12.0 bananas per group.
When Gunther wasn't looking, Arnold, the chimpanzee, stole 12 bananas / 2 = <|python_start|>12/2<|python_end|><|output_start|>6.0<|output_end|>6.0 bananas from the pile.
The next day, Gunther added 25 bananas + 6.0 banan
```

**Example 3** [WRONG] (pass rate: 0/4)

**Q:** Darrell and Allen's ages are in the ratio of 7:11. If their total age now is 162, calculate Allen's age 10 years from now.

**Ground truth:** ...18
Since the fraction of the ratio that represents Allen's age is 11/18, Allen's current age is 11/18*162 = 99
If Allen is currently 99 years old, in 10 years he will be 99+10 = 109 years old
#### 109

**Model output:**
```
Let's make the sum of their ages positive by using the formula for the sum of the first n terms of a finite sum: 

7 + (n+1) + (n+2)/2 - (n+1)n + n + (n+2)/2 + (n+3)/2 = 162

We also know that 7 + (n+2) is the sum of 7 and 11, making 7 + 11 = 162. Then 7 + 11 = 162 + 16 = 160.

So, Allen's age is 10 years from now.
```

**Example 4** [WRONG] (pass rate: 0/4)

**Q:** Lorraine and Colleen are trading stickers for buttons. Each large sticker is worth a large button or three small buttons. A small sticker is worth one small button. A large button is worth three small stickers. Lorraine starts with 30 small stickers and 40 large stickers. She trades 90% of her small

**Ground truth:** ...s because 100 - 50 = 50
She trades 20 large stickers for small buttons because 40 x .5 = 20
She gets 60 small buttons because 20 x 3 = 60
She has 89 buttons at the end because 9 + 20 + 60 = 89
#### 89

**Model output:**
```
Lorraine has 30 * 1.9 = <|python_start|>30*1.9<|python_end|><|output_start|>57.0<|output_end|>57.0 small stickers.
She also trades 5 * 0.5 = <|python_start|>5*0.5<|python_end|><|output_start|>2.5<|output_end|>2.5 large stickers for the small buttons.
She trades 0.5 * 50 = <|python_start|>0.5*50<|python_end|><|output_start|>25.0<|output_end|>25.0 for the large buttons.
She trades 1.5 * 100 = <|pyth
```

**Example 5** [WRONG] (pass rate: 0/4)

**Q:** Bethany can run 10 laps on the track in one hour.  Trey can run 4 more laps than Bethany.  Shaelyn can run half as many laps as Trey.  Quinn can run 2 fewer laps than Shaelyn.  How many more laps can Bethany run compared to Quinn?

**Ground truth:** ...Trey can run 10 + 4 = 14 laps in one hour.
Shaelyn can run 14 / 2 = 7 laps in one hour.
Quinn can run 7 - 2 = 5 laps.
Bethany can run 10 - 5 = 5 more laps compared to Quinn.
#### 5

**Model output:**
```
Bethany can run 10 laps in one hour and Trey can run 4 more laps than Bethany, so Trey can run 10+4 = <|python_start|>10+4<|python_end|><|output_start|>14<|output_end|>14 laps.
Shaelyn can run half as many laps as Trey, so Shaelyn can run 14*2 = <|python_start|>14*2<|python_end|><|output_start|>28<|output_end|>28 laps.
Quinn can run 2 fewer laps than Shaelyn, so Quinn can run 28-2 = <|python_start
```

## 3. Analysis

- **Solved** (at least 1/4 correct): 20/100 (20.0%)
- **Unsolved** (0/4 correct): 80/100 (80.0%)
- Large unsolved fraction with 4 samples indicates many problems are beyond the model's current capability

---
*Report generated by `python3 -m scripts.eval_report`*