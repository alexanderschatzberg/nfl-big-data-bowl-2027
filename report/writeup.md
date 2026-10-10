# How well do Combine drills predict the way receivers move in NFL games?

## Summary

I used the Combine's 10 Hz player tracking to measure how hard wide receivers brake in the comeback and curl drills. I then measured braking the same way on their hitch routes in regular-season NFL games and compared the two.

- Receivers who brake harder in the drills also brake harder in games. The rank correlation is 0.48 across 54 receivers, and 0.49 after I account for 40 time, height, weight and alignment. It is positive in each of the three draft classes.
- The drills predict game braking almost as well as a receiver's own previous NFL season does (0.48 compared with 0.56).
- Braking does not predict separation. I wrote that test down and committed it before I ran it, and the correlation against man coverage came out at 0.07. Braking measured in games does not predict separation either.

My conclusion is that Combine tracking gives a reliable early description of how a receiver moves, which is useful for judging scheme fit, and that it should not be used to forecast how open he will get.

## What the 40 predicts

I started by checking what the 40-yard dash predicts for receivers once they reach the league. I compared each standard Combine measurement with regular-season outcomes for the 2023, 2024 and 2025 draft classes. Because slot and outside receivers have different jobs, I measured every outcome relative to the average for the receiver's usual alignment.

![40-yard dash time and four game outcomes for wide receivers](figures/fig1_stopwatch.png)

A faster 40 goes with a higher top speed in games (−0.55), and that relationship holds up after correcting for every test I ran on receivers. The 40 has much weaker relationships with separation (−0.18), targets per route (−0.12) and yards per route (−0.13). The measurement most related to separation is body weight, where heavier receivers separate less (−0.51). Draft slot predicts targets (−0.41) and yards per route (−0.40) but has little relationship with separation (+0.16).

The 40 measures straight-line speed, and that speed shows up in games. I wanted to know whether the rest of what the tracking sensors record shows up as well.

## Measuring the same thing in the drill and in the game

My first approach, comparing drill measurements with related game statistics, found nothing. The approach that worked was to measure one movement trait in exactly the same way in a drill and in the game situation that the drill rehearses.

In the drill, I take each 10 Hz trace and compute the receiver's hardest sustained deceleration over half a second. I use the median across his attempts and express it relative to his own draft class and drill, since drill setups change from year to year. I then average the comeback and curl drills.

In games, I compute the same quantity on every regular-season hitch route, starting at the snap. For each receiver I use his 90th-percentile play. Most hitch routes do not require a full-effort stop, so a receiver's typical play says little about how hard he is able to brake.

I report each correlation after removing the effects of 40 time, height, weight and alignment, so a result that survives reflects something the standard measurements miss.

I set up 29 of these drill-to-game pairs across receivers, offensive and defensive linemen, and defensive backs. Two of them survive false-discovery correction across all 29, and both involve receivers.

## Braking carries over to games

![Braking in Combine drills and braking in NFL games](figures/fig2_braking_transfer.png)

Drill braking and game braking correlate at 0.48, or 0.49 after the controls. The bootstrap 95% interval is 0.22 to 0.66, the permutation p-value is 0.0005, and the false-discovery q-value is 0.005. Within the 2023, 2024 and 2025 classes the correlations are 0.55, 0.45 and 0.43.

The second surviving pair is about speed. A receiver's top speed in the tracked 40 predicts his top speed on go routes at 0.44 after I remove the official 40 time (q = 0.009). The sensor's peak-speed reading therefore carries information that the official time leaves out.

To judge how strong 0.48 is, I compared it with the same measurement taken from the player's own previous NFL season.

![Predicting game movement from the Combine and from the previous season](figures/fig3_benchmark.png)

A season of NFL hitch routes predicts the next season's braking at 0.56. The two Combine drills predict it at 0.48, and the intervals overlap. For top speed the corresponding numbers are 0.67 and 0.55. In both cases the Combine gets most of the way to what a season of game tracking provides.

### Checking the result

I reran the braking comparison under different analysis choices and added three placebo comparisons that should show no relationship.

![The braking result under different analysis choices](figures/fig4_robustness.png)

The result is specific to matching drills and routes. Braking in the stop drills does not predict braking on go routes (0.04), and braking measured in the go-route drill (−0.25) or the 40-yard dash (−0.05) does not predict braking on hitch routes.

The result also holds across definitions. Three other ways of measuring braking, each drill on its own, and different minimum play counts all give correlations between 0.33 and 0.49.

It does depend on using a receiver's hardest plays. When I use his median hitch route, the correlation drops to 0.15.

The result has two limitations. A version for tight ends using a different drill did not replicate, although it only had 23 players. Other receiver traits that I expected to carry over, including re-acceleration out of the break and speed carried through cuts, did not.

## Braking and separation

I expected a receiver who stops faster to leave his defender further behind, so I tested whether braking predicts separation.

My first test pooled all hitch routes and found nothing. Against zone coverage the defender is already sitting on the route regardless of how the receiver stops, so any effect should be clearer against man coverage. I wrote out a single test in full, including the players, plays, outcome, statistic and expected direction, committed it to the repository, and only then ran it.

![Braking carries over to games and is unrelated to results on hitch routes](figures/fig5_chain.png)

The pre-specified test found no relationship. The correlation between drill braking and cushion-adjusted hitch separation against man coverage is 0.07 (p = 0.68, 41 receivers). Target rate and yards after catch over expected show no relationship either.

Braking measured in games does not predict hitch separation either, as the bottom section of the figure shows. I extended that check to eight in-game movement measures across four groups of routes, tested against both separation and yards per route. Of those 64 tests, two pass false-discovery correction. One rests on 17 players, and the other runs in the opposite direction to what I expected, with harder braking going with less separation. Separation itself is a stable trait for a receiver, with a split-half reliability of 0.69, so these results cannot be put down to a noisy outcome.

This suggests the Combine measurement is working as intended and that the weak link is between movement and separation. Separation probably depends on things this dataset does not include, such as the defender's leverage and technique and the timing of the route.

## Building a view of scheme fit and coverage matchups

Braking and top speed are the two traits that carry over to games, and they are almost unrelated to each other across receivers (−0.10, 91 receivers). Together they give a two-part movement profile that is available before the draft. A receiver can be fast with ordinary brakes or the reverse, and the official 40 time only describes the first part.

The most direct use of this profile is judging fit with a route tree. An offence built on timing throws to hitches, curls and comebacks asks its receivers to stop hard on a large share of their routes, and the drills show which prospects are able to do that. An offence built on vertical routes can give more weight to the sensor's top speed. A position coach can use the same reading to plan a rookie's development.

I also tested whether the profile says anything about how a receiver does against particular coverages. I compared braking and top speed with separation against man and zone overall, and against Cover 1, Cover 2, Cover 3 and Cover 4 or 6, which made 12 tests. None showed a relationship, and the largest correlation was 0.15. A receiver's separation against man and his separation against zone are only moderately related (0.41), so receivers do differ in which coverages they beat, but these two traits do not explain the difference. For now I would not use Combine movement to grade separation against any coverage.

Matchups against individual defensive backs are the natural next step, and this dataset cannot support that analysis because it does not record which defender covered which receiver. A team with tracking for every player could measure the cornerback's braking on the same plays and test whether the gap between the two players predicts separation better than the receiver's braking does alone. I think that is the most useful open question from this work.

## Everything else I tested

The braking result is one positive finding out of many attempts. The notebook reproduces 5,821 statistical tests, and I estimate that I ran several hundred more while developing the analysis.

| What I tested | Tests | Result |
|---|---|---|
| Standard Combine numbers and 11 planned interactions against outcomes, five positions | 518 | 40 time predicts game speed; receiver weight predicts separation |
| Matched drill-to-game comparisons, four positions | 29 | Receiver braking and top speed survive; nothing for linemen or defensive backs after controlling for size and alignment |
| Variations and placebos for the braking comparison | 17 | Stable; placebos show no relationship |
| Whether braking, top speed or any in-game movement predicts results, including by coverage | 84 | No relationship, including the pre-specified test |
| Whether drill traits predict route or alignment usage | 17 | No relationship |
| Every pairwise interaction of Combine numbers and drill traits | 4,589 | Two pass correction, and neither holds up on closer inspection |
| Whether a position improves from season 1 to season 2 or 3, and whether the Combine predicts who improves | 567 | Nothing passes correction |

One result from the last group is worth checking with the 2026 class. Defensive linemen who played no snaps in their second season had much lower burst in the Combine pass-rush drill than those who played (0.7 standard deviations lower, p = 0.007). This came from a scan of 538 tests and does not pass correction, so I report it only as something to follow up.

I also trained two neural networks to match players across drills and games. They did not predict any outcome, so I have left them to the notebook appendix.

## Limitations

- The samples are small. The main result rests on 54 receivers, and the interval around 0.49 runs from 0.22 to 0.66.
- There is no holdout class left. I used all three draft classes while searching, so the braking result should be confirmed on the 2026 class before anyone relies on it. The man-coverage test is the only one I specified in advance.
- The sample only includes receivers who ran at least 20 hitch routes in NFL games, so players who rarely played are missing.
- The game tracking covers only the drafted players, so separation and cushion come from the play-level columns provided with the data.

## Code

The attached notebook reproduces every number and figure in this writeup from the competition CSVs in about ten minutes.
