# Predicting How Receivers Move: Combine Drills Against a Season of NFL Tracking

## Summary

The Combine comeback and curl drills predict how hard a wide receiver brakes on hitch routes in NFL games, with Spearman ρ = 0.42 against his braking pooled over his NFL seasons. A season of his own NFL tracking is a better guide. Predicting the same NFL seasons, the drills reach ρ = 0.33 and the previous season ρ = 0.56. For top speed, the tracked 40-yard dash reaches 0.47 and a previous season 0.67. The drills give a team a partial read on how a receiver moves before he plays a snap, and once a season of game tracking exists they add nothing to it.

- In both drills the receiver runs upfield, stops hard and comes back toward the ball, as on a hitch route. I measure braking from the Combine's 10 Hz tracking in the drills, and in the same way on hitch routes in regular-season games.
- Receivers who brake harder in the drills also brake harder in games. Spearman ρ = 0.42 across 54 receivers, and partial ρ = 0.43 controlling for 40 time, height, weight and alignment.
- Braking does not predict separation, in a test I committed before running it (0.10 against man coverage), and neither does braking measured in games.

I would use the drills as an early read on how a receiver moves, replaced by game tracking once he plays. I would not use them to forecast how open or productive he will be.

## Measuring braking the same way in the drill and in the game

My first approach, comparing drill measurements with related game statistics, found nothing. The approach that worked was to measure one movement trait in exactly the same way in a drill and in the game situation that the drill rehearses.

In the drill, I take each 10 Hz trace and compute the receiver's hardest sustained deceleration over half a second. I use the median across his attempts and express it relative to his own draft class and drill, since drill setups change from year to year. I then average the comeback and curl drills. I drop clips that last less than 65% of the usual time for their drill, which removes 29 of 6,307, because they end before the receiver has stopped.

In games, I compute the same quantity on every regular-season hitch route, starting at the snap, for the 2023, 2024 and 2025 draft classes. For each receiver I use his 90th-percentile play, because most hitch routes do not call for a full stop.

![Braking, measured the same way in a drill and in a game](figures/fig6_traces.png)

The figure shows the measurement for two receivers from the 2023 class. Marvin Mims ranks 6th of 54 for braking in the drills and 9th in games, and Puka Nacua ranks 38th and 44th. Each game route shown is the one closest to that receiver's 90th percentile.

Correlations are partial Spearman correlations controlling for 40 time, height, weight and alignment (slot or outside).

I set up 29 of these drill-to-game pairs across receivers, offensive and defensive linemen, and defensive backs. Two survive Benjamini–Hochberg correction across all 29 (q < 0.05), and both involve receivers. They are braking and top speed, where a receiver's peak speed in the tracked 40 is paired with his top speed on go routes.

## The Combine compared with a season of NFL tracking

To judge how strong these relationships are, I compared each with the same measurement from the receiver's previous NFL season. Both predict the same target, his braking or top speed in one NFL season, across 47 pairs of consecutive seasons from 33 receivers (48 from 34 for top speed).

![Predicting game movement from the Combine and from the previous season](figures/fig3_benchmark.png)

The previous season predicts braking at 0.56 and the Combine drills at 0.33. For top speed the numbers are 0.67 and 0.47. In R² terms, the Combine gives about a third of what a season gives for braking (0.35) and about half for top speed (0.48). The 95% CIs overlap, as only 33 receivers have two seasons. Controlling for the previous season, the drills add nothing (partial ρ = 0.10, p = 0.58).

Part of the gap is noise in the drills. The comeback and curl drills correlate at only 0.26 (Spearman–Brown reliability 0.41), so more repetitions per drill would likely sharpen the read.

## Braking carries over receiver by receiver

![Braking in Combine drills and braking in NFL games](figures/fig2_braking_transfer.png)

ρ = 0.42 and partial ρ = 0.43 (95% bootstrap CI 0.18 to 0.63, permutation p = 0.005, q = 0.018). Within each draft class ρ is 0.43 to 0.45.

The figure names the five hardest and five softest brakers in the drills. All five of the hardest are in the top half for game braking, and three of the five softest are in the bottom half. The relationship has clear misses. Rashee Rice ranked 7th in the drills and 48th in games across 141 hitch routes. Brian Thomas had the softest drill braking by a wide margin and sits near the middle in games.

### Checking the result

I reran the comparison under other analysis choices and three placebos.

![The braking result under different analysis choices](figures/fig4_robustness.png)

The result is specific to matching drills and routes. Braking in the stop drills does not predict braking on go routes (0.07). On hitch routes, braking in the 40-yard dash shows nothing (−0.05) and braking in the go-route drill runs the other way (−0.25).

The result holds across definitions. Three other ways of measuring braking, each drill on its own, and different minimum play counts all give correlations between 0.33 and 0.43. If I keep the clips that were cut off early, the correlation is 0.49, so those clips had been flattering the result. Harder game braking could simply mean arriving at the break faster, so I also controlled for each receiver's typical top speed on hitch routes. The correlation rises to 0.45.

It does depend on using a receiver's hardest plays. I set the 90th percentile before seeing any results. Across the 75th to 95th percentiles the correlation averages 0.34, and with his median hitch route it drops to 0.18 (p = 0.19). Some of the 29 drill-to-game pairs were added after earlier versions found nothing, so the q-value understates how much I searched. The 2026 class is the real test.

## What the sensors add to the stopwatch

I also checked what the standard Combine numbers predict for receivers. Because slot and outside receivers have different jobs, I measured every outcome relative to the average for the receiver's usual alignment.

![40-yard dash time and four game outcomes for wide receivers](figures/fig1_stopwatch.png)

A faster 40 goes with a higher top speed in games (−0.55), and survives family-wise correction across all receiver tests (family-wise p = 0.002). The 40 barely relates to separation, targets per route or yards per route (−0.12 to −0.18). The measurement most related to separation is body weight, where heavier receivers separate less (−0.51, family-wise p = 0.004).

The sensors add to the official time in two ways. The tracked 40's peak speed predicts top speed on go routes with the official 40 time among the controls (partial ρ = 0.44, q = 0.017), so it carries information the official time leaves out. Braking is almost unrelated to top speed across receivers (−0.13, 91 receivers), so it describes a second part of how a receiver moves that the 40 does not cover. It is also unrelated to the two stopwatch drills that involve stopping, the 3-cone (−0.03, 32 receivers) and the short shuttle (−0.08, 36). Too few of the 54 ran those drills to use them as controls.

## What the measurements should and should not be used for

The drills measure how a receiver moves, not how good he is. Puka Nacua ranks 38th of 54 in drill braking and 44th in games, and is one of the most productive receivers in his class. A timing offence built on hitches, curls and comebacks may still want prospects who stop hard, but whether that improves its results is a hypothesis for the 2026 class, not a finding.

I expected a receiver who stops faster to leave his defender further behind, so I also tested whether braking predicts separation. Pooled over all hitch routes, I found nothing. Against zone the defender is already sitting on the route, so any effect should be clearer against man coverage. I wrote out that single test in full, committed it, and only then ran it.

![Braking carries over to games but does not improve results on hitch routes](figures/fig5_chain.png)

The pre-specified test was null: partial ρ = 0.10 between drill braking and cushion-adjusted hitch separation against man coverage (p = 0.52, 41 receivers). Target rate and yards after catch over expected show no relationship either.

Game braking does not predict hitch separation either (bottom of the figure). I extended that check to eight in-game movement measures across four groups of routes, tested against both separation and yards per route. Of those 64 tests, one passes Benjamini–Hochberg correction at q < 0.05, and it rests on 17 players. One more passes at q < 0.10, and it runs opposite to what I expected: receivers who brake harder in games separate slightly less across all routes (−0.34, 77 receivers). The signs against zone coverage in the figure point the same way. One possible reason, which I have not tested, is that hard stoppers run hitches that zone defenders sit under. Separation is a stable trait for a receiver, with a split-half reliability of 0.69, so these results cannot be put down to a noisy outcome.

Splitting separation on all routes by coverage (man, zone, and Cover 1 to Cover 6) showed nothing for braking or top speed in 12 tests (largest |partial ρ| 0.15).

The Combine measurement is therefore doing its job, and the weak link is between movement and separation. Separation probably depends on things this dataset does not include, such as the defender's leverage and technique and the timing of the route. A team with tracking for all 22 players could test whether the gap between receiver and cornerback braking predicts separation. I think that is the most useful open question here.

## Everything else I tested

The braking result is one positive finding out of many attempts. The notebook reproduces 5,826 statistical tests, and I estimate that I ran several hundred more while developing the analysis.

| What I tested | Tests | Result |
|---|---|---|
| Standard Combine numbers and 11 planned interactions, five positions | 518 | 40 time predicts game speed; receiver weight predicts separation |
| Matched drill-to-game comparisons, four positions | 29 | Receiver braking and top speed survive; nothing for linemen or defensive backs |
| Variations and placebos for the braking comparison | 19 | Stable; placebos null |
| Whether braking, top speed or any in-game movement predicts results, including by coverage | 84 | No positive relationship, including the pre-specified test |
| Whether drill traits predict route or alignment usage | 17 | No relationship |
| Every pairwise interaction of Combine numbers and drill traits | 4,589 | Two pass correction; neither holds up on inspection |
| Season-to-season improvement, and whether the Combine predicts it | 567 | Nothing passes correction |

I also trained two neural networks to match players across drills and games. They did not predict any outcome and are in the notebook appendix.

## Limitations

- The samples are small. The main result rests on 54 receivers.
- There is no holdout class left. I used all three draft classes while searching, so the braking result should be confirmed on the 2026 class before anyone relies on it. The man-coverage test is the only one I specified in advance.
- The sample only includes receivers who ran at least 20 hitch routes in NFL games, so players who rarely played are missing.
- The game tracking covers only the drafted players, so separation and cushion come from the play-level columns provided with the data.

## Code

The attached notebook reproduces every number and figure in this writeup from the competition CSVs in about ten minutes.
