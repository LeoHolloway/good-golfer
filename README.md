# good-golfer

I am interested in what makes a good PGA Tour golfer. This Kaggle data has info about people who played PGA Tour events from 2015-2022, mostly related to performance in each tournament.

## Background

What makes a good PGA tour golfer has been very widely studied, but not always from a quantitative angle. A few of the books I've read about it are "The 8 Traits of Champion Golfers" (Graham, Stabler), "Golf Is Not a Game of Perfect" (Rotella), and "Dave Pelz's Short Game Bible" (Pelz). The first two books have a psychological focus, but Dave Pelz took a quantitative approach and made some very interesting discoveries.

If you play golf, I would highly recommend the entire book. Even if you don't care about golf, I would highly recommend reading chapters 1-3 and skimming the rest, reading any time you see numbers or charts. I have a huge amount of respect for Pelz, who put a massive amount of effort into collecting high-quality data, and made findings and tools that are now staples (2 ball putter, the 60 degree wedge, and the notion that short game disproportionately affects scoring). Other quantitative studies had been done before("The Search for the Perfect Swing", Cochran and Stobbs, 1968), but they were more limited and didn't really ask the right questions. I will summarize some pertinent information. 

Pelz was a college golfer, but not good enough to earn a living from it. He studied physics and eventually became a PI at NASA's Goddard Flight Center; most of his research focused on atmospheric studies of Earth and other planets. He left the job to try coaching pro golfers, which he did very successfully, with his students winning a total of 21 majors, including as Phil Mickelson, Tom Kite, Vijay Singh, and Lee Janzen.

Pelz collected a huge amount of data by caddying for tour players and recording every shot they made. When he made a name for himself and started coaching players, he also collected tons of their data in more controlled environments. I've inquired after these datasets, and will update this if I get a response.

Pelz's central finding is that "...short-game performance is more important than either putting or the power game in determining the score a player shoots and how much money he makes on the PGA Tour. Later, I would learn that the same is true for amateurs..." (Pelz 23). By power game, he roughly means full swing shots outside of 100 yards.

https://www.kaggle.com/datasets/robikscube/pga-tour-golf-data-20152022

## Golf performance metrics for non-golfers
Golf is a complicated game, so comparing how different players perform in different ways is a difficult task. Baseball is notorious for having relatively complicated statistics to quantify how good players are; this approach is newer to golf mostly because the quality of any one shot is particular to the player, hole, weather, and individual performance that day. Most of the statistics that are obvious to create don't really measure how good a shot is, but rather whether it had certain qualities that make them, on average, better than other shots.

The weather can have an enormous impact on how good shots are. You can really only compare the old-fashioned stats with those from other players on the same day. For instance, in terrible weather, playoff holes can sometimes be won by a player who comes in 5 or 7 over par; it just matters whether you do better than everyone else.

If you do play golf and are surprised by PGA tour stats being so low, you should remember that they only play the most difficult courses in the country. When they play normal courses they 

Old-fashioned golf stats:
- Driving distance: average distance off the tee. Although it's one of the most obvious statistics to collect, it really doesn't mean too much at a high level of play(and it only really matters at lower levels because it helps to measure the number of terrible first shots). For example, John Daly led the tour for 11 years in driving distance, but he did not have a stellar career(if you're interested in how this happened, read a little about him on Wikipedia. He is often said to have the greatest wasted talent in golf by a large margin, probably due to substance abuse and maybe a lack of effort).
- Putts: putts in a round. Lower is better generally but it can be misleading; if you have horrible approaches the whole round, you'll chip on and 1 putt a ton, which could give you super low putts.
- GIR: you got on the green in (par for this hole - 2) strokes. Better stat than putts, but occasionally misleading; sometimes(especially on par 3s with huge greens) it's better to miss the green in one particular area than to hit it in another.
- Fairways hit: whether your first shot came to rest in the fairway. Better than Putts and GIR, since you almost always want to hit the fairway. However, still very reductive; some players have good performance off the tee even when their fairways hit is poor(e.g. Bryson DeChambeau, who hits it farther enough than others on average that the loss in accuracy is justified). 
- Up and down %: percentage of times you make par when you don't get GIR. Probably the best stat on this list, since you always want to make par or better, although it's a little niche. Hypothetically if you had an incredible round and hit 17 GIR

Newer stats:
- Strokes gained: the king of golf statistics, invented by Columbia professor Mark Broadie in 2005. It isn't perfect, but there's really nothing better available currently. The SG(strokes gained) on any particular shot is the number of strokes, on average, you saved in comparison to the average PGA tour player. Note that a greater number is better, unlike raw score or handicap.
Imagine you got a hole in one on a hole where the mean score is 3.0. Then, you saved two strokes: SG=+2.0. Now, imagine the mean score on a different hole is 4.2. You hit a terrible drive to a position where, on average, it takes 3.9 strokes to finish the hole. Your strokes gained is 4.2(avg from former position) - 3.9(avg from curr position) - 1(the stroke you took) = -0.7. You are 0.7 strokes worse off than the average PGA Tour player would be if they took the same shot.

It's not too important to this project, but you can lose more than one stroke on a shot. Imagine if you blew a 20-foot putt by the hole and it wound up in the bunker.

To check your understanding, imagine what the strokes gained would be like for for whiffing a shot(which still counts as a stroke). Now, for making a putt from 1 foot. Now, for missing a putt from 1 foot, and it ends up 1 foot from the hole on the same incline.

A: Exactly -1.0, since you're in the same exact position but lost a stroke; a very small positive number like +0.01, since there was a tiny chance you missed it; a negative number just above -1, like -0.99, since it was very likely but not guaranteed that you made it.

Strokes gained is very powerful, and now most Vegas odds are informed heavily by strokes gained. 

## What are we trying to measure?

### pca.py
We are going to see whether we can get any predictive power out of the statistics provided in PCA. This was inspired by an example in my Risk Management class, where PCA on a yield curve actually yields interpretable results (level, slope, and curvature). This is a pretty well known result but it was the first time it was explained to me that way. I was very impressed by the recovery of intelligible principal components.

When I was thinking about golf some other time, I guessed that the principal components of being good at golf (as a function of goodness in different areas of play) were probably a level, representing generally how good you are, and 

Game plan:
- compute average strokes gained over the entire period for each golfer. Using strokes gained from each
- Try to predict # of wins, average finish, and winnings. For most events winnings follows a given proportion of purse (e.g. 1st gets 18%, 2nd 10.8%, ties get average between the several positions they cover)
- Mean winnings per event is a better target than the season total, because the total partly measures how many events someone played




## Considerations for improvement:
- Winnings tables are probably different year over year. Shouldn't be too hard to find old ones
- Newer data could be interesting
- Treat winnings properly
- Try to normalize winnings as a proportion of all prize money