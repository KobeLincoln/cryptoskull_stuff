# CryptoSkull... Stuff

Check out the .ipynb file to see how stuff works. Brief overview of the stuff:

### Mosaics

Create mosaics using the 24x24 cryptoskull images. Convenience methods for skulls as well as custom images that are upscaled by 24x. Option to turn anything into a flashy gif too!

### Assembly

Reassemble the existing skulls or clone an art folder (e.g. cryptoskulls_24) and customize a new art style for the skulls. Now you can recreate any or all skulls with your new custom art.

### Rarity (WIP)

Note: This section is a WIP and is completely naive and subjective.

**WARNING. This calculation method ALMOST CERTAINLY makes incorrect assumptions and should not be used to make any purchasing or selling decisions for anything ever forever-ever. This is not financial advice.**

Traditional rarity tools do a poor job with skulls for the following reasons:

* They don't consider game traits like intelligence and speed.
* They don't consider game skills like Archmage and Fire Lord.
* They treat uniqueness index like any other trait - which it is not. This leads to rewarding mediocre UIs just because there are few skulls with them.
* They don't consider the shape style of hair, just the genes. This unfairly treats the uncommon hair shape found in hair gene #32 (extra long mohawk) just like any other hair shape.
* They don't consider HIDDEN rarities where trait colors exactly match. There are even 30 triple matches and 10 rare double-double matches.

This rarity calculation tries to come up with a better composite score for rarity. It uses:

1) Average Jaccard distance for the basic aesthetic traits.  
2) Multiplicative score for numerical game traits. I.e. intelligence x ... x dexterity. This is based on the ASSUMPTION that the utility of these traits in-game will be compounding as they go up.  
3) Multiplicative score for game skill rarity. Same sort of assumption as above.  
4) Traditional rarity calculation for hidden rarities based on the "poker hand" (pair, 2-pair & 3-kind) frequency.  

These 4 scores are then linearly normalized to [0,1] and added together to create the rarity score.

This is worth repeating:

**WARNING. This calculation method ALMOST CERTAINLY makes incorrect assumptions and should not be used to make any purchasing or selling decisions for anything ever forever-ever. This is not financial advice.**
