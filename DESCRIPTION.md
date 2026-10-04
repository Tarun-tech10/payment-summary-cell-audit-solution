# Payment Summary Cell Audit — solution description

Run: `python3 solution.py <public_dir> <submission_out.csv>`

## Compliance

The pipeline contains exactly two neural models — a layout network and a sequence reader —
both defined inside `solution.py`, both built with randomly initialised weights and fitted
during the run. The gradient-boosted rankers are fitted in the same run. Training data is the
529 provided images plus text images the script renders itself at execution time.

The script reads no model file, installs nothing, makes no network request, and imports only numpy,
pandas, opencv, torch, lightgbm, scikit-learn, PIL and matplotlib. Every answer is derived
from pixels; there is no lookup keyed on test id. The only assets read from disk are the
challenge's own images and the `.ttf` files already present in the local matplotlib
installation, used purely as letter shapes for drawing training text.

## Method

**1. Layout network — grid, count, flags.**
A convolutional network reads the whole photo at 448x320 with five input planes: RGB, an
illumination-normalised grey plane, and a binarised text-ink plane. The grid head is
*spatially aligned* — the feature map is average-pooled to exactly 4x4 and a 1x1 convolution
emits four class logits per cell, so cell (r,c) is predicted from the image region it actually
covers. Two global heads predict the item count and the four field flags. Trained 5-fold,
three configurations (70/130/130 epochs, two widths), with photometric and small-affine
augmentation only, since layout must not be mirrored.

**2. Strip extraction.**
The predicted grid supplies a region of interest. Inside it the page is deskewed by the
rotation that maximises the sharpness of the horizontal ink projection (searched over +/-8
degrees), binarised adaptively at a block size tied to the estimated glyph height, and grouped
into text lines. Each line emits several competing crops — the whole line, three word-gap
thresholds, and the right-hand remainder — because any single segmentation rule silently
loses the answer on blurred or tightly-set receipts. Multi-hypothesis extraction raised the
fraction of images in which the true total is readable at all from 0.32 to 0.48.

**3. Sequence reader.**
A convolutional-recurrent network with a connectionist-temporal-classification loss reads
every strip over a 76-symbol alphabet. It is fitted from random initialisation on procedurally
rendered strings — receipt vocabulary, random words, and money patterns — drawn in 176
distinct locally installed fonts and pushed through a degradation chain. The most important element of that chain is a
**resolution bottleneck**: render large, downsample to a random 7-30 px line height, then
upsample back to 32 px. Without it the model learns to read sharp text and fails on real
crops. Fonts are screened for glyph distinctness first, after one font was found to render
every digit as an identical box and silently poison the training targets.

**4. Answer selection, in two stages.**
Every decoded string and every numeric substring becomes a candidate, together with its
canonical regroupings: these receipts group thousands, so once the digits are correct the
printed form is recoverable deterministically (25000 -> 25.000 / 25,000). Selection then
splits in two, because *which digits the receipt shows* and *how they were printed* are
different questions. Stage A ranks digit strings, pooling all candidate rows that share a
digit string into one. Stage B picks the printed form among the survivors using the receipt's
own formatting habits. Stage B is trained on stage A's out-of-fold picks so it never learns to
trust an in-sample digit choice.

Stage A also sees **receipt-arithmetic features**. A receipt is a self-consistent ledger: the
total is the value that equals subtotal + tax + service, and the value that cash minus change
lands on. These relations hold between candidate *values*, so they are checkable without any
labels, and they identify the total when per-strip confidence cannot.

**5. Grid refinement.**
The layout network cannot read, but "subtotal area" (class 2) versus "final-total area"
(class 3) is a semantic distinction carried by words such as SUBTOTAL / PPN / SERVICE versus
TOTAL. A per-cell gradient-boosted classifier consumes the network's own class probabilities,
those of the four neighbouring cells, and fuzzy keyword evidence for text landing in that
cell. This targets exactly the class-2 confusion that dominates the macro-IoU loss and moves
the grid from 0.741 to 0.766 out-of-fold.

Two structural constraints are applied at decode, both read off the training labels: class 3
is present in 100% of training grids, and class 2 never occurs when no field flag is set.

## Cross-validated results (529 rows, 5-fold)

| component | weight | out-of-fold |
|---|---|---|
| semantic_grid | 0.45 | 0.7658 |
| total_price | 0.35 | 0.6029 (exact-string 0.5312) |
| num_items | 0.12 | 0.4896 |
| fields_present | 0.08 | 0.7482 |
| **row score** | | **0.6742** |

Best constant submission scores 0.333, so every component carries a real model.

## Where the remaining loss is

For `total_price`, selection is no longer the constraint: the correct string is among the
candidates for 0.6465 of images, and the two-stage selector returns it for 82% of those.
Candidate recall is the ceiling, and a per-image diagnostic shows why it is hard rather than
careless — measuring the best digit edit-distance between any reading and the gold gives 0.656
exact, 0.110 at distance 1, 0.102 at distance 2, and only 0.013 at distance 5 or beyond. The
answer is on the page and very nearly read almost every time; the reader is confidently wrong
by a digit rather than uncertain, which is why widening the search recovers nothing and the
remaining headroom sits in reader accuracy itself.

For `semantic_grid`, the metric averages IoU over every class present in either map, so a
class predicted that the gold lacks contributes a whole zero term — and class 2 appears in
only 37% of receipts. Oracle bounds: knowing true class presence gives 0.771, knowing the true
active set gives 0.826, and knowing both gives 0.864.
