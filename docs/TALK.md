# Lost in Space: 5-minute talk

Draft talk script, shot list and question prep, built from the logged results in `docs/RESULTS.md`. Rewrite it in your own voice before presenting.

591 spoken words (the quoted script lines only, not the Questions section, shot list or comments), which is about 4 minutes at 150 words a minute, leaving time for demo clicks. Each
section says what to show. Numbers come from `docs/RESULTS.md` and `README.md` ("What we found"); if a run is repeated,
check them against those files.

The first paragraph of section 2 is the boundary statement, word for word the same as on poster v3, in the README and
on the web page.

## Script

### 1. The hook (0:00 to 0:25) · show the page, "Word order" example

> "The dog bit the man." "The man bit the dog." Different stories. A multilingual AI model scores them 0.98 alike, out
> of 1.
>
> Models like this one sit under search engines and chatbots. They turn every sentence into a point in space and
> compare points. You get one number and no reason. My project, Lost in Space, asks: which words drove that number,
> and which explanation of it should we believe?

### 2. Builds on, what is new, what it is (0:25 to 1:15) · type an English and a Spanish sentence, rotate the map


> Lost in Space builds on the NLP project, a group class project with Ananya Singh and Cherry Lian that asked whether you can visualize how a model understands the feeling of text across languages. New here: Runs in your browser; nothing leaves your device. XAI tools: attribution, maps, layer lens, faithfulness and sanity tests. Visualization to build understanding of how an AI learns meaning.
> The NLP project found the similarity score does not catch real translation errors, and its common-space idea
> came from a brain-decoding paper by d'Ascoli and colleagues.
>
> A twelve-layer transformer runs inside the tab, so nothing you type leaves your machine. Type two sentences in
> English, Spanish, Hindi or Chinese, and they land in a map of four thousand reference sentences. The map is a sketch:
> squeezed from 384 dimensions to three, a new sentence keeps only five to eight percent of its true neighbors.

### 3. Four explanations (1:15 to 1:35) · click through the four tabs

> Occlusion deletes a word and watches the score fall. The pair split divides
> the score exactly among word pairs. Integrated gradients uses real gradients; the browser runtime has
> none, so I wrote the backward pass by hand, and it matches PyTorch to 0.2 percent. Attention rollout follows the
> attention maps. The four tell different stories. Which one is right?

### 4. The test (1:50 to 2:55) · scroll to "Can we trust the explanation?", then "Break the model: does the explanation change?"

> Delete words in the order a method ranks them and re-run the model. If the method is right, losing its top words
> should hurt most. Over 1,224 explanations, occlusion leads with a gap of 0.45, then integrated
> gradients and attention at 0.33 and 0.32, and the pair split 0.20. Ranking words by length alone gets
> 0.13: that is the bar.
>
> Then I tested the test. It removes words, and so does occlusion. In an offline run that masks words instead, the
> ranking stays the same and the gaps shrink: occlusion 0.32 against 0.45. With eight of twelve layers replaced by random
> numbers, attention rollout's ranking barely changes, 0.97 agreement: a warning sign. No method wins everywhere, so
> the page puts a live deletion test next to every explanation.

### 5. Word order (2:55 to 3:50) · "Does word order matter to the model?", press "Scramble again", then "Can we trust the map?"

> I scrambled every word of 816 sentences, and the vector stayed 0.91 similar to the
> original. In the last layers the language of a sentence also becomes harder to read: a simple language guesser falls
> to 40 percent at layer 12, against 25 percent chance. Weakened, not removed. The header dropdown loads a different
> model trained only on masked words, where the guesser stays at 100 percent: the languages stay apart. And on
> re-ordered paraphrase pairs, the score barely beats a coin flip.

### 6. Limits and close (3:50 to 4:30) · the footer line and the "Using this responsibly" section of the About page (footer link), then the page title with the link

> What this does not show. One small model, news-style text, four languages. In the browser it runs with 8-bit
> weights, and that rounding can change the explanation of two nearly identical sentences even when the score hardly
> moves. The Hindi and Chinese examples are not checked by a native speaker, and without a user study I cannot say it
> changes how anyone trusts a score.
>
> A similarity score is not a verdict on meaning, translation or truth, and a neat explanation makes a wrong score
> easier to believe. So read the test before you believe the picture. Try it at timforcejr.github.io/lost_in_space.
> Thank you.

## Shot list for a recorded video

| Time | Screen | Note |
|---|---|---|
| 0:00 | Page loaded, click the "Word order" example | Let the 0.98 appear before speaking the number |
| 0:25 | Click "EN / ES", drag the map to rotate it | Point at the two labeled points and the lit neighbors; point at the caption under the map |
| 1:15 | Click the four method tabs in turn | Pause one second on each so the changing colors are visible |
| 1:50 | Scroll to "Can we trust the explanation?" | Curves first, then the bar chart; click through the language buttons |
| 2:30 | "Break the model: does the explanation change?" | Sanity chart: point at the flat attention line |
| 2:55 | Click "Word order" again, scroll to the word-order card, press "Scramble again" twice | Read one scrambled sentence aloud; then the "Where the shared space forms" chart under "Can we trust the map?" |
| 3:50 | Footer link "Using this responsibly" (opens the About page, section "Using this responsibly") | Scroll to the footer first; the card is no longer on the main page |
| 4:20 | Page title with the link | Say the URL slowly |

Record at 1440 × 900 or larger with browser zoom at 110%. Load the page once before recording so both models are cached.
On a laptop the first gradient run can take several seconds; the status pill at the top right shows when it is done.

## Questions to expect

- **Why is occlusion best? Is that not circular?** Partly, yes: a removal method is graded by a removal test. Its lead
  over integrated gradients is 0.11 when the test deletes words. In an offline test that masks words the ranking stays
  the same and the gaps are smaller (occlusion 0.319 against 0.448).
- **Is the pair split your idea?** No. It is the identity in Lee et al. (ACL 2022), who argue it explains poorly. The
  deletion test agrees.
- **What is actually new?** Over the NLP project, the group class project this builds on: it already had a
  local 3D view, checked only on synthetic errors and run through a local server. New here are a
  page that runs entirely in the browser and the tested XAI tools: four methods on one score, graded by deletion tests,
  a sanity check, plus map fidelity and a layer lens. Against the literature, as far as I could find: attribution of a
  similarity score fully in the browser, and a deletion test of such attributions per language including Hindi. I searched and found none, which is not proof that
  none exist. Exporting the backward pass as a plain graph is engineering; I do not claim it is new.
- **Does the word-order result show the model uses word order?** No. On PAWS-X pairs the score barely separates
  paraphrases from re-ordered non-paraphrases (AUC 0.57), and an offline linear probe on the two vectors does no better
  (0.54; not on the page).
- **Why trust the deletion test?** It has known flaws: sentences with words removed are unlike the training data. That
  is why I also ran an offline masking test and a randomized-model check.
- **Why this model?** It is small enough for a browser and uses mean pooling, which the pair split needs. The header dropdown also loads a masked-word model of the same size, trained only on masked words, never on translations (Microsoft Multilingual-MiniLM, MIT), so you can see what translation training adds. The offline benchmark (612 translation pairs, 1,224 explanations) and the sanity check were measured on the default model only. The map's trust numbers, the deletion curve for your sentences and the layer lens are computed for the loaded model.
- **Does it work in other languages?** The model's card lists about 50. Only four were measured, and they were not
  equal: Chinese had the lowest faithfulness for most methods, but Chinese "words" are single tokenizer pieces, so its
  word-length baseline is 0.03 against 0.13 to 0.20 for the others; compare each language with its own baseline. Every
  tested pair had English on one side.
- **Did you build a common space across languages?** No. Future work: a model trained on each language's text, with separate input layers into one shared space, to test whether untranslatable meaning shows up as distance. Not built; the page reads one frozen model and measures how shared its space already is.
- **Is the map wrong?** It is lossy. That is measured and shown; scores and neighbors never use it.
