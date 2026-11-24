# Publishing the page (GitHub Pages)

The page is live at https://timforcejr.github.io/lost_in_space/ (repository `timforcejr/lost_in_space`, published with GitHub Pages through GitHub Actions).
The steps below record what was done and how to redeploy.

## What was ready

- `web/` is the whole site. The largest file is 96 MB (`web/model/mmini/onnx/model_quantized.onnx_data`), under GitHub's
  100 MB per-file limit. GitHub prints a warning for files over 50 MB; that is expected.
- `.gitignore` leaves out the full-precision export (470 MB), the Python environment, caches, test output, the v0 poster
  draft and `RUBRIC.md` (course material).
- `.github/workflows/pages.yml` publishes `web/` as the site root on every push to `main`.
- The page uses relative paths only, so it works under `https://timforcejr.github.io/lost_in_space/`.

## What was done

```bash
git init -b main
git add .
git commit -m "Lost in Space: in-browser explainer for multilingual sentence similarity"
gh repo create lost_in_space --source=. --remote=origin
git push -u origin main
```

Then, once, in the repository on github.com: Settings, Pages, Build and deployment, Source: GitHub Actions.

## Redeploy

Push to `main` with a change under `web/` or to the workflow file; the workflow runs on its own. To start it by hand:
Actions, "Deploy site to GitHub Pages", Run workflow.

```bash
node tests/smoke.mjs https://timforcejr.github.io/lost_in_space/ tests/out_pages     # must exit 0
node poster/make_qr.js                       # writes poster/fig/qr_live.png for the live URL
```

## Things to know

- GitHub Pages serves over HTTPS, so the browser caches the model after the first visit (about 170 MB in total:
  96 + 22 MB for the main model, a 9 MB tokenizer, and 43 MB for the gradient model, which loads only after the
  first result).
- The first workflow run, triggered by the push, fails at "configure-pages" if Pages is not yet set to "GitHub
  Actions". Set the source as described above, then re-run the workflow. One red run is expected.
- The browser caches the model files by address. If a model is ever re-exported after people have visited the page,
  rename its folder (for example `web/model/mmini` to `mmini2`, and the name in `web/js/engine.js`) so that visitors
  do not keep the old file. (How long the cache is kept was not tested.)
- The GitHub Actions versions in the workflow file were not checked against the current releases.
- Pages cannot send the headers that allow multi-threaded WebAssembly, so the model runs on one thread. The timings
  quoted on the poster were measured that way.
