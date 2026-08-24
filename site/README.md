# RingFence showcase site

Two static pages and one data bundle. No backend, no build step, no runtime.

```
site/
  index.html            the showcase page
  console.html          the analyst console, generated from ringfence/api/console.html
  data/console-data.js  baked API responses (~1.9 MB)
```

## Regenerating

```bash
python -m ringfence.cli site
```

That reruns the scoring service in-process, bakes the alert queue, reasons,
what-ifs and evidence subgraphs into `data/console-data.js`, and regenerates
`console.html` from the canonical console with a script tag injected. Do not edit
`site/console.html` by hand; edit `ringfence/api/console.html` and rebuild.

The console reads from the bundle when one is present and falls back to the live
API when it is not, so the same page serves both `ringfence.cli serve` and a CDN.

## Deploying

**Vercel.** Point the project at this repo, set the output directory to `site`,
leave the build command empty. `vercel.json` at the repo root handles caching.

**Render.** New Static Site, publish directory `site`, no build command.

Everything here is derived from the synthetic corpus. No real payment data,
and no IEEE-CIS data, is included.
