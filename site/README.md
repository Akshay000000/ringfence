# RingFence showcase site

Two static pages and one data bundle. No backend, no build step, no runtime.

```
site/
  index.html                 the showcase page
  console.html               the analyst console, generated from ringfence/api/console.html
  data/console-data.js       baked API responses (~2.1 MB)
  favicon.svg                the tab icon, plus favicon-32.png and apple-touch-icon.png
  assets/og.png              the 1200x630 social card
  assets/og-template.html    the card's source, rendered by tools/render_og.py
  assets/console-*.png       console screenshots used on the showcase page
```

## The social card

`assets/og.png` is what Slack, WhatsApp and most forms show when the link is
pasted. It is a rendered PNG, but the source of truth is the HTML beside it, so
changing the headline is an edit rather than a rebuild from scratch:

```bash
python tools/render_og.py
```

The `og:` and `twitter:` tags on both pages point at absolute URLs. They have to:
an unfurler fetches the image from its own servers, where a relative path
resolves to nothing. If the site ever moves off `ringfence-razor.vercel.app`,
those tags and `tools/render_og.py` are the two places the domain is written down.

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
