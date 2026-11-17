/*
 * WikiVisage Service Worker
 *
 * Strategy:
 *   - Scripts (/static/*.js): network-first, cache only as a fallback when the network fails or
 *     answers 5xx, so a page and the shared script it calls always come from the same release
 *   - Other static assets (icons, logos): stale-while-revalidate
 *   - All other requests (routes, API): network-only (app requires auth + live data)
 *
 * This SW exists primarily to enable PWA install (Add to Home Screen)
 * and cache static assets for faster repeat loads. The app is not
 * designed for offline use (requires DB, OAuth, Commons API).
 */

/* Bump this whenever the caching rules below change: activate deletes every other
 * cache, so existing clients drop their stale copies instead of keeping them. */
const CACHE_NAME = "wikivisage-v3";

/* Not cached up front: shared scripts must always match the page that loads them. */
const PRECACHE_URLS = [
  "/static/wikivisage-logo.svg",
  "/static/wikivisage-logo-notext.svg",
  "/static/powered-by-toolforge.svg",
  "/static/view-it-tool.png",
  "/static/icon-192.png",
  "/static/icon-512.png"
];

/* Install — precache static assets */
self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(PRECACHE_URLS))
  );
  event.waitUntil(self.skipWaiting());
});

/* Activate — clean up old caches */
self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((names) =>
      Promise.all(
        names
          .filter((name) => name !== CACHE_NAME)
          .map((name) => caches.delete(name))
      )
    )
  );
  self.clients.claim();
});

/* Store the fresh copy of a script and drop the other cached versions of the same file, as
   one serialized step per pathname. Two tabs can ask for two releases at the same moment:
   without serializing, both `put()` calls finish before either `keys()` call, so each job
   deletes the copy the other just stored and the cache is left empty — a later 5xx or an
   offline load then has no fallback at all. A failed write must not block the next one, so
   each link of the chain swallows its own error. */
const scriptWrites = new Map();

function storeScriptAndPrune(pathname, request, body) {
  const previous = scriptWrites.get(pathname) || Promise.resolve();
  const write = previous.then(async () => {
    const cache = await caches.open(CACHE_NAME);
    await cache.put(request, body);
    const keys = await cache.keys();
    await Promise.all(
      keys
        .filter((key) => new URL(key.url).pathname === pathname && key.url !== request.url)
        .map((key) => cache.delete(key))
    );
  });
  const settled = write.catch(() => {});
  scriptWrites.set(pathname, settled);
  settled.then(() => {
    if (scriptWrites.get(pathname) === settled) {
      scriptWrites.delete(pathname);
    }
  });
  return write;
}

/* Fetch — network-first for scripts, stale-while-revalidate for other static files,
   network-only for everything else */
self.addEventListener("fetch", (event) => {
  const url = new URL(event.request.url);

  /* Only handle same-origin GET requests */
  if (event.request.method !== "GET" || url.origin !== self.location.origin) {
    return;
  }

  /* Scripts — network first. A page and the script it calls are released together, so a
     cached copy served first (stale-while-revalidate) pairs a new page with an old script
     after every update and breaks it until the next reload. Fall back to the cache when the
     network fails or answers 5xx (a broken deploy or proxy error page must not replace a
     working script); with nothing cached, the 5xx response is passed on as is. */
  if (url.pathname.startsWith("/static/") && url.pathname.endsWith(".js")) {
    event.respondWith(
      fetch(event.request)
        .then((response) => {
          if (response && response.ok) {
            const copy = response.clone();
            /* Templates ask for a new ?v=<release> URL after every update, so the cache
               would keep one copy per release for ever. Keep only the version just
               fetched. waitUntil lets the worker finish the write after the response
               has been handed to the page. */
            event.waitUntil(storeScriptAndPrune(url.pathname, event.request, copy));
            return response;
          }
          if (response && response.status >= 500) {
            return caches.match(event.request).then((cached) => cached || response);
          }
          return response;
        })
        .catch(() => caches.match(event.request))
    );
    return;
  }

  /* Static assets — stale-while-revalidate */
  if (url.pathname.startsWith("/static/")) {
    event.respondWith(
      caches.open(CACHE_NAME).then((cache) =>
        cache.match(event.request).then((cached) => {
          // Always try to update the cache in the background
          const fetchPromise = fetch(event.request)
            .then((response) => {
              if (response && response.ok) {
                cache.put(event.request, response.clone());
              }
              return response;
            })
            .catch(() => cached);

          // If we have something cached, return it immediately; otherwise wait for network
          return cached || fetchPromise;
        })
      )
    );
    return;
  }

  /* Everything else — network only (auth-gated, live data) */
});
