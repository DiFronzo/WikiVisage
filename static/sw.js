/*
 * WikiVisage Service Worker
 *
 * Strategy:
 *   - Static assets (icons, logos): cache-first with long TTL
 *   - All other requests (routes, API): network-only (app requires auth + live data)
 *
 * This SW exists primarily to enable PWA install (Add to Home Screen)
 * and cache static assets for faster repeat loads. The app is not
 * designed for offline use (requires DB, OAuth, Commons API).
 */

const CACHE_NAME = "wikivisage-v1";

const PRECACHE_URLS = [
  "/static/wikivisage-logo.svg",
  "/static/wikivisage-logo-notext.svg",
  "/static/icon-192.png",
  "/static/icon-512.png"
];

/* Install — precache static assets */
self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(PRECACHE_URLS))
  );
  self.skipWaiting();
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

/* Fetch — cache-first for static assets, network-only for everything else */
self.addEventListener("fetch", (event) => {
  const url = new URL(event.request.url);

  /* Only handle same-origin GET requests */
  if (event.request.method !== "GET" || url.origin !== self.location.origin) {
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
