// Makes the page cross-origin isolated on GitHub Pages, which cannot set
// response headers. Isolation enables SharedArrayBuffer, so the CPU (WASM)
// fallback can run the model on several threads instead of one.
//
// Adds COOP/COEP to same-origin responses. COEP is "credentialless", so the
// model (huggingface.co) and libraries (cdn.jsdelivr.net) still load: they are
// fetched without cookies, which they do not need. Nothing is cached here.
self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", (event) => event.waitUntil(self.clients.claim()));

self.addEventListener("fetch", (event) => {
  const request = event.request;
  if (new URL(request.url).origin !== self.location.origin) return; // cross-origin: browser default
  if (request.cache === "only-if-cached" && request.mode !== "same-origin") return;
  event.respondWith(
    fetch(request).then((response) => {
      if (response.status === 0) return response;
      const headers = new Headers(response.headers);
      headers.set("Cross-Origin-Opener-Policy", "same-origin");
      headers.set("Cross-Origin-Embedder-Policy", "credentialless");
      headers.set("Cross-Origin-Resource-Policy", "same-origin");
      return new Response(response.body, { status: response.status, statusText: response.statusText, headers });
    }),
  );
});
